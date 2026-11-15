"""Train a distance heuristic: regression on non-backtracking random walks from the goal.

Labels:
  - A state inside the goal ball has its exact distance from the ball (no estimate).
  - Any other state gets the number of moves its walk took. That is an upper bound on its
    distance, because the walk may not be a shortest path.

Walks are generated on the GPU during training and never stored, so the amount of training
data is limited by compute, not by disk or RAM.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from core.beam import OneHot
from core.ball import GoalBall
from core.hashing import hash_states
from core.torch_engine import TorchGatherPuzzle

NO_MOVE = 6  # sentinel for "no previous move"; real moves are 0..5


class DistanceMLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 1024, depth: int = 3) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = [torch.nn.Linear(in_dim, hidden), torch.nn.ReLU()]
        for _ in range(depth - 1):
            layers += [torch.nn.Linear(hidden, hidden), torch.nn.ReLU()]
        layers.append(torch.nn.Linear(hidden, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@torch.no_grad()
def nonbacktracking_walks(tp: TorchGatherPuzzle, n: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """n walks of `length` moves from the goal, never undoing the previous move.

    Returns states (length+1, n, D) uint8 and moves (length, n) uint8, where moves[t] took
    states[t] to states[t+1].
    """
    device = tp.device
    inv = tp.inverse.to(device)
    states = tp.goal_batch(n)
    out_states = torch.empty((length + 1, n, tp.dim), dtype=torch.uint8, device=device)
    out_moves = torch.empty((length, n), dtype=torch.uint8, device=device)
    out_states[0] = states
    last = torch.full((n,), NO_MOVE, dtype=torch.long, device=device)
    for t in range(length):
        excluded = torch.where(last == NO_MOVE, torch.full_like(last, NO_MOVE), inv[last.clamp(max=5)])
        n_allowed = torch.where(last == NO_MOVE, torch.full_like(last, NO_MOVE), torch.full_like(last, 5))
        r = (torch.rand(n, device=device) * n_allowed).long().clamp_(max=n_allowed - 1)
        move = r + (r >= excluded).long()
        # Apply every move to every walk, then keep each walk's chosen move. No host syncs.
        candidates = torch.stack([tp.apply(states, m) for m in range(tp.num_moves)])  # (M, n, D)
        pick = move.view(1, n, 1).expand(1, n, tp.dim)
        states = candidates.gather(0, pick).squeeze(0)
        out_states[t + 1] = states
        out_moves[t] = move.to(torch.uint8)
        last = move
    return out_states, out_moves


class ExactLabeler:
    """Label states with exact ball depth when known, else with the walk step."""

    def __init__(self, ball: GoalBall) -> None:
        self.hash_vec = ball.hash_vec
        self.hashes, self.depths = ball.device_hashes_with_depth()

    @torch.no_grad()
    def __call__(self, states: torch.Tensor, steps: torch.Tensor, return_mask: bool = False):
        h = hash_states(states, self.hash_vec)
        idx = torch.searchsorted(self.hashes, h).clamp_(max=self.hashes.numel() - 1)
        found = self.hashes[idx] == h
        exact = self.depths[idx].float()
        labels = torch.where(found, exact, steps.float())
        return (labels, found) if return_mask else labels


@torch.no_grad()
def bellman_target(
    walk_label: torch.Tensor,
    is_exact: torch.Tensor,
    states: torch.Tensor,
    target_model: DistanceMLP,
    tp: TorchGatherPuzzle,
    onehot: OneHot,
    use_amp: bool,
) -> torch.Tensor:
    """Refined target: min(walk label, 1 + best neighbour value), except exact ball labels.

    The neighbour values come from a frozen copy of the network, so the target does not
    move with every gradient step.
    """
    neighbours = torch.stack([tp.apply(states, m) for m in range(tp.num_moves)])  # (M, B, D)
    flat_n = neighbours.reshape(-1, tp.dim)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        values = target_model(onehot(flat_n)).float().reshape(tp.num_moves, -1)
    backup = values.min(dim=0).values + 1.0
    refined = torch.minimum(walk_label, backup)
    return torch.where(is_exact, walk_label, refined)


def train(
    tp: TorchGatherPuzzle,
    cardinality: np.ndarray,
    ball: GoalBall,
    out_path: str | Path,
    iters: int,
    walks: int = 4096,
    length: int = 24,
    batch_size: int = 8192,
    hidden: int = 1024,
    lr: float = 1e-3,
    checkpoint_every: int = 50,
    log: Callable[[str], None] = print,
    seed: int = 0,
    amp: bool = True,
    bellman: bool = False,
    bellman_start: int = 300,
    target_sync: int = 100,
) -> DistanceMLP:
    torch.manual_seed(seed)
    device = tp.device
    onehot = OneHot(cardinality, device)
    model = DistanceMLP(onehot.width, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    labeler = ExactLabeler(ball)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    use_amp = amp and is_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    target_model = copy.deepcopy(model).eval()
    for p in target_model.parameters():
        p.requires_grad_(False)
    t0 = time.perf_counter()
    loss_sum = torch.zeros((), device=device)
    loss_count = 0
    for it in range(1, iters + 1):
        if it % target_sync == 0:
            target_model.load_state_dict(model.state_dict())
        states_t, _ = nonbacktracking_walks(tp, walks, length)
        flat = states_t.reshape(-1, tp.dim)
        steps = torch.arange(length + 1, device=device).repeat_interleave(walks)
        labels, exact = labeler(flat, steps, return_mask=True)
        use_bellman = bellman and it > bellman_start
        perm = torch.randperm(flat.shape[0], device=device)
        for s in range(0, flat.shape[0], batch_size):
            idx = perm[s:s + batch_size]
            target = labels[idx]
            if use_bellman:
                target = bellman_target(target, exact[idx], flat[idx], target_model, tp, onehot, use_amp)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                pred = model(onehot(flat[idx]))
            loss = torch.nn.functional.mse_loss(pred.float(), target)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.detach()  # stays on the GPU; read back only at checkpoints
            loss_count += 1
        if it % checkpoint_every == 0 or it == iters:
            mean_loss = float(loss_sum) / max(loss_count, 1)
            loss_sum.zero_()
            loss_count = 0
            peak = torch.cuda.max_memory_allocated() / 2**30 if is_cuda else float("nan")
            log(f"iter {it:6d}  loss {mean_loss:8.3f}  elapsed {time.perf_counter() - t0:8.1f} s  "
                f"peak VRAM {peak:5.2f} GB")
            torch.save({"model": model.state_dict(), "iter": it, "hidden": hidden,
                        "cardinality": cardinality.tolist(), "hash_seed": ball.hash_seed}, out_path)
    return model


@torch.no_grad()
def calibration(
    model: DistanceMLP,
    tp: TorchGatherPuzzle,
    cardinality: np.ndarray,
    ball: GoalBall,
    depths: tuple[int, ...] = (4, 8, 12, 16, 20, 24, 30, 40, 50, 60),
    n: int = 2000,
) -> list[tuple[int, float, float]]:
    """For each walk length d, the mean and std of the predicted value over fresh walks.

    A useful heuristic's mean should keep rising with d, including beyond the ball depth.
    """
    device = tp.device
    onehot = OneHot(cardinality, device)
    model.eval()
    states_t, _ = nonbacktracking_walks(tp, n, max(depths))
    rows = []
    for d in depths:
        pred = model(onehot(states_t[d])).float()
        rows.append((d, float(pred.mean()), float(pred.std())))
    model.train()
    return rows
