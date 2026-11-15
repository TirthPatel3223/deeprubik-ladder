"""Training pipeline: walk generator, exact labels, and a short training run (CPU)."""
from __future__ import annotations

import torch

from core.ball import GoalBall
import copy

from core.beam import OneHot
from core.train import DistanceMLP, ExactLabeler, bellman_target, nonbacktracking_walks, train
from core.torch_engine import TorchGatherPuzzle
from puzzles.maltese_gear.puzzle import build_maltese_gear_puzzle, cardinality

PUZZLE = build_maltese_gear_puzzle()
TP = TorchGatherPuzzle(PUZZLE, device="cpu")
CARD = cardinality()


def _silent(_: str) -> None:
    pass


def test_walks_never_undo_the_previous_move() -> None:
    states, moves = nonbacktracking_walks(TP, 500, 20)
    inv = TP.inverse
    for t in range(1, moves.shape[0]):
        prev = moves[t - 1].long()
        cur = moves[t].long()
        assert bool((cur != inv[prev]).all()), f"walk undid its previous move at step {t}"


def test_walk_states_follow_their_moves() -> None:
    states, moves = nonbacktracking_walks(TP, 50, 10)
    for t in range(moves.shape[0]):
        for m in range(TP.num_moves):
            sel = moves[t] == m
            if bool(sel.any()):
                expected = TP.apply(states[t][sel], m)
                assert torch.equal(expected, states[t + 1][sel])


def test_labels_are_exact_inside_the_ball(tmp_path) -> None:
    ball = GoalBall(TP, CARD, tmp_path, depth=3)
    ball.build(log=_silent)
    labeler = ExactLabeler(ball)
    states, _ = nonbacktracking_walks(TP, 200, 10)
    flat = states.reshape(-1, PUZZLE.dim)
    steps = torch.arange(11).repeat_interleave(200)
    labels = labeler(flat, steps)
    for i in range(flat.shape[0]):
        depth = ball.depth_of(flat[i].numpy())
        if depth is not None:
            assert float(labels[i]) == float(depth)
        else:
            assert float(labels[i]) == float(steps[i])


def test_short_training_run_produces_finite_loss_and_checkpoint(tmp_path) -> None:
    ball = GoalBall(TP, CARD, tmp_path / "ball", depth=3)
    ball.build(log=_silent)
    logs: list[str] = []
    out = tmp_path / "model.pt"
    model = train(TP, CARD, ball, out, iters=2, walks=32, length=6, batch_size=64,
                  hidden=32, checkpoint_every=1, log=logs.append)
    assert out.exists()
    assert len(logs) == 2
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_bellman_target_never_exceeds_walk_label_and_keeps_exact_labels(tmp_path) -> None:
    ball = GoalBall(TP, CARD, tmp_path, depth=3)
    ball.build(log=_silent)
    labeler = ExactLabeler(ball)
    onehot = OneHot(CARD, "cpu")
    net = DistanceMLP(onehot.width, hidden=16)
    target_net = copy.deepcopy(net).eval()
    states, _ = nonbacktracking_walks(TP, 100, 8)
    flat = states.reshape(-1, PUZZLE.dim)
    steps = torch.arange(9).repeat_interleave(100)
    labels, exact = labeler(flat, steps, return_mask=True)
    refined = bellman_target(labels, exact, flat, target_net, TP, onehot, use_amp=False)
    assert bool((refined <= labels + 1e-5).all())
    assert torch.equal(refined[exact], labels[exact])


def test_short_run_with_bellman_refinement_is_finite(tmp_path) -> None:
    ball = GoalBall(TP, CARD, tmp_path / "ball", depth=3)
    ball.build(log=_silent)
    model = train(TP, CARD, ball, tmp_path / "bellman.pt", iters=3, walks=32, length=6, batch_size=64,
                  hidden=32, checkpoint_every=1, log=_silent_log, bellman=True, bellman_start=1,
                  target_sync=1)
    assert all(torch.isfinite(p).all() for p in model.parameters())


def _silent_log(_: str) -> None:
    pass
