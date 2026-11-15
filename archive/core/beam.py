"""Batched beam search on a GatherPuzzle, with per-stage timing.

One step: expand every beam state under every move, deduplicate by a 64-bit hash,
score the survivors, keep the best `width`.

Dedup is exact up to hash collisions. A collision can only drop a candidate, never
produce a wrong answer, because every returned solution is replayed on the simulator.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator

import torch

from core.torch_engine import TorchGatherPuzzle

ScoreFn = Callable[[torch.Tensor], torch.Tensor]


class NullTimer:
    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        yield


class CudaStageTimer:
    """Times named stages with CUDA events. Call totals_ms() to synchronise and read them."""

    def __init__(self) -> None:
        self._pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        self._pending.append((name, start, end))

    def totals_ms(self) -> dict[str, float]:
        torch.cuda.synchronize()
        totals: dict[str, float] = {}
        for name, start, end in self._pending:
            totals[name] = totals.get(name, 0.0) + start.elapsed_time(end)
        self._pending.clear()
        return totals


class OneHot:
    """One-hot features. cardinality[i] is how many values coordinate i can take."""

    def __init__(self, cardinality, device: str | torch.device) -> None:
        card = torch.as_tensor(cardinality, dtype=torch.long, device=device)
        self.offsets = torch.cumsum(card, 0) - card
        self.width = int(card.sum())

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        n = states.shape[0]
        idx = states.to(torch.long) + self.offsets
        out = torch.zeros((n, self.width), dtype=torch.float32, device=states.device)
        out.scatter_(1, idx, 1.0)
        return out


@torch.no_grad()
def beam_step(
    tp: TorchGatherPuzzle,
    beam: torch.Tensor,
    score_fn: ScoreFn,
    hash_vec: torch.Tensor,
    width: int | None = None,
    chunk: int = 1 << 17,
    timer=None,
) -> torch.Tensor:
    """Expand, dedupe, score and select. Lower score is better.

    beam: (B, D) uint8 on tp.device. Returns (min(width, unique), D) uint8.
    chunk bounds the working memory of the hash and score stages.
    """
    timer = timer or NullTimer()
    b = beam.shape[0]
    width = b if width is None else width
    n = tp.num_moves * b
    device = beam.device

    with timer.stage("expand"):
        cand = torch.empty((n, tp.dim), dtype=torch.uint8, device=device)
        for m in range(tp.num_moves):
            cand[m * b:(m + 1) * b] = tp.apply(beam, m)

    with timer.stage("hash"):
        keys = torch.empty(n, dtype=torch.int64, device=device)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            keys[s:e] = (cand[s:e].to(torch.int64) * hash_vec).sum(1)

    with timer.stage("dedup"):
        order = torch.argsort(keys)
        sorted_keys = keys[order]
        first = torch.ones(n, dtype=torch.bool, device=device)
        first[1:] = sorted_keys[1:] != sorted_keys[:-1]
        keep = order[first]
        del order, sorted_keys, first, keys
        cand = cand[keep]

    with timer.stage("score"):
        count = cand.shape[0]
        scores = torch.empty(count, dtype=torch.float32, device=device)
        for s in range(0, count, chunk):
            e = min(s + chunk, count)
            scores[s:e] = score_fn(cand[s:e]).reshape(-1).float()

    with timer.stage("select"):
        k = min(width, count)
        top = torch.topk(scores, k, largest=False).indices
        return cand[top]
