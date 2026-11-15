"""Throughput of the GPU gather engine on the Maltese Gear Cube.

Reports state-move applications per second (one application = one state under one move).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.gather_engine import random_walk_states  # noqa: E402
from core.torch_engine import TorchGatherPuzzle  # noqa: E402
from puzzles.maltese_gear.puzzle import build_maltese_gear_puzzle  # noqa: E402


def bench(batch: int, reps: int, trials: int = 5) -> tuple[float, float, float]:
    """Return (median, min, max) state-move applications/s over several trials."""
    puzzle = build_maltese_gear_puzzle()
    gpu = TorchGatherPuzzle(puzzle, device="cuda")
    states = torch.as_tensor(random_walk_states(puzzle, batch, 20, np.random.default_rng(0)), device="cuda")
    for m in range(puzzle.num_moves):  # warm-up
        gpu.apply(states, m)
    torch.cuda.synchronize()
    rates = []
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(reps):
            for m in range(puzzle.num_moves):
                gpu.apply(states, m)
        torch.cuda.synchronize()
        rates.append(batch * puzzle.num_moves * reps / (time.perf_counter() - t0))
    return float(np.median(rates)), min(rates), max(rates)


def main() -> None:
    print(f"device: {torch.cuda.get_device_name(0)}")
    for batch in (2**16, 2**18, 2**20):
        med, lo, hi = bench(batch, reps=20)
        print(f"batch {batch:>8}: median {med:,.0f}  (range {lo:,.0f} to {hi:,.0f}) state-move applications / s")


if __name__ == "__main__":
    main()
