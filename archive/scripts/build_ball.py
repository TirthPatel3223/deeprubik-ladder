"""Build (or resume) the goal ball on disk and report its layer sizes and timings.

Example:
    python scripts/build_ball.py --depth 12 --dir data/ball

Layers are written as they finish, so an interrupted build continues from the last saved layer.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.ball import GoalBall  # noqa: E402
from core.torch_engine import TorchGatherPuzzle  # noqa: E402
from puzzles.maltese_gear.puzzle import build_maltese_gear_puzzle, cardinality  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--dir", type=Path, default=Path("data/ball"))
    parser.add_argument("--hash-seed", type=int, default=12345)
    parser.add_argument("--chunk-exp", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tp = TorchGatherPuzzle(build_maltese_gear_puzzle(), device=args.device)
    ball = GoalBall(tp, cardinality(), args.dir, depth=args.depth, hash_seed=args.hash_seed, chunk=2**args.chunk_exp)
    t0 = time.perf_counter()
    ball.build()
    print(f"total build time {time.perf_counter() - t0:.1f} s")
    total = sum(ball.sizes())
    print(f"ball of depth {args.depth}: {total:,d} states")
    for k, size in enumerate(ball.sizes()):
        print(f"  depth {k:2d}: {size:>14,d}")


if __name__ == "__main__":
    main()
