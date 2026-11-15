"""Scramble the Maltese Gear Cube, solve it with the streamed beam search, and replay the result.

Example:
    python scripts/solve_demo.py --depth 6 --beam-exp 16

The heuristic here is only the number of stickers that differ from the goal. It is a
placeholder until the learned heuristic exists, so expect solutions only for shallow scrambles.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.ball import GoalBall  # noqa: E402
from core.gather_engine import random_walk_states  # noqa: E402
from core.hashing import make_hash_vec  # noqa: E402
from core.streamed_beam import StreamedBeam  # noqa: E402
from core.torch_engine import TorchGatherPuzzle  # noqa: E402
from puzzles.maltese_gear.puzzle import build_maltese_gear_puzzle, cardinality  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=6, help="scramble length in moves")
    parser.add_argument("--beam-exp", type=int, default=16, help="beam width = 2**beam-exp")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ball-depth", type=int, default=0, help="use the goal ball up to this depth (0 = off)")
    parser.add_argument("--ball-dir", type=Path, default=Path("data/ball"))
    parser.add_argument("--hash-seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    puzzle = build_maltese_gear_puzzle()
    tp = TorchGatherPuzzle(puzzle, device=args.device)
    start = random_walk_states(puzzle, 1, args.depth, np.random.default_rng(args.seed))[0]
    goal_t = tp.goal

    def mismatch(states: torch.Tensor) -> torch.Tensor:
        return (states != goal_t).sum(1).float()

    ball = None
    if args.ball_depth > 0:
        ball = GoalBall(tp, cardinality(), args.ball_dir, depth=args.ball_depth, hash_seed=args.hash_seed)
        ball.build(log=lambda _: None)  # loads saved layers; builds any that are missing
        print(f"goal ball of depth {args.ball_depth}: {sum(ball.sizes()):,d} states")
    hash_vec = make_hash_vec(puzzle.dim, args.hash_seed, args.device)
    beam = StreamedBeam(tp, cardinality(), mismatch, hash_vec, width=2**args.beam_exp, ball=ball)

    t0 = time.perf_counter()
    moves = beam.solve(start, max_steps=args.max_steps)
    elapsed = time.perf_counter() - t0

    if moves is None:
        print(f"not solved within {args.max_steps} steps ({elapsed:.1f} s)")
        return
    print(f"solved in {len(moves)} moves ({elapsed:.1f} s): {puzzle.word_str(moves)}")
    replay = puzzle.apply_word(start, moves)
    print("replay on reference simulator:", "goal reached" if np.array_equal(replay, puzzle.goal) else "FAILED")


if __name__ == "__main__":
    main()
