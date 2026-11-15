"""Solve rate by scramble depth: learned heuristic vs. a simple baseline.

Each scramble is a random non-backtracking walk of the given length from the goal. The goal
ball finishes any path that reaches it, so a solve only needs the beam to get within the ball.

Example:
    python scripts/eval_solve.py --checkpoint checkpoints/check_fp16_300.pt --n 20
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.ball import GoalBall  # noqa: E402
from core.beam import OneHot  # noqa: E402
from core.gather_engine import random_walk_states  # noqa: E402
from core.hashing import make_hash_vec  # noqa: E402
from core.streamed_beam import StreamedBeam  # noqa: E402
from core.torch_engine import TorchGatherPuzzle  # noqa: E402
from core.train import DistanceMLP  # noqa: E402
from puzzles.maltese_gear.puzzle import build_maltese_gear_puzzle, cardinality  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/check_fp16_300.pt"))
    parser.add_argument("--depths", default="13,16,20,24,30")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--beam-exp", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--ball-depth", type=int, default=12)
    parser.add_argument("--ball-dir", type=Path, default=Path("data/ball"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tp = TorchGatherPuzzle(build_maltese_gear_puzzle(), device=args.device)
    card = cardinality()
    ball = GoalBall(tp, card, args.ball_dir, depth=args.ball_depth)
    ball.build(log=lambda _: None)
    hash_vec = make_hash_vec(tp.dim, ball.hash_seed, args.device)

    onehot = OneHot(card, args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model = DistanceMLP(onehot.width, hidden=ckpt["hidden"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    use_amp = args.device == "cuda"

    @torch.no_grad()
    def learned(states: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            return model(onehot(states)).float()

    def mismatch(states: torch.Tensor) -> torch.Tensor:
        return (states != tp.goal).sum(1).float()

    depths = [int(x) for x in args.depths.split(",")]
    print(f"checkpoint {args.checkpoint} (iter {ckpt['iter']}), beam 2^{args.beam_exp}, "
          f"ball depth {args.ball_depth}, {args.n} scrambles per depth")
    print(f"{'depth':>5}  {'heuristic':>10}  {'solved':>7}  {'median len':>10}  {'median s':>9}")
    for d in depths:
        starts = random_walk_states(tp.puzzle, args.n, d, np.random.default_rng(args.seed + d))
        for name, score in (("learned", learned), ("mismatch", mismatch)):
            beam = StreamedBeam(tp, card, score, hash_vec, width=2**args.beam_exp, ball=ball)
            lengths, times, solved = [], [], 0
            for start in starts:
                t0 = time.perf_counter()
                moves = beam.solve(start, max_steps=args.max_steps)
                times.append(time.perf_counter() - t0)
                if moves is not None:
                    solved += 1
                    lengths.append(len(moves))
            med_len = f"{statistics.median(lengths):.0f}" if lengths else "-"
            print(f"{d:5d}  {name:>10}  {solved:3d}/{args.n:<3d}  {med_len:>10}  "
                  f"{statistics.median(times):9.2f}", flush=True)


if __name__ == "__main__":
    main()
