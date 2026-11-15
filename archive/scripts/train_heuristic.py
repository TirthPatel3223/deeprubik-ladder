"""Train the distance heuristic against the goal ball, then report calibration.

Examples:
    python scripts/train_heuristic.py --iters 200            # short measurement run
    python scripts/train_heuristic.py --iters 5000 --hidden 1024

Checkpoints go to checkpoints/ (not versioned). Resuming from a checkpoint is not implemented yet;
each run starts from scratch.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.ball import GoalBall  # noqa: E402
from core.train import calibration, train  # noqa: E402
from core.torch_engine import TorchGatherPuzzle  # noqa: E402
from puzzles.maltese_gear.puzzle import build_maltese_gear_puzzle, cardinality  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--walks", type=int, default=4096)
    parser.add_argument("--length", type=int, default=60,
                        help="walk length; labels cannot exceed this, so it must cover the depths to predict")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--ball-dir", type=Path, default=Path("data/ball"))
    parser.add_argument("--ball-depth", type=int, default=12)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/heuristic.pt"))
    parser.add_argument("--bellman", action="store_true", help="refine targets with neighbour values")
    parser.add_argument("--bellman-start", type=int, default=300, help="iterations before refinement starts")
    parser.add_argument("--target-sync", type=int, default=100, help="iterations between target-net copies")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tp = TorchGatherPuzzle(build_maltese_gear_puzzle(), device=args.device)
    card = cardinality()
    ball = GoalBall(tp, card, args.ball_dir, depth=args.ball_depth)
    ball.build(log=lambda _: None)
    print(f"goal ball depth {args.ball_depth}: {sum(ball.sizes()):,d} states")
    if args.device == "cuda":
        print(f"device: {torch.cuda.get_device_name(0)}")

    t0 = time.perf_counter()
    model = train(tp, card, ball, args.out, iters=args.iters, walks=args.walks, length=args.length,
                  batch_size=args.batch_size, hidden=args.hidden, lr=args.lr,
                  checkpoint_every=args.checkpoint_every, bellman=args.bellman,
                  bellman_start=args.bellman_start, target_sync=args.target_sync)
    print(f"training finished in {time.perf_counter() - t0:.1f} s; checkpoint {args.out}")

    print("calibration: mean predicted value by random-walk length")
    for d, mean, std in calibration(model, tp, card, ball):
        print(f"  walk length {d:3d}: mean {mean:7.2f}  std {std:6.2f}")


if __name__ == "__main__":
    main()
