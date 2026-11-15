"""Time each stage of one training iteration to find where the GPU time goes.

Example:
    python scripts/profile_train.py --walks 4096 --length 60 --batch-size 8192
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.ball import GoalBall  # noqa: E402
from core.beam import OneHot  # noqa: E402
from core.train import DistanceMLP, ExactLabeler, nonbacktracking_walks  # noqa: E402
from core.torch_engine import TorchGatherPuzzle  # noqa: E402
from puzzles.maltese_gear.puzzle import build_maltese_gear_puzzle, cardinality  # noqa: E402


def timed(fn, repeat: int = 3) -> float:
    """Median wall time in ms, synchronising the GPU before and after each call."""
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--walks", type=int, default=4096)
    parser.add_argument("--length", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--ball-dir", type=Path, default=Path("data/ball"))
    parser.add_argument("--ball-depth", type=int, default=12)
    args = parser.parse_args()

    device = "cuda"
    tp = TorchGatherPuzzle(build_maltese_gear_puzzle(), device=device)
    card = cardinality()
    ball = GoalBall(tp, card, args.ball_dir, depth=args.ball_depth)
    ball.build(log=lambda _: None)
    labeler = ExactLabeler(ball)
    onehot = OneHot(card, device)
    model = DistanceMLP(onehot.width, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # One full iteration's inputs, reused for the timing below.
    states_t, _ = nonbacktracking_walks(tp, args.walks, args.length)
    flat = states_t.reshape(-1, tp.dim)
    steps = torch.arange(args.length + 1, device=device).repeat_interleave(args.walks)
    labels = labeler(flat, steps)
    perm = torch.randperm(flat.shape[0], device=device)
    idx = perm[: args.batch_size]

    def walks_only():
        nonbacktracking_walks(tp, args.walks, args.length)

    def labels_only():
        labeler(flat, steps)

    def onehot_only():
        onehot(flat[idx])

    scaler = torch.amp.GradScaler("cuda")

    def mlp_train_step():
        # Same as core/train.py: fp16 autocast with gradient scaling.
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = model(onehot(flat[idx]))
        loss = torch.nn.functional.mse_loss(pred.float(), labels[idx])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    def mlp_train_step_fp32():
        pred = model(onehot(flat[idx]))
        loss = torch.nn.functional.mse_loss(pred, labels[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    def mlp_train_step_with_sync():
        mlp_train_step()
        float(labels[idx][0])  # the old per-batch loss.item() pattern forces this sync

    samples = flat.shape[0]
    batches = (samples + args.batch_size - 1) // args.batch_size
    results = {
        "walk generation (per iteration)": timed(walks_only),
        "labels (per iteration)": timed(labels_only),
        "one-hot encode (per batch)": timed(onehot_only),
        "MLP train step fp32 (per batch)": timed(mlp_train_step_fp32),
        "MLP train step fp16 (per batch)": timed(mlp_train_step),
        "MLP fp16 + host sync (per batch)": timed(mlp_train_step_with_sync),
    }
    print(f"samples per iteration: {samples:,d}, batches: {batches}")
    for name, ms in results.items():
        print(f"  {name:42s} {ms:9.2f} ms")
    per_iter_mlp = results["MLP train step fp16 (per batch)"] * batches
    total = results["walk generation (per iteration)"] + results["labels (per iteration)"] + per_iter_mlp
    print(f"estimated iteration: {total:,.0f} ms, of which MLP {100 * per_iter_mlp / total:.0f}%, "
          f"walks {100 * results['walk generation (per iteration)'] / total:.0f}%, "
          f"labels {100 * results['labels (per iteration)'] / total:.0f}%")


if __name__ == "__main__":
    main()
