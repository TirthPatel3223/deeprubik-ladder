"""Sweep the streamed beam width and record memory and cost to CSV.

Example:
    python scripts/sweep_streamed.py --min-exp 20 --max-exp 26 --out results/streamed_local.csv

A width that exceeds device memory is recorded as OOM and the sweep continues.
Run with the NVIDIA "Prefer No Sysmem Fallback" setting so that overflow is a real error.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.profile_streamed import profile_streamed  # noqa: E402
from core.torch_engine import TorchGatherPuzzle  # noqa: E402
from puzzles.maltese_gear.puzzle import build_maltese_gear_puzzle, cardinality  # noqa: E402

FIELDS = [
    "device", "beam_width", "status", "step_ms", "candidates_per_s",
    "peak_mem_gb", "gpu_util_mean", "gpu_util_max", "gpu_mem_mb_max", "kept",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-exp", type=int, default=20)
    parser.add_argument("--max-exp", type=int, default=26)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--beam-chunk-exp", type=int, default=17)
    parser.add_argument("--score-chunk-exp", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--no-amp", action="store_true", help="float32 heuristic instead of float16")
    parser.add_argument("--out", type=Path, default=Path("results/streamed_local.csv"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device_name = torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu"
    tp = TorchGatherPuzzle(build_maltese_gear_puzzle(), device=args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for exp in range(args.min_exp, args.max_exp + 1):
        res = profile_streamed(
            tp, cardinality(), width=2**exp, steps=args.steps,
            beam_chunk=2**args.beam_chunk_exp, score_chunk=2**args.score_chunk_exp,
            hidden=args.hidden, amp=not args.no_amp,
        )
        row = {
            "device": device_name,
            "beam_width": 2**exp,
            "status": res["status"],
            "step_ms": res.get("step_ms"),
            "candidates_per_s": res.get("candidates_per_s"),
            "peak_mem_gb": res.get("peak_mem_gb"),
            "gpu_util_mean": res.get("gpu_util_mean"),
            "gpu_util_max": res.get("gpu_util_max"),
            "gpu_mem_mb_max": res.get("gpu_mem_mb_max"),
            "kept": res.get("kept"),
        }
        rows.append(row)
        step = f"{row['step_ms']:.0f} ms" if row["step_ms"] else "-"
        rate = f"{row['candidates_per_s']:,.0f}/s" if row["candidates_per_s"] else "-"
        peak = f"{row['peak_mem_gb']:.2f} GB" if row["peak_mem_gb"] else "-"
        print(f"2^{exp:<2} {row['status']:<4} step {step:>10}  rate {rate:>16}  peak {peak:>9}", flush=True)
        if row["status"] == "OOM":
            break  # larger widths cannot fit either

    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
