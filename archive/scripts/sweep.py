"""Sweep beam widths and record cost per configuration to CSV.

Example (local or Colab):
    python scripts/sweep.py --min-exp 16 --max-exp 22 --out results/sweep.csv

Configurations that exceed device memory are recorded with status OOM and the sweep continues.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.profiling import profile_beam  # noqa: E402
from core.torch_engine import TorchGatherPuzzle  # noqa: E402
from puzzles.maltese_gear.puzzle import build_maltese_gear_puzzle, cardinality  # noqa: E402

FIELDS = [
    "device", "beam_width", "status", "step_ms", "candidates_per_s",
    "expand_ms", "hash_ms", "dedup_ms", "score_ms", "select_ms",
    "peak_mem_gb", "gpu_util_mean", "gpu_util_max", "gpu_mem_mb_max",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-exp", type=int, default=16)
    parser.add_argument("--max-exp", type=int, default=22)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--chunk-exp", type=int, default=17)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--out", type=Path, default=Path("results/sweep.csv"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device_name = torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu"
    tp = TorchGatherPuzzle(build_maltese_gear_puzzle(), device=args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for exp in range(args.min_exp, args.max_exp + 1):
        res = profile_beam(
            tp, cardinality(), beam_width=2**exp, steps=args.steps,
            chunk=2**args.chunk_exp, hidden=args.hidden,
        )
        stage = res.get("stage_ms", {})
        row = {
            "device": device_name,
            "beam_width": 2**exp,
            "status": res["status"],
            "step_ms": res.get("step_ms"),
            "candidates_per_s": res.get("candidates_per_s"),
            "expand_ms": stage.get("expand"),
            "hash_ms": stage.get("hash"),
            "dedup_ms": stage.get("dedup"),
            "score_ms": stage.get("score"),
            "select_ms": stage.get("select"),
            "peak_mem_gb": res.get("peak_mem_gb"),
            "gpu_util_mean": res.get("gpu_util_mean"),
            "gpu_util_max": res.get("gpu_util_max"),
            "gpu_mem_mb_max": res.get("gpu_mem_mb_max"),
        }
        rows.append(row)
        step = f"{row['step_ms']:.0f} ms" if row["step_ms"] else "-"
        rate = f"{row['candidates_per_s']:,.0f}/s" if row["candidates_per_s"] else "-"
        print(f"2^{exp:<2} {row['status']:<4} step {step:>10}  rate {rate:>16}  "
              f"peak {row['peak_mem_gb'] or 0:5.2f} GB  util {row['gpu_util_mean'] or 0:4.0f}%", flush=True)

    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
