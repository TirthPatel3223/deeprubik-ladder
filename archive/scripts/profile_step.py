"""Profile one beam-search configuration: per-stage time, peak memory, GPU utilisation.

Example:
    python scripts/profile_step.py --beam-exp 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.profiling import profile_beam  # noqa: E402
from core.torch_engine import TorchGatherPuzzle  # noqa: E402
from puzzles.maltese_gear.puzzle import build_maltese_gear_puzzle, cardinality  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam-exp", type=int, default=20, help="beam width = 2**beam-exp")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--chunk-exp", type=int, default=17, help="chunk size = 2**chunk-exp")
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--amp", action="store_true", help="float16 heuristic under autocast (CUDA only)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tp = TorchGatherPuzzle(build_maltese_gear_puzzle(), device=args.device)
    if args.device == "cuda":
        print(f"device: {torch.cuda.get_device_name(0)}")
    res = profile_beam(
        tp, cardinality(), beam_width=2**args.beam_exp, steps=args.steps,
        chunk=2**args.chunk_exp, hidden=args.hidden, amp=args.amp,
    )

    print(f"beam width 2^{args.beam_exp} = {2**args.beam_exp:,}  status={res['status']}")
    if res["status"] != "ok":
        return
    print(f"step time        {res['step_ms']:.1f} ms")
    print(f"candidates / s   {res['candidates_per_s']:,.0f}")
    total = sum(res["stage_ms"].values()) or 1.0
    print("per-stage time (ms, share of stages):")
    for name, ms in res["stage_ms"].items():
        print(f"  {name:>7}  {ms:9.1f}  {100 * ms / total:5.1f}%")
    if res.get("peak_mem_gb") is not None:
        print(f"peak GPU memory  {res['peak_mem_gb']:.2f} GB")
    if res.get("gpu_util_mean") is not None:
        print(f"GPU utilisation  mean {res['gpu_util_mean']:.0f}%  max {res['gpu_util_max']:.0f}%")


if __name__ == "__main__":
    main()
