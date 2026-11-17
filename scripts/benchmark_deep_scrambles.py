"""Benchmark a puzzle's ladder solver on deep random scrambles (depth 2000 to 5000).

    python scripts/benchmark_deep_scrambles.py                                     # Maltese Gear Cube
    python scripts/benchmark_deep_scrambles.py --puzzle tangram_extreme --out runs/tangram_benchmark.json

Generates 100 states with scramble depths uniformly sampled in [min_len, max_len],
runs the full ladder solver on each, verifies every solution on the reference
simulator, and records comprehensive statistics. Progress and intermediate results
are saved to disk after every single scramble.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.ladder import Ladder
from core.perm_model import build_point_model
from core.structure import analyse
from puzzles.registry import PUZZLES, get_puzzle


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100, help="number of test states (default: 100)")
    ap.add_argument("--min-len", type=int, default=2000, help="min scramble depth (default: 2000)")
    ap.add_argument("--max-len", type=int, default=5000, help="max scramble depth (default: 5000)")
    ap.add_argument("--beam", type=int, default=4096, help="base beam width (default: 4096)")
    ap.add_argument("--weight", type=float, default=0.3, help="weight on g-cost in beam search f = weight * g + h (default: 0.3)")
    ap.add_argument("--puzzle", choices=sorted(PUZZLES), default="maltese")
    ap.add_argument("--ladder", default=None, help="ladder name (default: the puzzle's)")
    ap.add_argument("--radius", type=int, default=7)
    ap.add_argument("--cache", default=None, help="ladder cache (default: see scripts/build_ladder.py)")
    ap.add_argument("--seed", type=int, default=42, help="random seed for scrambles (default: 42)")
    ap.add_argument("--out", default="runs/benchmark_100_results.json", help="path to save JSON results")
    return ap.parse_args()


def main():
    args = parse_args()
    entry = get_puzzle(args.puzzle)
    ladders = entry.ladders()
    ladder_name = args.ladder or entry.default_ladder
    if ladder_name not in ladders:
        sys.exit(f"{entry.title} has the ladders {sorted(ladders)}")
    print("=" * 80)
    print(f"{entry.title} Benchmark: {args.n} states, depth {args.min_len}..{args.max_len}, beam {args.beam}")
    print("=" * 80)

    puzzle = entry.build()
    model = build_point_model(puzzle)
    structure = analyse(model)
    cache = Path(args.cache) if args.cache else entry.cache_dir(ladder_name, args.radius)

    print(f"Loading ladder from {cache}...")
    ladder = Ladder(model, structure, ladders[ladder_name], mitm_radius=args.radius, log=lambda _msg: None)
    ladder.build(cache_dir=cache)
    if not ladder.ready():
        missing = [r.spec.name for r in ladder.rungs if r.table is None and r.heuristic is None]
        sys.exit(f"Ladder in {cache} is incomplete; untrained rungs: {missing}")

    notation = entry.notation(ladder.alg)
    metric_names = [m for m, _ in notation.metrics]
    keys = [m.replace(" ", "_") for m in metric_names]
    names = [r.spec.name for r in ladder.rungs]
    print(f"Ladder ready with {len(names)} rungs: {', '.join(names)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-generate scrambles with reproducible seed
    rng = np.random.default_rng(args.seed)
    scramble_specs = []
    for i in range(args.n):
        depth = int(rng.integers(args.min_len, args.max_len + 1))
        moves = rng.integers(0, model.num_moves, size=depth).tolist()
        scramble_specs.append((depth, moves))

    results = []
    solved_count = 0
    total_time = 0.0

    print(f"\nStarting benchmark on {args.n} states...")
    print("-" * 80)

    for idx, (depth, scramble) in enumerate(scramble_specs):
        state = puzzle.apply_word(puzzle.goal, scramble)
        t0 = time.time()
        err_msg = None
        word, per_rung = None, None
        verified = False

        try:
            word, per_rung = ladder.solve(state, beam=args.beam, weight=args.weight)
            dt = time.time() - t0
            verified = np.array_equal(puzzle.apply_word(state, list(word)), puzzle.goal)
            if not verified:
                err_msg = "Verification failed on reference simulator"
        except Exception as e:
            dt = time.time() - t0
            err_msg = str(e)

        solved = (word is not None) and verified
        if solved:
            solved_count += 1
            lengths = notation.lengths(word)
            rung_str = ", ".join(f"{n}:{c}" for n, c in zip(names, per_rung))
            status_str = f"SOLVED in {dt:5.1f}s | {notation.describe(word)} | [{rung_str}]"
        else:
            lengths = {m: None for m in metric_names}
            status_str = f"FAILED in {dt:5.1f}s | reason: {err_msg}"

        total_time += dt
        solve_rate = 100.0 * solved_count / (idx + 1)
        avg_time = total_time / (idx + 1)

        print(f"[{idx + 1:3d}/{args.n}] Depth {depth:4d} | {status_str} | Rate: {solve_rate:5.1f}% ({solved_count}/{idx + 1}) | Avg: {avg_time:5.1f}s", flush=True)

        res_entry = {
            "index": idx + 1,
            "depth": depth,
            "solved": solved,
            "time_s": round(dt, 2),
            **{k: lengths[m] for k, m in zip(keys, metric_names)},
            "per_rung": per_rung,
            "verified": verified,
            "error": err_msg,
        }
        results.append(res_entry)

        # Periodically flush results to disk
        with open(out_path, "w") as f:
            json.dump({
                "puzzle": entry.name,
                "n_total": args.n,
                "n_completed": idx + 1,
                "solved_count": solved_count,
                "solve_rate_pct": round(solve_rate, 2),
                "total_time_s": round(total_time, 2),
                "mean_time_s": round(avg_time, 2),
                "beam": args.beam,
                "min_len": args.min_len,
                "max_len": args.max_len,
                "results": results,
            }, f, indent=2)

        # Free GPU caches between scrambles
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final summary
    solved_results = [r for r in results if r["solved"]]
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Total Scrambles: {args.n} (depths {args.min_len}..{args.max_len})")
    print(f"Solved:          {solved_count}/{args.n} ({100.0 * solved_count / args.n:.1f}%)")
    print(f"Total Wall Time: {total_time / 60.0:.2f} minutes")
    print(f"Average Time:    {total_time / args.n:.2f} s per scramble")

    if solved_results:
        times = np.array([r["time_s"] for r in solved_results])
        print()
        for k, m in zip(keys, metric_names):
            vals = np.array([r[k] for r in solved_results])
            print(f"Solution Length ({m.title() + '):':15s} Mean {vals.mean():.1f} ± {vals.std():.1f} (Min {vals.min()}, Max {vals.max()})")
        print(f"Solve Time (Solved only):        Mean {times.mean():.1f}s ± {times.std():.1f}s (Min {times.min():.1f}s, Max {times.max():.1f}s)")

        per_rung_arr = np.array([r["per_rung"] for r in solved_results])
        print(f"\nMean {metric_names[0].title()} Per Rung:")
        for name, mean_c in zip(names, per_rung_arr.mean(axis=0)):
            print(f"  {name:14s}: {mean_c:6.1f} {metric_names[0]}")

    print(f"\nFull detailed results saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
