"""Build (or resume building) a ladder for the Maltese Gear Cube and test it on random scrambles.

    python scripts/build_ladder.py --ladder table --radius 7 --test 200

Each rung's actions and exact table are cached under --cache, so an interrupted build
resumes at the first unfinished rung. Rungs in "learned" mode are skipped here; train them
with scripts/train_rung.py.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.ladder import Ladder  # noqa: E402
from core.perm_model import build_point_model  # noqa: E402
from core.structure import analyse  # noqa: E402
from puzzles.maltese_gear.ladder import LADDERS  # noqa: E402
from puzzles.maltese_gear.puzzle import build_maltese_gear_puzzle  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ladder", choices=sorted(LADDERS), default="table")
    ap.add_argument("--radius", type=int, default=7, help="radius of the primitive ball used for meet in the middle")
    ap.add_argument("--cache", default=None, help="cache directory (default: data/ladders/<ladder>_r<radius>)")
    ap.add_argument("--upto", type=int, default=None, help="build only the first N rungs")
    ap.add_argument("--test", type=int, default=100, help="random scrambles to solve afterwards")
    ap.add_argument("--scramble-len", type=int, default=150)
    args = ap.parse_args()

    puzzle = build_maltese_gear_puzzle()
    model = build_point_model(puzzle)
    structure = analyse(model)
    print(structure.describe(), flush=True)
    cache = Path(args.cache) if args.cache else Path("data/ladders") / f"{args.ladder}_r{args.radius}"

    ladder = Ladder(model, structure, LADDERS[args.ladder], mitm_radius=args.radius)
    t0 = time.time()
    ladder.build(cache_dir=cache, upto=args.upto)
    print(f"build finished in {time.time() - t0:.0f} s; cache in {cache}", flush=True)

    if args.test and all(r.table is not None or r.heuristic is not None for r in ladder.rungs):
        rng = np.random.default_rng(0)
        rows, failures = [], 0
        for _ in range(args.test):
            scramble = rng.integers(0, model.num_moves, size=args.scramble_len).tolist()
            state = puzzle.apply_word(puzzle.goal, scramble)
            try:
                word, per_rung = ladder.solve(state)
            except LookupError as err:
                failures += 1
                print("FAILED:", err, flush=True)
                continue
            if len(ladder.rungs) == len(ladder.specs):
                assert np.array_equal(puzzle.apply_word(state, list(word)), puzzle.goal), "solution does not reach the goal"
            rows.append(per_rung + [len(word)])
        if rows:
            table = np.array(rows)
            print(f"\nsolved {len(rows)}/{args.test} scrambles; mean quarter turns per rung:")
            for rung, mean in zip(ladder.rungs, table[:, :-1].mean(0)):
                print(f"  {rung.spec.name:14s} {mean:7.1f}")
            print(f"  {'total':14s} {table[:, -1].mean():7.1f}  (after merging adjacent turns; min {table[:, -1].min()}, max {table[:, -1].max()})")
        print(f"failures: {failures}")


if __name__ == "__main__":
    main()
