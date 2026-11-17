"""Build (or resume building) a puzzle's ladder and test it on random scrambles.

    python scripts/build_ladder.py --ladder table --radius 7 --test 200              # Maltese Gear Cube
    python scripts/build_ladder.py --puzzle tangram_extreme --ladder table --test 100
    python scripts/build_ladder.py --puzzle tangram_extreme --ladder learned           # its table rungs only

Each rung's actions and exact table are cached under --cache (default: data/ladders/<ladder>_r<radius>
for the Maltese Gear Cube, data/ladders/<puzzle>/<ladder>_r<radius> for the others), so an interrupted
build resumes at the first unfinished rung. Rungs in "learned" mode are skipped here; train them with
scripts/train_rung.py.
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
from puzzles.registry import PUZZLES, get_puzzle  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puzzle", choices=sorted(PUZZLES), default="maltese")
    ap.add_argument("--ladder", default="table", help="ladder name (see the puzzle's ladder module)")
    ap.add_argument("--radius", type=int, default=7, help="radius of the primitive ball used for meet in the middle")
    ap.add_argument("--cache", default=None, help="cache directory (default: see above)")
    ap.add_argument("--upto", type=int, default=None, help="build only the first N rungs")
    ap.add_argument("--test", type=int, default=100, help="random scrambles to solve afterwards")
    ap.add_argument("--scramble-len", type=int, default=None, help="length of each random scramble (default: the puzzle's)")
    args = ap.parse_args()

    entry = get_puzzle(args.puzzle)
    ladders = entry.ladders()
    if args.ladder not in ladders:
        ap.error(f"{entry.title} has the ladders {sorted(ladders)}")
    puzzle = entry.build()
    model = build_point_model(puzzle)
    structure = analyse(model)
    print(structure.describe(), flush=True)
    cache = Path(args.cache) if args.cache else entry.cache_dir(args.ladder, args.radius)

    ladder = Ladder(model, structure, ladders[args.ladder], mitm_radius=args.radius)
    t0 = time.time()
    ladder.build(cache_dir=cache, upto=args.upto)
    print(f"build finished in {time.time() - t0:.0f} s; cache in {cache}", flush=True)

    if args.test and all(r.table is not None or r.heuristic is not None for r in ladder.rungs):
        notation = entry.notation(ladder.alg)
        unit = notation.metrics[0][0]
        rng = np.random.default_rng(0)
        rows, failures = [], 0
        for _ in range(args.test):
            scramble = rng.integers(0, model.num_moves, size=args.scramble_len or entry.scramble_len).tolist()
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
            print(f"\nsolved {len(rows)}/{args.test} scrambles; mean {unit} per rung:")
            for rung, mean in zip(ladder.rungs, table[:, :-1].mean(0)):
                print(f"  {rung.spec.name:14s} {mean:7.1f}")
            print(f"  {'total':14s} {table[:, -1].mean():7.1f}  (after merging turns across rungs; min {table[:, -1].min()}, max {table[:, -1].max()})")
        print(f"failures: {failures}")


if __name__ == "__main__":
    main()
