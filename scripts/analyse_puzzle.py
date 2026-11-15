"""Describe a puzzle's structure and propose a rung order, from its move tables alone.

    python scripts/analyse_puzzle.py --puzzle maltese
    python scripts/analyse_puzzle.py --puzzle toy --radius 4

Prints the orbits and pieces found, the features that could serve as rungs, and the rung
order chosen by the measured rules of core/discover.py (generator periods, shortest pure
macros, bits resolved per macro length).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.discover import generator_periods, linked_orbits, propose_ladder  # noqa: E402
from core.perm_model import build_point_model  # noqa: E402
from core.structure import analyse  # noqa: E402


def load(name: str):
    if name == "maltese":
        from puzzles.maltese_gear.puzzle import build_maltese_gear_puzzle

        return build_maltese_gear_puzzle()
    if name == "toy":
        from puzzles.toy.two_face_corners import build_two_face_corners

        return build_two_face_corners()
    raise SystemExit(f"unknown puzzle {name!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puzzle", choices=["maltese", "toy"], default="maltese")
    ap.add_argument("--radius", type=int, default=6, help="radius of the ball of primitive words searched for macros")
    ap.add_argument("--budget", type=int, default=10_000_000, help="largest coset count that gets an exact table")
    ap.add_argument("--no-order", action="store_true", help="only describe the structure")
    args = ap.parse_args()

    puzzle = load(args.puzzle)
    model = build_point_model(puzzle)
    structure = analyse(model)
    print(f"puzzle: {puzzle.name}, {puzzle.num_moves} moves ({' '.join(puzzle.move_names)}), state length {puzzle.dim}")
    print(structure.describe())
    print("orbits that move as the same pieces:", linked_orbits(model, structure))
    print("generator periods per orbit:", generator_periods(model, structure))
    print("candidate rung features (name, slots, values per slot):")
    for f in structure.candidate_features():
        print(f"  {f.name:18s} {f.n_slots:3d} slots, {sorted(set(f.radix.tolist()))} values")
    if args.no_order:
        return
    print("\nrung order by measurement:")
    t0 = time.time()
    specs, _ = propose_ladder(model, structure, radius=args.radius, table_budget=args.budget)
    print(f"\nproposed ladder ({time.time() - t0:.0f} s):")
    for i, spec in enumerate(specs):
        print(f"  rung {i}: {spec.name:18s} features {spec.features}  heuristic: {spec.mode}")


if __name__ == "__main__":
    main()
