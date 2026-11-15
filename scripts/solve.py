"""Solve Maltese Gear Cube scrambles with a built ladder, and verify every solution.

    python scripts/solve.py --scramble "U L F' U2 L'3 F"        # one scramble, in move notation
    python scripts/solve.py --random 50 --beam 4096             # statistics over random scrambles

Moves are U, L, F and their inverses U', L', F'. A number after a move repeats it (U2, F'3).
Every solution is replayed on the reference simulator (the verified move tables) before it
is reported; a solution that does not reach the solved state is an error, not a result.

Lengths are reported in quarter turns, and in face turns (a single or double turn of a face
counts one, a triple or quadruple turn two), the metric used by other solvers of this puzzle.
"""
from __future__ import annotations

import argparse
import re
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


def parse_word(text: str, move_names: tuple[str, ...]) -> list[int]:
    """'U L2 F'3' -> move indices. The count follows the move name, after the prime if any."""
    out: list[int] = []
    for token in text.split():
        match = re.fullmatch(r"([A-Za-z]'?)(\d*)", token)
        if match is None or match.group(1) not in move_names:
            raise ValueError(f"cannot read move {token!r}; moves are {' '.join(move_names)}")
        out.extend([move_names.index(match.group(1))] * int(match.group(2) or 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ladder", choices=sorted(LADDERS), default="learned")
    ap.add_argument("--radius", type=int, default=7)
    ap.add_argument("--cache", default=None, help="ladder cache (default: data/ladders/<ladder>_r<radius>)")
    ap.add_argument("--scramble", default=None, help="a scramble in move notation")
    ap.add_argument("--random", type=int, default=0, help="solve this many random scrambles and print statistics")
    ap.add_argument("--scramble-len", type=int, default=200, help="length of each random scramble")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beam", type=int, default=4096, help="beam width on the learned rungs")
    args = ap.parse_args()
    if args.scramble is None and not args.random:
        ap.error("give --scramble or --random")

    puzzle = build_maltese_gear_puzzle()
    model = build_point_model(puzzle)
    structure = analyse(model)
    cache = Path(args.cache) if args.cache else Path("data/ladders") / f"{args.ladder}_r{args.radius}"
    ladder = Ladder(model, structure, LADDERS[args.ladder], mitm_radius=args.radius, log=lambda _msg: None)
    ladder.build(cache_dir=cache)
    if not ladder.ready():
        missing = [r.spec.name for r in ladder.rungs if r.table is None and r.heuristic is None]
        sys.exit(f"the ladder in {cache} is not complete; untrained rungs: {missing} (see scripts/train_rung.py)")
    alg = ladder.alg
    names = [r.spec.name for r in ladder.rungs]

    def solve(scramble: list[int]):
        state = puzzle.apply_word(puzzle.goal, scramble)
        t0 = time.time()
        word, per_rung = ladder.solve(state, beam=args.beam)
        seconds = time.time() - t0
        if not np.array_equal(puzzle.apply_word(state, list(word)), puzzle.goal):
            raise AssertionError("the solution does not solve the scramble on the reference simulator")
        return word, per_rung, seconds

    if args.scramble is not None:
        scramble = parse_word(args.scramble, puzzle.move_names)
        word, per_rung, seconds = solve(scramble)
        print(f"scramble ({len(scramble)} quarter turns): {alg.format(scramble)}")
        print(f"solution: {alg.format(word)}")
        print(f"length: {len(word)} quarter turns, {alg.face_turns(word)} face turns; found in {seconds:.1f} s")
        print("per rung: " + ", ".join(f"{n} {c}" for n, c in zip(names, per_rung)))
        print("verified: replaying the solution on the reference simulator gives the solved state")

    if args.random:
        rng = np.random.default_rng(args.seed)
        rows, failures = [], 0
        for k in range(args.random):
            scramble = rng.integers(0, model.num_moves, size=args.scramble_len).tolist()
            try:
                word, per_rung, seconds = solve(scramble)
            except LookupError as err:
                failures += 1
                print(f"scramble {k}: no solution ({err})", flush=True)
                continue
            rows.append(per_rung + [len(word), alg.face_turns(word), seconds])
            print(f"scramble {k:3d}: {len(word):4d} quarter turns, {alg.face_turns(word):4d} face turns, {seconds:5.1f} s", flush=True)
        if rows:
            table = np.array(rows, dtype=float)
            n = len(names)
            print(f"\nsolved and verified {len(rows)}/{args.random} random scrambles (beam {args.beam})")
            print("mean quarter turns per rung (before merging across rungs):")
            for name, mean in zip(names, table[:, :n].mean(0)):
                print(f"  {name:12s} {mean:7.1f}")
            qt, ft, sec = table[:, n], table[:, n + 1], table[:, n + 2]
            print(f"solution length, quarter turns: mean {qt.mean():.1f}, sd {qt.std():.1f}, min {qt.min():.0f}, max {qt.max():.0f}")
            print(f"solution length, face turns:    mean {ft.mean():.1f}, sd {ft.std():.1f}, min {ft.min():.0f}, max {ft.max():.0f}")
            print(f"time per solve: mean {sec.mean():.1f} s, max {sec.max():.1f} s")
        print(f"failures: {failures}")


if __name__ == "__main__":
    main()
