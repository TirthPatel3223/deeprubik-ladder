"""Solve scrambles with a built ladder, and verify every solution.

    python scripts/solve.py --scramble "U L F' U2 L'3 F"                       # Maltese Gear Cube
    python scripts/solve.py --random 50 --beam 4096
    python scripts/solve.py --puzzle tangram_extreme --scramble "U2' F R U3 D' L"
    python scripts/solve.py --puzzle tangram_extreme --random 50 --replay-json runs/tangram_solutions.json

Each puzzle reads and writes its own notation. Maltese Gear Cube: moves U, L, F and their inverses
U', L', F'; a number after a move repeats it (U2, F'3); lengths in quarter turns and face turns (a single
or double turn of a face counts one, a triple or quadruple turn two). Tangram Cube: U, D (60° steps;
U2 = 120°, U3 = 180°, U2' = 120° anticlockwise) and the flips F, R, L, as in its simulator; lengths in
steps (each 60° step or flip) and turns (a corner-layer turn by any amount, or a flip).

Every solution is replayed on the reference simulator (the verified move tables) before it is
reported; a solution that does not reach the solved state is an error, not a result.

--markdown writes one solution as a page like solutions/maltese_solution.md. --replay-json writes
every scramble and solution in the Tangram simulator's notation, for tools/export/replay.gen.ts in the
Cube_Modeling project, which replays them on the 3D model's own engine.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.ladder import Ladder  # noqa: E402
from core.perm_model import build_point_model  # noqa: E402
from core.structure import analyse  # noqa: E402
from puzzles.registry import PUZZLES, get_puzzle  # noqa: E402


def write_markdown(path: Path, title: str, label: str, notation, scramble, word, per_rung, names) -> None:
    lengths = notation.describe(word)
    moves = notation.format(word).split()
    lines = [f"{k // 4 + 1:3d}. " + " ".join(moves[k:k + 4]) for k in range(0, len(moves), 4)]
    unit = notation.metrics[0][0]
    text = [
        f"# {title} solution",
        "",
        f"Solved with the ladder solver ({label}), then replayed on the reference simulator: the result is the solved state.",
        "",
        f"- Scramble ({notation.describe(scramble)}): `{notation.format(scramble)}`",
        f"- Solution: {len(moves)} moves as written below ({lengths})",
        f"- Per rung, in {unit} before merging across rungs: " + ", ".join(f"{n} {c}" for n, c in zip(names, per_rung)),
        *([f"- Notation: {notation.legend}"] if notation.legend else []),
        "- Read left to right, top to bottom, 4 moves per line.",
        "",
        "```",
        *lines,
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puzzle", choices=sorted(PUZZLES), default="maltese")
    ap.add_argument("--ladder", default=None, help="ladder name (default: the puzzle's)")
    ap.add_argument("--radius", type=int, default=7)
    ap.add_argument("--cache", default=None, help="ladder cache (default: see scripts/build_ladder.py)")
    ap.add_argument("--scramble", default=None, help="a scramble in the puzzle's notation")
    ap.add_argument("--random", type=int, default=0, help="solve this many random scrambles and print statistics")
    ap.add_argument("--scramble-len", type=int, default=None, help="length of each random scramble (default: the puzzle's)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beam", type=int, default=4096, help="beam width on the learned rungs")
    ap.add_argument("--markdown", default=None, help="write the --scramble solution as a markdown page")
    ap.add_argument("--replay-json", default=None, help="write scrambles and solutions for the 3D model's replay check")
    args = ap.parse_args()
    if args.scramble is None and not args.random:
        ap.error("give --scramble or --random")

    entry = get_puzzle(args.puzzle)
    ladders = entry.ladders()
    ladder_name = args.ladder or entry.default_ladder
    if ladder_name not in ladders:
        ap.error(f"{entry.title} has the ladders {sorted(ladders)}")
    puzzle = entry.build()
    model = build_point_model(puzzle)
    structure = analyse(model)
    cache = Path(args.cache) if args.cache else entry.cache_dir(ladder_name, args.radius)
    ladder = Ladder(model, structure, ladders[ladder_name], mitm_radius=args.radius, log=lambda _msg: None)
    ladder.build(cache_dir=cache)
    if not ladder.ready():
        missing = [r.spec.name for r in ladder.rungs if r.table is None and r.heuristic is None]
        sys.exit(f"the ladder in {cache} is not complete; untrained rungs: {missing} (see scripts/train_rung.py)")
    notation = entry.notation(ladder.alg)
    unit = notation.metrics[0][0]
    names = [r.spec.name for r in ladder.rungs]
    replays: list[dict] = []

    def solve(scramble: list[int]):
        state = puzzle.apply_word(puzzle.goal, scramble)
        t0 = time.time()
        word, per_rung = ladder.solve(state, beam=args.beam)
        seconds = time.time() - t0
        if not np.array_equal(puzzle.apply_word(state, list(word)), puzzle.goal):
            raise AssertionError("the solution does not solve the scramble on the reference simulator")
        if entry.geometry_variant is not None:
            replays.append({"variant": entry.geometry_variant, "scramble": notation.format(scramble),
                            "solution": notation.format(word)})
        return word, per_rung, seconds

    if args.scramble is not None:
        scramble = notation.parse(args.scramble)
        word, per_rung, seconds = solve(scramble)
        print(f"scramble ({len(scramble)} {unit}): {notation.format(scramble)}")
        print(f"solution: {notation.format(word)}")
        print(f"length: {notation.describe(word)}; found in {seconds:.1f} s")
        print("per rung: " + ", ".join(f"{n} {c}" for n, c in zip(names, per_rung)))
        print("verified: replaying the solution on the reference simulator gives the solved state")
        if args.markdown:
            write_markdown(Path(args.markdown), entry.title, f"{ladder_name}_r{args.radius}, beam {args.beam}",
                           notation, scramble, word, per_rung, names)
            print(f"wrote {args.markdown}")

    if args.random:
        rng = np.random.default_rng(args.seed)
        metric_names = [m for m, _ in notation.metrics]
        rows, failures = [], 0
        for k in range(args.random):
            scramble = rng.integers(0, model.num_moves, size=args.scramble_len or entry.scramble_len).tolist()
            try:
                word, per_rung, seconds = solve(scramble)
            except LookupError as err:
                failures += 1
                print(f"scramble {k}: no solution ({err})", flush=True)
                continue
            lengths = notation.lengths(word)
            rows.append(per_rung + [lengths[m] for m in metric_names] + [seconds])
            print(f"scramble {k:3d}: " + ", ".join(f"{lengths[m]:4d} {m}" for m in metric_names) + f", {seconds:5.1f} s", flush=True)
        if rows:
            table = np.array(rows, dtype=float)
            n = len(names)
            print(f"\nsolved and verified {len(rows)}/{args.random} random scrambles (beam {args.beam})")
            print(f"mean {unit} per rung (before merging across rungs):")
            for name, mean in zip(names, table[:, :n].mean(0)):
                print(f"  {name:12s} {mean:7.1f}")
            for j, m in enumerate(metric_names):
                col = table[:, n + j]
                print(f"solution length, {m + ':':15s} mean {col.mean():.1f}, sd {col.std():.1f}, min {col.min():.0f}, max {col.max():.0f}")
            sec = table[:, -1]
            print(f"time per solve: mean {sec.mean():.1f} s, max {sec.max():.1f} s")
        print(f"failures: {failures}")

    if args.replay_json:
        if not replays:
            sys.exit(f"--replay-json: {entry.title} has no 3D model to replay on")
        Path(args.replay_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.replay_json).write_text(json.dumps(replays, indent=1))
        print(f"wrote {len(replays)} solutions to {args.replay_json} (replay: tools/export/replay.gen.ts in Cube_Modeling)")


if __name__ == "__main__":
    main()
