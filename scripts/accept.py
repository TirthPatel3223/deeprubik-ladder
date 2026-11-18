"""Acceptance test of a built ladder: 100 deep scrambles, 60 s each, a second chance for at most 10.

    python scripts/accept.py --puzzle maltese                              # the puzzle's default ladder
    python scripts/accept.py --puzzle tangram_extreme --ladder learned
    python scripts/accept.py --tables my_puzzle.npz --ladder-file data/auto/my_puzzle/auto_r7/ladder.json

The protocol and what it can show are in core/acceptance.py. The verdict is PASS (all solved within 60 s),
PASS_SLOW (at least 90 within 60 s, the rest within 300 s with wider searches) or FAIL; the exit code is 0
for the first two and 1 for FAIL. The JSON report is rewritten after every state, so an interrupted run
resumes; --rerun starts over.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.acceptance import AcceptanceConfig, run_acceptance  # noqa: E402
from core.fingerprint import puzzle_fingerprint  # noqa: E402
from core.ladder import Ladder, specs_from_json  # noqa: E402
from core.perm_model import build_point_model  # noqa: E402
from core.structure import analyse  # noqa: E402
from puzzles.registry import add_puzzle_arguments, entry_from_args  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_puzzle_arguments(ap)
    ap.add_argument("--ladder", default=None, help="a registered ladder of the puzzle (default: the puzzle's)")
    ap.add_argument("--ladder-file", default=None, help="a ladder specification written by scripts/auto.py (ladder.json)")
    ap.add_argument("--radius", type=int, default=None,
                    help="meet-in-the-middle radius of the build (default: the ladder file's, else 7)")
    ap.add_argument("--cache", default=None, help="ladder cache (default: the registered one, or 'rungs' next to --ladder-file)")
    ap.add_argument("--out", default=None, help="JSON report (default: runs/acceptance/<puzzle>_<ladder>.json)")
    ap.add_argument("--rerun", action="store_true", help="start over even if the report has a verdict")
    defaults = AcceptanceConfig()
    ap.add_argument("--n", type=int, default=defaults.n_states)
    ap.add_argument("--min-depth", type=int, default=defaults.min_depth)
    ap.add_argument("--max-depth", type=int, default=defaults.max_depth)
    ap.add_argument("--seed", type=int, default=defaults.seed)
    ap.add_argument("--time-limit", type=float, default=defaults.time_limit, help="seconds per state, pass 1")
    ap.add_argument("--extended-time-limit", type=float, default=defaults.extended_time_limit, help="seconds per state, pass 2")
    ap.add_argument("--pass-fraction", type=float, default=defaults.pass_fraction, help="solved in pass 1 to get a pass 2")
    ap.add_argument("--beam", type=int, default=defaults.beam)
    ap.add_argument("--weight", type=float, default=defaults.weight,
                    help="weight of the path cost g in the beam search's f = weight * g + h")
    ap.add_argument("--no-fail-fast", action="store_true", help="run every state even once the verdict is decided")
    args = ap.parse_args()

    entry = entry_from_args(args)
    puzzle = entry.build()
    model = build_point_model(puzzle)
    if args.ladder_file:
        saved = json.loads(Path(args.ladder_file).read_text())
        if isinstance(saved, list):                         # a bare list of rung specifications
            saved = {"specs": saved}
        fp = saved.get("puzzle_fingerprint")
        if fp is not None and fp != puzzle_fingerprint(puzzle):
            ap.error(f"{args.ladder_file} was designed for another puzzle than {entry.title}")
        # the cache was built with the file's radius; another radius would not match it and rebuild every rung
        radius = saved.get("radius") or args.radius or 7
        if args.radius is not None and args.radius != radius:
            ap.error(f"the ladder in {args.ladder_file} was built with radius {radius}, not {args.radius}")
        specs = specs_from_json(saved["specs"])
        label = Path(args.ladder_file).parent.name or "ladder"
        cache = Path(args.cache) if args.cache else Path(args.ladder_file).parent / "rungs"
    else:
        radius = args.radius if args.radius is not None else 7
        ladders = entry.ladders()
        name = args.ladder or entry.default_ladder
        if name not in ladders:
            ap.error(f"{entry.title} has the ladders {sorted(ladders)}; give --ladder-file for an automatic one")
        specs, label = ladders[name], f"{name}_r{radius}"
        cache = Path(args.cache) if args.cache else entry.cache_dir(name, radius)
    ladder = Ladder(model, analyse(model), specs, mitm_radius=radius, log=lambda _msg: None).build(cache_dir=cache)
    if not ladder.ready():
        missing = [r.spec.name for r in ladder.rungs if r.table is None and r.heuristic is None]
        print(f"the ladder in {cache} is not complete; untrained rungs: {missing}", flush=True)
        sys.exit(2)
    if ladder.legacy:
        print(f"note: rungs {ladder.legacy} come from cache files without fingerprints (trusted as they are)", flush=True)

    cfg = AcceptanceConfig(n_states=args.n, min_depth=args.min_depth, max_depth=args.max_depth, seed=args.seed,
                           time_limit=args.time_limit, extended_time_limit=args.extended_time_limit,
                           pass_fraction=args.pass_fraction, beam=args.beam, weight=args.weight,
                           fail_fast=not args.no_fail_fast)
    out = Path(args.out) if args.out else Path("runs") / "acceptance" / f"{entry.name}_{label}.json"
    report = run_acceptance(ladder, puzzle, cfg, report_path=out, notation=entry.notation(ladder.alg), rerun=args.rerun)
    print(f"report: {out}", flush=True)
    sys.exit(0 if report["verdict"] in ("PASS", "PASS_SLOW") else 1)


if __name__ == "__main__":
    main()
