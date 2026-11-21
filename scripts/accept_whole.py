"""Acceptance test of one whole-puzzle network (core/whole.py): the protocol of scripts/accept.py.

    python scripts/accept_whole.py --puzzle tangram_extreme --checkpoint data/whole/tangram_extreme/learned_gated.best.pt

100 deep scrambles (2000..5000 random moves, seed 42: the ladder's acceptance states), 60 s each, a second chance
with wider searches for at most 10 (core/acceptance.py). The ladder's cache supplies only the actions and the
phases; no exact table and no rung network is used to solve. With --compare REPORT the ladder's acceptance report
(scripts/accept.py) is read and the two are compared state by state.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.acceptance import AcceptanceConfig, final_record, run_acceptance  # noqa: E402
from core.ladder import Ladder  # noqa: E402
from core.perm_model import build_point_model  # noqa: E402
from core.structure import analyse  # noqa: E402
from core.whole import MASKS, WholePuzzle, WholeSolver, WholeValue  # noqa: E402
from puzzles.registry import PUZZLES, get_puzzle  # noqa: E402


def compare(ours: dict, theirs: dict, metric: str) -> str:
    """Paired comparison on the states both solved."""
    a = {s["index"]: final_record(s) for s in ours["states"]}
    b = {s["index"]: final_record(s) for s in theirs["states"]}
    both = [i for i in a if i in b and a[i]["solved"] and b[i]["solved"]]
    if not both:
        return "no state solved by both"
    x = np.array([a[i]["lengths"][metric] for i in both], dtype=float)
    y = np.array([b[i]["lengths"][metric] for i in both], dtype=float)
    d = x - y
    return (f"on the {len(both)} states both solved: one network {x.mean():.1f} {metric}, ladder {y.mean():.1f}; "
            f"difference {d.mean():+.1f} (standard error {d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float('nan'):.1f}); "
            f"shorter on {(d < 0).sum()}, equal on {(d == 0).sum()}, longer on {(d > 0).sum()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puzzle", choices=sorted(PUZZLES), default="tangram_extreme")
    ap.add_argument("--ladder", default="learned", help="the registered ladder whose actions and phases the network used")
    ap.add_argument("--radius", type=int, default=7)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mask", choices=MASKS, default="rung", help="actions a state may take in the search (core/whole.py)")
    ap.add_argument("--out", default=None, help="JSON report (default: runs/acceptance/<puzzle>_whole_<checkpoint>.json)")
    ap.add_argument("--compare", default=None, help="the ladder's acceptance report, for a paired comparison")
    ap.add_argument("--budget", type=float, default=None,
                    help="children per beam state per step in the search (core/whole.py phased_beam_search)")
    ap.add_argument("--rerun", action="store_true")
    defaults = AcceptanceConfig()
    ap.add_argument("--n", type=int, default=defaults.n_states)
    ap.add_argument("--time-limit", type=float, default=defaults.time_limit)
    ap.add_argument("--beam", type=int, default=defaults.beam)
    ap.add_argument("--weight", type=float, default=defaults.weight)
    ap.add_argument("--no-fail-fast", action="store_true")
    args = ap.parse_args()

    entry = get_puzzle(args.puzzle)
    puzzle = entry.build()
    model = build_point_model(puzzle)
    cache = Path(args.cache) if args.cache else entry.cache_dir(args.ladder, args.radius)
    ladder = Ladder(model, analyse(model), entry.ladders()[args.ladder], mitm_radius=args.radius, log=lambda _m: None)
    ladder.build(cache_dir=cache)
    wp = WholePuzzle(ladder, mask=args.mask)
    value = WholeValue.from_checkpoint(wp, args.checkpoint, log=lambda _m: None)
    value.search_budget = args.budget
    solver = WholeSolver(ladder, value)
    print(f"one network ({args.checkpoint}, update {value.update}, baseline {value.cfg.baseline}) over "
          f"{len(wp.actions)} actions in {wp.n_phases} phases, mask {args.mask}"
          + (f", search budget {args.budget:g} children per state" if args.budget is not None else ""), flush=True)

    cfg = AcceptanceConfig(n_states=args.n, time_limit=args.time_limit, beam=args.beam, weight=args.weight,
                           fail_fast=not args.no_fail_fast)
    stem = (Path(args.checkpoint).stem + ("" if args.mask == "rung" else f"_search_{args.mask}")
            + ("" if args.budget is None else f"_budget{args.budget:g}"))
    out = Path(args.out) if args.out else Path("runs") / "acceptance" / f"{entry.name}_whole_{stem}.json"
    notation = entry.notation(ladder.alg)
    report = run_acceptance(solver, puzzle, cfg, report_path=out, notation=notation, rerun=args.rerun)
    print(f"report: {out}", flush=True)
    if args.compare and Path(args.compare).exists():
        print(compare(report, json.loads(Path(args.compare).read_text()), notation.metrics[0][0]), flush=True)
    sys.exit(0 if report["verdict"] in ("PASS", "PASS_SLOW") else 1)


if __name__ == "__main__":
    main()
