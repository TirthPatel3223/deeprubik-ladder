"""From a puzzle to a certified solver in one command: design, build, train and accept its ladder.

    python scripts/auto.py --tables my_puzzle.npz                      # any puzzle, ladder designed automatically
    python scripts/auto.py --geometry my_puzzle.json                   # ... or from its 3D model
    python scripts/auto.py --puzzle tangram_extreme                    # a registered puzzle, automatic ladder
    python scripts/auto.py --puzzle maltese --ladder learned           # a registered puzzle, its hand-written ladder

Everything is written to --out (default data/auto/<puzzle>/<ladder>_r<radius>/): ladder.json (the rungs),
rungs/ (macros, tables, networks and training records), acceptance.json, report.md, run.json and auto.log.
Every stage resumes, so the same command continues an interrupted run. The stages and the rules that replace
the hand-made decisions are in core/auto.py, core/autotrain.py and core/acceptance.py.

Exit code 0 when the acceptance test passes (PASS or PASS_SLOW), 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.acceptance import AcceptanceConfig  # noqa: E402
from core.auto import AutoConfig, run_auto  # noqa: E402
from core.autotrain import AutoTrainConfig  # noqa: E402
from puzzles.registry import add_puzzle_arguments, entry_from_args  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_puzzle_arguments(ap)
    ap.add_argument("--ladder", default="auto", help="'auto' (designed from the move tables) or a registered ladder")
    ap.add_argument("--out", default=None, help="output directory (default data/auto/<puzzle>/<ladder>_r<radius>)")
    ap.add_argument("--radius", type=int, default=7, help="meet-in-the-middle radius of the build")
    ap.add_argument("--propose-radius", type=int, default=6, help="meet-in-the-middle radius of the ladder design")
    t, a = AutoTrainConfig(), AcceptanceConfig()
    ap.add_argument("--rung-minutes", type=float, default=t.max_minutes, help="training cap per learned rung")
    ap.add_argument("--round-minutes", type=float, default=t.round_minutes, help="training between two tests")
    ap.add_argument("--test-states", type=int, default=t.test_states, help="test states per learned rung")
    ap.add_argument("--repair-rounds", type=int, default=AutoConfig.repair_rounds)
    ap.add_argument("--accept-n", type=int, default=a.n_states)
    ap.add_argument("--accept-time-limit", type=float, default=a.time_limit)
    ap.add_argument("--accept-extended-time-limit", type=float, default=a.extended_time_limit)
    args = ap.parse_args()

    entry = entry_from_args(args, default="maltese")
    cfg = AutoConfig(ladder=args.ladder, radius=args.radius, propose_radius=args.propose_radius,
                     repair_rounds=args.repair_rounds,
                     train=replace(t, max_minutes=args.rung_minutes, round_minutes=args.round_minutes,
                                   test_states=args.test_states),
                     accept=replace(a, n_states=args.accept_n, time_limit=args.accept_time_limit,
                                    extended_time_limit=args.accept_extended_time_limit))
    summary = run_auto(entry, args.out, cfg)
    sys.exit(0 if summary.get("verdict") in ("PASS", "PASS_SLOW") else 1)


if __name__ == "__main__":
    main()
