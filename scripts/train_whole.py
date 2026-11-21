"""Train one value network for the whole puzzle (core/whole.py), testing it on deep scrambles as it goes.

    python scripts/train_whole.py --puzzle tangram_extreme --baseline gated --minutes 180
    python scripts/train_whole.py --puzzle tangram_extreme --baseline none --minutes 180

The ladder (read from its cache) supplies the actions and the phases; the solver uses neither its tables nor
its networks. After every --round-minutes of training the network is tested on --test-states deep scrambles
(2000..5000 random moves, seed 1, so not the acceptance states) with the solver's first search, and the
network of the best complete test is kept as <out>.best.pt: most states solved, then the shortest mean
solution. Training stops after --minutes, or once every test state is solved and three tests in a row have not
shortened the mean solution by more than 1%. The same command resumes a stopped run.

Before training, the ladder itself solves the test states once (this is the only use of its tables), so every
test line can be compared with the ladder on the same states.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.acceptance import AcceptanceConfig, deep_scrambles  # noqa: E402
from core.ladder import Ladder, RungFailure  # noqa: E402
from core.perm_model import build_point_model  # noqa: E402
from core.structure import analyse  # noqa: E402
from core.whole import BASELINES, MASKS, WholeConfig, WholePuzzle, WholeSolver, WholeValue  # noqa: E402
from puzzles.registry import PUZZLES, get_puzzle  # noqa: E402

# Mean cost of each rung in the learned ladders' own solutions of deep scrambles (30 Maltese and 40 Tangram
# Extreme acceptance states, beam 4096, measured 2026-11-19), in quarter turns and steps. The baseline uses
# them as the cost of a whole rung; they are constants of the reward, not part of the solver.
MEASURED_RUNG_COSTS = {
    ("tangram_extreme", "learned"): (7.7, 47.4, 40.5),
    ("maltese", "learned"): (10.7, 27.5, 49.2, 86.9, 249.2),
}


def run_tests(solve, puzzle, scrambles, lengths, metric: str, time_limit: float, give_up: int = 4) -> dict:
    """Solve every test scramble with ``solve(state, time_limit)``; replay each solution on the simulator."""
    rows, tried = [], 0
    for _depth, moves in scrambles:
        if tried >= give_up and not any(r["solved"] for r in rows):
            break                                  # nothing solved so far: the network is not ready
        tried += 1
        state = puzzle.apply_word(puzzle.goal, moves)
        t0 = time.time()
        try:
            word, per_phase = solve(state, time_limit)
        except RungFailure as err:
            rows.append({"solved": False, "reason": err.reason, "seconds": time.time() - t0})
            continue
        if not np.array_equal(puzzle.apply_word(state, list(word)), puzzle.goal):
            raise AssertionError("a solution does not reach the goal on the reference simulator")
        rows.append({"solved": True, "length": lengths(word)[metric], "cost": int(sum(per_phase)),
                     "per_phase": [int(c) for c in per_phase], "seconds": time.time() - t0})
    ok = [r for r in rows if r["solved"]]
    return {"tested": len(rows), "n": len(scrambles), "complete": len(rows) == len(scrambles),
            "solved": len(ok) / max(1, len(rows)),
            "mean_length": float(np.mean([r["length"] for r in ok])) if ok else None,
            "mean_cost": float(np.mean([r["cost"] for r in ok])) if ok else None,
            "per_phase": np.mean([r["per_phase"] for r in ok], 0).round(1).tolist() if ok else None,
            "seconds_per_state": float(np.mean([r["seconds"] for r in rows])) if rows else None,
            "lengths": [r.get("length") for r in rows], "rows": rows}


def better(a: dict, b: dict | None) -> bool:
    """Is test a better than test b: more states solved, then a shorter mean solution."""
    if not a["complete"] or a["mean_length"] is None:
        return False
    if b is None:
        return True
    return (a["solved"], -a["mean_length"]) > (b["solved"], -b["mean_length"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puzzle", choices=sorted(PUZZLES), default="tangram_extreme")
    ap.add_argument("--ladder", default="learned", help="the registered ladder whose actions and phases are used")
    ap.add_argument("--radius", type=int, default=7)
    ap.add_argument("--cache", default=None, help="the ladder's cache (default: the registered one)")
    ap.add_argument("--out", default=None, help="checkpoint (default: data/whole/<puzzle>/<ladder>_<baseline>.pt)")
    ap.add_argument("--baseline", choices=BASELINES, default="gated", help="reward shaping (core/whole.py)")
    ap.add_argument("--mask", choices=MASKS, default="rung", help="actions a state may take (core/whole.py)")
    ap.add_argument("--init", default=None, help="start from this checkpoint's network (e.g. one trained under a tighter mask)")
    ap.add_argument("--rung-costs", default=None, help="mean cost of each rung, comma separated (default: measured)")
    ap.add_argument("--minutes", type=float, default=180.0, help="training time cap, over resumes")
    ap.add_argument("--round-minutes", type=float, default=10.0, help="training time between two tests")
    ap.add_argument("--test-states", type=int, default=16)
    ap.add_argument("--test-time", type=float, default=60.0, help="seconds per test state")
    ap.add_argument("--beam", type=int, default=4096)
    ap.add_argument("--weight", type=float, default=0.3)
    ap.add_argument("--res-dim", type=int, default=1000)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--states", type=int, default=200_000, help="states generated per update")
    ap.add_argument("--itrs", type=int, default=None,
                    help="training iterations per update (default 100; given when resuming, it replaces the checkpoint's)")
    ap.add_argument("--lr", type=float, default=None,
                    help="learning rate before decay (default 1e-3; overrides as --itrs)")
    ap.add_argument("--loss-thresh", type=float, default=None,
                    help="update the target network when the loss is below this (default 0.06; given when resuming "
                         "or with --init, it replaces the checkpoint's value)")
    ap.add_argument("--budget", type=float, default=None,
                    help="test searches: children per beam state per step (phased_beam_search); phases with many "
                         "actions then keep fewer states")
    ap.add_argument("--twin", action="store_true",
                    help="two networks, each valuing the children the other chooses (a new network only; core/whole.py)")
    ap.add_argument("--twin-search", choices=("a", "mean"), default="a",
                    help="with --twin, the value the search uses: network a alone (as fast as one network) or the mean")
    ap.add_argument("--target-tau", type=float, default=None,
                    help="> 0: the target network is a running average of the online one (Polyak), moved this "
                         "fraction after every step, and the search uses it (default 0: copies; overrides as "
                         "--loss-thresh)")
    ap.add_argument("--cross-weight", type=float, default=None,
                    help="twin or double backups: the share of the target taken from the other network's value of the "
                         "chosen child, the rest from the chooser's own (default 1; overrides as --loss-thresh)")
    ap.add_argument("--double", choices=("on", "off"), default=None,
                    help="choose each backup's action with the online network and value it with the target network "
                         "(default off; overrides as above)")
    ap.add_argument("--force-sync", type=int, default=None,
                    help="update the target network at least every this many updates (default 4; overrides as above)")
    ap.add_argument("--fresh", action="store_true", help="start over instead of resuming")
    ap.add_argument("--no-ladder-reference", action="store_true", help="do not solve the test states with the ladder")
    args = ap.parse_args()

    entry = get_puzzle(args.puzzle)
    ladders = entry.ladders()
    if args.ladder not in ladders:
        ap.error(f"{entry.title} has the ladders {sorted(ladders)}")
    puzzle = entry.build()
    model = build_point_model(puzzle)
    cache = Path(args.cache) if args.cache else entry.cache_dir(args.ladder, args.radius)
    ladder = Ladder(model, analyse(model), ladders[args.ladder], mitm_radius=args.radius, log=lambda _m: None)
    ladder.build(cache_dir=cache)
    notation = entry.notation(ladder.alg)
    metric = notation.metrics[0][0]
    tag = f"{args.ladder}_{args.baseline}" + ("" if args.mask == "rung" else f"_{args.mask}")
    out = Path(args.out) if args.out else Path("data") / "whole" / entry.name / f"{tag}.pt"
    best_path = out.with_name(out.stem + ".best.pt")
    record_path = out.with_name(out.stem + ".train.json")
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    wp = WholePuzzle(ladder, mask=args.mask)
    log(f"{entry.title}, ladder {args.ladder}: whole state of {wp.quotient.n_slots} slots, {len(wp.actions)} actions; "
        f"mask {args.mask}, actions per phase: " + ", ".join(f"{p.name} {len(p.actions)}" for p in wp.phases))
    rung_costs = (tuple(float(x) for x in args.rung_costs.split(",")) if args.rung_costs
                  else MEASURED_RUNG_COSTS.get((entry.name, args.ladder), ()))
    if args.baseline != "none" and len(rung_costs) != wp.n_phases:
        ap.error(f"--baseline {args.baseline} needs --rung-costs with {wp.n_phases} values")

    record = json.loads(record_path.read_text()) if record_path.exists() and not args.fresh else None
    if out.exists() and not args.fresh:
        value = WholeValue.from_checkpoint(wp, out, log=log)
        log(f"resuming {out} at update {value.update}")
    elif args.init:
        # a network trained under another mask: same input, same actions, so its weights carry over
        value = WholeValue.from_checkpoint(wp, args.init, log=log)
        if value.cfg.baseline != args.baseline:
            ap.error(f"{args.init} was trained with the {value.cfg.baseline} baseline, not {args.baseline}")
        log(f"starting from the network of {args.init} (update {value.update}; its learning-rate schedule continues)")
        record = None
    else:
        cfg = WholeConfig(res_dim=args.res_dim, num_blocks=args.blocks, batch_size=args.batch,
                          states_per_update=args.states, itrs_per_update=100 if args.itrs is None else args.itrs,
                          lr=1e-3 if args.lr is None else args.lr,
                          loss_thresh=0.06 if args.loss_thresh is None else args.loss_thresh,
                          force_sync=4 if args.force_sync is None else args.force_sync, baseline=args.baseline,
                          double=args.double == "on", twin=args.twin, twin_search=args.twin_search,
                          cross_weight=1.0 if args.cross_weight is None else args.cross_weight,
                          target_tau=0.0 if args.target_tau is None else args.target_tau,
                          rung_costs=rung_costs if args.baseline != "none" else ())
        value = WholeValue(wp, cfg, log=log)
        record = None
    asked = (("loss_thresh", args.loss_thresh), ("force_sync", args.force_sync),
             ("double", None if args.double is None else args.double == "on"), ("cross_weight", args.cross_weight),
             ("target_tau", args.target_tau), ("lr", args.lr), ("itrs_per_update", args.itrs))
    overrides = {k: v for k, v in asked if v is not None and getattr(value.cfg, k) != v}
    if overrides:
        log(f"backup rule changed from the checkpoint's: " + ", ".join(
            f"{k} {getattr(value.cfg, k)} -> {v}" for k, v in overrides.items()))
        value.cfg = replace(value.cfg, **overrides)
    n_par = sum(p.numel() for p in value.net.parameters())
    log(f"network {value.cfg.res_dim} x {value.cfg.num_blocks} blocks, {n_par / 1e6:.2f} M parameters, baseline "
        f"{value.cfg.baseline}; walks: depths {list(value.cfg.depths)}, mixing {list(value.cfg.mix_walks)}, "
        f"phase weights {[round(w, 3) for w in value.cfg.phase_weights]}"
        + (f"; mixed rungs unsolved {[round(x, 3) for x in value.cfg.mixed_unsolved]}" if value.cfg.mixed_unsolved
           else "")
        + (f"; twin networks (search with {value.cfg.twin_search}, cross weight {value.cfg.cross_weight:g})"
           if value.cfg.twin else "")
        + ("; double backup" if value.cfg.double else "")
        + (f"; averaged target network, tau {value.cfg.target_tau:g}" if value.cfg.target_tau > 0 else ""))

    value.search_budget = args.budget
    if args.budget is not None:
        log(f"test searches: at most {args.budget:g} children per beam state per step")
    scrambles = deep_scrambles(AcceptanceConfig(n_states=args.test_states, seed=1), puzzle.num_moves)
    solver = WholeSolver(ladder, value)

    def whole_solve(state, limit):
        return solver.solve(state, beam=args.beam, weight=args.weight, time_limit=limit)

    if record is None:
        record = {"puzzle": entry.name, "ladder": args.ladder, "config": asdict(value.cfg), "metric": metric,
                  "tests": [], "train_seconds": 0.0, "best": None, "ladder_reference": None}
    if overrides:
        record.setdefault("notes", []).append(f"from update {value.update}: " + ", ".join(
            f"{k} {v}" for k, v in overrides.items()))
        record["config"] = asdict(value.cfg)
    if record.get("ladder_reference") is None and not args.no_ladder_reference and ladder.ready():
        ref = run_tests(lambda s, t: ladder.solve(s, beam=args.beam, weight=args.weight, time_limit=t),
                        puzzle, scrambles, notation.lengths, metric, args.test_time)
        record["ladder_reference"] = ref
        log(f"the ladder on the {args.test_states} test states: solved {ref['solved']:.0%}, mean {ref['mean_length']:.1f} "
            f"{metric} (cost before merging {ref['mean_cost']:.1f}, per rung {ref['per_phase']})")
    ref = record.get("ladder_reference")

    def save_record() -> None:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(record, indent=1))

    while record["train_seconds"] < 60 * args.minutes:
        t0 = time.time()
        round_seconds = min(60 * args.round_minutes, 60 * args.minutes - record["train_seconds"])
        value.train(max_updates=10**9, time_limit=round_seconds, checkpoint=out, log_every=5)
        record["train_seconds"] += time.time() - t0
        test = run_tests(whole_solve, puzzle, scrambles, notation.lengths, metric, args.test_time)
        test.update(update=value.update, minutes=round(record["train_seconds"] / 60, 1))
        record["tests"].append(test)
        line = (f"test after {test['minutes']:.0f} min (update {value.update}): solved {test['solved']:.0%} of "
                f"{test['tested']}/{test['n']}")
        if test["mean_length"] is not None:
            line += (f", mean {test['mean_length']:.1f} {metric} (cost {test['mean_cost']:.1f}, per phase "
                     f"{test['per_phase']}), {test['seconds_per_state']:.1f} s per state")
            if ref and ref.get("mean_length"):
                line += f"; ladder {ref['mean_length']:.1f}"
        if better(test, record["best"]):
            shutil.copyfile(out, best_path)
            record["best"] = test
            line += "  [best so far, kept]"
        log(line)
        save_record()
        full = [t for t in record["tests"] if t["complete"] and t["solved"] >= 1.0]
        if len(full) > 3 and min(t["mean_length"] for t in full[-3:]) > min(t["mean_length"] for t in full[:-3]) * 0.99:
            log("every test state solved and the last three tests did not shorten the solutions by 1%: stopping")
            break
    save_record()
    if record["best"] is not None:
        b = record["best"]
        log(f"best network: {best_path} (update {b['update']}): solved {b['solved']:.0%}, mean {b['mean_length']:.1f} {metric}"
            + (f", ladder {ref['mean_length']:.1f}" if ref and ref.get("mean_length") else ""))
    else:
        log("no complete test yet; nothing kept as the best network")
    _ = torch


if __name__ == "__main__":
    main()
