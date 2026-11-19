"""The acceptance test: does the solver solve deep scrambles, every one of them, in time?

The protocol (every number is a field of ``AcceptanceConfig``):

1. 100 states, each a random scramble whose depth is uniform in 2000..5000 moves (seed 42, the same
   states as ``scripts/benchmark_deep_scrambles.py``). Scrambles this deep are effectively uniform random
   states, so every part of the state space the ladder has to cover is represented.
2. One untimed warm-up solve, so one-time GPU set-up is not charged to the first state.
3. Pass 1: each state gets 60 s with the standard search schedule, and every solution is replayed on the
   reference simulator. Once more than 10 states have failed the verdict is decided and the pass stops.
4. All 100 solved: PASS. At least 90 solved: pass 2 gives each unsolved state 300 s with the extended
   schedule (a greedier search, then wider beams: the same search given more time would fail the same
   way). All of them solved then: PASS_SLOW, otherwise FAIL. Fewer than 90 solved: FAIL.
5. A state outside a table rung is a coverage gap that no search can fix, so pass 2 skips it; a solution
   that does not replay is a bug. Either one makes the verdict FAIL.

What it can show: if a fraction p of random states falls in a part of the state space the solver never
solves, 100 states all miss it with probability (1 - p)^100, under 5% once p >= 3%. Rarer gaps need an exact
check on the group.

The report is a JSON file rewritten after every state, so an interrupted run resumes where it stopped (as
long as the puzzle, the ladder and the settings are unchanged).
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from core.fingerprint import puzzle_fingerprint
from core.gather_engine import GatherPuzzle
from core.ladder import Ladder, RungFailure, retire
from core.search_schedule import SearchSchedule

FORMAT = "acceptance/1"


@dataclass
class AcceptanceConfig:
    n_states: int = 100
    min_depth: int = 2000
    max_depth: int = 5000
    seed: int = 42
    time_limit: float = 60.0               # pass 1, seconds per state
    pass_fraction: float = 0.90            # solved in pass 1 to get a pass 2
    extended_time_limit: float = 300.0     # pass 2, seconds per remaining state
    beam: int = 4096                       # standard schedule: beam, then 8x wider up to max_width
    weight: float = 0.3
    max_width: int = 32768
    extended_max_width: int = 131072       # extended schedule
    greedy_weight: float = 0.1
    fail_fast: bool = True
    warmup: bool = True

    @property
    def allowed_failures(self) -> int:
        """Pass-1 failures that still earn a second pass (10 of 100 by default)."""
        return self.n_states - math.ceil(self.pass_fraction * self.n_states - 1e-9)

    def schedules(self) -> tuple[SearchSchedule, SearchSchedule]:
        return (SearchSchedule.standard(self.beam, self.weight, self.max_width),
                SearchSchedule.extended(self.max_width, self.weight, self.extended_max_width, self.greedy_weight))


def deep_scrambles(cfg: AcceptanceConfig, num_moves: int) -> list[tuple[int, list[int]]]:
    """(depth, moves) of every test state, drawn exactly as scripts/benchmark_deep_scrambles.py draws them."""
    rng = np.random.default_rng(cfg.seed)
    out = []
    for _ in range(cfg.n_states):
        depth = int(rng.integers(cfg.min_depth, cfg.max_depth + 1))
        out.append((depth, rng.integers(0, num_moves, size=depth).tolist()))
    return out


def _free_gpu() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def attempt(ladder: Ladder, puzzle: GatherPuzzle, state: np.ndarray, time_limit: float, schedule: SearchSchedule,
            lengths: Callable[[Sequence[int]], dict]) -> dict:
    """One timed solve of a visible state, replayed on the simulator. Returns a JSON-ready record."""
    t0 = time.time()
    rec: dict = {"solved": False, "reason": None, "rung": None}
    try:
        word, per_rung = ladder.solve(state, time_limit=time_limit, schedule=schedule)
    except RungFailure as err:
        rec.update(reason=err.reason, rung=err.rung, error=str(err))
    else:
        if np.array_equal(puzzle.apply_word(state, list(word)), puzzle.goal):
            rec.update(solved=True, lengths=lengths(word), per_rung=[int(c) for c in per_rung])
        else:
            rec.update(reason="replay_mismatch", error="the solution does not reach the goal on the reference simulator")
    rec["time_s"] = round(time.time() - t0, 3)
    return rec


def _describe(rec: dict) -> str:
    if rec["solved"]:
        lengths = ", ".join(f"{v} {k}" for k, v in rec["lengths"].items())
        return f"solved in {rec['time_s']:6.1f} s | {lengths}"
    where = f" in rung {rec['rung']}" if rec.get("rung") else ""
    return f"FAILED after {rec['time_s']:6.1f} s | {rec['reason']}{where}"


def run_acceptance(ladder: Ladder, puzzle: GatherPuzzle, cfg: AcceptanceConfig | None = None,
                   report_path: str | Path | None = None, notation=None, rerun: bool = False,
                   log: Callable[[str], None] | None = None) -> dict:
    """Run the protocol in the module docstring. Returns the report: per-state records, ``verdict`` (PASS,
    PASS_SLOW or FAIL) and ``summary``."""
    cfg = cfg or AcceptanceConfig()
    log = log or (lambda msg: print(msg, flush=True))
    if not ladder.ready():
        raise RuntimeError("the ladder is not complete: build it and train its learned rungs first")
    lengths = notation.lengths if notation is not None else (lambda w: {"moves": len(w)})
    header = {"format": FORMAT, "puzzle": puzzle.name, "puzzle_fingerprint": puzzle_fingerprint(puzzle),
              "ladder_fingerprint": ladder.fingerprint(), "rungs": [r.spec.name for r in ladder.rungs],
              "config": asdict(cfg)}
    path = Path(report_path) if report_path is not None else None
    report: dict | None = None
    if path is not None and path.exists():
        old = json.loads(path.read_text())
        if not rerun and all(old.get(k) == v for k, v in header.items()):
            report = old
            if report.get("verdict"):
                log(f"acceptance: {report['verdict']} (already decided in {path})")
                return report
            log(f"acceptance: resuming {path} after {len(report['states'])} states")
        else:
            log(f"acceptance: {path} was for another ladder or other settings: set aside as {retire(path, None).name}")
    if report is None:
        report = {**header, "states": [], "verdict": None, "summary": None}

    def save() -> None:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=1))

    standard, extended = cfg.schedules()
    scrambles = deep_scrambles(cfg, puzzle.num_moves)
    allowed = cfg.allowed_failures
    if cfg.warmup:
        try:
            ladder.solve(puzzle.apply_word(puzzle.goal, scrambles[0][1][:60]), schedule=standard)
        except (RungFailure, RuntimeError):
            pass

    # -- pass 1 -------------------------------------------------------------------------------------------
    log(f"acceptance pass 1: {cfg.n_states} states of {cfg.min_depth}..{cfg.max_depth} random moves, "
        f"{cfg.time_limit:g} s each (beams {[w for w, _ in standard.attempts]}, weight {cfg.weight:g})")
    done = {s["index"] for s in report["states"]}
    for k, (depth, scramble) in enumerate(scrambles):
        if k in done:
            continue
        failures = sum(not s["pass1"]["solved"] for s in report["states"])
        if cfg.fail_fast and failures > allowed:
            log(f"acceptance: {failures} failures, more than the {allowed} a second pass allows: stopping pass 1")
            break
        rec = attempt(ladder, puzzle, puzzle.apply_word(puzzle.goal, scramble), cfg.time_limit, standard, lengths)
        report["states"].append({"index": k, "depth": depth, "pass1": rec})
        solved = sum(s["pass1"]["solved"] for s in report["states"])
        log(f"  [{k + 1:3d}/{cfg.n_states}] depth {depth:4d} | {_describe(rec)} | {solved}/{len(report['states'])} solved")
        save()
        _free_gpu()

    # -- the decision, with pass 2 when it is allowed ----------------------------------------------------
    states = report["states"]
    failed = [s for s in states if not s["pass1"]["solved"]]
    bug = any(s["pass1"]["reason"] == "replay_mismatch" for s in states)
    if len(states) < cfg.n_states or len(failed) > allowed or bug:
        verdict = "FAIL"
    elif not failed:
        verdict = "PASS"
    else:
        log(f"acceptance pass 2: {len(failed)} unsolved states, {cfg.extended_time_limit:g} s each "
            f"(searches {list(extended.attempts)})")
        for s in failed:
            if "pass2" in s:
                continue
            if s["pass1"]["reason"] == "outside_table":
                s["pass2"] = {"solved": False, "reason": "outside_table", "rung": s["pass1"]["rung"],
                              "skipped": "the state is outside an exact table: no search can solve it", "time_s": 0.0}
            else:
                state = puzzle.apply_word(puzzle.goal, scrambles[s["index"]][1])
                s["pass2"] = attempt(ladder, puzzle, state, cfg.extended_time_limit, extended, lengths)
            log(f"  state {s['index'] + 1:3d}: {_describe(s['pass2'])}")
            save()
            _free_gpu()
        verdict = "PASS_SLOW" if all(s["pass2"]["solved"] for s in failed) else "FAIL"
    report["verdict"] = verdict
    report["summary"] = summarize(report)
    save()
    log(format_summary(report))
    return report


def final_record(s: dict) -> dict:
    """The record that decides a state: pass 2 if it ran, else pass 1."""
    return s.get("pass2") or s["pass1"]


def summarize(report: dict) -> dict:
    states = report["states"]
    cfg = report["config"]
    solved = [final_record(s) for s in states if final_record(s)["solved"]]
    out: dict = {
        "verdict": report["verdict"], "states_run": len(states), "states_planned": cfg["n_states"],
        "solved_pass1": sum(s["pass1"]["solved"] for s in states), "solved": len(solved),
        "slow_states": [{"index": s["index"], "depth": s["depth"], "time_s": s["pass2"]["time_s"]}
                        for s in states if s.get("pass2", {}).get("solved")],
        "failures": [{"index": s["index"], "depth": s["depth"], "rung": final_record(s).get("rung"),
                      "reason": final_record(s).get("reason")} for s in states if not final_record(s)["solved"]],
    }
    out["failures_by_rung"] = dict(Counter(f"{f['rung']}: {f['reason']}" for f in out["failures"]))
    if solved:
        metrics = list(solved[0]["lengths"])
        out["lengths"] = {}
        for m in metrics:
            v = np.array([r["lengths"][m] for r in solved], dtype=float)
            out["lengths"][m] = {"mean": round(float(v.mean()), 1), "sd": round(float(v.std()), 1),
                                 "min": int(v.min()), "max": int(v.max())}
        t = np.array([r["time_s"] for r in solved])
        out["time_s"] = {"mean": round(float(t.mean()), 2), "max": round(float(t.max()), 2)}
        per_rung = np.array([r["per_rung"] for r in solved], dtype=float)
        out["per_rung_mean"] = dict(zip(report["rungs"], [round(float(x), 1) for x in per_rung.mean(0)]))
    return out


def format_summary(report: dict) -> str:
    s = report["summary"]
    lines = [f"acceptance verdict: {s['verdict']}  ({s['solved']}/{s['states_planned']} solved; "
             f"{s['solved_pass1']} within the first time limit; {s['states_run']} states run)"]
    for m, st in s.get("lengths", {}).items():
        lines.append(f"  length in {m}: mean {st['mean']}, sd {st['sd']}, min {st['min']}, max {st['max']}")
    if "time_s" in s:
        lines.append(f"  time per solved state: mean {s['time_s']['mean']} s, max {s['time_s']['max']} s")
    if s.get("per_rung_mean"):
        lines.append("  mean cost per rung: " + ", ".join(f"{k} {v}" for k, v in s["per_rung_mean"].items()))
    if s["slow_states"]:
        lines.append("  solved only with more time: " + ", ".join(f"#{x['index'] + 1} ({x['time_s']} s)" for x in s["slow_states"]))
    if s["failures_by_rung"]:
        lines.append("  unsolved, by rung and reason: " + ", ".join(f"{k} x{v}" for k, v in s["failures_by_rung"].items()))
    return "\n".join(lines)
