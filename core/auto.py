"""One command from a puzzle to a certified solver: analyse, design the ladder, build it, train it, accept it.

Stages, each resumable because its results are files in the output directory:

1. analyse  the labelled model and its structure (orbits, pieces, candidate features);
2. ladder   the rung specification: one of the puzzle's registered ladders, or one designed automatically
            (core/discover.py). It is written to ``ladder.json``, which later stages and re-runs read;
3. build    macros and exact tables (core/ladder.py), cached in ``rungs/``;
4. train    every learned rung in order, by the autopilot (core/autotrain.py);
5. accept   the acceptance test (core/acceptance.py), report in ``acceptance.json``;
6. repair   if the test fails in learned rungs, those rungs are trained again with the failing states
            added to their tests, and the test is run again (at most ``repair_rounds`` times). A failure in
            an exact table is a gap in the ladder itself and is reported, not repaired here.

``report.md`` summarises the run, ``run.json`` records it, and ``auto.log`` has everything that was printed.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from core.acceptance import AcceptanceConfig, deep_scrambles, format_summary, run_acceptance
from core.autotrain import AutoTrainConfig, autotrain_rung
from core.fingerprint import puzzle_fingerprint
from core.ladder import Ladder, RungFailure, RungSpec, retire, specs_from_json, specs_to_json
from core.perm_model import build_point_model
from core.structure import analyse


@dataclass
class AutoConfig:
    ladder: str = "auto"                  # "auto" (designed here) or the name of a registered ladder
    radius: int = 7                       # meet-in-the-middle radius of the build
    propose_radius: int = 6               # ... and of the ladder design
    table_budget: int = 10_000_000        # largest coset count the design gives an exact table
    repair_rounds: int = 2
    repair_minutes: float = 60.0          # training per failing rung and repair round
    train: AutoTrainConfig = field(default_factory=AutoTrainConfig)
    accept: AcceptanceConfig = field(default_factory=AcceptanceConfig)


def default_out_dir(entry, cfg: AutoConfig) -> Path:
    return Path("data") / "auto" / entry.name / f"{cfg.ladder}_r{cfg.radius}"


def _tee(log: Callable[[str], None] | None, path: Path) -> Callable[[str], None]:
    echo = log or (lambda msg: print(msg, flush=True))

    def both(msg: str) -> None:
        echo(msg)
        with path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")
    return both


def _design(entry, model, structure, out: Path, cfg: AutoConfig, log) -> list[RungSpec]:
    """The rung specification, from ladder.json if a previous run wrote one for this puzzle and source."""
    path = out / "ladder.json"
    source = "auto" if cfg.ladder == "auto" else f"registered:{cfg.ladder}"
    fp = puzzle_fingerprint(model.puzzle)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("puzzle_fingerprint") == fp and saved.get("source") == source and saved.get("radius") == cfg.radius:
            log(f"ladder: {len(saved['specs'])} rungs from {path}")
            return specs_from_json(saved["specs"])
        log(f"ladder: {path} is for another puzzle or design: set aside as {retire(path, None).name}")
    if cfg.ladder == "auto":
        from core.discover import propose_ladder

        log(f"ladder: designing it from the move tables (meet in the middle to radius {cfg.propose_radius})")
        specs, _ = propose_ladder(model, structure, radius=cfg.propose_radius, table_budget=cfg.table_budget, log=log)
    else:
        ladders = entry.ladders()
        if cfg.ladder not in ladders:
            raise KeyError(f"{entry.title} has no ladder {cfg.ladder!r}; it has {sorted(ladders)} (or use 'auto')")
        specs = list(ladders[cfg.ladder])
    _save_design(path, entry, fp, source, cfg.radius, specs)
    for i, s in enumerate(specs):
        log(f"  rung {i}: {s.name:20s} {s.mode:8s} features {s.features}")
    return specs


def _save_design(path: Path, entry, fp: str, source: str, radius: int, specs: list[RungSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"puzzle": entry.name, "puzzle_fingerprint": fp, "source": source, "radius": radius,
                                "specs": specs_to_json(specs)}, indent=1))


def failing_states(ladder: Ladder, puzzle, accept_cfg: AcceptanceConfig, report: dict, index: int) -> np.ndarray:
    """Rung ``index``'s feature vectors of the acceptance states that failed in that rung."""
    name = ladder.rungs[index].spec.name
    scrambles = deep_scrambles(accept_cfg, puzzle.num_moves)
    out = []
    for f in report["summary"]["failures"]:
        if f["rung"] != name:
            continue
        state = puzzle.apply_word(puzzle.goal, scrambles[f["index"]][1])
        try:
            lab, _, _ = ladder.descend(ladder.structure.decode(state), upto=index)
        except RungFailure:
            continue
        out.append(ladder.rungs[index].quotient.project(lab))
    q = ladder.rungs[index].quotient
    return np.stack(out) if out else np.empty((0, q.n_slots), dtype=np.uint8)


def run_auto(entry, out_dir: str | Path | None = None, cfg: AutoConfig | None = None,
             log: Callable[[str], None] | None = None) -> dict:
    """Run every stage for the puzzle of ``entry`` (a ``puzzles.registry.PuzzleEntry``). Returns the summary
    also written to run.json; its ``verdict`` is the acceptance verdict, or FAIL if a stage could not finish."""
    cfg = cfg or AutoConfig()
    out = Path(out_dir) if out_dir is not None else default_out_dir(entry, cfg)
    out.mkdir(parents=True, exist_ok=True)
    log = _tee(log, out / "auto.log")
    t_start = time.time()
    summary: dict = {"puzzle": entry.name, "title": entry.title, "ladder_source": cfg.ladder, "config": asdict(cfg),
                     "started": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": {}}

    def save_summary() -> None:
        summary["minutes"] = round((time.time() - t_start) / 60, 1)
        (out / "run.json").write_text(json.dumps(summary, indent=1, default=str))

    log(f"=== {entry.title}: automatic run in {out}")
    puzzle = entry.build()
    model = build_point_model(puzzle)
    structure = analyse(model)
    log(f"analyse: {puzzle.num_moves} moves ({' '.join(puzzle.move_names)}), state length {puzzle.dim}")
    log(structure.describe())
    summary["stages"]["analyse"] = {"points": model.n, "orbits": len(structure.orbits)}

    specs = _design(entry, model, structure, out, cfg, log)
    summary["stages"]["ladder"] = {"rungs": [s.name for s in specs], "modes": [s.mode for s in specs]}
    save_summary()

    rungs_dir = out / "rungs"

    def make() -> Ladder:
        return Ladder(model, structure, specs, mitm_radius=cfg.radius, log=log).build(cache_dir=rungs_dir)

    def set_spec(i: int, spec: RungSpec) -> Ladder:
        specs[i] = spec
        _save_design(out / "ladder.json", entry, puzzle_fingerprint(puzzle), "auto" if cfg.ladder == "auto"
                     else f"registered:{cfg.ladder}", cfg.radius, specs)
        return make()

    t0 = time.time()
    try:
        ladder = make()
    except (RuntimeError, AssertionError) as err:
        log(f"build failed: {err}")
        summary["stages"]["build"] = {"error": str(err)}
        summary["verdict"] = "FAIL"
        save_summary()
        return summary
    summary["stages"]["build"] = {"seconds": round(time.time() - t0, 1),
                                  "rungs": [{"name": r.spec.name, "mode": r.spec.mode, "actions": len(r.actions),
                                             "cost_min": int(r.actions.costs.min()), "cost_max": int(r.actions.costs.max()),
                                             "cosets": r.info.get("cosets")} for r in ladder.rungs]}
    save_summary()

    training: dict = {}
    for i in range(len(ladder.rungs)):
        if ladder.rungs[i].spec.mode == "learned":
            ladder, record = autotrain_rung(ladder, i, rungs_dir, cfg.train, set_spec=set_spec, log=log)
            training[ladder.rungs[i].spec.name] = record
            summary["stages"]["train"] = {k: _train_brief(v) for k, v in training.items()}
            save_summary()
    if not ladder.ready():
        summary["verdict"] = "FAIL"
        log("the ladder is not complete after training")
        save_summary()
        return summary

    notation = entry.notation(ladder.alg)
    report = run_acceptance(ladder, puzzle, cfg.accept, report_path=out / "acceptance.json", notation=notation, log=log)
    rounds = []
    for round_ in range(cfg.repair_rounds):
        if report["verdict"] in ("PASS", "PASS_SLOW"):
            break
        names = [r.spec.name for r in ladder.rungs]
        failing = {}
        for f in report["summary"]["failures"]:
            if f["rung"] in names and ladder.rungs[names.index(f["rung"])].spec.mode == "learned":
                failing[f["rung"]] = failing.get(f["rung"], 0) + 1
        if not failing:
            log("acceptance failed outside the learned rungs (a state outside an exact table, or a solution that "
                "does not replay): that is a gap in the ladder, not something more training can fix")
            break
        log(f"repair round {round_ + 1}: retraining {failing} with the states they failed on")
        for name in sorted(failing, key=lambda n: -failing[n]):
            i = names.index(name)
            extra = failing_states(ladder, puzzle, cfg.accept, report, i)
            ladder, record = autotrain_rung(ladder, i, rungs_dir, cfg.train, set_spec=set_spec, extra_states=extra,
                                            budget_minutes=cfg.repair_minutes, log=log)
            training[name] = record
        ladder = make()
        report = run_acceptance(ladder, puzzle, cfg.accept, report_path=out / "acceptance.json", notation=notation, log=log)
        rounds.append({"retrained": failing, "verdict": report["verdict"]})

    summary["stages"]["train"] = {k: _train_brief(v) for k, v in training.items()}
    summary["stages"]["accept"] = report["summary"]
    summary["stages"]["repair"] = rounds
    summary["verdict"] = report["verdict"]
    save_summary()
    (out / "report.md").write_text(_report_md(entry, structure, ladder, training, report, summary), encoding="utf-8")
    log(f"=== verdict {summary['verdict']} after {summary['minutes']} min; report in {out / 'report.md'}")
    return summary


def _train_brief(record: dict) -> dict:
    fin = record.get("final") or {}
    return {"status": record.get("status"), "minutes": round(record.get("minutes", 0), 1), "updates": record.get("updates"),
            "walk_max": record.get("walk_max"), "actions": record.get("actions"),
            "repairs": [r["kind"] for r in record.get("repairs", [])],
            "final_solved": fin.get("solved"), "final_mean_cost": fin.get("mean_cost")}


def _report_md(entry, structure, ladder: Ladder, training: dict, report: dict, summary: dict) -> str:
    lines = [f"# Automatic run: {entry.title}", "",
             f"Verdict **{summary['verdict']}** after {summary['minutes']} minutes (ladder: {summary['ladder_source']}).", "",
             "## Structure", "", "```", structure.describe(), "```", "",
             "## Ladder", "", "| # | Rung | Features | Heuristic | Actions | Costs |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(ladder.rungs):
        heur = f"table, {len(r.table):,} cosets" if r.table is not None else "value network"
        lines.append(f"| {i} | {r.spec.name} | {', '.join(f.name for f in r.features)} | {heur} | {len(r.actions)} | "
                     f"{int(r.actions.costs.min())}..{int(r.actions.costs.max())} |")
    if training:
        lines += ["", "## Training", "", "| Rung | Status | Minutes | Updates | Walks | Repairs | Final test solved | Mean cost |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, rec in training.items():
            b = _train_brief(rec)
            cost = "-" if b["final_mean_cost"] is None else f"{b['final_mean_cost']:.1f}"
            solved = "-" if b["final_solved"] is None else f"{100 * b['final_solved']:.0f}%"
            lines.append(f"| {name} | {b['status']} | {b['minutes']} | {b['updates']} | {b['walk_max']} | "
                         f"{', '.join(b['repairs']) or 'none'} | {solved} | {cost} |")
    lines += ["", "## Acceptance test", "", "```", format_summary(report), "```", ""]
    return "\n".join(lines)
