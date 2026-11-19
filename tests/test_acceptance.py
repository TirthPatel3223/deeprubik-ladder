"""The acceptance test's decision rule, with a scripted solver, and once for real on the toy ladder."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from core.acceptance import AcceptanceConfig, deep_scrambles, run_acceptance
from core.harvest import WordAlgebra
from core.ladder import Ladder, RungFailure
from puzzles.toy.two_face_corners import LADDER

CFG = AcceptanceConfig(n_states=20, min_depth=10, max_depth=30, warmup=False)    # 2 failures allowed


class ScriptedLadder:
    """Solves the toy scrambles by undoing them, except where the script says otherwise.
    ``script[k] = (pass-1 outcome, pass-2 outcome)``, each 'ok', 'wrong' or a RungFailure reason."""

    def __init__(self, toy, cfg: AcceptanceConfig, script: dict[int, tuple[str, str]], stop_at: int | None = None):
        puzzle, model, _ = toy
        self.alg = WordAlgebra(model)
        self.rungs = [SimpleNamespace(spec=SimpleNamespace(name="r0")), SimpleNamespace(spec=SimpleNamespace(name="r1"))]
        self.extended = cfg.schedules()[1]
        self.index = {puzzle.apply_word(puzzle.goal, w).tobytes(): (k, w) for k, (_, w) in enumerate(deep_scrambles(cfg, 4))}
        self.script, self.stop_at, self.calls = script, stop_at, 0

    def ready(self) -> bool:
        return True

    def fingerprint(self) -> str:
        return "scripted"

    def solve(self, state, time_limit=None, schedule=None):
        k, word = self.index[state.tobytes()]
        if self.stop_at is not None and self.calls == self.stop_at:
            raise KeyboardInterrupt
        self.calls += 1
        outcome = self.script.get(k, ("ok", "ok"))[1 if schedule == self.extended else 0]
        if outcome == "ok":
            return self.alg.invert(word), [len(word), 0]
        if outcome == "wrong":
            return tuple(word), [len(word), 0]                     # does not solve: replay must catch it
        raise RungFailure("r1", outcome, f"r1: {outcome}")


def _run(toy, script, **kw):
    puzzle, _, _ = toy
    ladder = ScriptedLadder(toy, CFG, script, kw.pop("stop_at", None))
    report = run_acceptance(ladder, puzzle, CFG, log=lambda _m: None, **kw)
    return report, ladder


def test_all_solved_passes(toy) -> None:
    report, _ = _run(toy, {})
    assert report["verdict"] == "PASS" and report["summary"]["solved"] == 20
    assert report["summary"]["per_rung_mean"]["r1"] == 0.0


def test_a_few_slow_states_pass_after_the_second_chance(toy) -> None:
    report, _ = _run(toy, {3: ("timeout", "ok"), 11: ("search_exhausted", "ok")})
    s = report["summary"]
    assert report["verdict"] == "PASS_SLOW" and s["solved_pass1"] == 18 and s["solved"] == 20
    assert [x["index"] for x in s["slow_states"]] == [3, 11]


def test_a_state_unsolved_after_the_second_chance_fails(toy) -> None:
    report, _ = _run(toy, {3: ("timeout", "ok"), 11: ("timeout", "timeout")})
    assert report["verdict"] == "FAIL"
    assert report["summary"]["failures_by_rung"] == {"r1: timeout": 1}


def test_too_many_failures_stop_early(toy) -> None:
    report, ladder = _run(toy, {k: ("timeout", "ok") for k in (1, 2, 4, 5, 6, 7)})
    assert report["verdict"] == "FAIL"
    assert report["summary"]["states_run"] == 5 and ladder.calls == 5    # stopped once 3 > 2 had failed
    assert not any("pass2" in s for s in report["states"])


def test_a_coverage_gap_is_not_given_a_second_chance(toy) -> None:
    report, ladder = _run(toy, {5: ("outside_table", "ok")})
    assert report["verdict"] == "FAIL" and ladder.calls == 20             # pass 2 skipped the state
    assert report["states"][5]["pass2"]["reason"] == "outside_table"


def test_a_solution_that_does_not_replay_fails_the_test(toy) -> None:
    report, _ = _run(toy, {7: ("wrong", "ok")})
    assert report["verdict"] == "FAIL"
    assert report["states"][7]["pass1"]["reason"] == "replay_mismatch"


def test_an_interrupted_run_resumes(toy, tmp_path) -> None:
    path = tmp_path / "acceptance.json"
    with pytest.raises(KeyboardInterrupt):
        _run(toy, {}, report_path=path, stop_at=8)
    assert len(json.loads(path.read_text())["states"]) == 8
    report, ladder = _run(toy, {}, report_path=path)
    assert report["verdict"] == "PASS" and ladder.calls == 12             # only the 12 states not yet run
    again, ladder = _run(toy, {}, report_path=path)
    assert again["verdict"] == "PASS" and ladder.calls == 0               # decided: not run again
    _run(toy, {}, report_path=path, rerun=True)
    assert any(".stale-" in p.name for p in tmp_path.iterdir())          # the old report is kept


def test_the_real_toy_ladder_passes(toy) -> None:
    puzzle, model, st = toy
    ladder = Ladder(model, st, LADDER, mitm_radius=4, log=lambda _m: None).build()
    cfg = AcceptanceConfig(n_states=10, min_depth=50, max_depth=200, time_limit=10.0)
    report = run_acceptance(ladder, puzzle, cfg, log=lambda _m: None)
    assert report["verdict"] == "PASS"
    assert report["summary"]["lengths"]["moves"]["max"] <= 14 + 20       # place (<= 14) then twist
    assert np.isfinite(report["summary"]["time_s"]["max"])
