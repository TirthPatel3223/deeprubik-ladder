"""The training autopilot and the one-command driver, end to end on the toy puzzle."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from core.acceptance import AcceptanceConfig
from core.auto import AutoConfig, run_auto
from core.autotrain import AutoTrainConfig, _plateaued, _stalled, autotrain_rung, match_curve, mixing_length
from core.ladder import Ladder, RungSpec
from core.quotient import TorchQuotient
from puzzles.registry import PuzzleEntry, get_puzzle

TINY = AutoTrainConfig(test_states=8, final_states=8, min_depth=50, max_depth=200, round_minutes=0.05, eval_beam=64,
                       eval_time=10.0, stall_tests=2, min_updates_per_repair=2, plateau_tests=2, max_minutes=2.0,
                       res_dim=64, num_blocks=1, batch_size=256, states_per_update=4000, itrs_per_update=20, walk_min=5)
QUICK_ACCEPT = AcceptanceConfig(n_states=8, min_depth=50, max_depth=200, time_limit=20.0, extended_time_limit=20.0)
TOY_LEARNED = [RungSpec("place", [(0, "multiset")], max_actions=16),
               RungSpec("twist", [(0, "full")], mode="learned", max_actions=64)]


def _quiet(_msg: str) -> None:
    pass


def test_progress_rules() -> None:
    rows = lambda *xs: [{"solved": x, "mean_cost": c} for x, c in xs]  # noqa: E731
    assert not _stalled(rows((0.1, None), (0.2, None)), 2)
    assert _stalled(rows((0.5, None), (0.4, None), (0.5, None)), 2)
    assert not _stalled(rows((0.5, None), (0.4, None), (0.6, None)), 2)
    assert not _plateaued(rows((1.0, 50.0), (1.0, 45.0)), 2, 0.01, 1.0)
    assert _plateaued(rows((1.0, 45.0), (1.0, 45.2), (1.0, 44.9)), 2, 0.01, 1.0)
    assert not _plateaued(rows((1.0, 45.0), (1.0, 46.0), (1.0, 40.0)), 2, 0.01, 1.0)
    assert not _plateaued(rows((0.9, 30.0), (1.0, 45.0), (1.0, 45.0)), 2, 0.01, 1.0)   # partial solves do not count


def test_walk_length_rule_on_the_toy(toy) -> None:
    _, model, st = toy
    ladder = Ladder(model, st, TOY_LEARNED, mitm_radius=4, log=_quiet).build()
    curve = match_curve(TorchQuotient(ladder.rungs[1].quotient, "cpu"), n_walks=2048, max_len=300)
    # one twisting macro already scrambles the twists: the curve starts near its mixed value (1/3)
    assert curve[0] < 1.0 and abs(curve[-50:].mean() - 1 / 3) < 0.02
    assert 1 <= mixing_length(curve) <= 60


def test_autopilot_trains_a_rung_and_does_not_redo_it(toy, tmp_path) -> None:
    _, model, st = toy
    ladder = Ladder(model, st, TOY_LEARNED, mitm_radius=4, log=_quiet).build(cache_dir=tmp_path)
    ladder, record = autotrain_rung(ladder, 1, tmp_path, TINY, log=_quiet)
    assert record["phase"] == "B" and record["final"]["solved"] == 1.0
    assert (tmp_path / "rung01_twist.train.json").exists() and (tmp_path / "rung01_twist.pt").exists()
    assert ladder.ready()

    again = Ladder(model, st, TOY_LEARNED, mitm_radius=4, log=_quiet).build(cache_dir=tmp_path)
    updates = again.rungs[1].heuristic.update
    _, record2 = autotrain_rung(again, 1, tmp_path, TINY, log=_quiet)
    if record["status"] == "done":
        assert again.rungs[1].heuristic.update == updates                    # finished: not trained again
    assert record2["final"]["solved"] == 1.0


def test_action_repair_rebuilds_the_rung_and_keeps_the_network(toy, tmp_path) -> None:
    """Force the second repair (more actions) by allowing no walk repairs and a target nothing can reach."""
    _, model, st = toy
    ladder = Ladder(model, st, TOY_LEARNED, mitm_radius=4, log=_quiet).build(cache_dir=tmp_path)
    cfg = replace(TINY, target=2.0, max_walk_repairs=0, max_width_repairs=0, max_minutes=0.6, stall_tests=1,
                  min_updates_per_repair=1)
    ladder, record = autotrain_rung(ladder, 1, tmp_path, cfg, log=_quiet)
    kinds = [r["kind"] for r in record["repairs"]]
    assert kinds[:1] == ["actions"]
    assert ladder.specs[1].max_actions == 128 and ladder.specs[1].max_bridges == 512
    assert record["fingerprint"] == ladder.rungs[1].info["fingerprint"]["actions"]


def test_one_command_on_a_hand_written_ladder(toy, tmp_path) -> None:
    puzzle, _, _ = toy
    toy_entry = get_puzzle("toy")
    entry = PuzzleEntry("toy", toy_entry.title, toy_entry.build, lambda: {"learned": TOY_LEARNED}, toy_entry.notation)
    cfg = AutoConfig(ladder="learned", radius=4, train=TINY, accept=QUICK_ACCEPT)
    summary = run_auto(entry, tmp_path, cfg, log=_quiet)
    assert summary["verdict"] == "PASS"
    for name in ("ladder.json", "acceptance.json", "report.md", "run.json", "auto.log"):
        assert (tmp_path / name).exists(), name
    assert json.loads((tmp_path / "ladder.json").read_text())["source"] == "registered:learned"
    assert "PASS" in (tmp_path / "report.md").read_text()
    again = run_auto(entry, tmp_path, cfg, log=_quiet)                         # everything resumes from files
    assert again["verdict"] == "PASS"


def test_one_command_with_an_automatic_ladder(tmp_path) -> None:
    cfg = AutoConfig(ladder="auto", radius=4, propose_radius=4, train=TINY, accept=QUICK_ACCEPT)
    summary = run_auto(get_puzzle("toy"), tmp_path, cfg, log=_quiet)
    assert summary["verdict"] == "PASS"
    assert summary["stages"]["ladder"]["rungs"] == ["first"]                 # the whole toy fits one exact table


def test_a_test_stops_early_only_after_a_failure(toy, monkeypatch) -> None:
    """With no time budget left, a test still runs to the end while every state is solved, and stops once one
    has failed: a weak network costs little, a perfect one is always checked on every state."""
    import numpy as np

    import core.autotrain as at
    from core.davi import DaviConfig, RungValue

    _, model, st = toy
    ladder = Ladder(model, st, TOY_LEARNED, mitm_radius=4, log=_quiet).build()
    q = ladder.rungs[1].quotient
    value = RungValue(q, DaviConfig(res_dim=16, num_blocks=1), device="cpu")
    goal = q.goal[None, :]
    other = q.step(goal, 0)                                        # one action away: not the goal
    monkeypatch.setattr(at, "beam_search", lambda tq, vf, start, **kw: ([], 0.0) if bool(tq.is_goal(start[None])[0]) else None)

    # budget -1: already spent (a zero budget can read as unspent on a coarse clock)
    perfect = at.evaluate(value, np.repeat(goal, 20, axis=0), 64, 0.3, 5.0, give_up=8, budget=-1.0)
    assert perfect["complete"] and perfect["tested"] == 20 and perfect["solved"] == 1.0
    mixed = at.evaluate(value, np.concatenate([np.repeat(goal, 10, axis=0), np.repeat(other, 20, axis=0)]),
                        64, 0.3, 5.0, give_up=8, budget=-1.0)
    assert not mixed["complete"] and mixed["tested"] == 11 and mixed["solved"] == 10 / 11


def test_the_best_network_is_kept(toy, tmp_path, monkeypatch) -> None:
    """Phase B keeps training while the cost falls; if later networks score worse, the best one is restored."""
    import core.autotrain as at

    _, model, st = toy
    ladder = Ladder(model, st, TOY_LEARNED, mitm_radius=4, log=_quiet).build(cache_dir=tmp_path)
    scores = iter([50.0, 40.0, 45.0, 46.0, 47.0])              # TINY stops after 2 tests without a gain
    seen: list[int] = []

    def scripted(value, states, *a, **kw):
        if "budget" not in kw:                                  # the final test on the other states
            return {"solved": 1.0, "tested": len(states), "n": len(states), "complete": True, "mean_cost": 40.0,
                    "mean_actions": 5.0, "seconds_per_state": 0.0, "mean_value": 0.0, "update": value.update}
        seen.append(value.update)
        return {"solved": 1.0, "tested": len(states), "n": len(states), "complete": True, "mean_cost": next(scores),
                "mean_actions": 5.0, "seconds_per_state": 0.0, "mean_value": 0.0}

    monkeypatch.setattr(at, "evaluate", scripted)
    cfg = replace(TINY, round_minutes=0.02, max_minutes=5.0)
    ladder, record = at.autotrain_rung(ladder, 1, tmp_path, cfg, log=_quiet)
    assert record["status"] == "done" and record["best"]["mean_cost"] == 40.0
    assert ladder.rungs[1].heuristic.update == seen[1]                 # the network of the second test
    assert record["final"]["update"] == seen[1]


@pytest.mark.parametrize("bad", ["no_such_ladder"])
def test_unknown_registered_ladder_is_refused(tmp_path, bad) -> None:
    with pytest.raises(KeyError):
        run_auto(get_puzzle("toy"), tmp_path, AutoConfig(ladder=bad, radius=4), log=_quiet)
