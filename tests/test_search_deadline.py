"""Search schedules, deadlines, and the reasons a ladder gives when it cannot solve a state."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from core.coset_table import CosetTable
from core.davi import DaviConfig, RungValue
from core.ladder import Ladder, RungFailure, RungSpec
from core.search import SearchSchedule, SearchTimeout, beam_search
from puzzles.toy.two_face_corners import LADDER

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _quiet(_msg: str) -> None:
    pass


@pytest.fixture(scope="module")
def learned_toy(toy):
    """The toy puzzle as one learned rung with a small, untrained network."""
    _, model, st = toy
    ladder = Ladder(model, st, [RungSpec("all", [(0, "full")], mode="learned", max_actions=4)],
                    mitm_radius=4, log=_quiet).build()
    rung = ladder.rungs[0]
    rung.heuristic = RungValue(rung.quotient, DaviConfig(res_dim=16, num_blocks=1), device=DEVICE)
    return ladder


def test_schedules() -> None:
    assert SearchSchedule.standard().attempts == ((4096, 0.3), (32768, 0.3))
    assert SearchSchedule.standard(256).attempts == ((256, 0.3), (2048, 0.3), (16384, 0.3))   # w, 8w, 64w as before
    assert SearchSchedule.standard(64, max_width=100).attempts == ((64, 0.3), (100, 0.3))
    assert SearchSchedule.extended().attempts == ((32768, 0.1), (131072, 0.3), (131072, 0.1))


def test_beam_search_raises_when_out_of_time(learned_toy) -> None:
    rung = learned_toy.rungs[0]
    start = torch.as_tensor(rung.quotient.goal.copy())
    start[0] = (start[0] + 1) % int(rung.quotient.radix[0])              # some state that is not the goal
    with pytest.raises(SearchTimeout):
        beam_search(rung.heuristic.tq, rung.heuristic.value, start, width=8, deadline=0.0)
    goal = torch.as_tensor(rung.quotient.goal)
    assert beam_search(rung.heuristic.tq, rung.heuristic.value, goal, width=8, deadline=0.0) == ([], 0.0)


def test_ladder_reports_a_timeout_with_the_rung(toy, learned_toy) -> None:
    puzzle, _, _ = toy
    state = puzzle.apply_word(puzzle.goal, [0, 2, 0, 3, 1, 2, 2, 0, 3, 3])
    with pytest.raises(RungFailure) as err:
        learned_toy.solve(state, time_limit=-1.0)
    assert (err.value.rung, err.value.reason) == ("all", "timeout")
    assert isinstance(err.value, LookupError)                           # callers that catch LookupError still do


def test_ladder_reports_an_exhausted_search(toy, learned_toy, monkeypatch) -> None:
    puzzle, _, _ = toy
    state = puzzle.apply_word(puzzle.goal, [0, 2, 0, 3, 1])
    monkeypatch.setattr(learned_toy.rungs[0].heuristic, "solve", lambda *a, **k: None)
    with pytest.raises(RungFailure) as err:
        learned_toy.solve(state)
    assert (err.value.rung, err.value.reason) == ("all", "search_exhausted")


def test_ladder_reports_a_state_outside_a_table(toy) -> None:
    puzzle, model, st = toy
    ladder = Ladder(model, st, LADDER, mitm_radius=4, log=_quiet).build()
    full = ladder.rungs[1].table
    goal = int(np.flatnonzero(full.action < 0)[0])
    ladder.rungs[1].table = CosetTable(keys=full.keys[goal:goal + 1], dist=full.dist[goal:goal + 1],
                                       action=full.action[goal:goal + 1], complete=True)   # only the goal
    rng = np.random.default_rng(0)
    state = puzzle.apply_word(puzzle.goal, rng.integers(0, 4, size=60).tolist())
    with pytest.raises(RungFailure) as err:
        ladder.solve(state)
    assert (err.value.rung, err.value.reason) == ("twist", "outside_table")


def test_learned_rung_solves_within_a_generous_time_limit(toy) -> None:
    """A trained toy rung still solves when the solve has a deadline far away."""
    puzzle, model, st = toy
    ladder = Ladder(model, st, [RungSpec("all", [(0, "full")], mode="learned", max_actions=4)],
                    mitm_radius=4, log=_quiet).build()
    rung = ladder.rungs[0]
    cfg = DaviConfig(res_dim=128, num_blocks=1, batch_size=1024, states_per_update=10_000, itrs_per_update=30,
                     walk_max=30, max_updates=20, force_sync=1, test_states=50)
    rung.heuristic = RungValue(rung.quotient, cfg, device=DEVICE)
    rung.heuristic.train(log=_quiet)
    rng = np.random.default_rng(3)
    for _ in range(5):
        state = puzzle.apply_word(puzzle.goal, rng.integers(0, 4, size=40).tolist())
        word, _ = ladder.solve(state, time_limit=60.0, schedule=SearchSchedule.standard(256, max_width=4096))
        assert np.array_equal(puzzle.apply_word(state, list(word)), puzzle.goal)
