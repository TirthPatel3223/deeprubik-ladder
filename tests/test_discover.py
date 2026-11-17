"""The measured rung-ordering rules, on the toy puzzle where the right answer is known."""
from __future__ import annotations

import numpy as np

from core.discover import forced_all, generator_periods, linked_orbits, propose_ladder
from core.ladder import Ladder
from puzzles.tangram.ladder import O_BOTTOM
from puzzles.toy.two_face_corners import STATE_COUNT


def test_toy_periods_and_groups(toy) -> None:
    _, model, st = toy
    assert generator_periods(model, st) == [[4, 4, 4, 4]]
    assert linked_orbits(model, st) == [[0]]


def test_proposed_ladder_places_then_twists_and_solves(toy) -> None:
    puzzle, model, st = toy
    specs, report = propose_ladder(model, st, radius=4, log=lambda _msg: None)
    # Placing the corners first leaves twist macros of 6 to 8 moves; solving everything at
    # once leaves nothing. So the proposal is the ladder a person would write: place, then twist.
    assert [s.features for s in specs] == [[(0, "multiset")], [(0, "full")]]
    assert all(s.mode == "table" for s in specs)
    first = report["steps"][0]["candidates"]
    by_name = {c.name: c for c in first}
    assert by_name["orbit0.multiset"].cosets == 120 and by_name["orbit0.full"].cosets == STATE_COUNT
    assert by_name["orbit0.multiset"].score > 0 and by_name["orbit0.full"].score == 0

    ladder = Ladder(model, st, specs, mitm_radius=4, log=lambda _msg: None).build()
    assert [len(r.table) for r in ladder.rungs] == [120, 243]
    rng = np.random.default_rng(0)
    for _ in range(50):
        state = puzzle.apply_word(puzzle.goal, rng.integers(0, 4, size=40).tolist())
        word, _ = ladder.solve(state)
        assert np.array_equal(puzzle.apply_word(state, list(word)), puzzle.goal)


def test_proposal_is_completed_by_the_exact_check(tangram_original) -> None:
    """With short macros the greedy rules leave the original Tangram Cube's bottom turning corner
    unplaced: once the other rungs are solved no short word reaches it, so it looked solved. The
    exact check finds it is not and puts it in a first rung; the proposal then solves everything."""
    _, model, st = tangram_original
    specs, report = propose_ladder(model, st, radius=3, log=lambda _msg: None)
    assert O_BOTTOM in report["not_forced"] and specs[0].name == "first"
    feats = [st.feature(*fs) for spec in specs for fs in spec.features]
    everything = [st.feature(o.index, "full") for o in st.orbits]
    assert forced_all(model, feats, everything) == [True] * len(everything)
