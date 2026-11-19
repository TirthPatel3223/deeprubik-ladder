"""The measured rung-choosing rules, on the toy puzzle where the right answer is known, and the pieces of
the automatic ladder design that can be checked exactly."""
from __future__ import annotations

import itertools
import math

import numpy as np

from core.discover import (_assignments, _permanent, forced_all, generator_periods, goal_arrangements,
                           linked_orbits, orbit_links, propose_ladder)
from core.ladder import Ladder
from puzzles.tangram import ladder as tl
from puzzles.tangram.ladder import O_BOTTOM
from puzzles.toy.two_face_corners import STATE_COUNT


def _quiet(_msg: str) -> None:
    pass


def test_toy_periods_and_groups(toy) -> None:
    _, model, st = toy
    assert generator_periods(model, st) == [[4, 4, 4, 4]]
    assert linked_orbits(model, st) == [[0]]


def test_proposed_ladder_places_then_twists_and_solves(toy) -> None:
    puzzle, model, st = toy
    specs, report = propose_ladder(model, st, radius=4, bundle=False, log=_quiet)
    # Placing the corners first leaves twist macros of 6 to 8 moves; solving everything at
    # once leaves nothing. So the proposal is the ladder a person would write: place, then twist.
    assert [s.features for s in specs] == [[(0, "multiset")], [(0, "full")]]
    assert all(s.mode == "table" for s in specs)
    first = report["steps"][0]["candidates"]
    by_name = {c.name: c for c in first}
    assert by_name["orbit0.multiset"].cosets == 120 and by_name["orbit0.full"].cosets == STATE_COUNT
    assert by_name["orbit0.multiset"].score > 0 and by_name["orbit0.full"].score == 0

    ladder = Ladder(model, st, specs, mitm_radius=4, log=_quiet).build()
    assert [len(r.table) for r in ladder.rungs] == [120, 243]
    rng = np.random.default_rng(0)
    for _ in range(50):
        state = puzzle.apply_word(puzzle.goal, rng.integers(0, 4, size=40).tolist())
        word, _ = ladder.solve(state)
        assert np.array_equal(puzzle.apply_word(state, list(word)), puzzle.goal)


def test_a_puzzle_that_fits_one_table_gets_one_rung(toy) -> None:
    """Rule 0: the whole toy (29,160 states) fits one exact table, which is optimal."""
    _, model, st = toy
    specs, _ = propose_ladder(model, st, radius=4, log=_quiet)
    assert [(s.name, s.mode, s.features) for s in specs] == [("first", "table", [(0, "full")])]


def test_schreier_words_reach_the_corner_short_macros_miss(tangram_original) -> None:
    """Once the other pieces are placed, no word of up to 6 moves turns the original Tangram Cube's bottom
    corner (the old proposal left it unplaced). The pool now also holds the Schreier words of the table rungs,
    as the builder's does, so even without rule 0 the corner gets its rung and everything is forced."""
    _, model, st = tangram_original
    specs, report = propose_ladder(model, st, radius=3, bundle=False, log=_quiet)
    assert any((O_BOTTOM, "full") in s.features for s in specs) and report["not_forced"] == []
    feats = [st.feature(*fs) for spec in specs for fs in spec.features]
    everything = [st.feature(o.index, "full") for o in st.orbits]
    assert forced_all(model, feats, everything) == [True] * len(everything)


def test_an_orbit_left_unforced_joins_the_first_rung(tangram_original, monkeypatch) -> None:
    """If the completeness check finds an orbit not forced, the proposal is made again with it in the first rung."""
    import core.discover as discover

    _, model, st = tangram_original
    real, calls = discover.forced_all, {"n": 0}
    everything = [st.feature(o.index, "full") for o in st.orbits]

    def first_check_misses_the_corner(m, fixed, features):
        out = real(m, fixed, features)
        if len(features) == len(everything) and calls["n"] == 0 and out is not None:
            calls["n"] += 1
            out = [ok and i != O_BOTTOM for i, ok in enumerate(out)]
        return out

    monkeypatch.setattr(discover, "forced_all", first_check_misses_the_corner)
    specs, _ = propose_ladder(model, st, radius=3, bundle=False, log=_quiet)
    assert calls["n"] == 1 and specs[0].name == "first" and specs[0].features == [(O_BOTTOM, "full")]


def test_small_kinds_of_piece_go_first_together(tangram_original) -> None:
    """Rule 0 puts the turning corners and the middle layer (tiny coset spaces) in a first rung from the start."""
    _, model, st = tangram_original
    specs, report = propose_ladder(model, st, radius=3, log=_quiet)
    assert specs[0].name == "first" and (O_BOTTOM, "full") in specs[0].features
    feats = [st.feature(*fs) for spec in specs for fs in spec.features]
    everything = [st.feature(o.index, "full") for o in st.orbits]
    assert forced_all(model, feats, everything) == [True] * len(everything)
    assert report["predicted_cost"] > 0


def test_permanent_against_brute_force() -> None:
    rng = np.random.default_rng(0)
    for n in range(1, 6):
        a = rng.integers(0, 3, size=(n, n))
        brute = sum(math.prod(int(a[i, p[i]]) for i in range(n)) for p in itertools.permutations(range(n)))
        assert _permanent(a.tolist()) == brute == _assignments(a)


def test_goal_arrangements_count_look_alikes(maltese, tangram) -> None:
    _, mmodel, mst = maltese
    assert goal_arrangements([mst.feature(2, "full")]) == math.factorial(4) ** 6     # 6 colours x 4 identical gears
    ring = goal_arrangements([mst.feature(1, "full")])
    assert 1024 <= ring <= 4 ** 6 * 2 ** 12                                         # the true stabiliser has 1024
    _, tmodel, tst = tangram
    assert goal_arrangements([tst.feature(tl.LARGE_TRI_A, "full")]) == math.factorial(6)   # the 6 empty places

    # A gear's colour and rotation move together: one arrangement of the gears serves both orbits. Counted
    # apart, the rotation orbit alone allows 24! goal-looking arrangements (every gear looks alike there).
    gears = [mst.feature(2, "full"), mst.feature(3, "full")]
    assert goal_arrangements(gears, orbit_links(mmodel, mst)) == math.factorial(4) ** 6
    assert goal_arrangements(gears) == math.factorial(4) ** 6 * math.factorial(24)
    # The two large-triangle classes of the Tangram also move together, and the empty places of one are the
    # occupied places of the other, so only the solved arrangement shows the goal: the joint rung has
    # |H| = 12!/2 cosets, which is what the table ladder measured (665,280 x 360).
    large = [tst.feature(tl.LARGE_TRI_A, "full"), tst.feature(tl.LARGE_TRI_B, "full")]
    assert goal_arrangements(large, orbit_links(tmodel, tst)) == 1
    assert 665_280 * 360 == math.factorial(12) // 2
