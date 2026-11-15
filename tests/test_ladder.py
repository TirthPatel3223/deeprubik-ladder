"""Harvesting macros and building a ladder, end to end on the toy puzzle."""
from __future__ import annotations

import numpy as np

from core.coset_table import build_table
from core.harvest import ball, collisions, schreier_elements
from core.ladder import Ladder
from core.quotient import ActionSet, Quotient
from puzzles.toy.two_face_corners import LADDER, STATE_COUNT


def _quiet(_msg: str) -> None:
    pass


def test_meet_in_the_middle_finds_only_stabiliser_elements(toy) -> None:
    _, model, st = toy
    place = st.feature(0, "multiset")
    tables, words, costs = ball(model, ActionSet.primitives(model), 5)
    assert costs.max() == 5
    found = collisions(model, tables, words, [place])
    assert found
    for key, word in found.items():
        table = np.frombuffer(key, dtype=np.int16).astype(np.int32)
        assert np.array_equal(model.compile(word), table)                  # the word produces the element
        assert np.array_equal(place.compute(table), place.goal)            # and leaves every corner in its place
        assert len(word) <= 10


def test_schreier_words_fix_the_rung(toy) -> None:
    _, model, st = toy
    place = st.feature(0, "multiset")
    q = Quotient([place], ActionSet.primitives(model))
    table = build_table(q)
    words = schreier_elements(model, q, table, max_cost=12, n_random=30)
    assert len(words) > 30
    tables = model.compile_many(words)
    assert (place.compute(tables) == place.goal).all()
    assert all(len(w) > 0 for w in words)


def test_ladder_solves_every_scramble_and_indices_multiply_to_the_group_order(toy) -> None:
    puzzle, model, st = toy
    ladder = Ladder(model, st, LADDER, mitm_radius=4, log=_quiet).build()
    counts = [len(r.table) for r in ladder.rungs]
    assert counts == [120, 243]
    assert counts[0] * counts[1] == STATE_COUNT
    rng = np.random.default_rng(0)
    for _ in range(200):
        scramble = rng.integers(0, 4, size=50).tolist()
        state = puzzle.apply_word(puzzle.goal, scramble)
        word, per_rung = ladder.solve(state)
        assert np.array_equal(puzzle.apply_word(state, list(word)), puzzle.goal)
        assert len(word) <= sum(per_rung)                                  # merging turns never lengthens


def test_second_rung_actions_keep_the_first_rung_solved(toy) -> None:
    _, model, st = toy
    ladder = Ladder(model, st, LADDER, mitm_radius=4, log=_quiet).build()
    place = st.feature(0, "multiset")
    tables = ladder.rungs[1].actions.tables
    assert (place.compute(tables) == place.goal).all()
    assert ladder.rungs[1].actions.is_symmetric()


def test_ladder_cache_round_trip(toy, tmp_path) -> None:
    puzzle, model, st = toy
    first = Ladder(model, st, LADDER, mitm_radius=4, log=_quiet).build(cache_dir=tmp_path)
    again = Ladder(model, st, LADDER, mitm_radius=4, log=_quiet).build(cache_dir=tmp_path)
    for a, b in zip(first.rungs, again.rungs):
        assert a.actions.words == b.actions.words
        assert np.array_equal(a.table.keys, b.table.keys) and np.array_equal(a.table.dist, b.table.dist)
    state = puzzle.apply_word(puzzle.goal, [0, 2, 0, 3, 1, 2, 2, 0])
    assert first.solve(state)[0] == again.solve(state)[0]
