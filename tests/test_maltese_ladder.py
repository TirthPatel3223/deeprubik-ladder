"""Regression tests for the measured facts the Maltese Gear Cube ladder rests on.

Each number below was computed from the verified move tables (docs/hierarchical_solver.md).
The full ladder needs trained networks and minutes of table building, so it is exercised by
scripts/solve.py rather than here; the toy puzzle covers the machinery end to end.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from core.coset_table import build_table_auto, table_solve
from core.discover import generator_periods, linked_orbits
from core.harvest import ball, collisions
from core.quotient import ActionSet, Quotient

# Number of corner positions at each distance, quarter turns of three faces. These are the
# known figures for the 2x2x2 cube with one corner fixed (3,674,160 positions, diameter 14).
CORNER_DISTANCES = [1, 6, 27, 120, 534, 2256, 8969, 33058, 114149, 360508, 930588, 1350852, 782536, 90280, 276]


@pytest.fixture(scope="module")
def corner_rung(maltese):
    _, model, st = maltese
    q = Quotient([st.feature(0, "full")], ActionSet.primitives(model))
    return q, build_table_auto(q)


def test_corner_rung_is_the_pocket_cube(corner_rung) -> None:
    _, table = corner_rung
    assert table.complete and len(table) == 3_674_160
    assert np.bincount(table.dist).tolist() == CORNER_DISTANCES


def test_corner_table_solves_scrambles(maltese, corner_rung) -> None:
    puzzle, model, st = maltese
    q, table = corner_rung
    rng = np.random.default_rng(0)
    for _ in range(50):
        state = puzzle.apply_word(puzzle.goal, rng.integers(0, 6, size=80).tolist())
        labels = st.decode(state)
        acts = table_solve(q, table, q.project(labels))
        assert len(acts) <= 14
        solved = puzzle.apply_word(state, acts)                       # actions are the six primitive moves
        assert np.array_equal(solved[[s for c in range(7) for s in st.orbits[0].pieces[c]]],
                              puzzle.goal[[s for c in range(7) for s in st.orbits[0].pieces[c]]])


def test_orbit_kinds_and_periods(maltese) -> None:
    _, model, st = maltese
    assert linked_orbits(model, st) == [[0], [1], [2, 3]]             # gear colour and rotation are the same pieces
    periods = generator_periods(model, st)
    assert periods[0] == [4] * 6                                      # corners: X^4 fixes them
    assert periods[1] == periods[2] == periods[3] == [8] * 6


def test_short_corner_stabiliser_elements(maltese) -> None:
    """All elements that fix the corners, up to 10 quarter turns: X^4 and its relatives."""
    _, model, st = maltese
    tables, words, _ = ball(model, ActionSet.primitives(model), 5)
    found = collisions(model, tables, words, [st.feature(0, "full")])
    assert dict(Counter(len(w) for w in found.values())) == {4: 3, 6: 12, 8: 75, 10: 762}
    assert sorted(w for w in found.values() if len(w) == 4) == [(0, 0, 0, 0), (2, 2, 2, 2), (4, 4, 4, 4)]


def test_pure_ring_words_of_twelve_moves(maltese) -> None:
    """72 words of 12 quarter turns fix every corner and every gear. Each is three 3-cycles
    of ring pieces. These are the cheap macros that make the ring the last rung."""
    _, model, st = maltese
    tables, words, _ = ball(model, ActionSet.primitives(model), 6)
    fixed = [st.feature(0, "full"), st.feature(3, "full"), st.feature(2, "full")]
    found = collisions(model, tables, words, fixed)
    assert len(found) == 72 and {len(w) for w in found.values()} == {12}
    ring = st.orbits[1]
    for key in found:
        table = np.frombuffer(key, dtype=np.int16).astype(np.int32)
        slot_perm = ring.slot_of[table[ring.ref_pos]]
        moved = int((slot_perm != np.arange(18)).sum())
        assert moved == 9
        cube = slot_perm[slot_perm[slot_perm]]
        assert np.array_equal(cube, np.arange(18))                    # order 3 on the slots


def test_gear_rotation_before_colour(maltese) -> None:
    """Within 12 moves, words that fix the corners and every gear rotation still move gears,
    but no word fixes corners and gear colours while changing a rotation."""
    _, model, st = maltese
    tables, words, _ = ball(model, ActionSet.primitives(model), 6)
    corners, rot, col = st.feature(0, "full"), st.feature(3, "full"), st.feature(2, "full")

    def count(fixed, changed) -> int:
        found = collisions(model, tables, words, fixed)
        tabs = np.frombuffer(b"".join(found.keys()), dtype=np.int16).reshape(-1, model.n).astype(np.int32)
        return int((~(changed.compute(tabs) == changed.goal).all(axis=1)).sum())

    assert count([corners, rot], col) > 100
    assert count([corners, col], rot) == 0


def test_cheap_gear_moves_keep_gears_in_pairs(maltese) -> None:
    """The words of up to 10 moves that fix the corners and every gear rotation move the gears
    only as 12 rigid pairs: on the 24 gear slots they generate a group of order 2^11 * 12!,
    a sliver of A24. Exactly 24 words of 12 moves split pairs. Keeping one of them is enough
    to generate A24, and that is why generation alone does not make a usable action set
    (see ``Ladder._select_actions``)."""
    from math import factorial

    from sympy.combinatorics import Permutation, PermutationGroup

    _, model, st = maltese
    tables, words, _ = ball(model, ActionSet.primitives(model), 6)
    found = collisions(model, tables, words, [st.feature(0, "full"), st.feature(3, "full")])
    gear = st.orbits[2]
    shortest: dict[tuple, int] = {}
    for key, word in found.items():
        table = np.frombuffer(key, dtype=np.int16).astype(np.int32)
        perm = tuple(int(x) for x in gear.slot_of[table[gear.ref_pos]])
        if perm != tuple(range(24)):
            shortest[perm] = min(len(word), shortest.get(perm, 99))
    assert dict(Counter(shortest.values())) == {4: 3, 6: 12, 8: 27, 10: 90, 12: 370}

    cheap = PermutationGroup([Permutation(list(p)) for p, n in shortest.items() if n <= 10])
    assert cheap.order() == 2 ** 11 * factorial(12)
    (blocks,) = cheap.minimal_blocks()                                # one block system: the pairs
    assert sorted(Counter(blocks).values()) == [2] * 12

    splitting = [p for p, n in shortest.items() if n == 12 and not cheap.contains(Permutation(list(p)))]
    assert len(splitting) == 24
    whole = PermutationGroup(list(cheap.generators) + [Permutation(list(splitting[0]))])
    assert whole.order() == factorial(24) // 2                        # one of them already gives all of A24
