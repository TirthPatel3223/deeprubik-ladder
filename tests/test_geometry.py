"""Compiling a 3D model into move tables (core/geometry.py), on the Tangram Cube's exported geometry.

The geometry comes from the Cube_Modeling simulator (tools/export/geometry.gen.ts there). Each export
carries walks played on that simulator's own engine; compile_geometry replays them on the tables it
builds and raises if any differs, so every test below runs on tables that match the 3D model.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from sympy.combinatorics import Permutation, PermutationGroup

from core.geometry import GeometryError, compile_geometry, load_geometry
from puzzles.tangram.puzzle import EMPTY, GROUP_ORDER, compiled, geometry_file


def test_extreme_pieces_stickers_and_positions() -> None:
    c = compiled("extreme")
    r = c.report
    assert (r["pieces"], r["stickers"], r["positions"]) == (45, 66, 120)
    assert int((c.puzzle.goal == EMPTY).sum()) == 54                    # where stickers go when the shape changes
    assert r["orientations"] == 12                                       # D6
    assert sorted(set(r["poses_per_piece"])) == [1, 2, 6, 12]           # core, middle edges, turning corners, the rest
    assert r["model_walks"] == 20


def test_original_never_changes_shape() -> None:
    c = compiled("original")
    assert (c.report["positions"], c.report["stickers"]) == (42, 42)
    assert not (c.puzzle.goal == EMPTY).any()


def test_a_sticker_centre_alone_is_not_a_position() -> None:
    """On the Extreme, 12 pairs of positions share a centre and differ only in their normal: a sticker
    turned 60° lands where another piece's sticker sits, tilted. So positions are (centre, normal)."""
    assert compiled("extreme").report["centre_collisions"] == 12
    assert compiled("original").report["centre_collisions"] == 0


@pytest.mark.parametrize("variant", ["extreme", "original"])
def test_group_order_matches_the_documented_counts(variant: str) -> None:
    puzzle = compiled(variant).puzzle
    gens = [Permutation(np.argsort(puzzle.idx[m]).tolist()) for m in range(puzzle.num_moves)]
    assert PermutationGroup(gens).order() == GROUP_ORDER[variant]


def test_moves_orders_inverses_and_commuting_layers() -> None:
    puzzle = compiled("extreme").puzzle
    names = list(puzzle.move_names)
    assert names == ["U", "U'", "D", "D'", "F", "R", "L"]
    goal = puzzle.goal
    rng = np.random.default_rng(1)
    state = puzzle.apply_word(goal, rng.integers(0, 7, 30).tolist())
    u, d, f = names.index("U"), names.index("D"), names.index("F")
    assert np.array_equal(puzzle.apply_word(state, [u] * 6), state)
    assert not np.array_equal(puzzle.apply_word(state, [u] * 3), state)
    assert np.array_equal(puzzle.apply_word(state, [f, f]), state)
    assert np.array_equal(puzzle.apply_word(state, [u, d]), puzzle.apply_word(state, [d, u]))
    for m in range(7):
        assert np.array_equal(puzzle.apply_word(state, [m, int(puzzle.inverse[m])]), state)
    # one 60° turn changes the shape: stickers leave the cube for positions that were empty
    turned = puzzle.apply_word(goal, [u])
    assert (turned[goal == EMPTY] != EMPTY).any()
    assert np.array_equal(puzzle.apply_word(goal, [u, u]) == EMPTY, goal == EMPTY)   # 120°: a cube again


def test_refuses_a_move_that_would_cut_pieces() -> None:
    """Move the F flip's cut plane: the flip would then cut through pieces, as on a jumbling puzzle."""
    geom = load_geometry(geometry_file("extreme"))
    moves = tuple(dataclasses.replace(m, offset=0.5 * m.offset) if m.name == "F" else m for m in geom.moves)
    with pytest.raises(GeometryError, match="cut a piece"):
        compile_geometry(dataclasses.replace(geom, moves=moves))


def test_refuses_tables_that_disagree_with_the_model() -> None:
    geom = load_geometry(geometry_file("extreme"))
    names, orient = geom.checks[0]
    wrong = orient.copy()
    moved = int(np.flatnonzero(np.abs(orient - np.eye(3)).max(axis=(1, 2)) > 1e-6)[0])
    wrong[moved] = np.eye(3)                                            # claim that a moved piece stayed home
    checks = ((names, wrong),) + geom.checks[1:]
    with pytest.raises(GeometryError, match="check walk 0"):
        compile_geometry(dataclasses.replace(geom, checks=checks))


def test_positions_are_far_apart_and_clear_of_every_cut() -> None:
    r = compiled("extreme").report
    assert r["separation"] > 0.2                                         # matched with a tolerance of 1e-7
    assert r["boundary_margin"] > 0.08                                   # no position is near a move's boundary
