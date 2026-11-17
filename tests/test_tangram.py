"""The Tangram Cube Extreme in the pipeline: structure of the compiled tables, the facts its ladder rests
on, notation and the puzzle registry.

The full ladders take minutes to build (and the learned one needs trained networks), so they are run
by scripts/build_ladder.py and scripts/solve.py; the numbers they produced are in docs/tangram.md.
"""
from __future__ import annotations

import numpy as np
import pytest
from sympy.combinatorics import Permutation, PermutationGroup

from core.coset_table import build_table_auto
from core.discover import forced, forced_all, linked_orbits, stabiliser_points
from core.harvest import WordAlgebra
from core.quotient import ActionSet, Quotient
from puzzles.registry import PUZZLES, get_puzzle
from puzzles.tangram import ladder as tl
from puzzles.tangram.puzzle import compiled


def _pieces(st, orbit: int) -> list[int]:
    c = compiled("extreme")
    return sorted({int(c.geometry.sticker_piece[c.home_sticker[p]]) for p in st.orbits[orbit].points
                   if c.home_sticker[p] >= 0})


def _sticker_sides(st, orbit: int) -> set[int]:
    c = compiled("extreme")
    return {len(c.geometry.sticker_polygon[c.home_sticker[p]]) for p in st.orbits[orbit].points if c.home_sticker[p] >= 0}


def test_orbit_names_hold_the_pieces_they_say(tangram) -> None:
    _, _, st = tangram
    g = compiled("extreme").geometry
    assert len(st.orbits) == 14
    stickers_per_piece = np.bincount(g.sticker_piece, minlength=g.n_pieces)
    for name in ("SMALL_TRI_A", "SMALL_TRI_B", "LARGE_TRI_A", "LARGE_TRI_B", "PENTAGON", "CORNER", "CORNER_2", "CORNER_3"):
        orbit = getattr(tl, name)
        assert (st.orbits[orbit].n_pieces, st.orbits[orbit].k) == (12, 1), name        # 12 places, 6 of them empty
        assert len(_pieces(st, orbit)) == 6, name
    for name in ("SMALL_TRI_A", "SMALL_TRI_B", "LARGE_TRI_A", "LARGE_TRI_B"):
        assert _sticker_sides(st, getattr(tl, name)) == {3}
        assert set(stickers_per_piece[_pieces(st, getattr(tl, name))]) == {1}
    assert _sticker_sides(st, tl.PENTAGON) == {5}
    assert _pieces(st, tl.CORNER) == _pieces(st, tl.CORNER_2) == _pieces(st, tl.CORNER_3)
    assert set(stickers_per_piece[_pieces(st, tl.CORNER)]) == {3}
    for name, direction in (("TOP", 1), ("BOTTOM", -1)):
        (piece,) = _pieces(st, getattr(tl, name))
        centroid = g.piece_centroid[piece]
        assert np.allclose(centroid / np.linalg.norm(centroid), direction * np.ones(3) / np.sqrt(3))
    for name, faces in (("MID_EDGE_F", {"+x", "-y"}), ("MID_EDGE_R", {"+y", "-z"}), ("MID_EDGE_L", {"-x", "+z"})):
        (piece,) = _pieces(st, getattr(tl, name))
        assert {g.colours[c] for c in g.sticker_colour[g.sticker_piece == piece]} == faces
    assert st.orbits[tl.MIDDLE].n_pieces == 3


def test_turning_corners_and_middle_edges_need_a_rung_of_their_own(tangram) -> None:
    """With every other piece solved a turning corner can still be turned 120° (3 states) and two middle
    edges flipped together (4 states). No word of up to 12 moves does either once the other rungs are
    solved, so a ladder that leaves them for the end cannot be built from short macros: they go first."""
    _, model, st = tangram
    group = PermutationGroup([Permutation(np.argsort(g).tolist()) for g in model.gens])

    def freedom(orbits) -> int:
        rest = [int(p) for o in st.orbits if o.index not in orbits for p in o.points]
        return int(group.pointwise_stabilizer(rest).order())

    assert freedom({tl.TOP}) == 3 and freedom({tl.BOTTOM}) == 3
    assert freedom({tl.MID_EDGE_F, tl.MID_EDGE_R, tl.MID_EDGE_L}) == 4
    assert freedom({tl.MID_EDGE_F}) == 1                                  # one edge alone cannot flip
    full = {o.index: st.feature(o.index, "full") for o in st.orbits}
    assert forced(model, [full[o] for o in full if o != tl.TOP], full[tl.TOP]) is False


def test_each_ladders_features_solve_the_whole_puzzle(tangram) -> None:
    """Exact: the stabiliser of every rung's goal moves nothing else (the other two corner stickers
    follow from the first, and the last middle edge from the other pieces)."""
    _, model, st = tangram
    everything = [st.feature(o.index, "full") for o in st.orbits]
    for specs in tl.LADDERS.values():
        feats = [st.feature(*fs) for spec in specs for fs in spec.features]
        assert forced_all(model, feats, everything) == [True] * len(everything)


def test_stabiliser_points_pins_pieces_but_not_empty_places(tangram, maltese) -> None:
    _, _, st = tangram
    assert len(stabiliser_points(st.feature(tl.LARGE_TRI_A, "full"))) == 6        # 6 pieces; 6 empty places are free
    _, _, mst = maltese
    assert stabiliser_points(mst.feature(2, "full")) is None                      # identical gears of six colours


def test_linked_orbits_keep_pieces_that_turn_in_place_apart(tangram) -> None:
    _, model, st = tangram
    groups = linked_orbits(model, st)
    assert [tl.TOP] in groups and [tl.MID_EDGE_F] in groups
    assert sorted(map(sorted, groups))[:3] == [[tl.SMALL_TRI_A, tl.SMALL_TRI_B], [tl.LARGE_TRI_A, tl.LARGE_TRI_B],
                                              [tl.PENTAGON, tl.CORNER, tl.CORNER_2, tl.CORNER_3]]


def test_frame_rung_is_a_small_exact_table(tangram) -> None:
    puzzle, model, st = tangram
    q = Quotient([st.feature(*fs) for fs in tl.FRAME], ActionSet.primitives(model))
    table = build_table_auto(q)
    assert table.complete and len(table) == 3456 and int(table.dist.max()) == 13


def test_tangram_notation_matches_the_simulator() -> None:
    entry = get_puzzle("tangram_extreme")
    from core.perm_model import build_point_model

    notation = entry.notation(WordAlgebra(build_point_model(entry.build())))
    text = "U2' F R U3 D' L"
    word = notation.parse(text)
    assert notation.format(word) == text
    assert notation.format(notation.parse("U U U U")) == "U2'"                 # four steps one way = two the other
    assert notation.format(notation.parse("U'2 F' F'")) == "U2'"              # both prime orders; F' is F; F F cancels
    assert notation.lengths(word) == {"steps": 9, "turns": 6}
    with pytest.raises(ValueError):
        notation.parse("X")


def test_turns_merge_across_the_commuting_corner_layers(tangram, maltese) -> None:
    puzzle, model, _ = tangram
    alg = WordAlgebra(model)
    notation = get_puzzle("tangram_extreme").notation(alg)
    word = notation.parse("L D' U' D F U D U'")
    short = alg.simplify(word)
    assert notation.format(alg.reduce(word)) == "L D' U' D F U D U'"            # adjacent merging finds nothing
    assert notation.format(short) == "L U' F D"
    assert np.array_equal(puzzle.apply_word(puzzle.goal, list(word)), puzzle.apply_word(puzzle.goal, list(short)))
    # No two faces of the Maltese Gear Cube commute, so its solutions are exactly as before.
    _, mmodel, _ = maltese
    malg = WordAlgebra(mmodel)
    n = mmodel.num_moves
    assert not any(malg.commutes[a][b] for a in range(n) for b in range(n) if malg.base[a] != malg.base[b])
    rng = np.random.default_rng(0)
    for _ in range(20):
        w = rng.integers(0, n, size=60).tolist()
        assert malg.simplify(w) == malg.reduce(w)


def test_maltese_notation_is_unchanged(maltese) -> None:
    _, model, _ = maltese
    notation = get_puzzle("maltese").notation(WordAlgebra(model))
    word = notation.parse("U L F' U2 L'3 F")
    assert notation.format(word) == "U L F' U2 L'3 F"
    assert notation.describe(word) == "9 quarter turns, 7 face turns"


def test_registry_builds_every_puzzle_and_keeps_the_maltese_cache_path() -> None:
    for name, entry in PUZZLES.items():
        puzzle = entry.build()
        ladders = entry.ladders()
        assert entry.default_ladder in ladders, name
        assert puzzle.num_moves == len(puzzle.move_names)
    assert get_puzzle("maltese").cache_dir("learned", 7).as_posix() == "data/ladders/learned_r7"
    assert get_puzzle("tangram_extreme").cache_dir("learned", 7).as_posix() == "data/ladders/tangram_extreme/learned_r7"
    with pytest.raises(KeyError):
        get_puzzle("no such puzzle")
