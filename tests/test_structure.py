"""Structure discovery: orbits, pieces, decoding and features, all from the move tables."""
from __future__ import annotations

import numpy as np
import pytest


def test_maltese_orbits_and_pieces(maltese) -> None:
    _, model, st = maltese
    shape = [(len(o.points), o.n_pieces, o.k) for o in st.orbits]
    assert shape == [(21, 7, 3), (72, 18, 4), (24, 24, 1), (96, 24, 4)]   # corners, ring, gear colour, gear rotation
    assert len(st.fixed_points) == 3                                       # the pegged corner
    ring = st.orbits[1]
    centres = [b for b in range(18) if len(set(model.visible[ring.pieces[b]].tolist())) == 1]
    assert len(centres) == 6                                               # 6 one-colour pieces, 12 two-colour
    assert [len(o.sub_blocks) for o in st.orbits] == [0, 1, 0, 1]          # half-turn sub-blocks on ring and gears


def test_decode_reproduces_the_visible_state(maltese) -> None:
    puzzle, model, st = maltese
    rng = np.random.default_rng(2)
    for _ in range(100):
        word = rng.integers(0, 6, size=rng.integers(1, 80)).tolist()
        state = puzzle.apply_word(puzzle.goal, word)
        assert np.array_equal(model.labels_to_state(st.decode(state)), state)


def test_features_do_not_depend_on_which_identical_piece_is_chosen(maltese) -> None:
    puzzle, model, st = maltese
    feats = st.candidate_features() + [st.feature(2, "full", (2, 1)), st.feature(1, "multiset", (0,))]
    rng = np.random.default_rng(3)
    for _ in range(100):
        word = rng.integers(0, 6, size=rng.integers(1, 80)).tolist()
        true_labels = model.compile(word)
        decoded = st.decode(model.labels_to_state(true_labels))
        for f in feats:
            assert np.array_equal(f.compute(true_labels), f.compute(decoded)), f.name


def test_features_are_closed_under_every_move(maltese) -> None:
    _, model, st = maltese
    rng = np.random.default_rng(4)
    for f in st.candidate_features():
        steps = [f.transition(model.gens[m]) for m in range(model.num_moves)]
        for _ in range(40):
            word = rng.integers(0, 6, size=rng.integers(1, 40)).tolist()
            vec = f.goal.copy()
            for m in word:
                src, lut = steps[m]
                vec = lut[np.arange(f.n_slots), vec[src]]
            assert np.array_equal(vec, f.compute(model.compile(word))), f.name


def test_expected_feature_sizes(maltese) -> None:
    _, _, st = maltese
    radix = {f.name: sorted(set(f.radix.tolist())) for f in st.candidate_features()}
    assert radix["orbit0.full"] == [21]         # 7 corners x 3 twists
    assert radix["orbit1.full"] == [54]         # 12 edges x 4 spins + 6 centres
    assert radix["orbit1.type"] == [2]          # centre or edge
    assert radix["orbit2.full"] == [6]          # gear colour
    assert radix["orbit3.full"] == [4]          # gear rotation
    assert radix["orbit3.sub0"] == [2]          # gear rotation parity


def test_decode_rejects_impossible_patterns(maltese) -> None:
    puzzle, _, st = maltese
    bad = puzzle.goal.copy()
    bad[0] = bad[60]                              # a corner showing the same colour twice
    with pytest.raises(ValueError):
        st.decode(bad)
