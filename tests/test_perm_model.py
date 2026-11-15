"""The labelled point model must agree with the reference simulator move for move."""
from __future__ import annotations

import numpy as np

from core.harvest import WordAlgebra
from core.perm_model import move_orders


def test_point_model_shape(maltese) -> None:
    _, model, _ = maltese
    # 96 stickers + 24 gear colours as single points, 24 gear rotations as 4 points each
    assert model.n == 96 + 24 + 4 * 24
    assert move_orders(model) == [8] * 6


def test_labelled_goal_shows_goal_state(maltese) -> None:
    puzzle, model, _ = maltese
    assert np.array_equal(model.labels_to_state(model.identity()), puzzle.goal)


def test_compiled_words_match_simulator(maltese) -> None:
    puzzle, model, _ = maltese
    rng = np.random.default_rng(0)
    for _ in range(200):
        word = rng.integers(0, 6, size=rng.integers(1, 60)).tolist()
        reference = puzzle.apply_word(puzzle.goal, word)
        labels = model.compile(word)
        assert np.array_equal(model.labels_to_state(labels), reference)
        assert np.array_equal(model.state_to_colours(reference), model.visible[labels])


def test_word_reduction_keeps_the_effect(maltese) -> None:
    _, model, _ = maltese
    alg = WordAlgebra(model)
    rng = np.random.default_rng(1)
    for _ in range(500):
        word = rng.integers(0, 6, size=rng.integers(0, 40)).tolist()
        reduced = alg.reduce(word)
        assert len(reduced) <= len(word)
        assert np.array_equal(model.compile(reduced), model.compile(word))
    assert alg.reduce([0] * 8) == ()                      # eight quarter turns of one face
    assert alg.reduce([0] * 5) == (1, 1, 1)               # five clockwise = three anticlockwise
    assert alg.reduce([0, 2, 3, 1]) == ()                 # U, L, L inverse, U inverse
    word = (0, 2, 4, 0)
    assert alg.concat(word, alg.invert(word)) == ()
