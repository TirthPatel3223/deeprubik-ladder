"""Goal ball: layer sizes, resumable builds, exact paths, and beam integration (CPU)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from core.ball import GoalBall
from core.gather_engine import bfs_level_sizes, random_walk_states
from core.hashing import hash_states, make_hash_vec
from core.streamed_beam import StreamedBeam
from core.torch_engine import TorchGatherPuzzle
from puzzles.maltese_gear.puzzle import build_maltese_gear_puzzle, cardinality

PUZZLE = build_maltese_gear_puzzle()
TP = TorchGatherPuzzle(PUZZLE, device="cpu")
CARD = cardinality()


def _silent(_: str) -> None:
    pass


def test_layer_sizes_match_reference_bfs(tmp_path) -> None:
    ball = GoalBall(TP, CARD, tmp_path, depth=6)
    ball.build(log=_silent)
    assert ball.sizes() == bfs_level_sizes(PUZZLE, PUZZLE.goal, 6)


def test_build_resumes_from_saved_layers(tmp_path) -> None:
    GoalBall(TP, CARD, tmp_path, depth=4).build(log=_silent)
    resumed = GoalBall(TP, CARD, tmp_path, depth=6)
    resumed.build(log=_silent)
    assert resumed.sizes() == bfs_level_sizes(PUZZLE, PUZZLE.goal, 6)


def test_metadata_guard_rejects_different_hash_seed(tmp_path) -> None:
    GoalBall(TP, CARD, tmp_path, depth=2, hash_seed=1).build(log=_silent)
    with pytest.raises(ValueError):
        GoalBall(TP, CARD, tmp_path, depth=2, hash_seed=2)


def test_numpy_and_torch_hashes_agree() -> None:
    hv = make_hash_vec(PUZZLE.dim, 12345)
    states = random_walk_states(PUZZLE, 50, 30, np.random.default_rng(2))
    torch_hashes = hash_states(torch.as_tensor(states, dtype=torch.uint8), hv).numpy()
    host = (states.astype(np.int64) * hv.numpy()).sum(1)
    assert np.array_equal(torch_hashes, host)


def test_depth_and_path_are_exact(tmp_path) -> None:
    ball = GoalBall(TP, CARD, tmp_path, depth=6)
    ball.build(log=_silent)
    for state in random_walk_states(PUZZLE, 30, 5, np.random.default_rng(4)):
        k = ball.depth_of(state)
        assert k is not None and k <= 5
        path = ball.path_to_goal(state)
        assert path is not None and len(path) == k
        assert np.array_equal(PUZZLE.apply_word(state, path), PUZZLE.goal)


def test_state_outside_ball_has_no_path(tmp_path) -> None:
    ball = GoalBall(TP, CARD, tmp_path, depth=3)
    ball.build(log=_silent)
    far = random_walk_states(PUZZLE, 1, 20, np.random.default_rng(9))[0]
    if ball.depth_of(far) is None:
        assert ball.path_to_goal(far) is None


def test_beam_with_ball_solves_scramble(tmp_path) -> None:
    ball = GoalBall(TP, CARD, tmp_path, depth=6)
    ball.build(log=_silent)
    hv = ball.hash_vec

    def mismatch(states: torch.Tensor) -> torch.Tensor:
        return (states != TP.goal).sum(1).float()

    beam = StreamedBeam(TP, CARD, mismatch, hv, width=4096, beam_chunk=64, ball=ball)
    start = random_walk_states(PUZZLE, 1, 7, np.random.default_rng(13))[0]
    moves = beam.solve(start, max_steps=10)
    assert moves is not None
    assert np.array_equal(PUZZLE.apply_word(start, moves), PUZZLE.goal)


def test_beam_refuses_ball_with_different_hash(tmp_path) -> None:
    ball = GoalBall(TP, CARD, tmp_path, depth=2)
    ball.build(log=_silent)

    def mismatch(states: torch.Tensor) -> torch.Tensor:
        return (states != TP.goal).sum(1).float()

    other = make_hash_vec(PUZZLE.dim, 999)
    with pytest.raises(ValueError):
        StreamedBeam(TP, CARD, mismatch, other, width=64, ball=ball)
