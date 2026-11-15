"""Beam-step correctness on CPU (no GPU needed)."""
from __future__ import annotations

import numpy as np
import torch

from core.beam import OneHot, beam_step
from core.gather_engine import random_walk_states
from core.torch_engine import TorchGatherPuzzle
from puzzles.maltese_gear.puzzle import build_maltese_gear_puzzle, cardinality

PUZZLE = build_maltese_gear_puzzle()
TP = TorchGatherPuzzle(PUZZLE, device="cpu")
HASH = torch.randint(1, 2**62, (PUZZLE.dim,), dtype=torch.int64, generator=torch.Generator().manual_seed(0))


def mismatch_score(states: torch.Tensor) -> torch.Tensor:
    """Number of coordinates that differ from the goal. Lower is better."""
    return (states != TP.goal).sum(1).float()


def test_onehot_has_one_bit_per_coordinate() -> None:
    enc = OneHot(cardinality(), "cpu")
    assert enc.width == 816
    states = torch.as_tensor(PUZZLE.goal[None, :], dtype=torch.uint8)
    x = enc(states)
    assert x.shape == (1, 816)
    assert int(x.sum()) == PUZZLE.dim


def test_beam_step_removes_duplicates_and_respects_width() -> None:
    beam = TP.goal_batch(4)  # four copies of the goal: every candidate collapses heavily
    out = beam_step(TP, beam, mismatch_score, HASH, width=100)
    assert out.shape[0] <= 100
    rows = {row.numpy().tobytes() for row in out}
    assert len(rows) == out.shape[0], "beam contains duplicate states"


def test_beam_finds_goal_two_moves_away() -> None:
    start = random_walk_states(PUZZLE, 1, 2, np.random.default_rng(3))
    beam = torch.as_tensor(start, dtype=torch.uint8)
    found = False
    for _ in range(2):
        beam = beam_step(TP, beam, mismatch_score, HASH, width=4096)
        if bool((beam == TP.goal).all(1).any()):
            found = True
            break
    assert found, "beam of width 4096 should contain the goal after two steps"
