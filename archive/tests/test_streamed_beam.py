"""Streamed beam search: packing, equivalence with the plain beam step, and end-to-end solving (CPU)."""
from __future__ import annotations

import numpy as np
import torch

from core.beam import beam_step
from core.gather_engine import random_walk_states
from core.streamed_beam import PackCodec, StreamedBeam, bits_for_cardinality
from core.torch_engine import TorchGatherPuzzle
from puzzles.maltese_gear.puzzle import build_maltese_gear_puzzle, cardinality

PUZZLE = build_maltese_gear_puzzle()
TP = TorchGatherPuzzle(PUZZLE, device="cpu")
CARD = cardinality()
HASH = torch.randint(1, 2**62, (PUZZLE.dim,), dtype=torch.int64, generator=torch.Generator().manual_seed(0))


def mismatch_score(states: torch.Tensor) -> torch.Tensor:
    return (states != TP.goal).sum(1).float()


def test_pack_unpack_round_trip_uses_54_bytes() -> None:
    bits = bits_for_cardinality(CARD)
    assert bits == 3
    codec = PackCodec(PUZZLE.dim, bits, torch.device("cpu"))
    assert codec.nbytes == 54
    states = torch.as_tensor(random_walk_states(PUZZLE, 500, 40, np.random.default_rng(5)), dtype=torch.uint8)
    # include the extreme values explicitly: colours 0 and 5, rotations 0 and 3
    states[0] = torch.zeros(PUZZLE.dim, dtype=torch.uint8)
    states[1] = torch.full((PUZZLE.dim,), 5, dtype=torch.uint8)
    assert torch.equal(codec.unpack(codec.pack(states)), states)


def test_streamed_step_matches_plain_beam_step_when_nothing_is_dropped() -> None:
    states = torch.as_tensor(random_walk_states(PUZZLE, 40, 8, np.random.default_rng(7)), dtype=torch.uint8)
    beam = StreamedBeam(TP, CARD, mismatch_score, HASH, width=100_000, beam_chunk=16, no_undo=False)
    run = beam.run_from_states(states)
    nxt, hit = beam.step(run)
    assert hit is None
    streamed = {row.numpy().tobytes() for row in beam.codec.unpack(nxt.packed)}
    plain = beam_step(TP, states, mismatch_score, HASH, width=100_000)
    plain_set = {row.numpy().tobytes() for row in plain}
    assert streamed == plain_set


def test_solves_short_scramble_with_valid_path() -> None:
    start = random_walk_states(PUZZLE, 1, 4, np.random.default_rng(11))[0]
    beam = StreamedBeam(TP, CARD, mismatch_score, HASH, width=4096, beam_chunk=64)
    moves = beam.solve(start, max_steps=8)
    assert moves is not None, "a width-4096 beam must solve a depth-4 scramble"
    assert len(moves) <= 4
    assert np.array_equal(PUZZLE.apply_word(start, moves), PUZZLE.goal)
