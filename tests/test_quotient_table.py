"""Quotient puzzles and exact coset tables, checked against brute force on the toy puzzle."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from core.coset_table import build_table, build_table_torch, table_solve
from core.gather_engine import bfs_level_sizes
from core.quotient import ActionSet, Quotient, TorchQuotient
from puzzles.toy.two_face_corners import STATE_COUNT

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _full_quotient(toy):
    _, model, st = toy
    return Quotient([st.feature(0, "full")], ActionSet.primitives(model))


def test_quotient_follows_the_labelled_model(toy) -> None:
    _, model, st = toy
    q = _full_quotient(toy)
    rng = np.random.default_rng(0)
    for _ in range(100):
        word = rng.integers(0, 4, size=rng.integers(1, 30)).tolist()
        assert np.array_equal(q.apply_word(q.goal, word), q.project(model.compile(word)))


def test_pack_round_trip(toy) -> None:
    q = _full_quotient(toy)
    states = q.step_all(q.goal[None, :]).reshape(-1, q.n_slots)
    assert np.array_equal(q.unpack(q.pack(states)), states)


def test_table_is_the_whole_puzzle_with_bfs_distances(toy) -> None:
    puzzle, _, _ = toy
    q = _full_quotient(toy)
    table = build_table(q)
    assert table.complete and len(table) == STATE_COUNT
    sizes = bfs_level_sizes(puzzle, puzzle.goal, 30)
    assert np.bincount(table.dist).tolist() == [s for s in sizes if s]      # unit costs: Dijkstra = BFS


def test_table_walk_is_optimal_and_reaches_the_goal(toy) -> None:
    q = _full_quotient(toy)
    table = build_table(q)
    rng = np.random.default_rng(1)
    for row in rng.integers(0, len(table), 200):
        state = q.unpack(table.keys[row:row + 1])[0]
        acts = table_solve(q, table, state)
        assert len(acts) == table.dist[row]
        assert q.is_goal(q.apply_word(state, acts)[None])[0]


@pytest.mark.parametrize("device", DEVICES)
def test_torch_quotient_and_table_match_numpy(toy, device: str) -> None:
    q = _full_quotient(toy)
    tq = TorchQuotient(q, device)
    rng = np.random.default_rng(2)
    states = np.stack([q.apply_word(q.goal, rng.integers(0, 4, size=20).tolist()) for _ in range(64)])
    t_states = torch.as_tensor(states, device=device)
    assert np.array_equal(tq.step_all(t_states).cpu().numpy(), q.step_all(states))
    acts = rng.integers(0, 4, size=64)
    assert np.array_equal(tq.step(t_states, torch.as_tensor(acts, device=device)).cpu().numpy(), q.step_each(states, acts))
    assert np.array_equal(tq.pack(t_states).cpu().numpy().astype(np.uint64), q.pack(states))
    a, b = build_table(q), build_table_torch(q, device)
    assert np.array_equal(a.keys, b.keys) and np.array_equal(a.dist, b.dist)


def test_weighted_actions_give_dijkstra_distances(toy) -> None:
    """With macros of different costs the table must hold cheapest costs, not fewest steps."""
    _, model, st = toy
    words = [(0,), (1,), (2,), (3,), (0, 0), (1, 1), (0, 2), (3, 1)]          # the last two are mutual inverses
    actions = ActionSet.from_words(model, words)
    q = Quotient([st.feature(0, "full")], actions)
    table = build_table(q)
    unit = build_table(_full_quotient(toy))
    assert np.array_equal(table.keys, unit.keys)
    assert np.array_equal(table.dist, unit.dist)          # a macro never beats its own primitive moves
