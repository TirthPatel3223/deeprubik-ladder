"""Coset value iteration on a rung small enough to know the right answer."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from core.coset_table import build_table
from core.davi import DaviConfig, RungValue
from core.quotient import ActionSet, Quotient
from core.search import beam_search, greedy_rollout

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def trained(toy):
    _, model, st = toy
    q = Quotient([st.feature(0, "full")], ActionSet.primitives(model))
    # Walks must be long enough to reach the deep states: with 20 steps on this 4-move puzzle
    # the values of states beyond distance 10 flatten out.
    cfg = DaviConfig(res_dim=256, num_blocks=2, batch_size=1024, states_per_update=20_000,
                     itrs_per_update=40, walk_max=40, max_updates=80, force_sync=1, test_states=100)
    value = RungValue(q, cfg, device=DEVICE)
    value.train(log=lambda _msg: None)
    return q, value, build_table(q)


def test_scramble_walk_cost_bounds_the_true_distance(trained) -> None:
    q, value, table = trained
    states, cost = value.scramble(2000)
    pos, found = table.lookup(q.pack(states.cpu().numpy()))
    assert found.all()
    assert (cost.cpu().numpy() >= table.dist[pos]).all()


def test_bellman_targets_with_the_exact_value_are_exact(trained) -> None:
    """Feeding the true distances as the 'target network' must reproduce them (fixed point)."""
    q, value, table = trained

    class Exact(torch.nn.Module):
        def forward(self, states):
            pos, _ = table.lookup(q.pack(states.cpu().numpy()))
            return torch.as_tensor(table.dist[pos], dtype=torch.float32, device=states.device)[:, None]

    states, _ = value.scramble(1000)
    pos, _ = table.lookup(q.pack(states.cpu().numpy()))
    targets = value.bellman_targets(states, net=Exact())
    assert np.allclose(targets.cpu().numpy(), table.dist[pos])


def test_learned_value_tracks_the_exact_distance(trained) -> None:
    q, value, table = trained
    rows = np.random.default_rng(0).integers(0, len(table), 4000)
    states = torch.as_tensor(q.unpack(table.keys[rows]), device=value.device)
    v = value.value(states).cpu().numpy()
    d = table.dist[rows]
    near = d <= 8
    assert np.abs(v[near] - d[near]).mean() < 0.75        # close to the goal the values are nearly exact
    assert np.corrcoef(v, d)[0, 1] > 0.5                  # and they keep rising further out
    assert v[d >= 12].mean() > v[d <= 6].mean() + 4
    assert value.value(value.tq.goal[None, :]).item() == 0.0


def test_search_with_the_learned_value_solves_near_optimally(trained) -> None:
    q, value, table = trained
    rows = np.random.default_rng(1).integers(0, len(table), 40)
    excess = []
    for row in rows:
        state = q.unpack(table.keys[row:row + 1])[0]
        found = beam_search(value.tq, value.value, torch.as_tensor(state), width=64, max_steps=80)
        assert found is not None
        acts, cost = found
        assert q.is_goal(q.apply_word(state, acts)[None])[0]
        assert cost == len(acts) >= table.dist[row]
        excess.append(cost - table.dist[row])
    assert np.mean(excess) < 1.0
    solved, _ = greedy_rollout(value.tq, value.value, torch.as_tensor(q.unpack(table.keys[rows])), max_steps=80)
    assert solved.float().mean().item() > 0.2          # greedy has no memory and can cycle; beam search above does not


def test_checkpoint_round_trip(trained, tmp_path) -> None:
    q, value, _ = trained
    path = tmp_path / "rung.pt"
    value.save(path)
    again = RungValue.from_checkpoint(q, path, device=DEVICE)
    states, _ = value.scramble(256)
    assert torch.allclose(value.value(states), again.value(states), atol=1e-3)
    assert again.update == value.update
