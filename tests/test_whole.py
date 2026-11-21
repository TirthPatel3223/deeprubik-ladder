"""One network for the whole puzzle (core/whole.py), checked on the toy puzzle against brute force.

The toy's two rungs (place the corners, then twist them) give a whole state of 12 slots and 29,160 reachable
states, few enough to compute the exact cost-to-go under the per-rung mask for every one of them."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from core.ladder import Ladder
from core.nnet import evaluate
from core.whole import TorchBaseline, WholeConfig, WholePuzzle, WholeSolver, WholeValue, phased_beam_search
from puzzles.toy.two_face_corners import LADDER, STATE_COUNT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _quiet(_msg: str) -> None:
    pass


@pytest.fixture(scope="module")
def toy_ladder(toy):
    puzzle, model, st = toy
    return puzzle, Ladder(model, st, LADDER, mitm_radius=4, log=_quiet).build()


@pytest.fixture(scope="module")
def whole(toy_ladder):
    puzzle, ladder = toy_ladder
    return puzzle, ladder, WholePuzzle(ladder)


def exact_values(wp: WholePuzzle):
    """Every reachable whole state, its packed key and its exact cost-to-go under the puzzle's mask (value
    iteration on the masked graph)."""
    q = wp.quotient
    states = q.goal[None].copy()
    seen = {int(q.pack(states)[0])}
    frontier = states
    while len(frontier):
        kids = q.step_all(frontier).reshape(-1, q.n_slots)
        keys = q.pack(kids)
        _, first = np.unique(keys, return_index=True)
        fresh = [i for i in first if int(keys[i]) not in seen]
        seen.update(int(keys[i]) for i in fresh)
        frontier = kids[fresh]
        states = np.concatenate([states, frontier])
    keys = q.pack(states)
    order = np.argsort(keys)
    states, keys = states[order], keys[order]
    phase = wp.phase_of(states)
    J = np.full(len(states), np.inf)
    J[phase == wp.n_phases] = 0.0
    for _ in range(200):
        new = J.copy()
        for k, p in enumerate(wp.phases):
            rows = np.nonzero(phase == k)[0]
            kids = np.stack([q.step(states[rows], int(a)) for a in p.actions], 1)          # (n, A_k, S)
            idx = np.searchsorted(keys, q.pack(kids.reshape(-1, q.n_slots))).reshape(len(rows), -1)
            new[rows] = (q.costs[p.actions][None, :] + J[idx]).min(1)
        if np.array_equal(new, J):
            break
        J = new
    assert np.isfinite(J).all(), "every state must be solvable under the mask"
    return states, keys, J


@pytest.fixture(scope="module")
def exact(whole):
    return exact_values(whole[2])


class _ExactNet(torch.nn.Module):
    """A 'network' whose output r makes H + r the exact cost-to-go."""

    def __init__(self, value: WholeValue, keys: np.ndarray, J: np.ndarray) -> None:
        super().__init__()
        self.value, self.keys, self.J = value, keys, J

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        q = self.value.wp.quotient
        pos = np.searchsorted(self.keys, q.pack(states.cpu().numpy()))
        j = torch.as_tensor(self.J[pos], dtype=torch.float32, device=states.device)
        u = self.value.tw.unsolved(states)
        return (j - self.value.baseline(u, self.value.tw.phase(unsolved=u)))[:, None]


def test_whole_puzzle_collects_every_rung_and_matches_the_simulator(whole, toy) -> None:
    puzzle, ladder, wp = whole
    _, model, st = toy
    assert [p.name for p in wp.phases] == [r.spec.name for r in ladder.rungs]
    assert wp.quotient.n_slots == sum(r.quotient.n_slots for r in ladder.rungs)
    assert len(wp.actions) <= sum(len(r.actions) for r in ladder.rungs)
    # every action, applied to the whole feature vector, agrees with the action applied to the labels
    rng = np.random.default_rng(0)
    for _ in range(20):
        lab = st.decode(puzzle.apply_word(puzzle.goal, rng.integers(0, puzzle.num_moves, 40).tolist()))
        vec = wp.project(lab)
        for a in rng.integers(0, len(wp.actions), 10):
            assert np.array_equal(wp.quotient.step(vec[None], int(a))[0], wp.project(lab[wp.actions.tables[a]]))
    # the second phase's actions keep the first rung solved
    p0, p1 = wp.phases
    goal = wp.quotient.goal
    for a in p1.actions:
        assert np.array_equal(wp.quotient.step(goal[None], int(a))[0][p0.start:p0.stop], goal[p0.start:p0.stop])


def test_exact_masked_values_cover_the_whole_puzzle(exact) -> None:
    states, _, J = exact
    assert len(states) == STATE_COUNT
    assert J.max() > 10 and (J == 0).sum() == 1


def test_bellman_targets_with_the_exact_value_are_exact(whole, exact) -> None:
    _, _, wp = whole
    states, keys, J = exact
    for kind in ("none", "gated", "smooth"):
        cfg = WholeConfig(res_dim=64, num_blocks=1, baseline=kind, rung_costs=(4.0, 12.0), depths=(10, 10),
                          mix_walks=(20, 20))
        value = WholeValue(wp, cfg, device=DEVICE, log=_quiet)
        rows = np.random.default_rng(1).integers(0, len(states), 3000)
        t = value.bellman_targets(torch.as_tensor(states[rows], device=value.device), net=_ExactNet(value, keys, J))
        assert np.allclose(t.cpu().numpy(), J[rows]), kind


def test_double_backup_with_the_exact_value_is_exact(whole, exact) -> None:
    _, _, wp = whole
    states, keys, J = exact
    cfg = WholeConfig(res_dim=64, num_blocks=1, baseline="gated", rung_costs=(4.0, 12.0), depths=(10, 10),
                      mix_walks=(20, 20), double=True)
    value = WholeValue(wp, cfg, device=DEVICE, log=_quiet)
    value.net = value.target = _ExactNet(value, keys, J)        # online chooses, target values: both exact
    value.target_is_zero = False
    rows = np.random.default_rng(5).integers(0, len(states), 3000)
    t = value.bellman_targets(torch.as_tensor(states[rows], device=value.device))
    assert np.allclose(t.cpu().numpy(), J[rows])


@pytest.mark.parametrize("cross_weight", [1.0, 0.5])
def test_twin_targets_with_the_exact_value_are_exact(whole, exact, cross_weight) -> None:
    _, _, wp = whole
    states, keys, J = exact
    cfg = WholeConfig(res_dim=64, num_blocks=1, baseline="gated", rung_costs=(4.0, 12.0), depths=(10, 10),
                      mix_walks=(20, 20), twin=True, cross_weight=cross_weight)
    value = WholeValue(wp, cfg, device=DEVICE, log=_quiet)
    value.target.a = _ExactNet(value, keys, J)                 # each half chosen by one, valued by the other
    value.target.b = _ExactNet(value, keys, J)
    value.target_is_zero = False
    rows = np.random.default_rng(6).integers(0, len(states), 3001)
    t = value.training_targets(torch.as_tensor(states[rows], device=value.device))
    assert np.allclose(t.cpu().numpy(), J[rows])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused inference copy is built on the GPU only")
@pytest.mark.parametrize("twin", [False, True])
def test_fused_inference_copy_matches_the_network_and_follows_it(whole, twin) -> None:
    _, _, wp = whole
    cfg = WholeConfig(res_dim=128, num_blocks=2, baseline="gated", rung_costs=(4.0, 12.0), depths=(10, 10),
                      mix_walks=(20, 20), batch_size=512, states_per_update=4096, itrs_per_update=20, force_sync=1,
                      twin=twin, twin_search="mean")
    value = WholeValue(wp, cfg, device="cuda", log=_quiet)
    value.train(max_updates=3)                        # weights and batch-norm statistics away from their start
    states = value.sample(5000)
    for net in (value.net, value.target):
        ref = evaluate(net, states, amp=True).reshape(-1)
        fast = value._fast(net)
        assert fast is not None and fast is not net
        out = evaluate(fast, states, amp=True).reshape(-1)
        assert float((out - ref).abs().max()) <= 0.02 * (float(ref.abs().max()) + value.scale)
    online, target = value._fast(value.net), value._fast(value.target)
    assert value._fast(value.net) is online           # cached while the networks are unchanged
    value.train(max_updates=1)                        # one more update, and a refresh (force_sync=1)
    assert value._fast(value.net) is not online and value._fast(value.target) is not target


def test_looser_masks_allow_more_and_never_cost_more(toy_ladder, exact) -> None:
    _, ladder = toy_ladder
    states, keys, J_rung = exact
    by_mask = {"rung": J_rung}
    for mask in ("progress", "none"):
        wp = WholePuzzle(ladder, mask=mask)
        goal = wp.quotient.goal
        for k, p in enumerate(wp.phases):
            assert set(p.walk.tolist()) <= set(p.actions.tolist())         # the rung's own actions stay allowed
            if mask == "progress":                                         # and nothing allowed undoes a finished rung
                for a in p.actions:
                    moved = wp.quotient.step(goal[None], int(a))[0]
                    for done in wp.phases[:k]:
                        assert np.array_equal(moved[done.start:done.stop], goal[done.start:done.stop])
            else:
                assert len(p.actions) == len(wp.actions)
        s2, k2, J = exact_values(wp)
        assert np.array_equal(k2, keys)
        by_mask[mask] = J
        # exact values are a fixed point of the training targets under this mask too
        value = WholeValue(wp, WholeConfig(res_dim=64, num_blocks=1, depths=(10, 10), mix_walks=(20, 20)),
                           device=DEVICE, log=_quiet)
        rows = np.random.default_rng(4).integers(0, len(s2), 2000)
        t = value.bellman_targets(torch.as_tensor(s2[rows], device=value.device), net=_ExactNet(value, k2, J))
        assert np.allclose(t.cpu().numpy(), J[rows]), mask
    assert (by_mask["progress"] <= by_mask["rung"]).all() and (by_mask["none"] <= by_mask["progress"]).all()
    assert by_mask["none"].mean() < by_mask["rung"].mean()


def test_excursions_only_with_a_looser_mask(toy_ladder) -> None:
    _, ladder = toy_ladder
    cfg = dict(res_dim=64, num_blocks=1, depths=(10, 10), mix_walks=(20, 20), excursion_share=1.0, excursion_max=2)
    tight = WholeValue(WholePuzzle(ladder, mask="rung"), WholeConfig(**cfg), device=DEVICE, log=_quiet)
    loose = WholeValue(WholePuzzle(ladder, mask="none"), WholeConfig(**cfg), device=DEVICE, log=_quiet)
    one = torch.ones(500, dtype=torch.long, device=tight.device)
    x = tight._walk(tight.tw.tq.goal_batch(500), 1, one)                 # one action from the goal: phase 1
    assert torch.equal(tight._excursions(x.clone()), x)
    assert not torch.equal(loose._excursions(x.clone()), x)


def test_baseline_is_zero_at_the_goal_and_gated_counts_only_the_current_rung(whole) -> None:
    _, _, wp = whole
    cfg = WholeConfig(res_dim=64, num_blocks=1, depths=(10, 10), mix_walks=(20, 20))
    value = WholeValue(wp, cfg, device=DEVICE, log=_quiet)
    tw = value.tw
    goal = tw.tq.goal_batch(1)
    for kind in ("pieces", "gated", "smooth"):
        base = TorchBaseline(kind, (4.0, 12.0), tw, mixed_unsolved=(0.8, 0.6))
        u = tw.unsolved(goal)
        assert float(base(u, tw.phase(unsolved=u))[0]) == 0.0
    gated = TorchBaseline("gated", (4.0, 12.0), tw)
    states = value.sample(2000)
    u = tw.unsolved(states)
    ph = tw.phase(unsolved=u)
    h = gated(u, ph)
    p0, p1 = wp.phases
    expect = torch.where(ph == 0, 4.0 / p0.n_slots * u[:, 0] + 12.0,
                         torch.where(ph == 1, 12.0 / p1.n_slots * u[:, 1], torch.zeros_like(h)))
    assert torch.allclose(h, expect)


def test_smooth_baseline_is_continuous_on_average_at_the_rung_boundary(whole) -> None:
    _, _, wp = whole
    costs = (4.0, 12.0)
    cfg = WholeConfig(res_dim=64, num_blocks=1, baseline="smooth", rung_costs=costs, depths=(10, 10),
                      mix_walks=(20, 20))
    value = WholeValue(wp, cfg, device=DEVICE, log=_quiet)
    f = value.cfg.mixed_unsolved
    assert len(f) == 2 and all(0.0 < x <= 1.0 for x in f)
    tw = value.tw
    n = 20000
    # rung 0 just finished, rung 1 mixed: the states phase 0 ends in
    start = value._walk(tw.tq.goal_batch(n), 1, torch.full((n,), 40, dtype=torch.long, device=value.device))
    u = tw.unsolved(start)
    ph = tw.phase(unsolved=u)
    assert bool((ph == 1).float().mean() > 0.95)
    rows = ph == 1
    smooth = float(value.baseline(u, ph)[rows].mean())
    gated = float(TorchBaseline("gated", costs, tw)(u, ph)[rows].mean())
    # the limit of phase 0's potential as rung 0 is finished is the tail: rung 1's mean cost
    assert abs(smooth - costs[1]) < 0.1 * costs[1]
    assert abs(gated - f[1] * costs[1]) < 0.1 * costs[1]


def test_phased_search_with_the_exact_value_is_optimal(whole, exact) -> None:
    _, _, wp = whole
    states, keys, J = exact
    cfg = WholeConfig(res_dim=64, num_blocks=1, depths=(10, 10), mix_walks=(20, 20))
    value = WholeValue(wp, cfg, device=DEVICE, log=_quiet)
    net = _ExactNet(value, keys, J)

    def exact_value(x: torch.Tensor) -> torch.Tensor:
        return value._values(x, net)

    q = wp.quotient
    for i, row in enumerate(np.random.default_rng(2).integers(0, len(states), 30)):
        # every other search builds its next beam from many small chunks (the pool path of a very wide beam)
        found = phased_beam_search(value.tw, exact_value, torch.as_tensor(states[row]), width=64, weight=1.0,
                                   expand_children=(1 << 21) if i % 2 else 25)
        assert found is not None
        acts, cost = found
        assert q.is_goal(q.apply_word(states[row], acts)[None])[0]
        assert cost == pytest.approx(J[row])
        # the path respects the mask: each action belongs to the phase of the state it was taken in
        cur = states[row]
        for a in acts:
            k = int(wp.phase_of(cur)[0])
            assert a in set(wp.phases[k].actions.tolist())
            cur = q.step(cur[None], a)[0]


def test_search_with_a_children_budget_respects_it_and_still_solves(whole, exact) -> None:
    _, _, wp = whole
    states, keys, J = exact
    cfg = WholeConfig(res_dim=64, num_blocks=1, depths=(10, 10), mix_walks=(20, 20))
    value = WholeValue(wp, cfg, device=DEVICE, log=_quiet)
    net = _ExactNet(value, keys, J)
    n_kids = [len(p.actions) for p in wp.phases]
    width, budget = 64, min(n_kids) + 0.5 * (max(n_kids) - min(n_kids))     # cuts the beam in the bigger phase
    q = wp.quotient
    for row in np.random.default_rng(7).integers(0, len(states), 20):
        stats: dict = {}
        found = phased_beam_search(value.tw, lambda x: value._values(x, net), torch.as_tensor(states[row]),
                                   width=width, weight=1.0, budget=budget, stats=stats)
        assert found is not None
        acts, cost = found
        assert q.is_goal(q.apply_word(states[row], acts)[None])[0] and cost >= J[row] - 1e-6
        assert all(made <= width * budget for _s, _lo, _hi, made, _t in stats["per_step"])


@pytest.mark.parametrize("kind,twin,tau", [("none", False, 0.0), ("gated", False, 0.0), ("gated", True, 0.0),
                                          ("smooth", True, 0.05)])
def test_trained_network_solves_scrambles_of_the_real_puzzle(whole, exact, kind, twin, tau) -> None:
    puzzle, ladder, wp = whole
    states, keys, J = exact
    cfg = WholeConfig(res_dim=256, num_blocks=2, batch_size=1024, states_per_update=20_000, itrs_per_update=40,
                      force_sync=1, baseline=kind, rung_costs=(4.0, 12.0), twin=twin, target_tau=tau)
    value = WholeValue(wp, cfg, device=DEVICE, log=_quiet)
    value.train(max_updates=60)
    solver = WholeSolver(ladder, value)
    rng = np.random.default_rng(3)
    excess = []
    for _ in range(20):
        state = puzzle.apply_word(puzzle.goal, rng.integers(0, puzzle.num_moves, 300).tolist())
        word, per_phase = solver.solve(state, beam=256)
        assert np.array_equal(puzzle.apply_word(state, list(word)), puzzle.goal)
        vec = wp.project(ladder.structure.decode(state))
        j = J[np.searchsorted(keys, wp.quotient.pack(vec[None]))[0]]
        excess.append(sum(per_phase) - j)
    assert min(excess) >= 0
    assert np.mean(excess) < 1.0
