"""One value network for the whole puzzle: the ladder's macros and rung order, without its tables.

The ladder solver (core/ladder.py) gives every large rung its own network and walks the rungs one at a
time. Here one network sees every piece and estimates the cost of solving the whole puzzle, and one beam
search runs from the scramble to the goal. The ladder still supplies two things:

* the actions: the union of its rungs' action sets (primitive moves and macros), each costing its length in
  primitive moves, so solution costs are in the puzzle's own metric;
* the phases: a state's phase is its first rung that is not solved. With the per-rung mask a state in phase
  k may use only rung k's actions. These are the ladder's moves, but they are chosen by a network that sees
  the whole puzzle, in one search across all rungs, so a rung can be finished in the way that makes the
  later ones cheaper. Looser masks allow every action that keeps the finished rungs solved, or every action
  (``WholePuzzle``). The mask is applied by the trainer and the search; the network's input is only the
  pieces (every rung's feature vector side by side).

Reward shaping. The value is V(s) = H(s) + r(s): H is a fixed count of unsolved pieces weighted by rung
(``Baseline``) and r is the network's output. In terms of rewards this is potential-based shaping with the
potential -H (Ng, Harada & Russell 1999): the cost-to-go the training converges to is unchanged, and it
starts from H instead of from zero (Wiewiora 2003). ``Baseline(kind="none")`` is plain DAVI.

Training follows core/davi.py (DeepCubeA / DeepXube's loop) with two differences. Training states come from
stratified walks: pick a phase k, mix every later rung with a long walk of rung k+1's actions, then take a
uniform 0..D_k of rung k's actions, so each state has the earlier rungs solved, rung k at a controlled depth
and the later rungs mixed, which is how the search meets states near the end of rung k. And there is no
walk-cost bound (``ub_walk``): under the mask the reversed walk is not always a legal path (an action can
solve a later rung by accident and change the phase), so the bound could be wrong.
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Sequence

import numpy as np
import torch

from core.autotrain import mixing_length
from core.fingerprint import digest
from core.ladder import Ladder, RungFailure
from core.nnet import ResnetFC, evaluate, fuse
from core.quotient import ActionSet, Quotient, TorchQuotient
from core.search_schedule import SearchSchedule, SearchTimeout

ValueFn = Callable[[torch.Tensor], torch.Tensor]


# -- the whole puzzle ------------------------------------------------------------------------------------

MASKS = ("rung", "progress", "none")


@dataclass
class Phase:
    """Rung k as a phase of the whole puzzle: its slots in the whole feature vector, the actions a state in this
    phase may take (``actions``), and the rung's own actions, which the training walks use (``walk``)."""
    name: str
    start: int
    stop: int
    actions: np.ndarray        # (A_k,) indices into WholePuzzle.actions allowed in this phase
    walk: np.ndarray           # (W_k,) the rung's own actions, in the rung's order
    walk_inverse: np.ndarray   # (W_k,) position in ``walk`` of each walk action's inverse on the rung, or -1

    @property
    def n_slots(self) -> int:
        return self.stop - self.start


class WholePuzzle:
    """The whole puzzle as one quotient: every rung's features side by side, the union of the rungs' actions.

    Two words with the same permutation of the points are one action; a word kept by several rungs belongs to
    each of their phases. ``mask`` says which actions a state in phase k may take:

    ``rung``      rung k's own actions (the ladder's moves, chosen by one network across all rungs);
    ``progress``  every action that leaves rungs 0 .. k-1 solved: rung k's and every later rung's macros, and any
                  other action that happens not to disturb the finished rungs;
    ``none``      every action, including the ones that break finished rungs.

    The mask is applied by the trainer and the search, never given to the network. Training walks always use the
    rungs' own actions, so the training states are the states the solver meets."""

    def __init__(self, ladder: Ladder, mask: str = "rung") -> None:
        if len(ladder.rungs) != len(ladder.specs):
            raise ValueError("build every rung of the ladder first: its actions are needed")
        if mask not in MASKS:
            raise ValueError(f"mask must be one of {MASKS}")
        self.ladder = ladder
        self.mask = mask
        model = ladder.model
        words: list[tuple[int, ...]] = []
        index: dict[bytes, int] = {}
        feats = []
        own: list[tuple[str, int, int, np.ndarray, np.ndarray]] = []
        start = 0
        for rung in ladder.rungs:
            ids = []
            for w, t in zip(rung.actions.words, rung.actions.tables):
                key = t.tobytes()
                if key not in index:
                    index[key] = len(words)
                    words.append(tuple(int(m) for m in w))
                ids.append(index[key])
            n = sum(f.n_slots for f in rung.features)
            own.append((rung.spec.name, start, start + n, np.asarray(ids, dtype=np.int64),
                        np.asarray(rung.actions.inverse, dtype=np.int64)))
            feats.extend(rung.features)
            start += n
        self.actions = ActionSet.from_words(model, words)
        self.quotient = Quotient(feats, self.actions, name="whole")
        q = self.quotient
        moved = q.step_all(q.goal[None])[0]                                    # (A, S): every action on the goal
        keeps = np.stack([(moved[:, a:b] == q.goal[a:b]).all(1) for _, a, b, _, _ in own])   # (K, A)
        all_ids = np.arange(len(words), dtype=np.int64)
        self.phases: list[Phase] = []
        for k, (name, a, b, ids, inv) in enumerate(own):
            if mask == "rung":
                allowed = ids
            elif mask == "progress":
                allowed = all_ids[keeps[:k].all(0)] if k else all_ids
            else:
                allowed = all_ids
            self.phases.append(Phase(name, a, b, allowed, ids, inv))

    @property
    def n_phases(self) -> int:
        return len(self.phases)

    def project(self, lab: np.ndarray) -> np.ndarray:
        """Whole feature vector(s) of labelled state(s)."""
        return self.quotient.project(lab)

    def phase_of(self, vecs: np.ndarray) -> np.ndarray:
        """First rung with a slot off its goal value, per vector (n_phases when solved)."""
        vecs = np.atleast_2d(vecs)
        bad = np.stack([(vecs[:, p.start:p.stop] != self.quotient.goal[p.start:p.stop]).any(1) for p in self.phases], 1)
        return np.where(bad.any(1), bad.argmax(1), self.n_phases)

    def fingerprint(self) -> str:
        """What a network trained on this puzzle depends on: the slots and every phase's actions. A network
        trained under one mask can be trained further under another, so the mask is not part of it."""
        return digest("whole", [w for w in self.actions.words], [(p.name, p.start, p.stop, p.walk.tolist())
                                                                for p in self.phases])


class TorchWhole:
    """WholePuzzle on a torch device: the dynamics, the phase of states and each phase's moves."""

    def __init__(self, wp: WholePuzzle, device: str | torch.device) -> None:
        self.wp = wp
        self.tq = TorchQuotient(wp.quotient, str(device))
        self.device = self.tq.device
        self.K = wp.n_phases
        self.n_slots = self.tq.n_slots
        self.ranges = [(p.start, p.stop) for p in wp.phases]
        self.ids = [torch.as_tensor(p.actions, dtype=torch.long, device=self.device) for p in wp.phases]
        self.walk_ids = [torch.as_tensor(p.walk, dtype=torch.long, device=self.device) for p in wp.phases]
        self.walk_inv = [torch.as_tensor(p.walk_inverse, dtype=torch.long, device=self.device) for p in wp.phases]
        self.src = [self.tq.src[i] for i in self.ids]
        self.base = [self.tq.base[i] for i in self.ids]
        self.costs = [self.tq.costs[i] for i in self.ids]
        self.n_slots_per_phase = torch.tensor([p.n_slots for p in wp.phases], dtype=torch.float32, device=self.device)

    def unsolved(self, states: torch.Tensor) -> torch.Tensor:
        """(N, K) number of each rung's slots that are off their goal value."""
        diff = states != self.tq.goal
        return torch.stack([diff[:, a:b].sum(1) for a, b in self.ranges], 1)

    def phase(self, states: torch.Tensor | None = None, unsolved: torch.Tensor | None = None) -> torch.Tensor:
        """(N,) first rung that is not solved; K for the goal."""
        u = self.unsolved(states) if unsolved is None else unsolved
        bad = u > 0
        first = torch.argmax(bad.to(torch.int32), dim=1)
        return torch.where(bad.any(1), first, torch.full_like(first, self.K))

    def step_all_phase(self, states: torch.Tensor, k: int) -> torch.Tensor:
        """(n, S) -> (n, A_k, S): every state under every action of phase k."""
        idx = states[:, self.src[k]].long() + self.base[k][None]
        return self.tq.lut[idx]


# -- reward shaping ----------------------------------------------------------------------------------------

BASELINES = ("none", "pieces", "gated", "smooth")


class TorchBaseline:
    """H(s), the potential's negative: an estimate of the remaining cost from counts of unsolved slots.

    ``none``    H = 0 (plain DAVI).
    ``pieces``  every unsolved slot of rung j costs c_j / n_j, with c_j the rung's mean cost and n_j its slots.
    ``gated``   only the current rung's unsolved slots count (at c_k / n_k each), plus the mean cost of every
                later rung. Later rungs are reshuffled by every early move (measured on the Maltese ladder:
                5 slots per corner move), so counting them early adds noise, not information. This is the
                reward-machine construction (Toro Icarte et al. 2022) with a within-rung progress term.
    ``smooth``  ``gated`` with each slot at c_k / (n_k f_k), f_k the fraction of rung k's slots unsolved when the
                rung is mixed (measured). A mixed rung then counts c_k on average, so finishing rung k lands, on
                average, where phase k+1 starts. Under ``gated`` a mixed rung counts only f_k c_k, and the potential
                jumps by (1 - f_k) c_k at each phase boundary (14 to 25 quarter turns on the Maltese cube), a step
                the network has to learn to cancel.
    H is 0 at the goal under each kind, as a potential must be."""

    def __init__(self, kind: str, rung_costs: Sequence[float], tw: TorchWhole,
                 mixed_unsolved: Sequence[float] = ()) -> None:
        if kind not in BASELINES:
            raise ValueError(f"baseline must be one of {BASELINES}")
        self.kind = kind
        self.K = tw.K
        if kind == "none":
            return
        if len(rung_costs) != tw.K:
            raise ValueError(f"the {kind} baseline needs one mean cost per rung ({tw.K}), got {len(rung_costs)}")
        c = torch.tensor([float(x) for x in rung_costs], dtype=torch.float32, device=tw.device)
        self.w = c / tw.n_slots_per_phase                                         # (K,) cost per unsolved slot
        if kind == "smooth":
            if len(mixed_unsolved) != tw.K:
                raise ValueError("the smooth baseline needs the unsolved fraction of each mixed rung")
            self.w = self.w / torch.tensor([float(x) for x in mixed_unsolved], dtype=torch.float32, device=tw.device)
        tail = torch.flip(torch.cumsum(torch.flip(c, [0]), 0), [0]) - c           # cost of the rungs after k
        self.w_ext = torch.cat([self.w, torch.zeros(1, device=tw.device)])        # phase K (the goal) has no rung
        self.tail_ext = torch.cat([tail, torch.zeros(1, device=tw.device)])

    def __call__(self, unsolved: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        if self.kind == "none":
            return torch.zeros(unsolved.shape[0], device=unsolved.device)
        u = unsolved.float()
        if self.kind == "pieces":
            return (u * self.w).sum(1)
        u_ext = torch.cat([u, torch.zeros((u.shape[0], 1), device=u.device)], 1)
        current = u_ext.gather(1, phase[:, None]).reshape(-1)
        return self.w_ext[phase] * current + self.tail_ext[phase]


# -- search -------------------------------------------------------------------------------------------------

def _cheapest_copy(h: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Indices of one copy per distinct hash, the one with the least g, in increasing hash order."""
    o = torch.argsort(g, stable=True)
    o = o[torch.argsort(h[o], stable=True)]
    hs = h[o]
    first = torch.ones(hs.shape[0], dtype=torch.bool, device=h.device)
    first[1:] = hs[1:] != hs[:-1]
    return o[first]


@torch.no_grad()
def phased_beam_search(tw: TorchWhole, value_fn: ValueFn, start: torch.Tensor, width: int = 4096,
                       max_steps: int = 400, weight: float = 0.3, expand_children: int = 1 << 21,
                       deadline: float | None = None, stats: dict | None = None,
                       budget: float | None = None) -> tuple[list[int], float] | None:
    """Beam search on the whole puzzle in which each state is expanded with its own phase's actions.

    Otherwise as core.search.beam_search: duplicates are dropped (64-bit hash), children are scored by
    f = weight * g + h, the ``width`` best are kept, and the search goes on while a beam state could still beat
    the best solution found. One difference: a state already expanded is dropped only if it is not reached more
    cheaply now. Macros cost several moves and primitive moves one, so the first path to a state is often not
    the cheapest, and core.search.beam_search would keep the first one. Returns (global action ids, cost) or
    None. Raises ``SearchTimeout`` if ``deadline`` passes before any solution is found. With ``budget``, the next
    beam is also cut, best first, where its states would make more than ``width * budget`` children, so phases with
    many actions keep fewer states (the time of a step is the time to value its children). A ``stats`` dict is filled
    with the steps, the step of the first solution, the children made and valued, and the time spent valuing them
    (synchronised, so only for profiling)."""
    tq = tw.tq
    if stats is not None:
        stats.update(steps=0, first_solution_step=None, children=0, valued=0, value_seconds=0.0, per_step=[])
    dev = tw.device
    start = start.to(dev).reshape(1, -1)
    if bool(tq.is_goal(start)[0]):
        return [], 0.0
    min_cost = float(tq.costs.min())
    n_kids = torch.tensor([t.shape[0] for t in tw.ids] + [0], device=dev)     # children per state, by phase
    beam = start
    g = torch.zeros(1, device=dev)
    visited = tq.hash(beam)                           # sorted hashes of every expanded state ...
    visited_g = g.clone()                             # ... and the cheapest cost it was expanded at
    parents: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    best_cost = float("inf")
    best_path: tuple[int, int, int] | None = None     # (step, parent index, global action)

    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            if best_path is not None:
                break
            raise SearchTimeout
        t_step = time.time()
        ph = tw.phase(beam)
        made = 0
        # The next beam is built chunk by chunk in a pool of at most ``width`` states (one copy per state, the
        # cheapest), so memory does not grow with the beam width times the number of actions.
        pool = None                                   # (kids, g, parent, action, hash, f), sorted by hash
        for k in torch.unique(ph).tolist():
            if k >= tw.K:
                continue
            rows = torch.nonzero(ph == k).reshape(-1)
            n_act = tw.ids[k].shape[0]
            chunk = max(1, expand_children // n_act)
            for s in range(0, rows.shape[0], chunk):
                r = rows[s:s + chunk]
                kids = tw.step_all_phase(beam[r], k).reshape(r.shape[0] * n_act, -1)
                made += kids.shape[0]
                kg = (g[r][:, None] + tw.costs[k][None, :]).reshape(-1)
                par = r.repeat_interleave(n_act)
                act = tw.ids[k].repeat(r.shape[0])
                solved = tq.is_goal(kids)
                if solved.any():
                    cand = torch.where(solved, kg, torch.full_like(kg, float("inf")))
                    i = int(torch.argmin(cand))
                    if float(kg[i]) < best_cost:
                        best_cost = float(kg[i])
                        best_path = (step, int(par[i]), int(act[i]))
                        if stats is not None and stats["first_solution_step"] is None:
                            stats["first_solution_step"] = step
                keep = ~solved & (kg < best_cost)
                kids, kg, par, act = kids[keep], kg[keep], par[keep], act[keep]
                if kids.shape[0] == 0:
                    continue
                h = tq.hash(kids)
                u = _cheapest_copy(h, kg)                                     # sorted by hash
                kids, kg, par, act, h = kids[u], kg[u], par[u], act[u], h[u]
                # drop states expanded before at no greater cost, and states the pool holds at no greater cost
                pos = torch.searchsorted(visited, h).clamp_(max=visited.shape[0] - 1)
                fresh = (visited[pos] != h) | (kg < visited_g[pos])
                if pool is not None:
                    pos = torch.searchsorted(pool[4], h).clamp_(max=pool[4].shape[0] - 1)
                    fresh &= (pool[4][pos] != h) | (kg < pool[1][pos])
                kids, kg, par, act, h = kids[fresh], kg[fresh], par[fresh], act[fresh], h[fresh]
                if kids.shape[0] == 0:
                    continue
                if stats is not None:
                    torch.cuda.synchronize(dev) if dev.type == "cuda" else None
                    t_val = time.time()
                f = weight * kg + value_fn(kids).reshape(-1)
                if stats is not None:
                    torch.cuda.synchronize(dev) if dev.type == "cuda" else None
                    stats["value_seconds"] += time.time() - t_val
                    stats["valued"] += kids.shape[0]
                cand = (kids, kg, par, act, h, f)
                if pool is not None:
                    cand = tuple(torch.cat([a, b]) for a, b in zip(pool, cand))
                    u = _cheapest_copy(cand[4], cand[1])
                    cand = tuple(x[u] for x in cand)
                if cand[0].shape[0] > width:
                    top = torch.topk(cand[5], width, largest=False).indices
                    top = top[torch.argsort(cand[4][top])]                    # back in hash order
                    cand = tuple(x[top] for x in cand)
                pool = cand
        if stats is not None:
            stats["steps"] = step + 1
            stats["children"] += made
            stats["per_step"].append((step, int(ph.min()), int(ph.max()), made, round(time.time() - t_step, 3)))
        if pool is None:
            break
        if budget is not None:
            order = torch.argsort(pool[5])
            keep = order[torch.cumsum(n_kids[tw.phase(pool[0][order])], 0) <= width * budget]
            pool = tuple(x[keep if keep.numel() else order[:1]] for x in pool)
        beam, g, par, act, bh = pool[0], pool[1], pool[2], pool[3], pool[4]
        del pool
        parents.append(par.cpu()); actions.append(act.cpu())
        # merge the new beam into the expanded set, keeping the cheapest cost per state
        all_h = torch.cat([visited, bh])
        all_g = torch.cat([visited_g, g])
        u = _cheapest_copy(all_h, all_g)
        visited, visited_g = all_h[u], all_g[u]
        if best_path is not None and float(g.min()) + min_cost >= best_cost:
            break

    if best_path is None:
        return None
    step, idx, action = best_path
    out = [action]
    for t in range(step - 1, -1, -1):
        out.append(int(actions[t][idx]))
        idx = int(parents[t][idx])
    out.reverse()
    return out, best_cost


# -- the value network ------------------------------------------------------------------------------------

class TwinNet(torch.nn.Module):
    """Two value networks, ``a`` and ``b``, with their own weights and their own half of every update's states.

    Each one's training targets take the action it values cheapest and the other network's value of that child
    (``WholeValue.training_targets``), so a child chosen because one network underrates it is not also valued by
    that network: the double estimator of double Q-learning and TD3, which removes the downward bias of a min over
    hundreds of noisy children. The output, used by the search, is ``a`` alone or the mean of both (``mean``)."""

    def __init__(self, make: Callable[[], torch.nn.Module], mean: bool = False) -> None:
        super().__init__()
        self.a = make()
        self.b = make()
        self.mean = mean

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return (self.a(states) + self.b(states)) / 2 if self.mean else self.a(states)


@dataclass
class WholeConfig:
    res_dim: int = 1000
    num_blocks: int = 4
    batch_norm: bool = True
    batch_size: int = 8192
    states_per_update: int = 200_000
    itrs_per_update: int = 100
    lr: float = 1e-3
    lr_decay: float = 0.99995            # per training iteration, as in DeepXube
    loss_thresh: float = 0.06            # target network update when the (scaled) loss is below this
    force_sync: int = 4                  # ... or after this many updates without one
    amp: bool = True
    target_children: int = 1 << 22       # children evaluated per chunk when computing targets
    # stratified walks (measured by ``measure_walks`` when left empty)
    walk_factor: float = 3.5             # the autopilot's rule: walks of 3.5 x the measured mixing length
    walk_min: int = 10
    walk_cap: int = 400
    depths: tuple[int, ...] = ()         # per phase: longest walk with the phase's own actions
    mix_walks: tuple[int, ...] = ()      # per phase: walk with the phase's actions that mixes it and every later rung
    phase_weights: tuple[float, ...] = ()   # share of training states per phase (default: in proportion to depths)
    excursion_share: float = 0.25        # with mask "none": share of states that also take a few allowed actions
    excursion_max: int = 3
    # reward shaping
    baseline: str = "none"
    rung_costs: tuple[float, ...] = ()   # mean cost of each rung, for the baseline
    mixed_unsolved: tuple[float, ...] = ()  # fraction of each rung's slots unsolved once it is mixed (measured)
    double: bool = False                 # choose the backup's action with the online network (bellman_targets)
    twin: bool = False                   # two networks, each valuing the children the other chooses (TwinNet)
    twin_search: str = "a"               # the twin's value in the search: network "a" alone, or the "mean" of both
    cross_weight: float = 1.0            # a double backup's target: this share from the other network's value of the
                                         # chosen child, the rest from the chooser's own (weighted double Q-learning)
    target_tau: float = 0.0              # > 0: after the first refresh the target network follows the online one as
                                         # an average, target += tau * (online - target) after every step (Polyak),
                                         # instead of being replaced by a copy; the search then uses the average
    seed: int = 0


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"


class WholeValue:
    """The whole puzzle's value network, its training, and search with it."""

    def __init__(self, wp: WholePuzzle, cfg: WholeConfig | None = None, device: str | None = None,
                 log: Callable[[str], None] | None = None) -> None:
        self.wp = wp
        self.cfg = cfg or WholeConfig()
        self.log = log or (lambda msg: print(msg, flush=True))
        self.tw = TorchWhole(wp, device or _default_device())
        self.device = self.tw.device
        torch.manual_seed(self.cfg.seed)
        q = wp.quotient
        self.scale = float(q.costs.mean())

        def make() -> ResnetFC:
            return ResnetFC(q.n_slots, q.n_classes, res_dim=self.cfg.res_dim, num_blocks=self.cfg.num_blocks,
                            batch_norm=self.cfg.batch_norm, scale=self.scale).to(self.device)

        self.net = TwinNet(make, mean=self.cfg.twin_search == "mean") if self.cfg.twin else make()
        self.target = copy.deepcopy(self.net).eval()
        self.target_is_zero = True
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        self.itr = 0
        self.update = 0
        self.target_updates = 0
        self.since_sync = 0
        self.history: list[dict] = []
        self._fused: dict[int, tuple] = {}             # id(network) -> (version, fused inference copy or None)
        self.search_budget: float | None = None         # children per beam state per step (phased_beam_search)
        self.fingerprint = wp.fingerprint()
        if not self.cfg.depths or not self.cfg.mix_walks:
            self.measure_walks()
        if self.cfg.baseline == "smooth" and not self.cfg.mixed_unsolved:
            self.cfg.mixed_unsolved = tuple(self._mixed_unsolved(k) for k in range(self.tw.K))
        if not self.cfg.phase_weights:
            d = np.asarray(self.cfg.depths, dtype=float)
            self.cfg.phase_weights = tuple((d / d.sum()).tolist())
        self.baseline = TorchBaseline(self.cfg.baseline, self.cfg.rung_costs, self.tw, self.cfg.mixed_unsolved)

    # -- walks ---------------------------------------------------------------------------------------------
    @torch.no_grad()
    def _walk(self, states: torch.Tensor, k: int, lengths: torch.Tensor) -> torch.Tensor:
        """Random walks with rung k's own actions, ``lengths[i]`` actions for state i, never undoing the last one."""
        tw = self.tw
        n = states.shape[0]
        n_act = tw.walk_ids[k].shape[0]
        last = torch.full((n,), -1, dtype=torch.long, device=self.device)
        for t in range(int(lengths.max()) if n else 0):
            pos = torch.randint(0, n_act, (n,), device=self.device)
            undo = (last >= 0) & (pos == tw.walk_inv[k][last.clamp(min=0)])
            pos = torch.where(undo, (pos + 1) % n_act, pos)
            active = t < lengths
            nxt = tw.tq.step(states, tw.walk_ids[k][pos])
            states = torch.where(active[:, None], nxt, states)
            last = torch.where(active, pos, last)
        return states

    @torch.no_grad()
    def _excursions(self, states: torch.Tensor) -> torch.Tensor:
        """A share ``excursion_share`` of the states gets 1 .. ``excursion_max`` random actions allowed in its phase,
        so that with a looser mask the network has seen the states the search can step into."""
        cfg = self.cfg
        if cfg.excursion_share <= 0 or self.wp.mask == "rung":
            return states
        tw = self.tw
        pick = torch.rand(states.shape[0], device=self.device) < cfg.excursion_share
        left = torch.where(pick, torch.randint(1, cfg.excursion_max + 1, (states.shape[0],), device=self.device),
                           torch.zeros(states.shape[0], dtype=torch.long, device=self.device))
        for _ in range(cfg.excursion_max):
            ph = tw.phase(states)
            for k in range(tw.K):
                rows = torch.nonzero((ph == k) & (left > 0)).reshape(-1)
                if rows.shape[0]:
                    a = tw.ids[k][torch.randint(0, tw.ids[k].shape[0], (rows.shape[0],), device=self.device)]
                    states[rows] = tw.tq.step(states[rows], a)
            left = left - 1
        return states

    @torch.no_grad()
    def _match_curve(self, k: int, lo: int, hi: int, n_walks: int = 4096, max_len: int = 600) -> np.ndarray:
        """Fraction of slots lo:hi at their goal value after 1 .. max_len random actions of rung k from the goal."""
        tw = self.tw
        states = tw.tq.goal_batch(n_walks)
        goal = tw.tq.goal[lo:hi]
        n_act = tw.walk_ids[k].shape[0]
        last = torch.full((n_walks,), -1, dtype=torch.long, device=self.device)
        out = np.empty(max_len)
        for t in range(max_len):
            pos = torch.randint(0, n_act, (n_walks,), device=self.device)
            undo = (last >= 0) & (pos == tw.walk_inv[k][last.clamp(min=0)])
            pos = torch.where(undo, (pos + 1) % n_act, pos)
            states = tw.tq.step(states, tw.walk_ids[k][pos])
            last = pos
            out[t] = float((states[:, lo:hi] == goal).float().mean())
        return out

    def measure_walks(self) -> None:
        """Walk lengths from mixing measurements, as the autopilot chooses them: D_k = 3.5 x the mixing length of
        phase k's actions on rung k's own slots, and the mixing walk = 3.5 x their mixing length on the slots of
        rung k and every later rung."""
        cfg = self.cfg
        depths, mixes = [], []
        end = self.tw.n_slots
        for k, p in enumerate(self.wp.phases):
            own = mixing_length(self._match_curve(k, p.start, p.stop))
            later = mixing_length(self._match_curve(k, p.start, end))
            depths.append(int(min(cfg.walk_cap, max(cfg.walk_min, round(cfg.walk_factor * own)))))
            mixes.append(int(min(cfg.walk_cap, max(cfg.walk_min, round(cfg.walk_factor * later)))))
            self.log(f"  phase {k} [{p.name}]: {len(p.actions)} actions, mixes its own slots in {own} and the "
                     f"later ones in {later}; walks up to {depths[-1]}, mixing walk {mixes[-1]}")
        cfg.depths, cfg.mix_walks = tuple(depths), tuple(mixes)

    def _mixed_unsolved(self, k: int, tail: int = 100) -> float:
        """Fraction of rung k's slots not at their goal value once rung k's own walks have mixed it."""
        p = self.wp.phases[k]
        return float(1.0 - self._match_curve(k, p.start, p.stop)[-tail:].mean())

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        """Training states by stratified walks (module docstring)."""
        cfg = self.cfg
        K = self.tw.K
        counts = np.random.multinomial(n, np.asarray(cfg.phase_weights) / np.sum(cfg.phase_weights))
        parts = []
        for k, m in enumerate(counts.tolist()):
            if m == 0:
                continue
            x = self.tw.tq.goal_batch(m)
            if k + 1 < K:
                x = self._walk(x, k + 1, torch.full((m,), cfg.mix_walks[k + 1], device=self.device))
            x = self._walk(x, k, torch.randint(0, cfg.depths[k] + 1, (m,), device=self.device))
            parts.append(x)
        states = self._excursions(torch.cat(parts))
        return states[torch.randperm(states.shape[0], device=self.device)]

    # -- values ----------------------------------------------------------------------------------------------
    def _fast(self, net: torch.nn.Module) -> torch.nn.Module | None:
        """A fused inference copy (core.nnet.FusedResnet) of one of this value's networks, rebuilt when that network
        has changed: the online one after every training iteration, the target one at every refresh. None for
        any other module (a test's exact values) or off the GPU."""
        twin = self.cfg.twin
        online = (self.net, self.net.a, self.net.b) if twin else (self.net,)
        target = (self.target, self.target.a, self.target.b) if twin else (self.target,)
        if any(net is n for n in online):
            version = ("online", self.itr)
        elif any(net is n for n in target):
            version = ("target", self.target_updates, self.itr if self.cfg.target_tau > 0 else 0)
        else:
            return None
        hit = self._fused.get(id(net))
        if hit is not None and hit[0] == version:
            return hit[1]
        if isinstance(net, TwinNet):
            pair = [fuse(net.a), fuse(net.b)]
            fused = None if None in pair else TwinNet(lambda: pair.pop(0), mean=net.mean)
        else:
            fused = fuse(net)
        self._fused[id(net)] = (version, fused)
        return fused

    def _values(self, states: torch.Tensor, net: torch.nn.Module | None) -> torch.Tensor:
        """H + r, at least 0, exactly 0 at the goal. ``net=None`` means r = 0 (the zero target network)."""
        u = self.tw.unsolved(states)
        ph = self.tw.phase(unsolved=u)
        v = self.baseline(u, ph)
        if net is not None:
            v = v + evaluate(self._fast(net) or net, states, amp=self.cfg.amp).reshape(-1)
        v = v.clamp_(min=0)
        return torch.where(ph == self.tw.K, torch.zeros_like(v), v)

    @torch.no_grad()
    def value(self, states: torch.Tensor) -> torch.Tensor:
        """Estimated cost to solve the whole puzzle: the online network, or with ``target_tau`` its running average
        (the target network), which averages out the step-to-step jitter of training."""
        if self.cfg.target_tau > 0 and not self.target_is_zero:
            return self._values(states, self.target)
        return self._values(states, self.net)

    @torch.no_grad()
    def _follow(self, tau: float) -> None:
        """target += tau * (online - target), for the weights and the batch-norm statistics."""
        for t, o in zip(self.target.parameters(), self.net.parameters()):
            t.lerp_(o, tau)
        for t, o in zip(self.target.buffers(), self.net.buffers()):
            if t.dtype.is_floating_point:
                t.lerp_(o, tau)
            else:
                t.copy_(o)

    @torch.no_grad()
    def bellman_targets(self, states: torch.Tensor, net: torch.nn.Module | None = None,
                        zero: bool | None = None, chooser: torch.nn.Module | None = None) -> torch.Tensor:
        """min over the phase's actions of cost + value(child), 0 at the goal; the target network by default.

        With ``chooser``, the action is the one ``chooser`` values cheapest and the target is its cost plus ``net``'s
        value of that child (a double backup, as in double Q-learning): the min over hundreds of noisy children
        is otherwise biased low, by more the more actions a phase has. The other network's value of a chosen child
        is biased high instead (the chooser often misses the cheapest child), so ``cfg.cross_weight`` below 1 mixes
        in the chooser's own value, as in weighted double Q-learning (Zhang, Pan and Kochenderfer 2017).
        ``cfg.double`` makes the online network the chooser of the default backup."""
        tw = self.tw
        select = chooser
        if net is None:
            net = self.target
            zero = self.target_is_zero if zero is None else zero
            if self.cfg.double and not zero:
                select = self.net
        net = None if zero else net
        ph = tw.phase(states)
        out = torch.zeros(states.shape[0], device=self.device)
        for k in range(tw.K):
            rows = torch.nonzero(ph == k).reshape(-1)
            if rows.shape[0] == 0:
                continue
            n_act = tw.ids[k].shape[0]
            chunk = max(1, self.cfg.target_children // n_act)
            for s in range(0, rows.shape[0], chunk):
                r = rows[s:s + chunk]
                kids = tw.step_all_phase(states[r], k).reshape(r.shape[0] * n_act, -1)
                q = tw.costs[k][None, :] + self._values(kids, select if select is not None else net).reshape(
                    r.shape[0], n_act)
                if select is None:
                    out[r] = q.min(dim=1).values
                else:
                    a = q.argmin(dim=1)
                    chosen = kids.reshape(r.shape[0], n_act, -1)[torch.arange(r.shape[0], device=self.device), a]
                    other = tw.costs[k][a] + self._values(chosen, net)
                    w = self.cfg.cross_weight
                    out[r] = other if w >= 1.0 else w * other + (1.0 - w) * q.min(dim=1).values
        return out

    @torch.no_grad()
    def training_targets(self, states: torch.Tensor) -> torch.Tensor:
        """The targets ``train`` fits. With twin networks the first half of the states is network a's: a chooses each
        action and b values the child; the second half is b's, the other way round."""
        if not self.cfg.twin or self.target_is_zero:
            return self.bellman_targets(states)
        h = states.shape[0] // 2
        return torch.cat([self.bellman_targets(states[:h], net=self.target.b, zero=False, chooser=self.target.a),
                          self.bellman_targets(states[h:], net=self.target.a, zero=False, chooser=self.target.b)])

    # -- training --------------------------------------------------------------------------------------------
    def train(self, max_updates: int, time_limit: float | None = None, checkpoint: str | Path | None = None,
              log_every: int = 1) -> None:
        cfg = self.cfg
        use_amp = cfg.amp and self.device.type == "cuda"
        t_start = time.time()
        stop_at = self.update + max_updates
        while self.update < stop_at:
            t0 = time.time()
            states = self.sample(cfg.states_per_update)
            y = self.training_targets(states)
            u = self.tw.unsolved(states)
            resid = y - self.baseline(u, self.tw.phase(unsolved=u))       # the network learns r = J - H
            t_data = time.time() - t0
            self.net.train()
            loss_sum, n = 0.0, states.shape[0]
            h = n // 2
            for _ in range(cfg.itrs_per_update):
                for group in self.opt.param_groups:
                    group["lr"] = cfg.lr * (cfg.lr_decay ** self.itr)
                if cfg.twin:                                           # each network fits its own half
                    ia = torch.randint(0, h, (cfg.batch_size,), device=self.device)
                    ib = torch.randint(h, n, (cfg.batch_size,), device=self.device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                        pa = self.net.a(states[ia]).reshape(-1)
                        pb = self.net.b(states[ib]).reshape(-1)
                    loss = (torch.nn.functional.mse_loss(pa.float() / self.scale, resid[ia] / self.scale)
                            + torch.nn.functional.mse_loss(pb.float() / self.scale, resid[ib] / self.scale)) / 2
                else:
                    idx = torch.randint(0, n, (cfg.batch_size,), device=self.device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                        pred = self.net(states[idx]).reshape(-1)
                    loss = torch.nn.functional.mse_loss(pred.float() / self.scale, resid[idx] / self.scale)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()
                if cfg.target_tau > 0 and not self.target_is_zero:
                    self._follow(cfg.target_tau)
                loss_sum += float(loss.detach())
                self.itr += 1
            last_loss = float(loss.detach())
            self.update += 1
            self.since_sync += 1
            synced = False
            if cfg.target_tau > 0 and not self.target_is_zero:
                self.since_sync = 0                    # the average follows every step; no copies after the first
            elif last_loss < cfg.loss_thresh or self.since_sync >= cfg.force_sync:
                self.target.load_state_dict(self.net.state_dict())
                self._fused.clear()
                self.target_is_zero = False
                self.target_updates += 1
                self.since_sync = 0
                synced = True
            row = dict(update=self.update, itr=self.itr, loss=loss_sum / cfg.itrs_per_update,
                       target_mean=float(y.mean()), target_max=float(y.max()), synced=synced,
                       target_updates=self.target_updates, seconds=time.time() - t0, data_seconds=t_data)
            self.history.append(row)
            if self.update % log_every == 0:
                self.log(f"    update {self.update:5d}  loss {row['loss']:.4f}  target mean {row['target_mean']:7.2f} "
                         f"max {row['target_max']:7.1f}  sync {self.target_updates:4d}{'*' if synced else ' '}  "
                         f"data {t_data:5.1f}s  total {row['seconds']:5.1f}s")
            if checkpoint is not None and self.update % 5 == 0:
                self.save(checkpoint)
            if time_limit is not None and time.time() - t_start > time_limit:
                break
        if checkpoint is not None:
            self.save(checkpoint)

    # -- solving ---------------------------------------------------------------------------------------------
    def max_steps(self) -> int:
        return int(max(200, 4 * sum(self.cfg.depths)))

    def search(self, vec, width: int = 4096, weight: float = 0.3, deadline: float | None = None,
               schedule: SearchSchedule | None = None, budget: float | None = None) -> tuple[list[int], float] | None:
        """Actions (global ids) taking the whole feature vector ``vec`` to the goal, or None. Tries each search of
        ``schedule`` (default: the given width, then 8x wider). Raises SearchTimeout past ``deadline``."""
        start = torch.as_tensor(np.asarray(vec), dtype=torch.uint8, device=self.device)
        schedule = schedule or SearchSchedule.standard(width, weight)
        for w, wt in schedule.attempts:
            found = phased_beam_search(self.tw, self.value, start, width=w, max_steps=self.max_steps(), weight=wt,
                                       deadline=deadline, budget=self.search_budget if budget is None else budget)
            if found is not None:
                return found
        return None

    # -- persistence -----------------------------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        torch.save({"net": self.net.state_dict(), "target": self.target.state_dict(), "opt": self.opt.state_dict(),
                    "cfg": asdict(self.cfg), "itr": self.itr, "update": self.update,
                    "target_updates": self.target_updates, "since_sync": self.since_sync,
                    "target_is_zero": self.target_is_zero, "history": self.history,
                    "fingerprint": self.fingerprint}, tmp)
        tmp.replace(path)

    def load(self, path: str | Path) -> "WholeValue":
        data = torch.load(Path(path), map_location=self.device, weights_only=False)
        if data.get("fingerprint") != self.fingerprint:
            raise ValueError(f"{path} was trained for other actions or phases than this ladder's")
        self.net.load_state_dict(data["net"])
        self.target.load_state_dict(data["target"])
        self.opt.load_state_dict(data["opt"])
        self.itr, self.update = data["itr"], data["update"]
        self.target_updates, self.since_sync = data["target_updates"], data["since_sync"]
        self.target_is_zero = data["target_is_zero"]
        self.history = data["history"]
        self._fused.clear()
        return self

    @staticmethod
    def read_config(path: str | Path) -> WholeConfig:
        data = torch.load(Path(path), map_location="cpu", weights_only=False)
        names = {f.name for f in fields(WholeConfig)}
        raw = {k: v for k, v in data["cfg"].items() if k in names}
        for key in ("depths", "mix_walks", "phase_weights", "rung_costs", "mixed_unsolved"):
            raw[key] = tuple(raw.get(key, ()))
        return WholeConfig(**raw)

    @staticmethod
    def from_checkpoint(wp: WholePuzzle, path: str | Path, device: str | None = None,
                        log: Callable[[str], None] | None = None) -> "WholeValue":
        return WholeValue(wp, WholeValue.read_config(path), device, log=log).load(path)


# -- the solver, as the acceptance test sees it -----------------------------------------------------------

class WholeSolver:
    """Solves visible states with one network. Offers the parts of ``Ladder`` that core/acceptance.py uses:
    ``ready``, ``fingerprint``, ``rungs`` (names only, for the per-phase costs) and ``solve``."""

    def __init__(self, ladder: Ladder, value: WholeValue) -> None:
        self.ladder = ladder
        self.value = value
        self.wp = value.wp
        self.alg = ladder.alg
        self.rungs = [SimpleNamespace(spec=SimpleNamespace(name=p.name)) for p in self.wp.phases]

    def ready(self) -> bool:
        return True

    def fingerprint(self) -> str:
        return digest("whole-solver", self.value.fingerprint, self.value.update, self.value.itr)

    def cost_by_phase(self, vec: np.ndarray, acts: Sequence[int]) -> list[int]:
        """Cost spent in each phase along a path (the phase of the state each action was taken in)."""
        q = self.wp.quotient
        out = [0] * self.wp.n_phases
        cur = np.asarray(vec, dtype=np.uint8)
        for a in acts:
            k = int(self.wp.phase_of(cur)[0])
            out[k] += int(q.costs[a])
            cur = q.step(cur[None], int(a))[0]
        return out

    def solve_vector(self, vec: np.ndarray, beam: int | None = None, weight: float = 0.3,
                     deadline: float | None = None, schedule: SearchSchedule | None = None) -> tuple[list[int], list[int]]:
        """Global action ids and per-phase costs for a whole feature vector. Raises RungFailure."""
        try:
            found = self.value.search(vec, width=beam or 4096, weight=weight, deadline=deadline, schedule=schedule)
        except SearchTimeout:
            raise RungFailure("whole", "timeout", "out of time before a solution was found") from None
        if found is None:
            raise RungFailure("whole", "search_exhausted", "search found no solution")
        acts, _ = found
        return acts, self.cost_by_phase(vec, acts)

    def solve(self, state: np.ndarray, beam: int | None = None, weight: float = 0.3,
              time_limit: float | None = None, schedule: SearchSchedule | None = None) -> tuple[tuple[int, ...], list[int]]:
        """Solve a visible state of the underlying GatherPuzzle. Returns the reduced primitive word (turns merged,
        as the ladder does) and the cost spent per phase. Raises RungFailure (a LookupError)."""
        deadline = None if time_limit is None else time.time() + time_limit
        vec = self.wp.project(self.ladder.structure.decode(state))
        acts, per_phase = self.solve_vector(vec, beam=beam, weight=weight, deadline=deadline, schedule=schedule)
        word = self.alg.simplify(self.alg.concat(*[self.wp.actions.words[a] for a in acts]))
        return word, per_phase
