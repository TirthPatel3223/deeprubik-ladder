"""Training a learned rung with no person in the loop: its settings, when to stop, and repairs.

A person used to decide three things for every learned rung (``scripts/train_rung.py --walk --minutes``, and
what to change when the network failed). The autopilot decides them from measurements.

Walk length
    DAVI trains on random walks from the goal, so walks must reach states as far out as those the solver
    hands the rung, or the values flatten there. Before training, walks with the rung's actions are run until
    the fraction of slots at their goal value is within 2% of its value on well mixed states; the walk length
    is 3.5 times that. On the four rungs trained by hand this gives 28, 38, 38 and 21, against the chosen 30,
    30, 40 and 20.
When to stop
    Every few minutes the network is tested on a fixed set of states as the solver hands them to the rung
    (deep scrambles with the earlier rungs solved, ``Ladder.sample_rung_states``), with the solver's first
    beam search, 15 s per state. Phase A: train until it solves every test state. Phase B: keep training while
    the mean solution cost still drops by more than 1% over three tests. A time cap per rung bounds both.
    At the end the network of the best test that solved every state is kept, which need not be the last
    (later networks can score worse, or fail some states). A test of a weak network stops once
    a state has failed and half the round's training time has gone into testing (on the Maltese colour rung
    the first test of the whole set took 16 minutes against 5 of training), so testing never dominates;
    a network that solves every state is always tested on all of them.
Repairs
    If the share of solved test states stops rising in phase A, the autopilot changes, in order: the walk
    length (x1.5, twice); the action set (twice the cheapest moves and twice the moves that leave the cheap
    moves' subgroup; the network keeps its weights, since its input does not depend on the actions); the
    network width (a new network). These are the remedies that fixed the rungs trained by hand. Splitting a
    rung into exact tables changes the ladder and is left to the ladder's designer.

Everything is recorded in ``rungNN_<name>.train.json`` beside the checkpoint, so a stopped run resumes and a
finished rung is not trained again.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from core.davi import DaviConfig, RungValue, default_device
from core.fingerprint import digest, feature_key
from core.ladder import Ladder, RungSpec, retire
from core.quotient import TorchQuotient
from core.search import SearchTimeout, beam_search


@dataclass
class AutoTrainConfig:
    # test states, as the solver hands them to the rung
    test_states: int = 32
    final_states: int = 64
    min_depth: int = 2000
    max_depth: int = 5000
    seed: int = 1
    # tests
    round_minutes: float = 5.0          # training time between two tests
    eval_beam: int = 4096
    eval_weight: float = 0.3
    eval_time: float = 15.0             # seconds per test state
    eval_give_up: int = 8               # a test stops early if none of its first states is solved
    eval_budget: float = 0.5            # ... or, once a state has failed, after this share of the round's training time
    target: float = 1.0                 # share of test states solved that ends phase A
    stall_tests: int = 3                # phase A: tests without progress before a repair
    min_updates_per_repair: int = 30    # ... and at least this many updates since the last one
    plateau_tests: int = 3              # phase B: tests without a gain before stopping
    plateau_gain: float = 0.01
    max_minutes: float = 120.0          # per rung
    # walk length
    walk_factor: float = 3.5
    walk_min: int = 10
    walk_cap: int = 400
    walk_growth: float = 1.5
    # repairs, tried in this order
    max_walk_repairs: int = 2
    action_growth: int = 2
    max_action_repairs: int = 1
    wider_res_dim: int = 2000
    max_width_repairs: int = 1
    # network and training data
    res_dim: int = 1000
    num_blocks: int = 4
    batch_size: int = 8192
    states_per_update: int = 200_000
    itrs_per_update: int = 100
    lr: float = 1e-3
    loss_thresh: float = 0.06
    force_sync: int = 4


# -- measurements --------------------------------------------------------------------------------------

@torch.no_grad()
def match_curve(tq: TorchQuotient, n_walks: int = 4096, max_len: int = 1000, seed: int = 0) -> np.ndarray:
    """Fraction of slots at their goal value after 1 .. max_len random actions (never undoing the last)."""
    gen = torch.Generator(device=tq.device).manual_seed(seed)
    states = tq.goal_batch(n_walks)
    last = torch.full((n_walks,), -1, dtype=torch.long, device=tq.device)
    out = np.empty(max_len)
    for t in range(max_len):
        a = torch.randint(0, tq.n_actions, (n_walks,), device=tq.device, generator=gen)
        undo = (last >= 0) & (a == tq.inverse[last.clamp(min=0)])
        a = torch.where(undo, (a + 1) % tq.n_actions, a)
        states = tq.step(states, a)
        last = a
        out[t] = float((states == tq.goal).float().mean())
    return out


def mixing_length(curve: np.ndarray, tol: float = 0.02) -> int:
    """Walk length at which ``match_curve`` comes within ``tol`` (relative) of its well mixed value."""
    ref = float(curve[-max(1, len(curve) // 10):].mean())
    hit = np.flatnonzero(curve <= ref + tol * (1 - ref))
    return int(hit[0]) + 1 if len(hit) else len(curve)


def evaluate(value: RungValue, states: np.ndarray, width: int, weight: float, time_cap: float,
             give_up: int = 8, budget: float | None = None) -> dict:
    """Beam search from every state with the network, in order; every path found is checked on the rung's
    quotient. Stops early when none of the first ``give_up`` states is solved, or when ``budget`` seconds
    have passed and some state has already failed (the test can no longer be perfect; the share solved so
    far is the estimate). A network that solves every state is always tested on all of them."""
    q = value.q
    solved, costs, actions, secs, tried = 0, [], [], [], 0
    steps = max(200, 6 * value.cfg.walk_max)
    t_start = time.time()
    for vec in states:
        if tried >= give_up and solved == 0:
            break
        if budget is not None and solved < tried and tried >= give_up and time.time() - t_start > budget:
            break
        tried += 1
        t0 = time.time()
        try:
            found = beam_search(value.tq, value.value, torch.as_tensor(vec, dtype=torch.uint8, device=value.device),
                                width=width, max_steps=steps, weight=weight, deadline=t0 + time_cap)
        except SearchTimeout:
            found = None
        secs.append(time.time() - t0)
        if found is not None:
            acts, cost = found
            if not q.is_goal(q.apply_word(vec, acts)[None])[0]:
                raise AssertionError("beam search returned a path that does not reach the rung's goal")
            solved += 1
            costs.append(float(cost))
            actions.append(len(acts))
    with torch.no_grad():
        v = value.value(torch.as_tensor(states, dtype=torch.uint8, device=value.device)).cpu().numpy()
    return {"solved": solved / tried, "tested": tried, "n": len(states), "complete": tried == len(states),
            "mean_cost": float(np.mean(costs)) if costs else None,
            "mean_actions": float(np.mean(actions)) if actions else None,
            "seconds_per_state": float(np.mean(secs)), "mean_value": float(v.mean())}


# -- the autopilot --------------------------------------------------------------------------------------

def _stalled(tests: list[dict], k: int) -> bool:
    """No test among the last k solved a larger share than the best before them."""
    if len(tests) <= k:
        return False
    before = max(t["solved"] for t in tests[:-k])
    return max(t["solved"] for t in tests[-k:]) <= before


def _full(t: dict, target: float) -> bool:
    """A test on every test state that solved (at least) the target share of them."""
    return t["solved"] >= target and t.get("complete", True) and t["mean_cost"] is not None


def _plateaued(tests: list[dict], k: int, gain: float, target: float) -> bool:
    """Among tests that solved everything, the last k did not lower the best mean cost by more than ``gain``."""
    full = [t["mean_cost"] for t in tests if _full(t, target)]
    if len(full) <= k:
        return False
    return min(full[-k:]) > min(full[:-k]) * (1 - gain)


def _warm_start(old: RungValue, quotient, device: str) -> RungValue:
    """A value network for a new action set that starts from ``old``'s weights (its input, the rung's feature
    vector, does not depend on the actions). The output scale is kept, so the values keep their meaning."""
    new = RungValue(quotient, replace(old.cfg), device)
    new.net.load_state_dict(old.net.state_dict())
    new.target.load_state_dict(old.target.state_dict())
    for obj in (new, new.net, new.target):
        obj.scale = old.scale
    new.target_is_zero = old.target_is_zero
    new.itr, new.update, new.target_updates = old.itr, old.update, old.target_updates
    new.history = list(old.history)
    return new


def autotrain_rung(ladder: Ladder, index: int, cache_dir: str | Path, cfg: AutoTrainConfig | None = None,
                   set_spec: Callable[[int, RungSpec], Ladder] | None = None, extra_states: np.ndarray | None = None,
                   budget_minutes: float | None = None, device: str | None = None,
                   log: Callable[[str], None] | None = None) -> tuple[Ladder, dict]:
    """Train learned rung ``index`` as described in the module docstring.

    ``set_spec(index, spec)`` must store a changed rung specification (a repair of the action set) and return
    the ladder rebuilt with it; without it the ladder's own specs are changed and it is rebuilt in place.
    ``extra_states`` are added to the test states (states the full solver failed on), and train the rung
    again even if it had finished. Returns the (possibly rebuilt) ladder and the training record."""
    cfg = cfg or AutoTrainConfig()
    log = log or (lambda msg: print(msg, flush=True))
    device = device or default_device()
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    def rebuild(spec: RungSpec) -> Ladder:
        if set_spec is not None:
            return set_spec(index, spec)
        ladder.specs[index] = spec
        return ladder.build(cache_dir=cache)

    rung = ladder.rungs[index]
    if rung.spec.mode != "learned":
        raise ValueError(f"rung {rung.spec.name} is not a learned rung")
    stem = f"rung{index:02d}_{rung.spec.name}"
    record_file, net_file, states_file = (cache / f"{stem}.train.json", cache / f"{stem}.pt", cache / f"{stem}.states.npz")
    fp = rung.info["fingerprint"]["actions"]

    record = json.loads(record_file.read_text()) if record_file.exists() else None
    if record is not None and record.get("fingerprint") != fp:
        log(f"  {record_file.name} is for other actions: set aside as {retire(record_file, record.get('fingerprint')).name}")
        record = None
    if record is not None and record["status"] == "done" and extra_states is None and rung.heuristic is not None:
        log(f"rung {index} [{rung.spec.name}]: already trained ({record['minutes']:.0f} min, final test "
            f"{100 * record['final']['solved']:.0f}% solved)")
        return ladder, record

    # -- test states, as the solver meets them -----------------------------------------------------------
    key = digest("states", rung.info["fingerprint"]["pool"], [feature_key(f) for f in rung.spec.features],
                 cfg.test_states, cfg.final_states, cfg.seed, cfg.min_depth, cfg.max_depth)
    test = final = None
    if states_file.exists():
        with np.load(states_file) as z:
            if "fingerprint" in z.files and str(z["fingerprint"]) == key:
                test, final = z["test"], z["final"]
    if test is None:
        t0 = time.time()
        log(f"rung {index} [{rung.spec.name}]: sampling {cfg.test_states} + {cfg.final_states} test states "
            f"(scrambles of {cfg.min_depth}..{cfg.max_depth} moves, earlier rungs solved)")
        test = ladder.sample_rung_states(index, cfg.test_states, seed=cfg.seed, min_depth=cfg.min_depth, max_depth=cfg.max_depth)
        final = ladder.sample_rung_states(index, cfg.final_states, seed=cfg.seed + 1000, min_depth=cfg.min_depth,
                                          max_depth=cfg.max_depth)
        np.savez(states_file, test=test, final=final, fingerprint=np.array(key))
        log(f"  done in {time.time() - t0:.0f} s")
    if extra_states is not None and len(extra_states):
        test = np.concatenate([test, extra_states])
        log(f"  {len(extra_states)} states the solver failed on are added to the test states")

    # -- network and walk length ---------------------------------------------------------------------------
    q = rung.quotient
    value = rung.heuristic
    if record is None:
        tq = TorchQuotient(q, device)
        curve = match_curve(tq)
        mix = mixing_length(curve)
        walk = int(min(cfg.walk_cap, max(cfg.walk_min, math.ceil(cfg.walk_factor * mix))))
        test_match = float((test == q.goal).mean())
        mixed_match = float(curve[-len(curve) // 10:].mean())
        log(f"rung {index} [{rung.spec.name}]: {q.n_slots} slots x {q.n_classes} classes, {q.n_actions} actions; "
            f"walks mix in {mix} actions (slots at goal: {mixed_match:.3f} mixed, {test_match:.3f} on solver states) "
            f"-> walks of up to {walk} actions")
        record = {"fingerprint": fp, "rung": rung.spec.name, "status": "running", "phase": "A", "mixing_length": mix,
                  "walk_max": walk, "res_dim": cfg.res_dim, "actions": q.n_actions, "repairs": [], "tests": [],
                  "minutes": 0.0, "config": asdict(cfg)}
        if value is not None:
            log(f"  continuing from the existing network ({value.update} updates)")
    if value is None:
        value = RungValue(q, DaviConfig(res_dim=record["res_dim"], num_blocks=cfg.num_blocks, batch_size=cfg.batch_size,
                                        states_per_update=cfg.states_per_update, itrs_per_update=cfg.itrs_per_update,
                                        walk_max=record["walk_max"], lr=cfg.lr, loss_thresh=cfg.loss_thresh,
                                        force_sync=cfg.force_sync), device)
    value.cfg.walk_max = record["walk_max"]
    value.fingerprint = fp
    best_file = cache / f"{stem}.best.pt"
    if extra_states is not None:
        record["phase"] = "A"
        record.pop("best", None)                # a larger test set: earlier best scores are not comparable
    record["status"] = "running"

    def save() -> None:
        record_file.write_text(json.dumps(record, indent=1))

    budget = 60 * (budget_minutes if budget_minutes is not None else max(0.0, cfg.max_minutes - record["minutes"]))
    t_start, minutes_before = time.time(), record["minutes"]
    # progress is judged on the tests since the last repair (or since this call, for a repair round)
    last = record["repairs"][-1] if record["repairs"] else None
    repair_mark = len(record["tests"]) if extra_states is not None else (last["test_index"] if last else 0)
    last_repair_update = value.update if extra_states is not None or last is None else last["restart_update"]
    quiet = lambda _msg: None  # noqa: E731
    while True:
        if time.time() - t_start >= budget:
            record["status"] = "done" if record["phase"] == "B" else "stopped"
            log(f"  time cap reached in phase {record['phase']}")
            break
        round_limit = min(60 * cfg.round_minutes, max(1.0, budget - (time.time() - t_start)))
        value.train(max_updates=value.update + 1_000_000, time_limit=round_limit, checkpoint=net_file, log=quiet)
        ev = evaluate(value, test, cfg.eval_beam, cfg.eval_weight, cfg.eval_time, cfg.eval_give_up,
                      budget=cfg.eval_budget * round_limit)
        hist = value.history[-1] if value.history else {}
        record["minutes"] = minutes_before + (time.time() - t_start) / 60
        row = {"update": value.update, "minutes": round(record["minutes"], 2), "walk_max": value.cfg.walk_max,
               "loss": hist.get("loss"), "target_mean": hist.get("target_mean"), **ev}
        record["tests"].append(row)
        cost = "-" if ev["mean_cost"] is None else f"{ev['mean_cost']:.1f}"
        log(f"  update {value.update:4d} ({record['minutes']:5.1f} min): solved {100 * ev['solved']:5.1f}% of "
            f"{ev['tested']} test states, mean cost {cost}, {ev['seconds_per_state']:.1f} s each; "
            f"loss {row['loss'] if row['loss'] is None else round(row['loss'], 4)}, walks {value.cfg.walk_max}")
        if _full(ev, cfg.target) and ev["mean_cost"] < record.get("best", {}).get("mean_cost", math.inf):
            value.save(best_file)               # the network to keep if later ones are not better
            record["best"] = {"update": value.update, "mean_cost": ev["mean_cost"], "test_index": len(record["tests"]) - 1}

        if record["phase"] == "A":
            if _full(ev, cfg.target):
                record["phase"] = "B"
                log("  phase B: every test state solved; training on while solutions get shorter")
            elif (_stalled(record["tests"][repair_mark:], cfg.stall_tests)
                  and value.update - last_repair_update >= cfg.min_updates_per_repair):
                done = Counter(r["kind"] for r in record["repairs"])
                if done["walk"] < cfg.max_walk_repairs:
                    walk = int(min(cfg.walk_cap, math.ceil(value.cfg.walk_max * cfg.walk_growth)))
                    record["repairs"].append({"kind": "walk", "update": value.update, "from": value.cfg.walk_max, "to": walk})
                    log(f"  repair: no progress in {cfg.stall_tests} tests; walks {value.cfg.walk_max} -> {walk}")
                    value.cfg.walk_max = record["walk_max"] = walk
                elif done["actions"] < cfg.max_action_repairs:
                    spec = rung.spec
                    new_spec = replace(spec, max_actions=spec.max_actions * cfg.action_growth,
                                       max_bridges=spec.max_bridges * cfg.action_growth)
                    log(f"  repair: no progress; actions: up to {new_spec.max_actions} cheapest and "
                        f"{new_spec.max_bridges} subgroup-leaving moves (network kept)")
                    value.save(net_file)
                    old = value
                    ladder = rebuild(new_spec)
                    rung = ladder.rungs[index]
                    q = rung.quotient
                    value = _warm_start(old, q, device)
                    del old
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    fp = value.fingerprint = rung.info["fingerprint"]["actions"]
                    record["repairs"].append({"kind": "actions", "update": value.update, "actions": q.n_actions,
                                              "max_actions": new_spec.max_actions, "max_bridges": new_spec.max_bridges})
                    record["fingerprint"], record["actions"] = fp, q.n_actions
                    record.pop("best", None)            # saved for the old action set
                elif done["width"] < cfg.max_width_repairs:
                    log(f"  repair: no progress; a new, wider network ({cfg.wider_res_dim} units)")
                    record["repairs"].append({"kind": "width", "update": value.update, "res_dim": cfg.wider_res_dim})
                    record["res_dim"] = cfg.wider_res_dim
                    value = RungValue(q, replace(value.cfg, res_dim=cfg.wider_res_dim), device)
                    value.fingerprint = fp
                    record.pop("best", None)            # saved for the narrower network
                else:
                    record["status"] = "failed"
                    log("  every repair has been tried and the rung still does not solve its test states")
                    save()
                    break
                last_repair_update = value.update
                repair_mark = len(record["tests"])
                record["repairs"][-1].update(test_index=repair_mark, restart_update=last_repair_update)
        elif _plateaued(record["tests"], cfg.plateau_tests, cfg.plateau_gain, cfg.target):
            record["status"] = "done"
            log(f"  done: mean cost has not dropped by {100 * cfg.plateau_gain:g}% in {cfg.plateau_tests} tests")
            break
        save()

    best = record.get("best")
    if best and best_file.exists() and value.update != best["update"]:
        # the network in hand is not the best one: a later test scored worse or failed some states
        last = record["tests"][-1]
        how = (f"scored {last['mean_cost']:.1f}" if _full(last, cfg.target)
               else f"solved {100 * last['solved']:.1f}% of {last['tested']} test states")
        log(f"  keeping the network of update {best['update']} (mean cost {best['mean_cost']:.1f}); "
            f"the last test (update {last['update']}) {how}")
        value.load(best_file)
        value.fingerprint = fp
    value.save(net_file)
    rung.heuristic = value
    log(f"  final test on {len(final)} other states:")
    record["final"] = evaluate(value, final, cfg.eval_beam, cfg.eval_weight, cfg.eval_time, give_up=len(final))
    fin = record["final"]
    log(f"  solved {100 * fin['solved']:.1f}%, mean cost "
        f"{'-' if fin['mean_cost'] is None else round(fin['mean_cost'], 1)}, {fin['seconds_per_state']:.1f} s each")
    record["updates"] = value.update
    save()
    return ladder, record
