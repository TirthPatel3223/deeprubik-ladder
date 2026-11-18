"""Train the value network of one learned rung by coset value iteration, then test it.

    python scripts/train_rung.py --ladder learned --radius 7 --rung gear_col --updates 200
    python scripts/train_rung.py --puzzle tangram_extreme --rung large --minutes 60

The ladder is built (from cache) up to the rung, the rung's network is trained, and the
checkpoint is written next to the rung's cached actions, where the ladder picks it up.
Training resumes from an existing checkpoint unless --fresh is given.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.davi import DaviConfig, RungValue  # noqa: E402
from core.ladder import Ladder  # noqa: E402
from core.perm_model import build_point_model  # noqa: E402
from core.search import beam_search  # noqa: E402
from core.structure import analyse  # noqa: E402
from puzzles.registry import PUZZLES, get_puzzle  # noqa: E402


def solver_states(ladder: Ladder, puzzle, model, structure, index: int, n: int,
                  scramble_len: int = 200, seed: int = 0) -> np.ndarray | None:
    """Feature vectors of rung ``index`` as the solver meets them: random scrambles of the
    whole puzzle with the earlier rungs solved. None if an earlier rung cannot be solved yet.

    Random walks with the rung's own actions are not a substitute. If a few of the actions
    are the only link between large parts of the rung (see the warning in
    ``Ladder._select_actions``), walks stay in one part, and a network that solves every
    walk can still fail on almost every scramble."""
    q = ladder.rungs[index].quotient
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        state = puzzle.apply_word(puzzle.goal, rng.integers(0, model.num_moves, size=scramble_len).tolist())
        try:
            lab, _, _ = ladder.descend(structure.decode(state), upto=index)
        except (LookupError, RuntimeError):
            return None
        out.append(q.project(lab))
    return np.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puzzle", choices=sorted(PUZZLES), default="maltese")
    ap.add_argument("--ladder", default="learned")
    ap.add_argument("--radius", type=int, default=7)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--rung", required=True, help="name of the learned rung to train")
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--minutes", type=float, default=None, help="stop after this much training time")
    ap.add_argument("--res-dim", type=int, default=1000)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--states", type=int, default=300_000, help="states generated per update")
    ap.add_argument("--itrs", type=int, default=100, help="training iterations per update")
    ap.add_argument("--walk", type=int, default=30, help="longest random walk, in actions")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss-thresh", type=float, default=0.06)
    ap.add_argument("--force-sync", type=int, default=4)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--test", type=int, default=50, help="states to solve with beam search afterwards")
    ap.add_argument("--beam", type=int, default=4096)
    ap.add_argument("--scramble-len", type=int, default=None, help="scramble length of the test states (default: the puzzle's)")
    ap.add_argument("--weight", type=float, default=0.3, help="weight on the cost so far in the test search (the solver's default)")
    args = ap.parse_args()

    entry = get_puzzle(args.puzzle)
    ladders = entry.ladders()
    if args.ladder not in ladders:
        ap.error(f"{entry.title} has the ladders {sorted(ladders)}")
    puzzle = entry.build()
    model = build_point_model(puzzle)
    structure = analyse(model)
    specs = ladders[args.ladder]
    names = [s.name for s in specs]
    if args.rung not in names:
        ap.error(f"the {args.ladder} ladder of the {entry.title} has the rungs {names}")
    index = names.index(args.rung)
    cache = Path(args.cache) if args.cache else entry.cache_dir(args.ladder, args.radius)

    ladder = Ladder(model, structure, specs, mitm_radius=args.radius)
    ladder.build(cache_dir=cache, upto=index + 1)
    rung = ladder.rungs[index]
    q = rung.quotient
    print(f"rung {index} [{args.rung}]: {q.n_slots} slots x {q.n_classes} classes, {q.n_actions} actions, "
          f"costs {int(q.costs.min())}..{int(q.costs.max())}", flush=True)

    ckpt = cache / f"rung{index:02d}_{args.rung}.pt"
    cfg = DaviConfig(res_dim=args.res_dim, num_blocks=args.blocks, batch_size=args.batch,
                     states_per_update=args.states, itrs_per_update=args.itrs, walk_max=args.walk,
                     lr=args.lr, loss_thresh=args.loss_thresh, force_sync=args.force_sync, max_updates=args.updates)
    if rung.heuristic is not None and not args.fresh:
        # The ladder loads a checkpoint only if it was trained for this rung's current actions.
        value = rung.heuristic
        value.cfg.max_updates = args.updates
        print(f"resuming from {ckpt} at update {value.update}", flush=True)
    else:
        value = RungValue(q, cfg)
    value.fingerprint = rung.info["fingerprint"]["actions"]
    n_par = sum(p.numel() for p in value.net.parameters())
    print(f"network: {n_par / 1e6:.2f} M parameters on {value.device}", flush=True)
    t0 = time.time()
    value.train(max_updates=args.updates, checkpoint=ckpt,
                time_limit=None if args.minutes is None else 60 * args.minutes)
    print(f"training took {time.time() - t0:.0f} s; checkpoint {ckpt}", flush=True)

    if args.test:
        # Test on the states the solver will hand this rung, not on the training distribution.
        real = solver_states(ladder, puzzle, model, structure, index, args.test,
                             scramble_len=args.scramble_len or entry.scramble_len)
        if real is not None:
            states = torch.as_tensor(real, dtype=torch.uint8, device=value.device)
            print(f"testing on {args.test} scrambles of the whole puzzle with the earlier rungs solved", flush=True)
        else:
            states, _ = value.scramble(args.test, walk_max=3 * args.walk)
            print(f"an earlier rung cannot be solved yet: testing on {args.test} random walks instead "
                  f"(this is the training distribution, so it can flatter the network)", flush=True)
        solved, costs, secs = 0, [], []
        for k in range(args.test):
            t1 = time.time()
            found = beam_search(value.tq, value.value, states[k], width=args.beam, max_steps=6 * args.walk, weight=args.weight)
            secs.append(time.time() - t1)
            if found is not None:
                acts, cost = found
                assert q.is_goal(q.apply_word(states[k].cpu().numpy(), acts)[None])[0]
                solved += 1
                costs.append(cost)
        mean = float(np.mean(costs)) if costs else float("nan")
        print(f"beam {args.beam}: solved {solved}/{args.test}, mean cost {mean:.1f} moves, "
              f"{np.mean(secs):.2f} s per state", flush=True)
    _ = torch


if __name__ == "__main__":
    main()
