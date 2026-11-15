"""Inspect what happens on a 1000-move scramble."""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from core.ladder import Ladder
from core.perm_model import build_point_model
from core.structure import analyse
from puzzles.maltese_gear.ladder import LADDERS
from puzzles.maltese_gear.puzzle import build_maltese_gear_puzzle

puzzle = build_maltese_gear_puzzle()
model = build_point_model(puzzle)
structure = analyse(model)
cache = Path("data/ladders/learned_r7")
ladder = Ladder(model, structure, LADDERS["learned"], mitm_radius=7, log=print)
ladder.build(cache_dir=cache)

# 1000 move random scramble
rng = np.random.default_rng(42)
scramble = rng.integers(0, model.num_moves, size=1000).tolist()
state = puzzle.apply_word(puzzle.goal, scramble)

print("\n--- 1000-move scramble generated ---")
print(f"Scramble length: {len(scramble)} moves")
scramble_str = ladder.alg.format(scramble)
print(f"Scramble (first 100 chars): {scramble_str[:100]}...")

print("\nSolving rung by rung...")
lab = structure.decode(state)
pieces = []
per_rung = []

for i, rung in enumerate(ladder.rungs):
    q = rung.quotient
    vec = q.project(lab)
    t0 = time.time()
    print(f"\nRung {i} [{rung.spec.name}] (mode: {rung.spec.mode})...", flush=True)
    if rung.table is not None:
        from core.coset_table import table_solve
        acts = table_solve(q, rung.table, vec)
        dt = time.time() - t0
        print(f"  table solved in {dt:.2f}s, {len(acts)} actions", flush=True)
    elif rung.heuristic is not None:
        # Let's inspect heuristic.solve
        start = torch.as_tensor(vec, dtype=torch.uint8, device=rung.heuristic.device)
        print(f"  goal check: is_goal={bool(rung.heuristic.tq.is_goal(start.unsqueeze(0))[0])}", flush=True)
        v0 = float(rung.heuristic.value(start.unsqueeze(0))[0])
        print(f"  initial heuristic estimate V(start): {v0:.2f}", flush=True)
        acts = rung.heuristic.solve(vec, width=4096)
        dt = time.time() - t0
        if acts is not None:
            print(f"  beam solved in {dt:.2f}s, {len(acts)} actions", flush=True)
        else:
            print(f"  beam search returned None after {dt:.2f}s!", flush=True)
            break
    cost = 0
    for a in acts:
        lab = lab[rung.actions.tables[a]]
        pieces.append(rung.actions.words[a])
        cost += int(rung.actions.costs[a])
    per_rung.append(cost)

if len(per_rung) == len(ladder.rungs):
    full_sol = ladder.alg.concat(*pieces)
    print("\n=== SOLVED SUCCESSFULLY ===")
    print(f"Solution: {ladder.alg.format(full_sol)}")
    print(f"Total length: {len(full_sol)} quarter turns, {ladder.alg.face_turns(full_sol)} face turns")
    print(f"Per rung: {per_rung}")
    valid = np.array_equal(puzzle.apply_word(state, list(full_sol)), puzzle.goal)
    print(f"Verified on reference simulator: {valid}")
