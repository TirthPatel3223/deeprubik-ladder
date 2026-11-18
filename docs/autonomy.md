# The one-command pipeline

The goal: give the pipeline a puzzle's states and moves, run one command, and get a solver that has been
designed, built, trained and tested with no person (or LLM) in the loop.

```bash
python scripts/auto.py --tables my_puzzle.npz      # the puzzle as gather tables (core/puzzle_io.py)
python scripts/auto.py --geometry my_puzzle.json   # or as a 3D model (core/geometry.py)
python scripts/auto.py --puzzle tangram_extreme    # or a registered puzzle
```

**Legend.** `[measured]` = computed in this project; the command or test that reproduces it is named.

---

## 1. What a person used to decide, and what decides it now

| Decision | Before | Now |
|---|---|---|
| Getting the puzzle in | a Python module, a registry entry and a notation | a file: `--tables` or `--geometry` (`core/puzzle_io.py`); missing inverse moves are added |
| The ladder (rungs, order, table or network) | `RungSpec` lists written by hand | `core/discover.py`, rules 0 to 4 and an exact completeness check (section 4) |
| Walk length of each network | `--walk`, chosen per rung | 3.5 times the measured mixing length of the rung's walks (section 3) |
| How long to train | `--minutes`, chosen per rung | until every test state is solved, then while solutions still get shorter (section 3) |
| What to do when a network fails | diagnosed by hand (the gear pairs) | the autopilot's repairs: longer walks, more actions, a wider network (section 3) |
| Whether the solver is good enough | a benchmark read by a person | the acceptance test with a fixed rule (section 2) |
| Stale cache files | moved aside by hand (`*.one_bridge.*.bak`) | fingerprints: a file is reused only if it was built for the same inputs |

Every stage resumes: the same command continues an interrupted run. Each run writes `ladder.json`, `rungs/`,
`acceptance.json`, `report.md`, `run.json` and `auto.log` under `data/auto/<puzzle>/<ladder>_r<radius>/`.

---

## 2. The acceptance test

`core/acceptance.py`, `scripts/accept.py`. The protocol is the deep-scramble test used to find the Maltese
ladder's `gear_col` problem, made into a rule:

1. 100 states, each scrambled with a number of random moves uniform in 2000..5000 (seed 42; the same states
   as `scripts/benchmark_deep_scrambles.py`), after one untimed warm-up solve.
2. Pass 1: 60 s per state. Every solution is replayed on the reference simulator. Once more than 10 states
   have failed the verdict is decided and the test stops.
3. All 100 solved: **PASS**. At least 90: each unsolved state gets 300 s with a stronger search (a greedier
   search, then beams up to 131,072 wide; the same search given more time would fail the same way). All
   solved then: **PASS_SLOW**; otherwise **FAIL**. Fewer than 90: **FAIL**.
4. A state outside an exact table is a gap that no search can close, and a solution that does not replay is
   a bug; either is a FAIL. Failures are reported by rung and reason (`outside_table`, `search_exhausted`,
   `timeout`), which is what the repair loop acts on.

What it can show: if a fraction p of random states fall in a part of the state space the solver cannot
solve, 100 states all miss it with probability (1 - p)^100, under 5% once p >= 3%. Rarer gaps need an exact
check on the group.

On the two hand-made learned ladders, and on the Maltese ladder with the `gear_col` network that first
failed (the archived one-bridge network, before every pair-splitting move was kept) `[measured:
scripts/accept.py; reports in runs/acceptance/]`:

| Ladder | Verdict | Solved | Length, mean (min-max) | Time per state, mean / max |
|---|---|---|---|---|
| Maltese, `learned` | PASS | 100 / 100 within 60 s | 406.1 quarter turns (332-510) | 12.2 s / 23.0 s |
| Tangram Extreme, `learned` | PASS | 100 / 100 within 60 s | 90.0 steps (76-107) | 4.7 s / 5.5 s |
| Maltese, one-bridge `gear_col` | FAIL | 19 of the first 30; stopped at 11 failures | 747.1 quarter turns (374-1275) | 26.3 s / 58.2 s |

The Maltese states are the benchmark's states, and the lengths are the benchmark's (406.1). The failing ladder
is caught in 30 states: all 11 failures are timeouts in `gear_col`, and where it does solve, `gear_col` alone
costs 451 quarter turns against 88 for the repaired network.

---

## 3. The training autopilot

`core/autotrain.py`. For each learned rung, in order:

- **Test states** are drawn as the solver hands them to the rung: deep scrambles of the whole puzzle with
  the earlier rungs solved (`Ladder.sample_rung_states`). 32 are used during training and 64 others for the
  final check.
- **Walk length.** Random walks with the rung's actions are run from the goal, and the fraction of slots at
  their goal value is followed until it is within 2% of its well-mixed value. The training walks go up to
  3.5 times that length. On the four rungs trained by hand `[measured]`:

  | Rung | Actions | Mixed in | 3.5 x | Chosen by hand |
  |---|---|---|---|---|
  | Maltese `gear_col` | 376 | 8 | 28 | 30 |
  | Maltese `ring` | 72 | 11 | 38 | 30 |
  | Tangram `large` | 163 | 11 | 38 | 40 |
  | Tangram `small` | 192 | 6 | 21 | 20 |

- **Phase A.** Train in rounds of 5 minutes; after each, beam search (width 4096, the solver's first search)
  on the test states. Continue until all are solved.
- **Repairs.** If the share solved has not risen in 3 tests (and at least 30 updates have passed), change,
  in order: the walk length (x1.5, twice); the action set (twice the cheapest moves and twice the moves that
  leave the cheap moves' subgroup, the remedy that fixed `gear_col`; the network keeps its weights, since its
  input does not depend on the actions); the network width (a new network of 2000 units). Splitting a rung
  into exact tables is a change of ladder, left to the ladder's design (and the last resort, by preference).
- **Phase B.** Keep training while the mean solution cost on the test states falls by more than 1% over 3
  tests, up to a cap of 120 minutes per rung.
- A record (`rungNN_<name>.train.json`) beside the checkpoint holds the settings, every test and the final
  check, so a stopped run resumes and a finished rung is not trained again.

If the acceptance test then fails in a learned rung, that rung is trained again with the failing states
added to its test states (two rounds at most).

**Results** `[measured: scripts/auto.py; auto.log and rungNN_<name>.train.json of each run under data/auto/]`.
Every network below was trained from scratch, with no setting given by hand. Run A is the Tangram Extreme
with its designed ladder, Run B the Maltese cube with its designed ladder, Run C the Maltese cube with the
hand-written `learned` ladder. The final test is 64 states not used in training, with the solver's first
search (beam 4096, 15 s each); costs are in steps (Tangram) and quarter turns (Maltese).

| Rung | Run | Actions | Walks: auto (by hand) | Every test state solved | Stopped | Repairs | Final test |
|---|---|---|---|---|---|---|---|
| Tangram `large` | A | 163 | 35 (40) | 8.5 min | 52.6 min, plateau | none | 100%, 47.3 |
| Maltese `gear_col` | C | 376 | 28 (30) | 34 min | 98 min, plateau | none | 96.9%, 87.6 |
| Maltese `ring` | C | 72 | 39 (30) | 8 min | 60 min, plateau | none | 100%, 249.4 |
| Maltese gear rotation | B | 157 | 42 (none) | 6.9 min | 40.5 min, plateau | none | 100%, 67.8 |
| Maltese gear colour | B | 372 | 28 (30) | 40 min | 70 min, plateau | none | 100%, 88.2 |
| Maltese ring | B | 138 | 35, then 53 (30) | 57 min | 121 min, time cap | walks x1.5 | 85.9%, 276.8 |

Where the rung is the same, the autopilot's networks are as good as the ones trained by hand, which got 45
minutes (`gear_col`, `large`) and 30 (`ring`) fixed in advance; the autopilot trained until the plateau
instead. Per solve on the 100 acceptance states: `large` 47.3 steps against 47.1, `gear_col` 88.3 quarter
turns against 88.4, `ring` 247.3 against 247.4. With both networks from the autopilot, the hand-written
Maltese ladder passes (100 of 100 within 60 s) at 405.6 quarter turns, against 406.1 with the hand-trained
networks; 206 minutes from the move tables to the verdict.

Run B's ring rung showed a fault, now fixed. Its best network (update 142: every test state solved, mean
cost 267.3) was not the one kept. The networks after it failed some test states until the time cap, and
the rule restored the best network only when a later one solved every state at a higher cost; none did.
The autopilot now keeps the network of the best complete test unless that is the last one tested. With it
restored, the final test solves 96.9% at a mean cost of 267.0 (85.9% and 276.8 before), and on the
acceptance states the ring costs 283.0 quarter turns per solve against 316.6 (shorter on 63 of the 100
states), 431.7 in all against 465.2, still with every state solved within 60 s
`[measured: runs/v3_maltese_best_ring.log]`. The training is unchanged by the fix, so these are the
numbers the same run gives with it.

---

## 4. Automatic ladder design

`core/discover.py`, `propose_ladder`. The rules of the design note, measured on the move tables, plus one
that both hand-made ladders follow:

- **Rule 0, small kinds of piece first, together.** Kinds of piece whose coset space under the primitive
  moves fits one table are joined into a first rung, from the smallest while the joint table still fits.
- **Rule 1** (ascending generator period) and **rule 2** (cheapest pure macros last) order the other kinds.
- **Rule 4** chooses each rung by bits resolved per mean length of the macros it leaves behind. The
  candidates are each feature of the next kind, the whole kind, and the kind merged with the next one.
  Features forced by the others are dropped (a corner's second and third sticker).
- A rung too large for a table is scored with `log2|H| - log2 A`, where H is the group its actions generate
  on its pieces and A the number of arrangements that show the goal. The old proposer used the size at
  which its table build stopped, which put every large rung at about 23 bits (the Maltese colour, rotation
  and ring rungs at 24.4, 24.2 and 23.3 instead of 51.5, 46 and 75.5).
- Each next rung's candidate macros are computed as the builder computes them, Schreier generators
  included. With only short words, the original Tangram Cube's bottom corner looked solved.
- **Completeness.** Orbits that the chosen features do not force (sympy, exact when decidable) join the first
  rung and the other rungs are chosen again.
- **Predicted cost** of a ladder: the sum over rungs of mean action cost x bits / log2(actions), a counting
  bound. On the cached Tangram ladders it gives 84.1 for the learned ladder and 125.6 for the table ladder,
  against the measured 90.8 and 137.2 `[measured]`.

**Results** `[measured: scripts/auto.py --puzzle tangram_extreme, --puzzle maltese; data/auto/<puzzle>/auto_r7/report.md]`.
Both ladders were designed, built, trained and accepted from the move tables alone.

| | Tangram Cube Extreme | Maltese Gear Cube |
|---|---|---|
| Designed ladder | orbits 7, 9 to 13 (table, 3,456 cosets); orbits 1, 2, 3, 8 (network, 163 actions); orbit 0 (table, 665,280 cosets); orbit 6 (table, 360) | corners (table, 3,674,160 cosets); gear rotation (network, 157 actions); gear colour (network, 372); ring (network, 138) |
| Hand ladder `learned` | `frame` and `large`: the same two rungs; `small`: orbits 0 and 6 as one network | corners; `gear_parity` and `gear_rot` (tables); `gear_col` (376 actions); `ring` (72) |
| Predicted cost | 100 | 284 |
| Acceptance, designed | PASS, 97.5 steps (77 to 112), 4.0 s per state | PASS, 465.2 quarter turns (328 to 810), 21.9 s per state; 431.7 (302 to 857) with the best ring network kept (section 3) |
| Acceptance, hand ladder | PASS, 90.0 steps (76 to 107), 4.7 s | PASS, 406.1 (332 to 510), 12.2 s |
| Move tables to verdict | 68 min | 311 min, in two runs of the command (stopped once to bound the length of tests) |

The Tangram's all-table ladder takes 137.2 steps (`docs/tangram.md`). Quarter turns per solve on the
Maltese acceptance states, rung by rung (every row sees the same 100 states):

| Maltese | corners | gear rotation | gear colour | ring | in all, after merging |
|---|---|---|---|---|---|
| hand ladder, networks trained by hand | 10.5 | 76.0 (two tables) | 88.4 | 247.4 | 406.1 |
| hand ladder, autopilot's networks (Run C) | 10.5 | 75.5 (two tables) | 88.3 | 247.3 | 405.6 |
| designed ladder (Run B) | 10.5 | 65.3 (one network) | 89.7 | 316.6 | 465.2 |
| designed ladder, best ring network kept | 10.5 | 65.3 (one network) | 89.7 | 283.0 | 431.7 |

(Run C rebuilt the two rotation tables: the same moves and distances, but where several moves are equally
short another one is stored, so a third of the states reach the rotation table in another arrangement.)

- **One network for the gear rotation is shorter than two tables**: 65.3 quarter turns against 76.0 (14%).
  `hierarchical_solver.md` §7 listed it as untried; this is its first measurement.
- **The Maltese difference is the ring's network, not its moves or its states.** The designed rung takes
  the 128 cheapest moves and their inverses: the hand ladder's 72 (all cost 12) and 66 of cost 14. Each
  ring network was run with each move set on the designed rung's 64 final-test states (beam 4096, 60 s per
  state, every state solved), quarter turns per solve `[measured: runs/ring_swap_72.log, ring_swap_138.log]`:

  | Ring network | 72 moves | 138 moves |
  |---|---|---|
  | trained with 72 moves (Run C) | 248.4 | 242.4 |
  | trained with 138 moves (Run B, best) | 268.5 | 274.2 |

  The network trained with 138 moves costs 20 more per solve with the same 72 moves and 32 more with the
  same 138 (paired over the 64 states, about 3.5 standard errors each); the move set changes either network
  by about 6, within the noise. Nor are the states harder: Run C's ring solves them at 248.4, as its own
  (249.4). Its training went differently. With 138 moves a test state took 9 to 13 s against the 15 s cap
  (4 s with 72), so tests ended early on time-outs (at the end, 2 of 64 states took longer than 15 s; with
  60 s none failed). The autopilot read that as no progress and lengthened the walks from 35 to 53, and with
  only two complete tests in phase B it ran to the time cap. Whether the longer walks or the training with
  138 moves made the network weaker is not yet separated.
- **The Tangram difference is the last step.** Orbits 0 and 6 become two tables (27.7 + 20.3 = 48.0 steps)
  where the hand ladder has one network (`small`, 40.4). Rule 4 divides by the length of the macros a rung
  leaves behind, so a rung that finishes the puzzle scores 0 and a merged last rung never wins. The
  predicted cost is reported but takes no part in the choice.

**From a file.** The original Tangram Cube, given only as an `.npz` of its move tables (written by
`core/puzzle_io.save_tables`), went from file to verdict in 12 seconds: a first table rung with the
primitive moves (the `frame` orbits and orbit 4; 311,040 cosets), then the cube corners (360 cosets, 359
macros), predicted 20 moves. PASS, 19.7 moves per solve (15 to 24); the hand-written two-table ladder
passes at 18.0 (10 to 23) `[measured: runs/v3_auto_tangram_original_npz.log,
runs/v3_accept_tangram_original_table.log]`.

---

## 5. Limits

- **Whether a network rung trains well is known only after training.** The design scores rungs before any
  network exists, and a poor choice costs GPU time before the autopilot or the acceptance test catches it.
- **The design is greedy.** It looks one rung ahead and proposes one ladder. If the acceptance test fails in
  an exact table (a state outside it), the run reports the gap; it does not redesign the ladder.
- **The last rung is not chosen by measurement.** The score cannot value a rung that finishes the puzzle,
  so it is never merged with the one before; the designed Tangram ladder loses its extra length there
  (section 4).
- **The autopilot's tests allow a fixed 15 s per state.** A rung with twice the moves searches about half
  as fast, so its tests time out on states its network can solve, and the autopilot reads that as no
  progress. On the designed Maltese ring this lengthened the walks, ran to the time cap and left a weaker
  network, which is where that ladder's extra length comes from (section 4).
- **Coverage is proved exactly only when pieces can be told apart.** With identical-looking pieces of several
  colours (the Maltese gears) the completeness check cannot decide, and the acceptance test stands in: it
  finds gaps that affect at least about 3% of states.
- **The walk-length rule is calibrated on four rungs** (3.5 x the mixing length). A wrong guess is repaired by
  the autopilot, at the cost of training time.
- **The puzzles it can take.** Every move must be a permutation of positions (always legal); bandaged and
  jumbling puzzles are outside the method. From a 3D model, `core/geometry.py` compiles moves that rotate one
  half-space; geared moves (several parts turning by different angles) and curved cuts need a compiler
  extension, once per kind of mechanism. The solved state must be unique (a fixed frame).
- **Scale.** Macros come from meet in the middle to radius 7 (at most 2 million elements) and from Schreier
  generators of table rungs; exact tables need a 63-bit key and at most 60 million cosets; beams go up to
  131,072 wide on a 12 GB GPU.
