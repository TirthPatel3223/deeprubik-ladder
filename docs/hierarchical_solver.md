# The hierarchical solver: what was built and what was measured

This note records how the outline in
`cayley_graphs_twisty_puzzles_and_hierarchical_rl_solvers.md` (the "design note") was turned
into a working solver for the Maltese Gear Cube, which of its claims were checked against the
move tables, and where the build departs from it.

**Legend.** `[measured]` = computed in this project from the verified move tables; the test
or script that reproduces it is named. `[src]` = from a source listed in §9. `[not done]` =
proposed in the design note or here, not implemented.

RESULTS_PLACEHOLDER

---

## 1. The idea in one paragraph

Solving a puzzle is finding a short path in its Cayley graph. One value network for the whole
Maltese Gear Cube fails: the earlier attempt (now in `archive/`) flattened out at a predicted
distance of about 18 and solved nothing beyond 24 quarter turns. The design note's remedy is
a ladder of subgroups G0 ⊃ G1 ⊃ … ⊃ {e}. Rung i only has to tell apart the cosets of G(i+1)
in G(i), using moves that stay inside G(i). Each rung is a much smaller graph (a Schreier
coset graph) with its own cost-to-go function. This project builds that, with one practical
rule the note does not state: **a rung gets an exact table when its coset space can be
enumerated, and a value network only when it cannot.** A table is the exact value function,
so nothing is lost, and the networks are spent where tables are impossible.

---

## 2. From the design note to code

| Design note | Module | What it does |
|---|---|---|
| Part I, Cayley and Schreier graphs | `core/perm_model.py`, `core/quotient.py` | Moves as permutations of labelled points; a rung as a quotient puzzle on its feature vector |
| Part II, DeepCubeA value iteration and batch search | `core/davi.py`, `core/nnet.py`, `core/search.py` | Deep approximate value iteration per rung; residual network; batched beam search |
| Part III, anatomy of the puzzle | `core/structure.py` | Orbits, pieces, orientation sub-blocks, found from the tables instead of written by hand |
| Part IV, projections π_i and macro walks (method A) | `core/quotient.py`, `core/harvest.py`, `core/ladder.py` | Feature projections; macros that keep earlier rungs solved; walks from the goal coset |
| Part V phase 1, candidate projections | `core/structure.py` | Candidate features: full, multiset, shape, type, sub-block, colour masks |
| Part V phase 2, ordering rules | `core/discover.py` | Generator periods, shortest pure macros, bits per macro length, all measured |

### 2.1 Labelled model

A visible state hides which of several identical pieces is where. `build_point_model` turns
every coordinate that is only moved into one point, and every counter (a gear rotation mod 4)
into four points that a move shifts cyclically. The Maltese cube becomes permutations of 216
points: 96 stickers, 24 gear colours, 24 × 4 gear rotation points. `[measured:
tests/test_perm_model.py compares 200 random words with the reference simulator]`

### 2.2 Structure discovery

Computed from the six generator permutations and the solved colouring only:

- **Orbits** of the points under the moves.
- **Pieces**: the piece of a point p is the set of points fixed by everything that fixes p.
  This picks out the three stickers of a corner or the four residues of a gear without being
  told. It is found by comparing orbit sizes of pairs of points.
- **Frames**: every way a piece can sit in a slot. What a slot holds is determined by the
  label at its first position.
- **Sub-blocks**: block systems finer than the pieces (Atkinson's algorithm), e.g. the two
  diagonals of a four-sticker piece. They give half-turn parity features.
- **Decoding**: a visible state is turned into labels slot by slot; identical-looking pieces
  get the first matching label. Features are functions of the visible pattern in a slot, so
  they do not depend on that choice. `[measured: tests/test_structure.py]`

Result for the Maltese cube `[measured: scripts/analyse_puzzle.py]`:

| Orbit | Points | Pieces × positions | Meaning |
|---|---|---|---|
| 0 | 21 | 7 × 3 | corners (the eighth is pegged: 3 fixed points) |
| 1 | 72 | 18 × 4 | ring pieces: 12 two-colour edges and 6 one-colour centres, any piece in any slot |
| 2 | 24 | 24 × 1 | gear colour |
| 3 | 96 | 24 × 4 | gear rotation |

Each quarter turn is a 4-cycle of corners, an 8-cycle plus a 4-cycle of ring pieces, and an
8-cycle plus two 4-cycles of gears, with period 4 on corners and 8 elsewhere. This confirms
Part III of the design note.

### 2.3 Features (the projections π_i)

A feature classifies what each slot holds, as a function of the colours it shows:

| Kind | Classifies a slot by | Example |
|---|---|---|
| `full` | everything visible | which corner and how it is twisted |
| `multiset` | the colours, ignoring their arrangement | which ring piece, ignoring its spin |
| `shape` | which positions show equal colours | whether an edge is flush or twisted a quarter turn |
| `type` | how many colours and how often | centre piece or edge piece |
| `sub<i>` | the colours inside each sub-block | gear rotation parity |
| any of these with a colour mask | the same, with other colours hidden | where the four gears of one colour are |

Every such feature is closed under every move: the feature vector after a move depends only
on the feature vector before it. So a rung can run on its feature vector alone
(`Feature.transition`, checked in `tests/test_structure.py`), which is what makes exact
tables and small networks possible. The reachable feature vectors are the cosets.

### 2.4 Macros: the moves of a rung

The moves of rung i must keep rungs 0 … i−1 solved, i.e. lie in the stabiliser H of their goal
features. Two exact sources (`core/harvest.py`):

- **Meet in the middle.** Enumerate the ball of radius R of primitive words. Two elements
  showing the same features differ by an element of H. This finds every element of H of
  length ≤ 2R. Radius 7 is 115,462 elements and takes a second.
- **Schreier generators.** With an exact table for the previous rung, t_c · a · t_(c·a)⁻¹
  lies in H for every coset c and action a, and these generate H (Schreier's lemma). The
  cheapest are kept as macros; 40 random ones are kept as witnesses.

Selection (`Ladder._select_actions`): two words that act identically on the rung's features
are one move, so only the cheapest is kept. The cheapest N distinct moves are taken, then a
permutation-group membership test (sympy) checks whether they produce every witness and
cheap candidate. This is what prevents a rung from being silently unsolvable.

When they do not, **every** cheap candidate outside the subgroup they generate is kept (up to
`max_bridges`), not just enough of them to restore generation. Generating the group is a weak
guarantee. On the gear colour rung the 132 moves of length ≤ 10 only move the gears as 12
rigid pairs (the two gears beside each cube edge): a subgroup of order 2¹¹·12! ≈ 9.8×10¹¹,
index 3.2×10¹¹ in A₂₄, which reaches less than 1/3000 of the 3.2×10¹⁵ colourings. 240 moves
of length 12 and 14 split pairs. Adding one of them restores generation, and the first build
did exactly that. Random walks with that action set stay inside the subgroup's orbit (90% of
25-action walks never leave it, and walks of 400 actions are still far from uniform), so the
network was trained on a sliver of the rung and failed on the states the solver hands it,
while solving 11 of 12 states drawn from its own walks. With all 240 kept, walks of about 15
actions match solver states. `[measured: the pair statistic of walk states against scrambles
with the earlier rungs solved]`

For the same reason a learned rung is tested on scrambles of the whole puzzle with the
earlier rungs solved (`scripts/train_rung.py`, `Ladder.descend`), not on walks with its own
actions.

### 2.5 Tables and networks

- **Table rung** (`core/coset_table.py`): Dijkstra from the goal over the quotient puzzle
  with integer macro costs, on the GPU. 8.4 million cosets × 700 actions takes about a
  minute. The table also stores the first action of a cheapest path, so solving is a table
  walk. The NumPy and torch builders agree exactly (`tests/test_quotient_table.py`).
- **Learned rung** (`core/davi.py`): described in §5.

---

## 3. Facts about the Maltese Gear Cube, checked against the tables

All `[measured]`; the reproducing test is in `tests/test_maltese_ladder.py` unless another
place is named.

1. **Corners are the pocket cube.** 3,674,160 positions; the number at each distance is
   1, 6, 27, 120, 534, 2256, 8969, 33058, 114149, 360508, 930588, 1350852, 782536, 90280, 276
   (diameter 14 quarter turns), the known figures for the 2×2×2 with one corner fixed.
2. **Short corner-fixing words.** Exactly 3, 12, 75, 762, 10,336 and 145,797 distinct
   elements of length 4, 6, 8, 10, 12 and 14 fix the corners (156,985 in all up to 14). The
   three of length 4 are U⁴, L⁴, F⁴, as the design note says.
3. **An invisible index-2 coset.** The elements of length ≤ 8 generate a subgroup of index 2
   in the corner stabiliser. The other coset contains a single centre piece turned 180°,
   which cannot be seen. So the short macros reach every visible state. (sympy membership
   test; the coverage tests over random scrambles agree: no failures.)
4. **Gear rotation parity has 2²³ cosets, and gear rotation given parity has 2²³ more.**
   Both are exact tables of 8,388,608 cosets.
5. **Rotation before colour.** Words that fix the corners and every gear rotation still
   permute the gears as the alternating group A₂₄, with generators of 4 to 12 moves. In the
   other order, no word of length ≤ 14 fixes corners and gear colours while changing a
   rotation. The design note solves "color + spin" in one rung; splitting it this way keeps
   the macros short.
6. **Pure ring words of 12 moves.** 72 words of length 12 (and 240 of length 14) fix every
   corner and every gear. Each is three 3-cycles of ring pieces. One of them:
   `U L U' L' F L F' U' F U L' F'`. The design note's shortest pure ring macro is
   W = (R⁴ U R⁴ U')³, 30 quarter turns. All 312 move centres into edge slots.
7. **The coset counts multiply to the number of visible states.**

   | Rung | Cosets | Bits |
   |---|---|---|
   | corners | 3,674,160 | 21.8 |
   | gear rotation parity | 8,388,608 | 23.0 |
   | gear rotation | 8,388,608 | 23.0 |
   | gear colour | 24!/(4!)⁶ = 3,246,670,537,110,000 | 51.5 |
   | ring | 53,707,003,286,859,546,624,000 | 75.5 |
   | product | 4.51 × 10⁵⁸ | 194.8 |

   The reference solver's write-up gives 194.9 bits of visible state `[src]`, and
   |G| / |invisible subgroup| = 8.82×10⁶⁹ / 1.96×10¹¹ = 4.5×10⁵⁸. The ring count is the order
   of the group the ring macros generate (5.4996×10²⁵) divided by the 1,024 combinations of
   centre turns it contains.

### Where the design note is corrected

- **The ladder.** The note's rungs are corners → gears (colour and spin) → centre/edge
  separation → flush parity → ring permutation → edge flips. Measured, that ring chain is
  the wrong shape: each split forces the next rung onto longer macros, because every cheap
  pure ring word mixes centres and edges. With exact tables the ring rungs cost 31, 53, 118,
  466 and 945 quarter turns on average and keep growing (`runs/build_table_r6.log`). One
  rung for the whole ring, using the 12-move macros, costs RING_COST_PLACEHOLDER.
- **"R⁴" and the generator set.** The tables model U, L, F with the opposite corner fixed.
  The note's R, D, B macros are not needed: the harvest finds shorter ones directly.
- **Value cliffs.** The note's example (a single edge flip at word distance 42) is the reason
  for the ladder and holds in spirit, but 42 is the length of one human algorithm, not a
  measured distance.
- **A typo in an earlier note** (`archive/docs/maltese_piece_structure.md`): the pure gear
  subgroup has order 2.18×10³⁷, not 10⁴⁰.

---

## 4. The ladder

`puzzles/maltese_gear/ladder.py`, `LEARNED_LADDER`:

| # | Rung | Feature | Cosets | Heuristic | Actions (cost in quarter turns) |
|---|---|---|---|---|---|
| 0 | corners | corner `full` | 3,674,160 | exact table | 6 primitive moves (1) |
| 1 | gear_parity | gear rotation `sub0` | 8,388,608 | exact table | 696 corner-fixing macros (4 to 10) |
| 2 | gear_rot | gear rotation `full` | 8,388,608 | exact table | 669 macros (4 to 12) |
| 3 | gear_col | gear colour `full` | 3.2 × 10¹⁵ | value network | 376 macros (4 to 18): the 132 of length ≤ 10 and 244 that split gear pairs (§2.4) |
| 4 | ring | ring `full` | 5.4 × 10²² | value network | 72 macros (12) |

`TABLE_LADDER` in the same file replaces rungs 3 and 4 by chains of small table rungs
(nested colour masks). It needs no training and is kept as the baseline that shows why the
networks are needed.

The order follows the design note's rules, measured by `core/discover.py`
(`python scripts/analyse_puzzle.py --puzzle maltese`):

- *Period hierarchy.* Generators have period 4 on corners and 8 on everything else, so
  corners go first and X⁴ comes for free.
- *Cheapest pure macros last.* The ring has pure words of 12 moves; gears and corners have
  none within 12. The ring goes last.
- *Bits per macro length* inside the gear pieces gives parity, then rotation, then colour.

---

## 5. Coset value iteration

`core/davi.py` follows the training loop of DeepXube `[src]`, which is DeepCubeA's:

```
repeat
    states  <- random walks from the goal coset, length uniform in 0..K, with the rung's macros
    target  <- 0 at the goal, else  min over macros a of  cost(a) + V_target(state · a)
    train V on (state, target) for some iterations (Adam, mean squared error)
    copy V into V_target when the loss is below a threshold, or after a few updates
```

Taken from DeepXube: the one-step Bellman backup against a target network, walk lengths
uniform in 0..K, the target starting as the zero function, the target refresh on a loss
check, the residual network (`resnet_fc`), the per-iteration learning-rate decay.

Different here:

- **States are a rung's feature vectors.** The network sees only the pieces its rung is
  responsible for (24 gear colours, or 18 ring slots), not the 144-entry state.
- **Walks use the rung's macros**, which keep earlier rungs solved, so every training state
  is one the solver can meet. This is the design note's method A. Because the action on
  cosets is well defined, walks start from the solved state itself.
- **Costs are macro lengths**, so the value is a number of quarter turns.
- **Everything is a GPU tensor.** DeepXube generates data with CPU worker processes and
  per-state Python objects; here a rung state is a short uint8 vector and one update
  (200,000 states × 376 actions) takes about 65 s.
- **`ub_walk`**: each target is capped by the cost of the walk that produced the state, a
  true upper bound on its cost-to-go.

What was learned about training `[measured: tests/test_davi.py and the runs below]`:

- On a rung small enough to check (the toy puzzle, 29,160 states), the learned values match
  the exact table closely near the goal (mean error below 0.75 up to distance 8) and beam
  search with them is within one move of optimal.
- **Walk length matters more than target refresh rate.** With walks too short to reach the
  deep states, values flatten beyond the covered depth whatever else is changed
  (correlation with the exact distance 0.30 with K = 20, 0.95 with K = 40 and more updates).
- Values still saturate at depth: the minimum over many noisy child estimates is biased low.
  Beam search only needs the ranking, and it works well before the values are calibrated.

TRAINING_PLACEHOLDER

### Search

`core/search.py`: batched beam search on the rung's quotient puzzle. One step expands every
beam state under every macro, removes duplicates and states already expanded (64-bit hash),
scores children by cost so far plus the network's estimate, and keeps the best W. Macro costs
differ, so the search continues while a beam state could still beat the best solution found.
If a width finds nothing, it is retried at 8× and 64× the width.

---

## 6. Results

EVAL_PLACEHOLDER

---

## 7. Limits and what to try next

- **Length.** COMPARISON_PLACEHOLDER
- **Gear rotation takes two table rungs and about 76 quarter turns.** One learned rung over
  the 2⁴⁶ rotation states with 157 cheap corner-fixing macros is shorter: 65.3 quarter turns
  per solve on the acceptance states, trained by `scripts/auto.py` in 40 minutes
  (`docs/autonomy.md`). `[measured]`
- **The ring uses 72 of its 304 distinct cheap moves.** More actions lengthen training; in the
  designed ladder of `docs/autonomy.md` they did not shorten solutions. Its ring rung has 138
  moves (costs 12 to 14): the network needed 57 minutes to solve every test state (8 with these
  72) and costs 283.0 quarter turns per solve against 247.4. `[measured, in another ladder]`
- **Each rung is solved without regard to the next.** The reference solver keeps several
  landings per rung and gains about 10 moves from it `[src]`. `[not done]`
- **Limited-horizon Bellman learning** (backing up a short search tree instead of one step)
  is the DeepXube option aimed at the value saturation seen here. `[not done]`
- **Design note method B** (primitive excursions that break and re-solve earlier rungs) is
  not implemented. Method A was enough to solve.
- **Rung ordering** (`core/discover.py`) is greedy and looks one rung ahead. For this puzzle it
  proposes corners, then gear rotation, gear colour and the ring as three value networks: the order
  above, with one network over the 2^46 rotation states in place of the two rotation tables. Its
  proposals are now built, trained and tested by `scripts/auto.py` (`docs/autonomy.md`); that
  ladder passes the acceptance test at 431.7 quarter turns against 406.1 for this one.
- **Other puzzles.** Nothing in `core/` is specific to the Maltese cube. A new puzzle needs
  its move tables as a `GatherPuzzle` and a rung list. The Tangram Cube Extreme is the
  second: its tables are compiled from its 3D model (`core/geometry.py`), see `docs/tangram.md`.

---

## 8. Reproducing

```bash
pytest                                                       # 42 tests, about a minute
python scripts/analyse_puzzle.py --puzzle maltese            # structure and measured rung order
python scripts/build_ladder.py --ladder learned --radius 7   # table rungs
python scripts/train_rung.py --rung gear_col --minutes 45 --states 200000
python scripts/train_rung.py --rung ring --minutes 30
python scripts/solve.py --random 50 --beam 4096
python scripts/build_ladder.py --ladder table --radius 6     # the all-table baseline (stops at the ring chain)
```

Machine used: RTX 5070 Ti laptop GPU (12 GB), Python 3.11, PyTorch 2.11 nightly.

---

## 9. Sources

- Agostinelli, McAleer, Shmakov, Baldi, "Solving the Rubik's cube with deep reinforcement
  learning and search", Nature Machine Intelligence 1 (2019): https://doi.org/10.1038/s42256-019-0070-z
- Agostinelli, "The DeepXube software package for solving pathfinding problems with learned
  heuristic functions and search" (2026): https://arxiv.org/abs/2603.23873. Code (MIT):
  https://github.com/forestagostinelli/deepxube. The training loop, the target-network rule
  and the network in `core/davi.py` and `core/nnet.py` follow it.
- Agostinelli et al., "Q* search: heuristic search with deep Q-networks":
  https://arxiv.org/abs/2102.04518 (the option for very large action sets; not used here).
- Chervov et al., CayleyPy, beam search with learned distances on Cayley graphs:
  https://arxiv.org/abs/2502.18663, https://github.com/cayleypy/cayleypy
- Bram Cohen, `maltese_scramble`, the table-based reference solver for this puzzle
  (mean 135 face turns, 35 s, about 50 GB of tables): https://github.com/bramcohen/maltese_scramble.
  Used for comparison and for the count of visible states. It has no licence file; no code
  was taken from it.
- The design note: `docs/cayley_graphs_twisty_puzzles_and_hierarchical_rl_solvers.md`.
