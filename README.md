# Cube Solving Pipeline

A solver for twisty and permutation puzzles that works from the move tables alone. It
splits the puzzle into a ladder of subgroups and solves one rung at a time: small rungs
with exact cost tables, large rungs with a value network trained by deep approximate value
iteration (the DeepCubeA / DeepXube method) on that rung's coset graph.

The first puzzle is Meffert's Maltese Gear Cube (about 4.5×10^58 visible states). The
approach is the one outlined in `docs/cayley_graphs_twisty_puzzles_and_hierarchical_rl_solvers.md`.
What was built, what was measured, and where the build departs from that outline are in
`docs/hierarchical_solver.md`.

Every solution is replayed on the reference simulator (the verified move tables) before it
is reported.

## Setup

```bash
pip install -e ".[test]"        # numpy, sympy, pytest
pip install torch               # the CUDA build for your driver; Colab has it already
pytest                          # about a minute; uses the GPU if there is one
```

## Solving the Maltese Gear Cube

```bash
# 1. Build the table rungs (corners, gear parity, gear rotation). About 3 minutes on a GPU.
python scripts/build_ladder.py --ladder learned --radius 7

# 2. Train the two value networks. Each run resumes from its checkpoint.
python scripts/train_rung.py --rung gear_col --minutes 45 --states 200000
python scripts/train_rung.py --rung ring --minutes 30

# 3. Solve.
python scripts/solve.py --scramble "U L F' U2 L'3 F"
python scripts/solve.py --random 50 --beam 4096
```

Everything a build produces is cached under `data/ladders/` (not versioned), so each step
resumes where it stopped. Moves are `U L F` and their inverses `U' L' F'`; the corner
opposite those three faces is held fixed, so the other three faces are not needed.

## How it works

1. **Labelled model** (`core/perm_model.py`). The move tables become permutations of 216
   labelled points, so identical-looking pieces can be told apart internally.
2. **Structure discovery** (`core/structure.py`). Orbits, pieces, orientation sub-blocks and
   the candidate rung features are computed from the permutations. For the Maltese cube it
   finds 7 corners × 3, 18 ring pieces × 4, and 24 gears with a colour and a rotation.
3. **Rung order** (`core/discover.py`). The note's ordering rules, measured: generator
   periods, the shortest macros each choice leaves behind, and bits resolved per macro length.
4. **Macro harvesting** (`core/harvest.py`). The moves of a rung must keep earlier rungs
   solved. They are found exhaustively by meet in the middle and as Schreier generators of
   the previous rung's table, then checked to generate the whole stabiliser.
5. **Rungs** (`core/quotient.py`, `core/coset_table.py`, `core/davi.py`). Each rung is a
   small puzzle on its own feature vector. It gets an exact Dijkstra table if its coset
   space fits in memory, otherwise a residual value network trained by value iteration.
6. **Search and solve** (`core/search.py`, `core/ladder.py`). Table rungs are a table walk;
   learned rungs use batched beam search on the GPU. The rung solutions are concatenated,
   adjacent turns are merged, and the result is replayed on the simulator.

## Layout

| Path | Contents |
|---|---|
| `core/gather_engine.py`, `core/torch_engine.py` | Reference simulator: a move is `next[i] = (state[idx[m,i]] + add[m,i]) mod modulus[i]` |
| `core/perm_model.py`, `core/structure.py`, `core/discover.py` | Labelled model, structure discovery, rung ordering |
| `core/harvest.py`, `core/quotient.py`, `core/coset_table.py`, `core/ladder.py` | Macros, quotient puzzles, exact tables, the ladder |
| `core/nnet.py`, `core/davi.py`, `core/search.py` | Value network, coset value iteration, beam search |
| `puzzles/maltese_gear/` | Verified move tables and the ladders for the Maltese Gear Cube |
| `puzzles/toy/` | A 29,160-state puzzle used to check everything against brute force |
| `scripts/` | `build_ladder.py`, `train_rung.py`, `solve.py`, `analyse_puzzle.py`, `group_order.py` |
| `tests/` | Unit and end-to-end tests |
| `docs/` | The design note and the implementation notes |
| `archive/` | The earlier attempts (single-network beam search), kept for reference |

## Adding a puzzle

Write its moves as a `GatherPuzzle` (index and addend tables plus the solved state), then:

```bash
python scripts/analyse_puzzle.py --puzzle <name>     # structure, candidate features, proposed rung order
```

Turn the proposal into a list of `RungSpec` (see `puzzles/maltese_gear/ladder.py`) and run
the three steps above.

## Conventions

- Runs on Windows and Linux (Colab). No hardcoded paths.
- Long jobs cache and resume.
- DeepXube (MIT) is credited where its design is followed: `core/nnet.py`, `core/davi.py`.
