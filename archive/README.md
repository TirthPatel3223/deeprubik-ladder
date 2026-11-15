# Archive

Earlier attempts, kept for reference. Nothing here is imported by the current solver, and
the tests in `archive/tests` are not collected by pytest.

| Folder | What it was |
|---|---|
| `core/`, `scripts/`, `tests/` | Track B: one distance network for the whole puzzle, trained on random-walk lengths (with an optional Bellman refinement), searched with a wide GPU beam over a depth-12 goal ball. It solved scrambles of up to about 24 quarter turns and nothing deeper: the predicted distance flattens out near 18 (`results/train_bellman.log`). |
| `docs/` | Research notes written before the ladder solver: general solver design, piece-isolation macros, and the first piece-structure measurements. `docs/hierarchical_solver.md` in the main tree supersedes them and re-verifies the facts it reuses. |
| `results/`, `checkpoints/` | Logs and networks from those runs. |
| `data/ball/` | Every state within 12 quarter turns of solved (353,559,133 states, about 6 GB). Not under version control. |

The archived modules import each other as `core.*`, so to run one, copy it back into `core/`
(or add `archive/` to `sys.path` ahead of the project root).

One number in `docs/maltese_piece_structure.md` is wrong: the pure gear subgroup has order
2.18e37, not 2.18e40 (3.10e23 x 2^46).
