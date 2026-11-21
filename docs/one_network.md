# One network for the whole puzzle

Branch `reward-shaped-rl`. The ladder solver (`docs/hierarchical_solver.md`) gives every large rung its own value
network and every small rung an exact table, and solves one rung at a time. This note is about replacing all of
them with one value network that sees the whole puzzle, trained like DAVI from the goal, with a reward shaped by
the ladder's rungs. The ladder's macro discovery is kept: its macros are the network's actions.

**Legend.** `[measured]` = computed in this project; the command that reproduces it is named. `[src]` = from a
source listed at the end.

The goal set for it: every acceptance state solved (100 deep scrambles, `core/acceptance.py`), with shorter
solutions than the ladder's (90.0 steps on the Tangram Cube Extreme, 406.1 quarter turns on the Maltese Gear Cube).

---

## 1. Design

`core/whole.py`; trained by `scripts/train_whole.py`, accepted by `scripts/accept_whole.py`.

- **State.** Every rung's feature vector side by side (Tangram Extreme: 80 slots; Maltese: 97). The network sees
  only these; it is never told which rungs are solved.
- **Actions.** The union of the rungs' action sets, one action per distinct permutation: 362 on the Tangram
  Extreme, 1,492 on the Maltese cube (its rungs' sets add up to 1,819; gear rotation and gear colour reuse gear
  parity macros). Each costs its length in primitive moves, so costs are in the puzzle's own metric. `[measured]`
- **Phases and masks.** A state's phase is its first rung that is not solved. The mask says which actions a
  state in phase k may take; it is applied by the trainer and the search, not given to the network:

  | Mask | Phase k may use | Maltese actions per phase |
  |---|---|---|
  | `rung` | rung k's own actions | 6 / 696 / 669 / 376 / 72 |
  | `progress` | every action that leaves rungs 0 .. k-1 solved | 1,492 / 1,486 / 973 / 682 / 72 |
  | `none` | every action | 1,492 in every phase |

  Training starts under `rung`; the mask is loosened once that works (`--mask`, `--init`).
- **Reward shaping.** The value is V(s) = H(s) + r(s), with r the network's output and H a fixed count of unsolved
  pieces weighted by rung (`--baseline`). This is potential-based shaping with potential -H `[src: Ng et al.]`: the
  cost-to-go the training converges to is unchanged, and training starts from H instead of zero
  `[src: Wiewiora]`. `gated` counts only the current rung's unsolved slots, at c_k / n_k each, plus the mean cost
  c_j of every later rung (the reward-machine construction `[src: Toro Icarte et al.]`); `smooth` is `gated`
  with each slot at c_k / (n_k f_k), f_k the measured fraction of a mixed rung's slots that are unsolved, so a
  mixed rung counts c_k on average and the potential does not step at a rung boundary (section 3, Maltese);
  `pieces` counts every unsolved slot; `none` is plain DAVI. The rung costs are the ladder's own means on deep scrambles:
  Tangram 7.7 / 47.4 / 40.5 steps, Maltese 10.7 / 27.5 / 49.2 / 86.9 / 249.2 quarter turns. `[measured]`
- **Training states.** Stratified walks: choose a phase k (in proportion to its walk length), mix every later rung
  with a long walk of rung k+1's actions, then take a uniform 0 .. D_k of rung k's actions. Each state then has
  the earlier rungs solved, rung k at a controlled depth and the later rungs mixed, which is how the search meets
  states near the end of rung k. D_k and the mixing walk are 3.5 times the measured mixing lengths, the
  autopilot's rule. Under the looser masks (`progress`, `none`) a quarter of the states also take 1 to 3 random
  allowed actions, so the network has seen the states those actions lead to (under `none`, states in which a
  finished rung is broken).
  Uniform walks over all actions would not do: on the Maltese cube 6 of the 1,492 actions are primitive moves, so
  a 100-action walk has none 67% of the time, and the network would almost never see the unsolved corners every
  real scramble starts with.
- **Targets.** The one-step Bellman backup of `core/davi.py` over the actions the mask allows. There is no
  walk-cost bound (`ub_walk`): under the mask the reversed walk is not always a legal path. The target network
  is refreshed when the loss falls below `--loss-thresh` (DeepXube's rule, with the loss in units of the mean
  action cost squared), and at least every 4 updates.
- **Twin networks** (`--twin`). A min over hundreds of noisy children is biased low, since it picks the
  children the network happens to underrate. Each refresh adds to the bias, and on the Maltese cube it sank the
  single network (section 3). With `--twin`, two networks with their own weights each fit half of every update's
  states. Network a's targets take the child that a values cheapest at b's value, and b's the other way round:
  the double estimator of double Q-learning and TD3. Each state's children are still valued once, so only the
  training step doubles. The search uses network a alone (`--twin-search mean` uses both, at twice the cost).
- **Search.** One beam search from the scramble to the goal, each state expanded with its own phase's actions.
  Unlike `core/search.py`, a state already expanded is expanded again when a cheaper path reaches it: macros cost
  several moves and primitive moves one, so the first path to a state is often not the cheapest. On the toy
  puzzle the old rule returned a 14-move path where 12 is optimal, with the exact values.
  `[measured: tests/test_whole.py]` The next beam is built from chunks of at most 2^21 children, merged into a
  pool of the `width` best distinct states, so memory does not grow with width x actions. The first version
  hashed all children at once, and the 8x wider second search (32,768 x 192 children on the Tangram cube) ran
  out of GPU memory. Now that search peaks at 3.2 GB, and the 4,096-wide search's solutions and speed are unchanged.
  `[measured]`

## 2. Checked against brute force

On the toy puzzle (29,160 states) `tests/test_whole.py` computes the exact cost-to-go under each mask by value
iteration. It then checks that:
- the training targets reproduce it, including the double and twin backups;
- the search with the exact values is optimal and respects the mask;
- a small network trained from scratch (with and without the shaping, and as twins) solves scrambles of the real
  puzzle within one move of optimal on average. `[measured]`

## 3. Results

### Tangram Cube Extreme, per-rung mask

`[measured: python scripts/train_whole.py --puzzle tangram_extreme --baseline gated --minutes 180; then
python scripts/accept_whole.py --puzzle tangram_extreme --checkpoint data/whole/tangram_extreme/learned_gated.best.pt
--compare runs/acceptance/tangram_extreme_learned_r7.json]`

One network of 1000 units x 4 residual blocks (8.6 M parameters), 200,000 states per update, about 25 s per
update on the RTX 5070 Ti laptop GPU (21 s of it computing targets). The measured walks: frame up to 42 primitive
moves, large 38 macros, small 18; mixing walks 28, 32 and 18. Tests every 10 minutes on 16 deep scrambles (seed 1),
against the ladder on the same states (91.2 steps):

| Minutes | 10 | 21 | 31 | 41 | 51 | 61 | 71 | 81 | 92 |
|---|---|---|---|---|---|---|---|---|---|
| solved | 16/16 | 16/16 | 16/16 | 16/16 | 16/16 | 16/16 | 16/16 | 16/16 | 16/16 |
| mean steps | 122.3 | 92.2 | 81.9 | 79.6 | 80.3 | **77.8** | 78.4 | 79.7 | 79.4 |

Training stopped on its plateau rule after 92 minutes; the network of minute 61 was kept. Acceptance test
(100 deep scrambles, seed 42, the ladder's own acceptance states):

| | Verdict | Solved within 60 s | Steps, mean (sd, min-max) | Turns | Time per state, mean / max | Per rung |
|---|---|---|---|---|---|---|
| One network | PASS | 100 / 100 | 80.2 (4.6, 66-89) | 75.6 | 7.9 s / 9.6 s | frame 8.1, large 46.0, small 30.4 |
| Ladder (`learned`) | PASS | 100 / 100 | 90.0 (4.5, 76-107) | 83.9 | 4.7 s / 5.5 s | frame 7.4, large 47.1, small 40.4 |

Paired over the 100 states the network is 9.9 steps shorter (standard error 0.7): shorter on 90 states, equal on 3,
longer on 7. Almost all of the gain is in the last rung (30.4 steps against 40.4): one search across the rungs
finishes the large pieces in the way that leaves the small triangles cheapest, which the ladder, solving one rung
at a time, cannot do. The first rung costs 0.7 steps more than the ladder's exact table (8.1 against 7.4), and the
network takes 7.9 s per state against 4.7.

### Tangram Cube Extreme without the shaping

`[measured: python scripts/train_whole.py --puzzle tangram_extreme --baseline none --minutes 92; then
scripts/accept_whole.py on its best network]`

The same network, training states, mask and search, but plain DAVI: the value starts from zero instead of from
the rung potential. Both runs converge to the same cost-to-go, so this run measures what the rung reward adds.
Tests on the same 16 states:

| Minutes | 10 | 21 | 31 | 41 | 51 | 61 | 71 | 81 | 91 |
|---|---|---|---|---|---|---|---|---|---|
| plain DAVI: solved | 0 of 4 | 0 of 4 | 2/16 | 9/16 | lost | 16/16 | 16/16 | 16/16 | 16/16 |
| plain DAVI: mean steps of the solved | | | 100.0 | 91.7 | | 82.1 | 80.4 | 79.4 | 78.2 |
| with the shaping (16/16 throughout) | 122.3 | 92.2 | 81.9 | 79.6 | 80.3 | **77.8** | 78.4 | 79.7 | 79.4 |

A test stops after 4 states if none of them is solved. The plain run's 51-minute test was lost to the search's
out-of-memory bug (fixed since, section 1), and the run resumed from its checkpoint.

Acceptance of plain DAVI's best network (minute 91): PASS, 100 / 100, 79.6 steps (sd 5.1, 65-89), 7.1 s per
state, per rung frame 8.5, large 44.5, small 31.9. Paired against the shaped network over the 100 states, the
shaped one is 0.5 steps longer (standard error 0.55; shorter on 34 states, equal on 26, longer on 40), which is no
measurable difference.

So on the Tangram cube the rung reward is a head start. Every test state was solved after 10 minutes of training
instead of between 41 and 61, and solutions were shorter until about minute 80. By minute 90 the two networks were
equally good. Potential-based shaping predicts exactly this: the same cost-to-go from a different starting point.
The rung structure also reaches the network through the training states (stratified by rung) and the mask, and
plain DAVI got there on those alone. This run cannot say whether the head start decides between learning and not
learning on the much larger Maltese cube.

### Tangram Cube Extreme, looser masks

`[measured: python scripts/train_whole.py --puzzle tangram_extreme --baseline gated --mask progress --init
data/whole/tangram_extreme/learned_gated.best.pt --round-minutes 15]`

What the extra actions change on this puzzle `[measured]`:
- Under `progress`, the frame phase can use all 362 actions instead of its 7 primitive moves. Every one of them can
  move frame pieces, but the 355 macros cost 7.9 moves on average.
- The large phase gains the 192 small-triangle macros. None of them can move a large piece; they only rearrange
  the small triangles.

**Search only.** The per-rung network searched with the `progress` mask, without retraining, on the 16 test
states: 78.8 steps against 78.1 under its own mask, at 26.7 s per state against 7.5. Single states moved by up to
14 steps in either direction: the extra actions open shortcuts, but the values still describe the per-rung mask.

**Trained under `progress`**, starting from the per-rung network. Each update takes 73 s instead of 25, because
the frame states now have 362 children instead of 7. Every test solved all 16 states, at about 27 s per state:

| Minutes | 16 | 32 | 48 | 64 | per-rung network |
|---|---|---|---|---|---|
| mean steps | 79.6 | 79.7 | **79.4** | 81.8 | **77.8** |
| cost before merging | 84.9 | 84.6 | 83.4 | 85.6 | 83.3 |
| per phase: frame, large, small | 16.1, 47.6, 21.2 | 17.1, 46.2, 21.2 | 15.1, 42.8, 25.6 | 15.2, 46.0, 24.4 | 7.9, 46.0, 29.4 |

The run stopped on its plateau rule. Before merging, its paths cost the same as the per-rung network's (83.4
against 83.3). The search moves work into the frame phase, using macros before the frame is finished: the small
triangles get 4 to 8 moves cheaper and the frame 7 to 9 moves dearer. After merging, the per-rung solutions are
shorter, because primitive moves next to a macro cancel more often than two macros do. On this puzzle `progress` buys
nothing and makes the search 3.5 times slower.

**No mask** (`--mask none`, also starting from the per-rung network): 81 s per update.
- After 12 updates the test solved 2 of the 16 states within 60 s, at 96.0 steps. Their cost per phase was
  79 / 6 / 15: the search did most of its work before the frame was solved.
- After 24 updates it solved none of the first 4.
- Over those 20 updates the mean training target fell from 58.4 to 36.9.

Without a mask, and with every macro costing its length, the cost-to-go is the true shortest distance in
primitive moves, since a macro is a word of primitive moves at the same cost. The values leave the ladder's shape and move toward that target, the one the archived
single network could not learn. Meanwhile the search, with 362 actions in every state, runs out of time. The run was
stopped after 32 minutes. On this puzzle the per-rung mask is part of what makes one network work.

### Maltese Gear Cube, per-rung mask

`[measured: python scripts/train_whole.py --puzzle maltese --baseline gated --round-minutes 20 (first run);
the same with --loss-thresh 0.6 --twin (second run)]`

The whole state has 97 slots, and there are 1,492 actions: 6 / 696 / 669 / 376 / 72 per rung under this mask.
The network is 1000 units x 4 blocks, 13.3 M parameters; the input embedding is larger than the Tangram cube's.
An update takes 75 s, 70 s of it computing targets. The measured walks are, for corners / gear parity / gear
rotation / gear colour / ring, 35 / 18 / 32 / 24 / 38 actions, with mixing walks of 24 / 21 / 60 / 32 / 38. The
ladder solves the 16 test states in 410.9 quarter turns (427.3 before merging, per rung 10.7 / 27.2 / 48.8 /
93.9 / 246.8).

**First run: one network.** Its loss (0.2 to 0.3) never fell below the threshold of 0.06, so the target network
was refreshed only every 4 updates:

| Minutes | 20 | 41 | 61 |
|---|---|---|---|
| solved | 0 of 4 | 2/16 | 6/16 |
| mean quarter turns of the solved | | 465.0 | 583.5 |
| ring phase of the solved | | 276 | 420 |

The threshold had been set on the Tangram cube, whose values are 4.5 times smaller. Relative to the value range,
the network's error (about 1.5%) was no larger than the Tangram network's, which was refreshed every update. So the
run was resumed at update 52 with a threshold of 0.6, and the target network was refreshed every update. The values then sank: the largest training
target went from 392 to 294 in 13 updates, and at minute 81 the test solved none of its first 4 states. The
values of the network at three points `[measured: a value check on fixed states]`:

| Update (refreshes) | 36 (9) | 52 (13) | 68 (29) | the ladder actually spends |
|---|---|---|---|---|
| deep test states | 396 | 366 | 239 | 427 |
| a fully mixed ring | 161 | 128 | 100 | 247 |

The drop per refresh was about the same before and after the change (5 to 7 quarter turns off the mean target),
so the faster refresh only made the bias add up faster. Each target is the least of 72 to 696 noisy child
values, which picks the children the network underrates. On the Tangram cube the network's error relative to an
action's cost was about 2.5 times smaller, and no drift showed. The second run therefore starts over with twin
networks (section 1).

**Second run: twin networks**, refreshed every update to update 50, then every 4 updates (threshold 0.06):

| Minutes | 21 | 42 | 63 | 86 | 106 |
|---|---|---|---|---|---|
| solved | 0 of 4 | 7/16 | 2/16 | 3/16 | 0 of 4 |
| mean quarter turns of the solved | | 555.9 | 564.5 | 479.0 | |
| corners phase of the solved | | 11.9 | 46.5 | 13.7 | |

The twins stopped the collapse. Deep test states were valued at 370 at update 49 and 363 at update 79, where the
single network had reached 239 by update 68. The corner values went wrong twice, though, and the search wandered
in the corners phase:
- At update 49 they were flat: 366.8 with the corners solved, 370.5 fully scrambled, against a true difference
  of about 11.
- At update 79 they were inverted: 374 with the corners solved, 359 to 363 unsolved.

The cause is the `gated` potential's step at each rung boundary. A mixed rung has only a fraction f_k of its slots
unsolved (measured: about 0.5 for gear parity and gear rotation, 0.83 for gear colour, 0.97 for the ring), so the
potential drops by (1 - f_k) c_k at the moment a rung is finished:
- corners to gear parity 13.8, gear parity to gear rotation 24.6, gear rotation to gear colour 14.6, gear colour
  to ring 7.1;
- the network has to learn a residual that cancels each step;
- at the corners the step is larger than the corners' whole cost (10.7), and an error of the network's usual size
  turns the corners' slope around.

The `smooth` baseline (section 1) removes the step on average, and the third run uses it.

**Search speed.** In a profile of three test states, 96% of a search was spent valuing children. That was 36 to
53 million per state, at 0.8 million a second, and they cluster in the phases with many actions:

| Phase the beam is in | Actions per state | Share of the search time |
|---|---|---|
| gear colour | 376 | 31 to 48% |
| gear parity | 696 | about 26% |
| gear rotation | 669 | about 15% |

The ladder values its gear parity and gear rotation rungs from exact tables. In the network, 40% of a forward pass
is the input lookup (97 rows of 1,000 numbers per state); raw bf16 matrix products run at 50 TFLOPS on this GPU.
`[measured]` Two changes followed:
- A fused inference copy of the network (batch norm folded into the layers, bf16) is used for every evaluation
  that needs no gradient, both targets and search: 1.22 times faster.
- An optional children budget for the search, `--budget`. The next beam is cut, best first, where its states
  would make more than width x budget children, so phases with many actions keep fewer states.

**Third run: `smooth` potential, twin networks**, refreshed every 4 updates, test searches with `--budget 250`.
It solved none of its first 4 test states at 21 and at 42 minutes. At update 30 the corners were still worth
nothing to it: 426 with the corners solved, 425 with them scrambled. From the update-38 checkpoint I tried three
backup rules. Each was stopped once the measured drift or the test showed it failing:

| From update 38 | What happened |
|---|---|
| weighted double backup, cross weight 0.5 | mean target 311 to 284 in 10 updates |
| target network as a running average (Polyak, tau 0.005) | a mixed ring's value 216 to 138 in 7 updates |
| refresh only when a batch's loss is below 0.06 (at most every 16 updates), learning rate 3e-4, 200 steps an update | at minute 62: 4/16 solved, 802.8 quarter turns, corners phase 100.8 |

**What limits the single network: precision per rung.** `[measured at update 38 and 45]`
- The twins disagree by 10 to 11 quarter turns RMS in the gear and ring phases, and 3 to 5 in the corners.
- The network's values move by 12 to 16 between refreshes.
- On fresh states, the targets after a refresh differ from the network's values by 19 to 27 RMS in the gear and
  ring phases, against the corners' 5.
- An action costs 10 to 12, so siblings differ by about one action, about the size of the noise. A min over 72 to
  696 such children then picks out the underrated ones, which is the drift above.

The ladder's own ring network is a useful comparison. It is trained by the same DAVI on the same 72 ring actions,
and fits its targets to 2.9 quarter turns RMS (loss 0.057, refreshed at 141 of its 155 updates). Its values are
just as low (largest training target 133, where the ring costs about 247), and yet its search solves the ring.
Precision, not calibration, is what the search needs. One network that covers all five rungs is 4 to 7 times less
precise per rung than the ladder's networks, each of which covers one.

## Sources

- Ng, Harada, Russell, "Policy invariance under reward transformations" (ICML 1999):
  https://mlanthology.org/icml/1999/ng1999icml-policy/
- Wiewiora, "Potential-based shaping and Q-value initialization are equivalent" (JAIR 2003):
  https://jair.org/index.php/jair/article/view/10338
- Toro Icarte, Klassen, Valenzano, McIlraith, "Reward machines" (JAIR 2022): https://arxiv.org/abs/2010.03950
- Agostinelli, "The DeepXube software package" (2026): https://arxiv.org/abs/2603.23873
- Agostinelli et al., "Q* search" (large action spaces): https://arxiv.org/abs/2102.04518
