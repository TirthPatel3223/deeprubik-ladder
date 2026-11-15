# How solvers for combinatorial puzzles are built: research notes

Purpose: a working model of how algorithms for twisty, permutation and other combinatorial puzzles are designed, so a new puzzle can be handled from first principles. The repo (Maltese Gear Cube) is the running example.

**Legend.** `[src]` = supported by a source found in this session's searches (see §10). `[bg]` = background knowledge that was not checked this session. Verify `[bg]` items before relying on them.

---

## 1. The model: a puzzle is a state graph

A puzzle is:

- **S**: the set of configurations (including unreachable ones).
- **M**: a set of moves, each a function S → S.
- **Goal**: a set of solved states (usually one state).
- **Cost**: usually the move count. The half-turn metric (HTM) counts a 180° turn as one move.

Solving means finding a path from the scramble to Goal. The Cayley-graph view: for a permutation puzzle the reachable states form the group G = ⟨M⟩, and the state graph is the Cayley graph of G with respect to M. God's number is the diameter of that graph. `[src: kociemba.org, cube20.org]`

- **Permutation puzzles**: states are group elements. A scramble is a word in the moves, and solving means factoring the scramble's inverse back into moves. Most twisty puzzles are permutation puzzles with extra orientation data (twists, flips), so their group sits inside a wreath product. `[bg]`
- **Non-group puzzles**: Lights Out (moves are XOR toggles, linear over GF(2)), sliding puzzles (group structure holds on the reachable set only), Sokoban (moves are irreversible, so there is no group). `[src: Carleton Lights Out notes; Demaine pushing-blocks]`

The first structural question for any puzzle is: **are the moves reversible, and do they act on the state as a group?** The answer selects the method family.

---

## 2. Step 1: classify the puzzle

| Question | If yes | If no |
|---|---|---|
| Does every move have an inverse in M? | The state graph is undirected; BFS and meet-in-the-middle apply. | Directed graph. Only forward search applies, and symmetric meet-in-the-middle does not. |
| Is the state a permutation of pieces (plus orientation)? | Group theory applies: stabilizer chains, subgroup chains, coset methods. | Look for linear algebra (GF(2) or Z_n) or generic search. |
| Are the moves linear over a finite field (Lights Out)? | Gaussian elimination solves it in polynomial time. `[src]` | Continue with the other questions. |
| Can a move create an unrecoverable dead end? | Standard search works. | Deadlock detection is needed. Such puzzles are often PSPACE-hard (Sokoban). `[src]` |

---

## 3. Step 2: measure before designing anything

The sizes decide the method, so measure first.

1. **Reachable-set size |⟨M⟩|.** For permutation puzzles, a base and strong generating set (BSGS) from the Schreier–Sims algorithm gives |G| as the product of the orbit sizes (transversal sizes) along the stabilizer chain. `[src: Magma handbook; Hulpke notes]`
   - Cube: about 4.3 × 10¹⁹. `[src: Kociemba/Thistlethwaite sources; cross-checked by the product of Thistlethwaite's indices in §4.1]`
   - Repo: Maltese group order about 8.82 × 10⁶⁹. `[repo README]`
2. **Invariants.** Find functions that are constant on every reachable state.
   - 15-puzzle: solvable iff the permutation parity matches the parity of the blank's row. `[src: Hsu slides]`
   - Cube: corner-twist sum ≡ 0 (mod 3), edge-flip sum ≡ 0 (mod 2), and corner and edge permutation parities are equal. `[bg]`
   - Raw piece arrangements are mostly illegal. Count the states that pass the invariants, not all arrangements. Check this count against the BSGS order.
3. **Sphere sizes and diameter.** For small spaces, BFS from the goal gives exact distances and the diameter. Sphere sizes act as a regression fingerprint. The repo already tests BFS sphere sizes. `[repo tests]`
4. **Move-table sanity.** Test move order, inverses, and commutation before writing any search code.

**Rule of thumb** `[bg]`:

- Up to roughly 10⁹ states: exact BFS or bidirectional BFS with a bitset or hash table.
- Roughly 10⁹ to 10²⁰ states: heuristic search with pattern databases (PDBs), or coset decomposition.
- Beyond 10²⁰ states, or no usable group structure: staged or algebraic methods that give guaranteed but not optimal solutions, or learned heuristics.

---

## 4. Method families

### 4.1 Subgroup chains (Thistlethwaite, 1981)

Build a chain G0 ⊃ G1 ⊃ G2 ⊃ G3 ⊃ G4 = {e}. At each stage, only the moves of the current subgroup are used to reduce the state into the next subgroup. `[src: jaapsch.net; cuberoot.me]`

- G0 is the full cube group.
- G1 = ⟨L, R, F, B, U2, D2⟩
- G2 = ⟨L, R, F2, B2, U2, D2⟩
- G3 = ⟨L2, R2, F2, B2, U2, D2⟩
- G4 = {e}

The index of each step is the number of cosets:

| Step | Index | Meaning (as reported by the sources) |
|---|---|---|
| G0 → G1 | 2048 = 2¹¹ | Edge orientations |
| G1 → G2 | 1,082,565 | |
| G2 → G3 | 29,400 | |
| G3 → G4 | 663,552 | |

The product of the four indices is about 4.3 × 10¹⁹, which equals |G|. This is the main sanity check for any chain: **the indices must multiply to |G|.** `[src, arithmetic checked]`

Move bounds: the original per-stage bounds were 7, 13, 15 and 17, giving 52 total. A refinement gives 50, and the best bound for this chain is 45. `[src]` Later computer work established the true worst case is 20 moves. `[src: cube20.org]`

Notation caveat: sources label the subgroups differently, so always check the generators, not the labels. `[src]`

### 4.2 Coordinates and two-phase solving (Kociemba, 1992)

Kociemba collapsed Thistlethwaite's four stages into two. `[src: cuberoot.me, secondary]`

- **G1 = ⟨U, D, R2, L2, F2, B2⟩.** Phase 1 moves the cube into G1. Phase 2 solves inside G1. `[src: kociemba.org]`
- **Phase 1 coordinates**: corner twist (0–2186), edge flip (0–2047), UD-slice position (0–494). Their product is 2187 × 2048 × 495 = 2,217,093,120 cosets. `[src]`
- **Phase 2 coordinates**: corner permutation (0–40319), edge permutation (0–40319), slice order (0–23). The phase-2 subgroup has 40320² × 24 / 2 = 19,508,428,800 states. `[src]`
- **Move tables**: each coordinate has a precomputed table giving its value after each move. This is the same "next state = table lookup" design as the repo's engine, where `next[i] = (state[idx[m,i]] + add[m,i]) mod modulus[i]`.
- **Pruning tables**: for each coordinate, BFS outward from the goal value gives exact distances. Using the max over several pruning tables is still a valid lower bound. `[src]`
- **Table sizes**: symmetry reduction keeps them small. One twist–flip table in a min2phase implementation uses 324 × 2048 cells at 4 bits each, about 332 KB. Total move tables are under 10 MB. `[src: min2phase]`
- **Optimality**: standard two-phase is not optimal. It returns the first solution found, typically 20–22 moves. `[src]`

The design lesson is that the coordinates are chosen so that the goal set is a single point, moves act as small tables, and "solved for this coordinate" defines a subgroup.

### 4.3 Stabilizer chains and BSGS (guaranteed, long solutions)

This is the algebraic baseline for any permutation puzzle. It is Track A in the repo README.

- **Base and strong generating set.** Choose a base b₁, …, bₖ of points. The stabilizer chain G = G₁ ⊃ G₂ ⊃ … ⊃ Gₖ₊₁ = 1 fixes b₁, …, bᵢ₋₁ at level i. Each level stores an orbit and a transversal (one group element per orbit point). `[src: Magma handbook; Hulpke]`
- **Order.** |G| is the product of the orbit sizes. `[src]`
- **Sifting (membership test).** To test whether g ∈ G, map each base point through the transversals, then check whether the residue is the identity. If the generators are not a strong generating set, sifting may return a nontrivial residue; that residue is evidence of a missing generator. `[src: DEJAVU arXiv 2108.04590]`
- **Random Schreier–Sims.** Random elements are sifted and the stabilizer chain is extended with the residues. The result is correct with high probability. One implementation reports a failure probability below 10⁻⁶. `[src: HaskellForMaths]`
- **Factorization.** Each transversal element is a word in the generators, via Schreier vectors. Sifting a state therefore produces a word in the moves, which is a solution. It is guaranteed to succeed, but it is long and not optimal. `[src: Ambrose et al., generalised sifting, arXiv math/0501346]`
- **Complexity.** Polynomial in the degree, with a large exponent. Hulpke's notes give roughly n⁷ naively and about n³ with Schreier trees. `[src]`
- **Known order as a check.** If the order is known, a randomized run that produces a smaller group signals a bug. `[src]`

### 4.4 Commutator and conjugate constructions (piece-by-piece)

This is how humans solve puzzles and how many generic solvers work.

- A commutator [a, b] = a·b·a⁻¹·b⁻¹ is often a small, local permutation, such as a 3-cycle or a swap, even when a and b are large moves.
- Conjugating a small permutation by a move c gives c·x·c⁻¹, which moves the same local effect to another location.
- A small set of such primitives (3-cycles and flips) lets you place pieces one at a time without disturbing those already placed. `[src: twisty-polyhedra README; forum discussion on the 3-cycle idea]`

The repo project `aditya-r-m/twisty-polyhedra` organizes this into tiers: simple commutators (L0), commutators conjugated by simple turns (L1), and pairs of commutators (L2), with special cases for leftover clusters. `[src: one search only; not read in full]`

### 4.5 Heuristic search: IDA* and pattern databases

- **IDA\***: depth-first search with a threshold that grows each iteration. Memory is linear in solution depth. The heuristic must be admissible (never overestimate) for optimal solutions. `[bg]`
- **Pattern database (PDB)**: abstract the state by keeping only a subset of pieces and marking the rest as wildcards. BFS from the abstract goal stores exact abstract distances in a table. Then h(s) = table[abstract(s)] is admissible, because removing pieces cannot increase distance. `[bg]`
- **Max vs sum**: the max of several PDBs is admissible. The sum is admissible only if each move's cost is charged to a single PDB, so that the costs split without double-counting. `[src: Korf & Felner, arXiv 1107.0050; Korf 1997]`
- **Korf 1997**: the first optimal solutions to random cube scrambles. It used IDA* with the max of three PDBs (corners, and two edge sets), giving a median optimal length of 18 moves. `[src]`
- **Cube-specific additive PDBs**: each cube move touches both corners and edges, so the naive sum is not admissible. The sources found did not settle whether a disjoint additive scheme works for the cube. `[src: not resolved]`
- **Sliding puzzles**: Manhattan distance is admissible. Linear conflict adds a penalty of 2 for each pair of reversed tiles in their goal row or column. PDBs cut searched nodes by about 1000× on a standard test set. `[src: Hsu slides; CEUR paper]`

### 4.6 Symmetry reduction

- The cube's rotation and reflection group has 48 elements (O_h). Conjugating a state by a symmetry that also preserves the move set gives a state at the same distance, so each symmetry class needs to be solved once. `[src: cuberoot.me; bg for the group identity]`
- Counting orbits with Burnside's lemma: #orbits = (1/|H|) Σ_h |Fix(h)|. Dividing the state count by |H| is only correct when no state is fixed by a nontrivial symmetry. `[src]`
- Example: for the 2×2×2 cube, a forum poster reports 893 classes by Burnside, and naive division gave 850.5. Naive division is wrong when the group acts with fixed points. `[src: GAP forum; unverified]`
- The 2010 God's-number proof used symmetry classes, reportedly reducing about 4.3 × 10¹⁹ states to roughly 9 × 10¹⁷ classes, close to dividing by 48. `[src: cuberoot.me; not verified against the paper]`

### 4.7 Learned heuristics (DeepCubeA, and the repo's Track B)

- **DeepCubeA** (Agostinelli, McAleer, Shmakov, Baldi, 2019): deep approximate value iteration learns the cost-to-goal from random walks outward from the solved state. At solve time it runs weighted A* with the learned value. `[src: Nature Mach. Intell. 1, 356–363]`
- **Results**: it solved 100% of test scrambles, with about 60.3% of solutions shortest. The same method applied to the 15, 24, 35 and 48 sliding puzzles, Lights Out and Sokoban. `[src]`
- **Trade-off**: no guarantee of optimality. It is generic, needs little domain knowledge, and needs substantial GPU training.
- **The repo's Track B** uses supervised random-walk regression with wide GPU beam search. It is in the same family, but beam search does not give admissibility, so it is good for finding solutions rather than optimal ones. `[repo README]`

### 4.8 Non-group puzzles: linear algebra (Lights Out)

- Let A be the n×n adjacency-plus-identity matrix over GF(2). A state x is solvable iff x is in the column space of A. The press set p satisfies A·p = x. `[src: Carleton notes; Anderson & Feil 1998, cited]`
- The kernel of A gives the redundancy: each solvable state has 2^dim(ker A) press sets. `[src]`
- For the 5×5 board, the kernel has dimension 2, so there are 2²³ solvable states. One blog gave 1,048,576 solvable states, which conflicts with this. The kernel argument supports 2²³. `[src, conflict noted]`
- To find the shortest press set, enumerate the 2^dim kernel translates of one solution and pick the minimum Hamming weight. Plain Gaussian elimination gives one solution, not the minimum. `[src]`

### 4.9 Hardness: when no fast general algorithm exists

- **Sokoban**: NP-hard (Dor & Zwick, 1995), PSPACE-complete (Culberson, 1997). Hearn & Demaine (2005) showed PSPACE-completeness even without barriers. `[src]`
- **Sliding-block puzzles**: PSPACE-hard even when every piece is a 1×2 domino (Hearn & Demaine). `[src]`
- **General n-puzzle**: optimal solving is NP-hard, but IDA* with Manhattan distance and PDBs is the practical standard. `[src]`
- **Push-pull puzzles**: 3D versions are PSPACE-complete. 2D versions with thin walls are NP-hard. `[src: arXiv 1709.01241]`
- **Other named puzzles**: 2048, Candy Crush and Rush Hour are reported as PSPACE-complete. `[src: summary of a 2017 paper]`

Practical consequence: for PSPACE-hard puzzles, expect exponential worst case. Solvers rely on deadlock pruning, good heuristics, and bounded instances.

---

## 5. The development workflow

This is the procedure I would follow for a new puzzle, based on the methods above.

1. **Formalize the moves.** Write each move as a permutation of pieces, with orientation as Z_k labels. Store as tables `next[m][i]`. This matches the repo's engine design.
2. **Validate the model.** Test move order, inverses, and sphere sizes (against known data where it exists) before any search code.
3. **Compute |G|.** Use a BSGS. Compare it with the count from the invariants in §3. A mismatch means a missing move or a wrong invariant.
4. **Pick the method family** using the rule of thumb in §3 and the structure from §2.
5. **If the puzzle is a group:** look for a subgroup chain or coordinate system whose zero sets are subgroups and whose index steps are small. Check that the index product equals |G|.
6. **Build exact tables.** BFS in each coordinate space gives pruning or coset tables. Check consistency: neighbors' heuristic values differ by at most 1. `[bg]`
7. **Search.** Use IDA* with the max of the tables, or a staged search for a fast non-optimal solver. For guaranteed-but-long solutions, use sifting and commutator constructions.
8. **Validate.** Check exhaustively on small instances against BFS ground truth. Check random scrambles against the simulator. Report move-count statistics, and state whether each solution is optimal. The repo already replays every solution on the reference simulator, which catches model bugs.

Pseudo-code for a generic pipeline:

```
given: move tables M, goal g0
1. validate M (order, inverses, sphere sizes)
2. order = BSGS(M); check against invariant count
3. if order is small (rule of thumb: <= ~1e9):
       exact bidirectional BFS / IDA* ; optimal
4. else:
       a. guaranteed: sift(state) via BSGS -> word in M (long, not optimal)
       b. staged: find chain G > G1 > ... > {e}; check index product == order
          solve stage by stage with coset tables
       c. optimal-ish: IDA* + PDBs from coordinates, symmetry-reduced
5. replay the solution on the reference simulator
```

---

## 6. Math toolkit

- **Permutation groups**: composition, inverse, orbits, stabilizers. Orbit–stabilizer: |G| = |orbit| · |Stab|. `[bg]`
- **Schreier's lemma**: generators of a subgroup from a transversal and generators of the group. `[src]`
- **Schreier–Sims (Sims 1970; Knuth 1991 improvement)**: BSGS construction, sifting, and randomized variants. `[src]`
- **Subgroup chains**: |G| = ∏ [Gᵢ : Gᵢ₊₁]. `[src]`
- **Cosets and coset BFS**: the state space of a coset is the coset itself, so BFS in a coset is much smaller than in G. `[bg]`
- **Commutators and conjugation**: [a, b] = aba⁻¹b⁻¹; c·x·c⁻¹. `[bg]`
- **Orientation groups**: Z_k labels, wreath products. `[bg]`
- **Counting**: Burnside / Cauchy–Frobenius lemma. `[src]`
- **GF(2) linear algebra**: A·x = b, kernel dimension, solvability iff b is in the column space. `[src]`
- **Heuristic search**: admissibility, consistency, A*, IDA* (linear memory), weighted A*, beam search. `[bg]`
- **Complexity**: NP-hard and PSPACE-complete reductions, e.g. Demaine & Hearn's constraint-logic framework. `[src]`
- **Learning**: value iteration, regression on random walks. `[src: DeepCubeA; bg for the general theory]`

---

## 7. Pitfalls

- **Treating heuristics as admissible when they are not.** Summing PDBs without splitting move costs, or using beam search and calling the result optimal.
- **Counting raw configurations.** Most arrangements are illegal. Filter by invariants first. `[bg]`
- **Naive division by a symmetry group.** Use Burnside whenever states can be fixed by symmetries. `[src]`
- **Leaving out reflections.** The cube's symmetry group has 48 elements including reflections, not 24 rotations. `[src: cuberoot; bg for the count of 24 rotations]`
- **Misreading chain labels.** Thistlethwaite and other sources label subgroups differently. Check the generators. `[src]`
- **Pattern database memory.** Pack entries into 4 bits where the range allows, and use symmetry to shrink tables. `[src]`
- **Shortest vs. any.** Two-phase and sifting solutions are not optimal. Say so in the output.
- **Trusting a secondary source.** Several sources here (cuberoot, Scribd, blogs) have arithmetic errors or conflicting numbers. Prefer primary papers and the first-party pages.

---

## 8. Applying this to the repo

- **Track A (algebraic) ↔ §4.3.** The repo's README says Track A uses stabilizer chains and factorization. The Maltese order of about 8.82 × 10⁶⁹ cannot be computed by BFS, so `scripts/` already computes it somehow. Check whether that method is a BSGS. If it is not, a BSGS with sifting is the first building block for Track A. `[to verify]`
- **Track B (learned) ↔ §4.7.** The repo uses supervised random-walk regression with beam search. DeepCubeA uses value iteration with weighted A*. The difference matters for optimality: beam search is not admissible.
- **Track C (staged) ↔ §4.1 and §4.2.** To build a staged solver for the Maltese cube, find a subgroup chain whose index product equals |G|. Each stage should have a small coset space with BFS tables.
- **Convention from memory**: gear rotations are slot-relative. The user's tables are ground truth. An absolute mismatch with zera-bot's convention is not, by itself, a bug.
- **Reference solver from memory**: bramcohen's algebraic subgroup-ladder solver is about 135 HTM. It has no LICENSE, so re-implement the idea and do not copy the code.

Proposed next steps (not started):

1. Check how `scripts/` computes the group order. If it is not a BSGS, implement one over the existing move tables and confirm it reproduces 8.82 × 10⁶⁹.
2. Implement sifting, and use it to factor random states into generator words. This gives a guaranteed baseline solver.
3. Search for a subgroup chain or coordinate system for Track C. Check the index product against |G|.

---

## 9. Known gaps in this research

- **Bidirectional and meet-in-the-middle optimal solvers.** The search returned nothing on this. The idea is in §3 and §4.2 from background knowledge only. `[bg]`
- **Holt–Eick–O'Brien handbook chapters.** The chapter covering stabilizer chains was not confirmed. `[unresolved]`
- **Cube-specific additive pattern databases.** Not resolved (§4.5).
- **2010 God's-number proof details** (symmetry-class counts, CPU time). Reported only by a secondary source. `[unverified]`
- **Thistlethwaite's 29-move claim** (G0 → G2 → G4). Reported by a source that I did not verify. `[unverified]`
- **Primary papers not read.** I have snippets from Kociemba's two-phase page, Korf (1997), and Hulpke's notes, not the full text. Reading them is the next step to deepen the expertise.
- **Lights Out 5×5 counts.** Sources conflict; the kernel argument (2²³) is the one I trust. `[bg check]`

---

## 10. Sources

Primary and first-party where available:

- Kociemba, two-phase coordinates: https://kociemba.org/math/twophase.htm
- Kociemba and collaborators, God's number is 20: https://kociemba.org/moves20.htm and https://cube20.org/
- Thistlethwaite's strategy: https://jaapsch.net/puzzles/thistle.htm
- Korf, Finding Optimal Solutions to Rubik's Cube Using Pattern Databases (AAAI 1997): https://www.cs.princeton.edu/courses/archive/fall06/cos402/papers/korfrubik.pdf (abstract page: https://mlanthology.org/aaai/1997/korf1997aaai-finding)
- Agostinelli et al., Solving the Rubik's cube with deep reinforcement learning and search (Nat. Mach. Intell. 2019): https://doi.org/10.1038/s42256-019-0070-z and https://deepcube.igb.uci.edu/static/files/SolvingTheRubiksCubeWithDeepReinforcementLearningAndSearch_Final.pdf
- Schreier–Sims and BSGS, Magma handbook: https://magma.maths.usyd.edu.au/magma/handbook/text/711
- Hulpke, lecture notes on permutation group algorithms (ICTS 2016): https://www.icts.res.in/sites/default/files/GTACM16-2016-11-07-Hulpke.pdf
- Ambrose, Neunhöffer, Praeger, Schneider, generalised sifting: https://arxiv.org/pdf/math/0501346v3
- DEJAVU, sifting as membership test: https://arxiv.org/pdf/2108.04590
- Holt, Eick, O'Brien, Handbook of Computational Group Theory (CRC, 2005): https://www.routledge.com/products/9780429147944
- Seress, Permutation Group Algorithms (Cambridge University Press): https://www.cambridge.org/core/books/abs/permutation-group-algorithms/further-lowlevel-algorithms/2EBE658AD2294CF1C8DAC888C2032AC1
- Felner, Korf et al., Additive pattern database heuristics: https://arxiv.org/pdf/1107.0050
- Rational deployment of multiple heuristics in IDA*: https://arxiv.org/pdf/1411.6593
- Hsu, sliding-puzzle heuristics and parity (lecture slides): https://homepage.iis.sinica.edu.tw/~tshsu/tcg/2022/slides/slide3.pdf
- Lights Out with GF(2) linear algebra (Carleton notes): https://people.carleton.edu/~rfjones/Math232w15/Math_232w15_Project1.pdf
- Demaine, pushing blocks: https://erikdemaine.org/pushingblocks
- Hearn & Demaine, PSPACE-completeness of sliding-block puzzles: https://arxiv.org/abs/cs/0205005
- Push-pull block puzzles: https://ar5iv.arxiv.org/html/1709.01241
- min2phase implementation notes: https://github.com/cs0x7f/min2phase
- twisty-polyhedra (generic commutator-based solver): https://github.com/aditya-r-m/twisty-polyhedra

Secondary (used with caution):

- cuberoot.me group-theory pages (symmetry counts, God's-number details, Thistlethwaite bounds): https://www.cuberoot.me/math/group/thistlethwaite and https://www.cuberoot.me/math/group/gods-number
- GAP forum, Burnside on 2×2×2: https://www.gap-system.org/ForumArchive2/2016/005277.html
- Pocket Cube symmetry tables: https://en.wikipedia.org/wiki/Pocket_Cube
- Speedsolving forum, how people design puzzle solutions: https://www.speedsolving.com/threads/how-does-one-figure-out-how-to-solve-a-puzzle.51543/
