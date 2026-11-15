# Piece-isolation algorithms: research notes

Scope: algorithms whose net effect touches only a few pieces (or stickers): a 3-cycle of pieces, a swap, a pair of opposite twists. Solvers use these as macros. This note covers how they are built, why they exist, how to search for them on any puzzle, and how to use them to solve.

It supersedes §4.4 of `puzzle_algorithm_research.md`, which was a brief sketch.

**Legend.** `[derived]` = I worked it out here, and the check is noted where one was run. `[src]` = from a source found in this session. `[bg]` = background knowledge, not checked this session.

---

## 1. Setup, trigger, undo

- **Conjugate.** S·T·S⁻¹ applies T's effect at relocated positions. For a cycle t = (x y z) and any permutation s:

  s t s⁻¹ = (s(x) s(y) s(z))  `[derived; checked numerically, 20,000 random cases, scripts/verify_one_point_commutator.py]`

  So any 3-cycle library entry can be moved to any target triple, by choosing S with S(x), S(y), S(z) equal to the target.
- **Commutator.** [A, B] = A·B·A⁻¹·B⁻¹. Setup/trigger/undo is S·[A, B]·S⁻¹, with the commutator as trigger. Cost is 2·|S| + |[A, B]|, minus any cancellation between S and the commutator's ends. Cancellation is what the Beyer–Hardwick-style solutions exploit. `[src: speedsolving wiki BH; search summary]`
- **Support bound.** supp([A, B]) ⊆ supp(B) ∪ A(supp(B)). Outside that set, B fixes the point and so does ABA⁻¹, so the commutator fixes it too. `[derived]`

The net effect of a commutator depends only on how the supports of A and B overlap. That is the key to designing isolating macros.

---

## 2. The one-point lemma (the core result)

**Statement.** If supp(A) ∩ supp(B) = {x}, then

[A, B] is the 3-cycle x → A(x) → B(x) → x, and it moves exactly the three points x, A(x), B(x).

**Proof sketch** (composition right to left, so the commutator is A∘B∘A⁻¹∘B⁻¹). `[derived]`

1. Write B = β·γ, where β = (x b₁ … b_m) is the cycle through x, and γ is the rest of B. Then γ is disjoint from supp(A), so A and A⁻¹ fix γ's points.
2. Let a = A(x). Then a ∉ supp(B), since a ∈ supp(A) and supp(A) ∩ supp(B) = {x}.
3. A fixes each b_i, so A·B·A⁻¹ = (a b₁ … b_m)·γ.
4. [A, B] = (a b₁ … b_m)·γ·β⁻¹·γ⁻¹ = (a b₁ … b_m)·β⁻¹, because γ commutes with everything involved.
5. Trace the result: x → a, a → b₁, b₁ → x, and every other point is fixed.

**Check.** The script `scripts/verify_one_point_commutator.py` runs 20,000 random cases each on 5, 8 and 12 points with no failures. Its worked example: A = (0 1 2), B = (2 3 4) gives C: 2 → 0 → 3 → 2, support {0, 2, 3}.

**Why it matters.** Two moves whose supports touch at exactly one point produce a clean 3-cycle at once. Everything else in the commutator cancels. The cost is the word length of the commutator, not the size of the moves.

**Where it applies.** The lemma uses only permutation algebra, so it holds for any point set. Whether it fires on a given puzzle depends on how the supports overlap, as §3 shows.

---

## 3. Stickers vs pieces: where the lemma applies

**Piece-level lemma.** `[derived]` The map from the puzzle's group to the permutation of its pieces (ignoring orientation) is a homomorphism. So the piece permutation of [A, B] equals [π(A), π(B)]. If A and B share exactly one piece p, the piece permutation of [A, B] is the 3-cycle (p, π(A)(p), π(B)(p)). The orientation part of the commutator has to be computed; it does not follow from the lemma.

**Sticker-level lemma on the 3×3 cube.** `[derived, not computer-checked]` For a legal move, the sticker support is a union of whole pieces. A moved corner has all three stickers moved, and a moved edge has both. So two moves share either no stickers or at least two (an edge) or three (a corner). A single shared sticker is impossible. The sticker-level one-point lemma therefore never fires on the 3×3. The piece-level version is the one that applies, and conjugates by setup moves are how a shared single piece is created.

**Caveat on the README's example.** The twisty-polyhedra README gives "3 sticker cycles which leave all other stickers unaffected" as a small-swap example. `[src: README, as relayed by the subagent]` On the 3×3, a single-corner 3-sticker cycle is a single corner twist. That breaks the corner-twist sum invariant (`[bg]`: the sum of corner twists is 0 mod 3 for any legal state). So the example cannot come from the 3×3 legal group. It must come from another puzzle or from a cluster model. I did not check which.

---

## 4. Orientation (twist) isolation

- **Invariants.** `[bg]` For the 3×3 cube, the corner-twist sum is 0 mod 3 and the edge-flip sum is 0 mod 2. So a single piece cannot be twisted alone. Isolated twists come in pairs.
- **Pair-twist commutators.** A commutator twists two pieces oppositely and leaves the rest alone. Heise's method builds X that twists corner A in place without touching the rest of its layer. Its inverse then twists corner B the other way, and a final conjugate returns both corners home. `[src: ryanheise.com corner twist page, search summary]`
- **Polster's recipe.** Your algorithm, then a quarter turn, then the algorithm's reverse, then the reverse quarter turn. This twists two adjacent corners oppositely and leaves the rest of the cube alone. `[src: qedcat.com, search summary]`
- **Pattern.** In every case the orientation macro is a commutator-like word whose net effect is a zero-sum set of twists on a few pieces.

---

## 5. Cycling pieces: 3-cycles and the groups they generate

**Building a library.** Find 3-cycle macros by the lemma or by setup search (§6). Then conjugate each by a setup S that maps its three points to any target triple (§1). One library entry gives every triple in its orbit under the setups you can afford.

**Generation facts.**

- **Jordan's theorem.** `[src]` A primitive permutation group that contains a 3-cycle contains the full alternating group A_n. Wielandt's book states it as Theorem 13.3, and a formal version exists in Mathlib.
- **Degree caveat.** `[src]` The prime-cycle version needs degree n ≥ q+3 for a q-cycle with q prime. Without that, counterexamples exist, for example a primitive group of degree p+1 containing a p-cycle but not A_{p+1}.
- **Connected 3-cycles.** `[bg, verify before relying]` A set of 3-cycles whose supports are connected as a hypergraph generates the alternating group on their union.

**Test for a block of pieces.** `[derived from the facts above]`

1. Take the moves restricted to the block, or the macros you have found.
2. Check that the generated group is primitive on the block.
3. Check that it contains a 3-cycle.
4. If both hold, it contains A_n on the block, so every even permutation of the block is reachable with macros.
5. Odd permutations need an odd macro. If none exists within the block, the parity has to be fixed using another block. Parity is linked across blocks in many puzzles, so check it globally.

**Parity in practice.** twisty-polyhedra's `correctParity` applies an atomic move with odd overlap to a cluster, and its solver assumes the result is then even. `[src: subagent reading of solver-preliminaries.js]`

---

## 6. Searching for isolating macros on any puzzle

**Inputs.** Generator cycles, a state model (stickers, or pieces with orientation), and the target set of points or pieces.

**Algorithm.**

```
atomic = all non-identity powers of each generator cycle       # twisty-polyhedra does this
macros = {}

# Level 0: direct one-point overlaps
for each pair (A, B) in atomic with |supp A ∩ supp B| == 1:
    C = commutator(A, B)                 # a 3-cycle by the lemma
    store C under its cycle, with cost = len(A)+len(B)+len(A^-1)+len(B^-1)

# Level 1: create a one-point overlap with a short setup
for each pair (A, B) with 2 <= |supp A ∩ supp B| <= k:
    for each setup S (BFS over words up to length L):
        A' = S·A·S⁻¹
        if |supp A' ∩ supp B| == 1:
            C = S·commutator(A', B)·S⁻¹     # relocated 3-cycle
            store C with cost = 2·len(S) + len(commutator)

# Keep the cheapest macro per target cycle, after word cancellation.
# Verify every stored macro by replaying it on the full state.
```

**Cost measures.** Word length after cancellation is the natural measure for a solver. Support size is an alternative, and twisty-polyhedra uses puzzle-wide size.

**Search size.** `[src: subagent reading of generateCommutators]` In twisty-polyhedra, conjugates have 1 or 3 atomic factors, and commutators have 4 or 8 atomic twists. The loops produce at most about n² conjugates and n³ candidate commutators for n atomic moves in a cluster. The depth is fixed, with no search, randomness or cap.

**Filter.** `[src: subagent reading of Cluster.js; read, not run]` twisty-polyhedra keeps the commutators that reach the smallest cluster overlap and the smallest total size at the same time. It also requires overlap < cluster size − 3. As a consequence, clusters of size 4 or less get no commutators, and a 3-cycle wholly inside a cluster needs size ≥ 7. The code does not state a purpose for the size − 3 rule. The subagent's reading is that it rejects commutators that move nearly the whole cluster.

---

## 7. Using macros to solve

**Solved-set discipline.** A macro must not disturb points that are already solved. Two ways to ensure that:

- choose macros whose support lies in the unsolved region, or
- conjugate the macro by a setup that carries its support into the unsolved region.

This is the standard piece-by-piece pattern.

**Progress.** `[src: subagent reading of solver-core.js]` twisty-polyhedra applies each accepted step only when a non-negative integer measure strictly drops (total size for normal clusters, overlap for special ones). This guarantees termination. It does not bound the solution length. The README claims quadratic length in the number of clusters, and the code does not prove it.

**Tiers.** `[src]` Its solver tries L0 (direct lookup), L1 (conjugate by one atomic move), L2 (two lookups linked through an intermediate sticker), L3 (two lookups with setup), and L4 (two-atomic setup). Two passes run with complexity limits 2 and then 5.

**Special clusters.** `[src]` Clusters with no useful commutator, or whose commutators spill onto other clusters, are solved first.

**License.** `[src: repo page]` twisty-polyhedra is GPL-3.0. Re-implement the ideas from this note. Do not copy its code into this project.

---

## 8. Worked example (checked)

On five points (0-indexed): A = (0 1 2), B = (2 3 4). The shared point is x = 2. Then A(2) = 0 and B(2) = 3, so [A, B] should be 2 → 0 → 3 → 2. The script prints exactly that, with support {0, 2, 3}.

---

## 9. Applying this to the repo

- The lemma and the macro search run on the Maltese Gear Cube's move tables. The first useful output is the list of atomic moves (all powers), and then the pairs whose piece supports meet in exactly one piece.
- Whether the Maltese blocks are primitive, and whether they contain 3-cycles, decides whether 3-cycle macros generate the even part of each block (§5).
- This is a constructive layer. It sits under the group-chain and search tracks from `puzzle_algorithm_research.md`, and it provides the local moves that a staged solver needs.

Proposed next steps (not started):

1. List atomic moves from the Maltese tables.
2. Find piece-level one-point overlaps and their 3-cycles.
3. Test primitivity and the 3-cycle condition per block, and analyse parity.

---

## 10. Gaps and unverified points

- twisty-polyhedra's commutator enumeration and small-swap filter were read from code by a subagent. I did not run them.
- The README's 3-sticker example conflicts with the 3×3 twist invariant. I did not resolve which puzzle it refers to.
- "Connected 3-cycles generate the alternating group on their union" is background, not checked.
- The claim that sticker supports of legal 3×3 moves are unions of whole pieces is derived, not computer-checked. A cube model would settle it.
- Orientation parts of piece-level commutators were not computed.

---

## 11. Sources

- twisty-polyhedra (GPL-3.0): https://github.com/aditya-r-m/twisty-polyhedra (files read: src/solver-worker.js, src/solver-preliminaries.js, src/solver-core.js, src/solver.js, src/classes/Solver/Cluster.js, src/classes/Solver/ComposableCycle.js)
- Corner twist via commutator (Heise): https://www.ryanheise.com/cube/corner_twists.html
- Polster, "Write Your Own Recipe for Rubik's Cube": https://www.qedcat.com/rubiks_cube
- Commutators and conjugates: https://ruwix.com/the-rubiks-cube/commutators-conjugates/
- Beyer–Hardwick method: https://www.speedsolving.com/wiki/index.php?title=BH
- Jones, primitive permutation groups containing a cycle: https://arxiv.org/pdf/1209.5169
- Cambridge Part III paper on primitive groups and p-cycles: https://www.maths.cam.ac.uk/postgrad/part-iii/files/pastpapers/2002/Paper1.pdf
- Verification script: `scripts/verify_one_point_commutator.py`
