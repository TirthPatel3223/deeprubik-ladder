"""Choosing the rungs and their order from measurements (Part V, phase 2 of the design note).

Phase 1 (``core/structure.py``) lists what could be a rung: every feature of every orbit. This module
chooses rungs with the note's rules, each turned into something measured on the move tables, and one rule
both hand-made ladders follow. Nothing about a particular puzzle is assumed.

Rule 0, small kinds of piece first, together
    Kinds of piece whose coset space under the primitive moves fits one exact table are joined into a first
    rung, from the smallest while their joint table still fits, and solved with the primitive moves. Such
    pieces are cheap to place first and can be very expensive later: on the Tangram Cube Extreme, with
    everything else solved, a turning corner can still be turned 120°, but no word of up to 12 moves does
    it. This gives the Tangram's hand-made ``frame`` rung and the Maltese Gear Cube's ``corners``.

Rule 1, period hierarchy (orders the other kinds of piece)
    The order of a generator on an orbit is its period there. If a generator has period 4 on one orbit and
    8 on another, its 4th power fixes the first and still moves the second: solving the first orbit first
    leaves short moves for the rest. Orbits are taken in ascending period. Orbits whose pieces move
    identically (a gear's colour and its rotation) are one kind of piece and stay together.

Rule 2, cheapest pure macros last (breaks ties between kinds of piece)
    The last kind solved needs words that move it and nothing else. The shortest such words are found
    exhaustively by meet in the middle; the kind with the shortest goes last.

Rule 4, bits per macro length (chooses each rung)
    score = bits the rung resolves / mean length of the shortest macros it leaves behind. The candidates
    are each feature of the first unsolved kind of piece, the whole kind as one rung, and the whole kind
    together with the next one (so a value network can take several kinds at once, as the Tangram's
    ``large`` rung does). Bits are exact when the rung's table fits the budget. Otherwise they are
    ``log2 |H| - log2 A``: H is the group the rung's actions generate on its pieces and A the number of
    arrangements of those pieces that show the goal, at least the goal's stabiliser in H, so this is a
    lower bound (within two bits of the documented 51.5 and 75.5 for the Maltese colour and ring rungs).
    When no candidate leaves short macros, the largest whole-kind candidate is taken: splitting further
    only lengthens the macros of what follows.

A rung gets an exact table when it fits the budget and a value network otherwise. The candidate pool of the
next rung is computed as the ladder builder computes it (the previous pool, meet in the middle, and the
Schreier generators of a table), so a kind that only longer macros reach is still found.

Completeness check
    At the end every orbit is tested exactly: does fixing the chosen features force it to its goal? (A
    pointwise stabiliser in sympy, exact when each feature's goal value pins its pieces down.) An orbit that
    is not forced joins the first rung, and the other rungs are chosen again.

The scores are greedy and look one rung ahead: a proposal that the build, the training and the acceptance
test then check, not a guarantee.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable, Sequence

import numpy as np

from core.coset_table import CosetTable, build_table_auto
from core.harvest import ball, collisions
from core.ladder import Ladder, RungSpec
from core.perm_model import PointModel, Word
from core.quotient import ActionSet, Quotient
from core.structure import Feature, Orbit, Structure


@dataclass
class Candidate:
    features: tuple              # the rung's feature specifications, (orbit, kind) each
    name: str
    bits: float                  # log2 of the coset count: exact if ``exact``, else a lower bound
    exact: bool
    cosets: int | None           # the exact coset count, when the table fits the budget
    macro_len: float             # mean length of the shortest macros left behind (inf: none within 2R)
    n_macros: int
    whole: bool = False          # a whole kind of piece, or two
    n_actions: int = 0
    mean_cost: float = 0.0
    quotient: Quotient | None = field(default=None, repr=False)
    table: CosetTable | None = field(default=None, repr=False)

    @property
    def spec(self) -> tuple:
        """The single feature specification of a one-feature candidate (older callers)."""
        return self.features[0]

    @property
    def score(self) -> float:
        return 0.0 if math.isinf(self.macro_len) else self.bits / self.macro_len

    @property
    def predicted_cost(self) -> float:
        """Primitive moves the rung needs, by counting: it must tell 2**bits cosets apart with actions of
        mean cost ``mean_cost``, and k actions reach at most n_actions**k cosets."""
        if self.n_actions < 2:
            return self.bits
        return self.mean_cost * self.bits / math.log2(self.n_actions)


def generator_periods(model: PointModel, structure: Structure) -> list[list[int]]:
    """For each orbit, the order of each primitive move on that orbit's points."""
    out = []
    for orb in structure.orbits:
        row = []
        for m in range(model.num_moves):
            g = model.gens[m]
            cur = g.copy()
            k = 1
            while not np.array_equal(cur[orb.points], orb.points):
                cur = cur[g]
                k += 1
            row.append(k)
        out.append(row)
    return out


def _slot_perms(model: PointModel, orb: Orbit) -> np.ndarray:
    """(M, n_pieces): the slot each slot is filled from, for every primitive move."""
    return np.stack([orb.slot_of[model.gens[m][orb.ref_pos]] for m in range(model.num_moves)])


def linked_orbits(model: PointModel, structure: Structure) -> list[list[int]]:
    """Groups of orbits whose pieces are permuted identically by every move. They describe
    the same physical pieces (for instance a counter and the colour that travels with it).
    An orbit with a single piece (a piece that only turns in place) is permuted trivially,
    which says nothing, so it is never grouped with another."""
    perms = [_slot_perms(model, o) for o in structure.orbits]
    groups: list[list[int]] = []
    for i, orb in enumerate(structure.orbits):
        placed = False
        for group in groups:
            j = group[0]
            if structure.orbits[j].n_pieces != orb.n_pieces or orb.n_pieces == 1:
                continue
            if _relabel(perms[i], perms[j]) is not None:
                group.append(i)
                placed = True
                break
        if not placed:
            groups.append([i])
    return groups


def _relabel(perm_a: np.ndarray, perm_b: np.ndarray) -> np.ndarray | None:
    """A relabelling pi of slots that turns one slot action into the other (perm_b[m][pi[x]] == pi[perm_a[m][x]]
    for every move m), or None. ``perm_*`` are (moves, slots) arrays: the slot each slot is filled from."""
    n = perm_a.shape[1]
    for image_of_zero in range(n):
        pi = np.full(n, -1)
        pi[0] = image_of_zero
        stack, ok = [0], True
        while stack and ok:
            x = stack.pop()
            for m in range(perm_a.shape[0]):
                y, target = int(perm_a[m][x]), int(perm_b[m][pi[x]])
                if pi[y] < 0:
                    pi[y] = target
                    stack.append(y)
                elif pi[y] != target:
                    ok = False
                    break
        if ok and (pi >= 0).all():
            return pi
    return None


def orbit_links(model: PointModel, structure: Structure) -> dict[int, tuple[int, np.ndarray]]:
    """For every orbit: the first orbit of its linked group and the map from that orbit's slots to its own.
    Linked orbits are permuted identically by every move, so their pieces can only be arranged together."""
    perms = [_slot_perms(model, o) for o in structure.orbits]
    links: dict[int, tuple[int, np.ndarray]] = {}
    for group in linked_orbits(model, structure):
        ref = group[0]
        links[ref] = (ref, np.arange(structure.orbits[ref].n_pieces))
        for o in group[1:]:
            pi = _relabel(perms[ref], perms[o])
            if pi is not None:
                links[o] = (ref, pi)
    return links


def _fixes(features: Sequence[Feature], tables: np.ndarray) -> np.ndarray:
    ok = np.ones(len(tables), dtype=bool)
    for f in features:
        ok &= (f.compute(tables) == f.goal).all(axis=-1)
    return ok


def stabiliser_points(feature: Feature) -> list[int] | None:
    """Points whose pointwise stabiliser is exactly the set of elements that leave ``feature`` at
    its goal value, or None if no such set exists.

    A slot is pinned when only its own home piece shows the goal class there. Fixing the feature
    then fixes the pinned slots' points, and the unpinned pieces fill the unpinned slots. This is
    the whole stabiliser when every unpinned piece shows the goal class in every unpinned slot it
    can reach: the unpinned pieces are look-alikes of one kind (for example empty places), so any
    arrangement of them shows the goal. Otherwise (identical pieces of several colours, say) None.
    """
    orb = feature.orbit
    hits = feature.table == feature.goal[:, None]                   # (slots, labels): shows the goal class
    possible = orb.arrangement[:, :, 0] >= 0
    pinned = [b for b in range(orb.n_pieces) if int(hits[b].sum()) == 1]
    loose = [b for b in range(orb.n_pieces) if int(hits[b].sum()) > 1]
    loose_labels = [int(x) for b in loose for x in orb.pieces[b]]
    for b in loose:
        for c in loose_labels:
            if possible[b, c] and not hits[b, c]:
                return None
    return sorted(int(p) for b in pinned for p in orb.pieces[b])


def forced_all(model: PointModel, fixed: Sequence[Feature], features: Sequence[Feature]) -> list[bool] | None:
    """For each of ``features``: does every element that keeps ``fixed`` at their goal values keep it
    at its goal too? Exact (one pointwise stabiliser in sympy); None when ``stabiliser_points``
    cannot express one of the fixed features."""
    from sympy.combinatorics import Permutation, PermutationGroup

    points: list[int] = []
    for f in fixed:
        p = stabiliser_points(f)
        if p is None:
            return None
        points.extend(p)
    group = PermutationGroup([Permutation(np.argsort(g).tolist()) for g in model.gens])   # image form
    stab = group.pointwise_stabilizer(sorted(set(points))) if points else group
    labels = [np.argsort(np.asarray(k.array_form)) for k in stab.generators]  # the labelled state each produces
    return [all((f.compute(lab) == f.goal).all() for lab in labels) for f in features]


def forced(model: PointModel, fixed: Sequence[Feature], feature: Feature) -> bool | None:
    """``forced_all`` for one feature."""
    out = forced_all(model, fixed, [feature])
    return None if out is None else out[0]


# -- how many cosets a rung has when its table does not fit ------------------------------------------------

def _permanent(a: list[list[int]]) -> int:
    """Permanent of a small square matrix of non-negative integers (Ryser's formula, Gray-code order)."""
    n = len(a)
    if n == 0:
        return 1
    sums = [0] * n
    total, prev = 0, 0
    for k in range(1, 1 << n):
        g = k ^ (k >> 1)
        j = (g ^ prev).bit_length() - 1
        step = 1 if g >> j & 1 else -1
        for i in range(n):
            sums[i] += step * a[i][j]
        prev = g
        prod = 1
        for s in sums:
            prod *= s
            if prod == 0:
                break
        total += prod if (n - bin(g).count("1")) % 2 == 0 else -prod
    return total


def _assignments(m: np.ndarray, max_exact: int = 16) -> int:
    """Ways to give every slot (row) its own piece (column), m[slot, piece] ways each: the permanent of m,
    computed block by block of the slot-piece graph. A block too large for the exact formula contributes the
    product of its row sums, an upper bound, so the coset estimate built on it stays a lower bound."""
    n = m.shape[0]
    parent = list(range(2 * n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for b, p in zip(*np.nonzero(m)):
        parent[find(int(b))] = find(n + int(p))
    blocks: dict[int, tuple[list[int], list[int]]] = {}
    for b in range(n):
        blocks.setdefault(find(b), ([], []))[0].append(b)
    for p in range(n):
        if find(n + p) in blocks:
            blocks[find(n + p)][1].append(p)
    total = 1
    for rows, cols in blocks.values():
        sub = m[np.ix_(rows, cols)].astype(np.int64)
        if len(rows) == len(cols) and sub.min() > 0 and (sub == sub[0, 0]).all():
            total *= math.factorial(len(rows)) * int(sub[0, 0]) ** len(rows)
        elif len(rows) == len(cols) and len(rows) <= max_exact:
            total *= _permanent(sub.tolist())
        else:
            total *= math.prod(int(s) for s in sub.sum(axis=1))
    return total


def _goal_matrix(feats: Sequence[Feature]) -> np.ndarray:
    """m[slot, piece] = ways piece can sit in slot showing the goal value of every feature of one orbit."""
    orb = feats[0].orbit
    labels = orb.points
    hits = orb.arrangement[:, labels, 0] >= 0                         # (slots, labels): the label can sit there
    for f in feats:
        hits &= f.table[:, labels] == f.goal[:, None]                 # ... and shows the goal class
    piece = orb.slot_of[labels]                                       # home piece of each label
    m = np.zeros((orb.n_pieces, orb.n_pieces), dtype=np.int64)
    for b in range(orb.n_pieces):
        np.add.at(m[b], piece[hits[b]], 1)
    return m


def goal_arrangements(features: Sequence[Feature], links: dict[int, tuple[int, np.ndarray]] | None = None) -> int:
    """How many arrangements of the features' pieces show the goal value of every feature: at least the size
    of the goal's stabiliser in any group acting on those pieces.

    Orbits that ``links`` (``orbit_links``) ties together are permuted identically by every move, so one
    arrangement of slots serves them all: they are counted together (a gear's colour and its rotation), not
    as independent arrangements, which would overcount by up to n!."""
    by_orbit: dict[int, list[Feature]] = {}
    for f in features:
        by_orbit.setdefault(f.orbit.index, []).append(f)
    clusters: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}     # reference orbit -> (matrix, slot map)
    for o, feats in by_orbit.items():
        ref, pi = links[o] if links and o in links else (o, np.arange(feats[0].orbit.n_pieces))
        clusters.setdefault(ref, []).append((_goal_matrix(feats), pi))
    total = 1
    for parts in clusters.values():
        joint = None
        for m, pi in parts:
            mapped = m[np.ix_(pi, pi)]                                # in the reference orbit's slots and pieces
            joint = mapped.astype(np.int64) if joint is None else joint * mapped
        total *= _assignments(joint)
    return total


def estimated_bits(features: Sequence[Feature], actions: ActionSet,
                   links: dict[int, tuple[int, np.ndarray]] | None = None) -> float | None:
    """log2 of the coset count, from below, when it is too large for a table: log2 |H| - log2 A (module
    docstring). None if the group order was not measured."""
    if actions.group_order is None:
        return None
    return max(0.0, math.log2(actions.group_order) - math.log2(max(goal_arrangements(features, links), 1)))


# -- the proposal ---------------------------------------------------------------------------------------------

def propose_ladder(model: PointModel, structure: Structure, radius: int = 6, table_budget: int = 10_000_000,
                   n_shortest: int = 12, max_actions: int = 64, bundle: bool = True, merge: bool = True,
                   pool_cap: int = 200_000, max_pairs: int = 4_000_000, open_sample: int = 5000,
                   log: Callable[[str], None] | None = None, _first: Sequence[int] = (),
                   _depth: int = 0) -> tuple[list[RungSpec], dict]:
    """Rungs from the rules above. Returns the rung specifications and the measurements (``report``:
    periods, groups, every scored candidate, the completeness verdicts and the predicted cost)."""
    log = log or (lambda msg: print(msg, flush=True))
    quiet = lambda _msg: None  # noqa: E731
    prim = ActionSet.primitives(model)
    tables, words, _ = ball(model, prim, radius)
    everything = [structure.feature(o.index, "full") for o in structure.orbits]
    links = orbit_links(model, structure)
    helper = Ladder(model, structure, [], mitm_radius=radius, log=quiet)
    helper._ball = (tables, words)

    def stabiliser_words(features: Sequence[Feature]) -> list[Word]:
        """Words within 2R that fix ``features`` and change something visible, shortest first."""
        if not features:
            return [(m,) for m in range(model.num_moves)]
        found = collisions(model, tables, words, features, max_pairs=max_pairs)
        if not found:
            return []
        keys = list(found.keys())
        tabs = np.frombuffer(b"".join(keys), dtype=np.int16).reshape(-1, model.n).astype(np.int32)
        change = ~_fixes(everything, tabs)
        return sorted((found[k] for k, ch in zip(keys, change) if ch), key=lambda w: (len(w), w))

    def full(orbits: Sequence[int]) -> list[Feature]:
        return [structure.feature(o, "full") for o in orbits]

    def prune(feats: list[Feature], base: list[Feature]) -> list[Feature]:
        """Drop features that the others (and ``base``) already force, e.g. the second and third sticker of a
        corner. Kept when that cannot be decided."""
        kept: list[Feature] = []
        for f in feats:
            if base or kept:
                verdict = forced_all(model, base + kept, [f])
                if verdict is not None and verdict[0]:
                    continue
            kept.append(f)
        return kept

    def describe(c: Candidate) -> str:
        size = f"{c.cosets:,}" if c.exact else f"~2^{c.bits:.1f}"
        macros = f"none within {2 * radius}" if math.isinf(c.macro_len) else f"{c.macro_len:.1f} ({c.n_macros:,} words)"
        return f"{c.name:24s} cosets {size:>14s}  bits {c.bits:5.1f}  macros left: {macros:26s} score {c.score:5.2f}"

    # -- rules 1 and 2: order the kinds of piece ----------------------------------------------------------
    periods = generator_periods(model, structure)
    groups = linked_orbits(model, structure)
    report: dict = {"periods": periods, "groups": groups, "steps": []}
    info = []
    for group in groups:
        others = [everything[o] for o in range(len(structure.orbits)) if o not in group]
        pure = stabiliser_words(others) if others else []
        pure_len = len(pure[0]) if pure else math.inf
        info.append((max(max(periods[o]) for o in group), pure_len, group))
    info.sort(key=lambda t: (t[0], -t[1] if not math.isinf(t[1]) else -1e9))     # ascending period; cheapest pure last
    report["group_order"] = [(g, p, pl) for p, pl, g in info]
    for period, pure_len, group in info:
        pure = f"none within {2 * radius}" if math.isinf(pure_len) else f"{pure_len} moves"
        log(f"orbits {group}: generator period {period}, shortest pure macro {pure}")
    remaining = [g for _, _, g in info]

    specs: list[RungSpec] = []
    chosen: list[Candidate] = []
    fixed: list[Feature] = []
    solved: set[int] = set()                       # orbits whose full feature is a rung feature
    pool: list[Word] = [(m,) for m in range(model.num_moves)]

    def done(group: Sequence[int]) -> bool:
        """Every orbit of the kind is a rung's full feature, or forced by the rung features so far."""
        return all(o in solved or forced_all(model, fixed, [everything[o]]) == [True] for o in group)

    def take(c: Candidate) -> None:
        """Emit candidate c as the next rung and hand its pool on, as the ladder builder does."""
        nonlocal pool, fixed
        solved.update(fs[0] for fs in c.features if fs[1] == "full" and len(fs) == 2)
        feats = [structure.feature(*fs) for fs in c.features]
        mode = "table" if c.exact else "learned"
        name = c.name.replace(".", "_").replace("+", "_")
        spec = RungSpec(name, list(c.features), mode=mode, max_actions=600 if mode == "table" else 128)
        specs.append(spec)
        chosen.append(c)
        pool = helper._next_pool(len(specs) - 1, spec, pool, fixed + feats,
                                 SimpleNamespace(table=c.table if c.exact else None, quotient=c.quotient))
        fixed = fixed + feats
        c.table = c.quotient = None                    # the measurements are kept, the tables are not

    # -- rule 0: small kinds of piece first, together ----------------------------------------------------------
    first = list(_first)
    if bundle:
        sizes = []
        for g in remaining:
            q = Quotient(full(g), prim)
            if q.packable:
                t = build_table_auto(q, max_states=table_budget)
                if t.complete:
                    sizes.append((len(t), g))
        for _, g in sorted(sizes, key=lambda x: x[0]):
            trial = prune(full(first + [o for o in g if o not in first]), [])
            q = Quotient(trial, prim)
            if not q.packable or not build_table_auto(q, max_states=table_budget).complete:
                break
            first += [o for o in g if o not in first]
    if first:
        feats = prune(full(first), [])
        q = Quotient(feats, prim)
        t = build_table_auto(q, max_states=table_budget) if q.packable else None
        exact = t is not None and t.complete
        left = stabiliser_words(feats)
        lens = [len(w) for w in left[:n_shortest]]
        c = Candidate(tuple((f.orbit.index, "full") for f in feats), "first",
                      math.log2(len(t)) if exact else (estimated_bits(feats, prim, links) or math.log2(table_budget)),
                      exact, len(t) if exact else None, float(np.mean(lens)) if lens else math.inf, len(left),
                      whole=True, n_actions=len(prim), mean_cost=1.0, quotient=q, table=t if exact else None)
        log(f"first rung, with the primitive moves: orbits {first}")
        log(f"    {describe(c)}  <- chosen")
        report["steps"].append({"group": first, "candidates": [c], "chosen": c.name})
        take(c)
        remaining = [g for g in remaining if not done(g)]

    # -- rule 4: choose each further rung ------------------------------------------------------------------
    singles = [(f.orbit.index, f.name.split(".", 1)[1]) for f in structure.candidate_features()]
    while remaining and pool:
        group = remaining[0]
        sample = pool[:open_sample] + pool[-200:] if len(pool) > open_sample + 200 else pool
        sample_tables = model.compile_many(sample)

        def is_open(feats: Sequence[Feature]) -> bool:
            return bool(len(sample_tables)) and not _fixes(feats, sample_tables).all()

        options: dict[tuple, list] = {}                      # feature specifications -> [name, whole kind]
        for spec in singles:
            if spec[0] in group and is_open([structure.feature(*spec)]):
                options[(spec,)] = [f"orbit{spec[0]}.{spec[1]}", False]
        kinds = [group] + ([remaining[1]] if merge and len(remaining) > 1 else [])
        for k in range(1, len(kinds) + 1):
            orbits = [o for g in kinds[:k] for o in g]
            feats = prune(full(orbits), fixed)
            if (len(orbits) > 1 or k > 1) and feats and is_open(feats):
                fspecs = tuple((f.orbit.index, "full") for f in feats)
                if fspecs in options:                         # pruned down to a candidate already listed
                    options[fspecs][1] = True
                else:
                    options[fspecs] = ["+".join(f"orbit{f.orbit.index}" for f in feats), True]
        if not options:
            remaining.pop(0)
            continue

        scored: list[Candidate] = []
        for fspecs, (name, whole) in options.items():
            feats = [structure.feature(*fs) for fs in fspecs]
            actions = helper._select_actions(RungSpec(name, list(fspecs), max_actions=max_actions, pool_cap=pool_cap),
                                             feats, pool)
            q = Quotient(feats, actions)
            t = build_table_auto(q, max_states=table_budget) if q.packable else None
            exact = t is not None and t.complete
            bits = math.log2(len(t)) if exact else (estimated_bits(feats, actions, links) or math.log2(table_budget))
            left = stabiliser_words(fixed + feats)
            lens = [len(w) for w in left[:n_shortest]]
            scored.append(Candidate(fspecs, name, bits, exact, len(t) if exact else None,
                                    float(np.mean(lens)) if lens else math.inf, len(left), whole=whole,
                                    n_actions=len(actions), mean_cost=float(actions.costs.mean()),
                                    quotient=q, table=t if exact else None))
        scored.sort(key=lambda c: (-c.score, -c.bits))
        best = scored[0]
        if best.score == 0.0:                          # nothing leaves short macros: the largest whole candidate
            wholes = [c for c in scored if c.whole] or [c for c in scored if c.features[0][1] == "full"] or scored
            best = max(wholes, key=lambda c: c.bits)
        log(f"orbits {group}, {len(fixed)} feature(s) fixed, {len(pool):,} allowed moves of length "
            f"{min(map(len, pool))}..{max(map(len, pool))}:")
        for c in scored:
            log(f"    {describe(c)}{'  <- chosen' if c is best else ''}")
        report["steps"].append({"group": group, "candidates": scored, "chosen": best.name})
        for c in scored:
            if c is not best:
                c.table = c.quotient = None
        take(best)
        remaining = [g for g in remaining if not done(g)]

    # -- completeness: an orbit no short word changes may still be unsolved --------------------------------------
    verdicts = forced_all(model, fixed, everything)
    if verdicts is None:
        loose, undecided = [], [o.index for o in structure.orbits]
    else:
        loose = [o.index for o, ok in zip(structure.orbits, verdicts) if not ok]
        undecided = []
    report["not_forced"], report["undecided"] = loose, undecided
    if undecided:
        log(f"orbits {undecided}: identical-looking pieces, so whether the rungs above solve them is not "
            "decided here; building the ladder and the acceptance test check it")
    if loose and _depth < 2:
        log(f"orbits {loose} are not forced by the rungs above (no word of up to {2 * radius} moves reaches them "
            "once those are solved): they join the first rung, and the other rungs are chosen again")
        return propose_ladder(model, structure, radius, table_budget, n_shortest, max_actions, bundle, merge,
                              pool_cap, max_pairs, open_sample, log, _first=list(dict.fromkeys(list(_first) + loose)),
                              _depth=_depth + 1)
    if loose:
        feats = [(o, "full") for o in loose]
        q = Quotient([structure.feature(*fs) for fs in feats], prim)
        table = build_table_auto(q, max_states=table_budget) if q.packable else None
        mode = "table" if table is not None and table.complete else "learned"
        log(f"orbits {loose} are still not forced: a first rung for them, with the primitive moves ({mode})")
        specs.insert(0, RungSpec("first", feats, mode=mode, max_actions=600 if mode == "table" else 128))
    report["predicted_cost"] = sum(c.predicted_cost for c in chosen)
    report["chosen"] = [{"name": s.name, "mode": s.mode, "features": s.features, "bits": round(c.bits, 1),
                         "exact": c.exact, "predicted_cost": round(c.predicted_cost, 1)} for s, c in zip(specs, chosen)]
    log(f"proposed ladder (predicted cost {report['predicted_cost']:.0f} moves):")
    for s, c in zip(specs, chosen):
        log(f"  {s.name:28s} {s.mode:8s} bits {c.bits:5.1f}  features {s.features}")
    return specs, report
