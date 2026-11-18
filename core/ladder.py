"""Subgroup ladders: build one rung after another, then solve by descending them.

A ladder is an ordered list of rungs. Rung i has

* features  F_i: what this rung puts right (a projection of the state);
* actions   S_i: words that keep F_0 .. F_(i-1) solved, with their costs;
* a heuristic: an exact cost table when the rung's coset space can be enumerated, or a
  trained value network when it cannot (see ``core/davi.py``).

Building is sequential, because the actions of rung i are harvested from what came before:

1. Candidates for S_i are every known word that fixes F_0 .. F_(i-1): all such words of
   length <= 2R found by meet in the middle on a ball of radius R, plus Schreier generators
   of the previous rung's table (cheap ones as macros, random ones as witnesses that the
   candidates generate the whole stabiliser).
2. Two candidates that act identically on F_i are the same move as far as this rung is
   concerned, so only the cheapest is kept. The cheapest ``max_actions`` moves are taken,
   then more are added until they generate the same group on the rung's pieces as *all*
   candidates do. That check is what stops a rung from being silently unsolvable.
3. The rung's quotient puzzle is built and, in table mode, solved exactly.

Solving walks down the rungs. The result is a word in the primitive moves; callers replay
it on the reference simulator before trusting it.

Caching. Every file a build writes is stamped with a fingerprint (``core/fingerprint.py``) of what
determined it: the puzzle's tables, the meet-in-the-middle radius and every earlier rung, then the rung's
own features and settings. A file is reused only when its stamp matches; otherwise it is renamed to
``*.stale-...`` (never deleted) and rebuilt. A learned rung's actions do not affect later rungs, so changing
its action settings leaves the later rungs' files valid. Files written before fingerprints existed carry no
stamp; they are trusted as they are, and the rungs that used them are listed in ``Ladder.legacy``.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from core.coset_table import CosetTable, build_table_auto, table_solve
from core.fingerprint import CACHE_VERSION, digest, feature_key, puzzle_fingerprint
from core.harvest import WordAlgebra, ball, collisions, schreier_elements
from core.perm_model import PointModel, Word
from core.quotient import ActionSet, Quotient
from core.search_schedule import SearchSchedule, SearchTimeout
from core.structure import Feature, Structure

FeatureSpec = tuple            # (orbit, kind) or (orbit, kind, mask)


class RungFailure(LookupError):
    """A rung could not take the state to its goal. ``reason`` is one of

    * ``outside_table``: the state is not in the rung's exact table, so its actions do not reach it
      (a coverage gap: more search cannot help);
    * ``search_exhausted``: every beam search of the schedule ended without reaching the goal;
    * ``timeout``: the deadline passed first.
    """

    def __init__(self, rung: str, reason: str, message: str) -> None:
        super().__init__(message)
        self.rung = rung
        self.reason = reason


@dataclass
class RungSpec:
    name: str
    features: list[FeatureSpec]
    mode: str = "table"                  # "table" (exact) or "learned" (value network)
    max_actions: int = 128               # cheapest distinct moves to start from
    max_cost: int | None = None          # ignore candidates costlier than this
    schreier_cost: int = 40              # budget for cheap Schreier generators passed on
    check_sample: int = 60               # cheapest unused moves tested first, with the witnesses
    check_witnesses: int = 60            # the longest candidates: the random Schreier generators
    check_candidates: int = 1500         # if any of those is missing, how many unused moves to consider adding
    max_bridges: int = 256               # ... and how many of them to keep when they leave the cheap moves' subgroup
    max_states: int = 60_000_000         # table mode: give up beyond this many cosets


@dataclass
class Rung:
    spec: RungSpec
    features: list[Feature]
    actions: ActionSet
    quotient: Quotient
    table: CosetTable | None = None
    heuristic: object | None = None      # set by the learned-rung trainer
    info: dict = field(default_factory=dict)


def _signature(features: Sequence[Feature], table: np.ndarray) -> bytes:
    """How a permutation acts on the rung's feature vectors, as hashable bytes."""
    parts = []
    for f in features:
        src, lut = f.transition(table)
        parts.append(src.tobytes())
        parts.append(lut.tobytes())
    return b"".join(parts)


def _identity_signature(features: Sequence[Feature], n: int) -> bytes:
    return _signature(features, np.arange(n, dtype=np.int32))


def _restricted_perms(model: PointModel, features: Sequence[Feature], tables: np.ndarray) -> list[tuple[int, ...]]:
    """Each table as a permutation of the points of the orbits the features live on."""
    pts = np.unique(np.concatenate([f.orbit.points for f in features]))
    pos = np.full(model.n, -1)
    pos[pts] = np.arange(len(pts))
    return [tuple(pos[t[pts]].tolist()) for t in tables]


def _group_order(perms: Sequence[tuple[int, ...]]) -> int:
    from sympy.combinatorics import Permutation, PermutationGroup

    uniq = list(dict.fromkeys(perms))
    if not uniq:
        return 1
    return int(PermutationGroup([Permutation(list(p)) for p in uniq]).order())


def retire(path: Path, stamp: str | None) -> Path:
    """Move a stale cache file out of the way (``<name>.stale-<stamp>-<time>``) instead of deleting it."""
    base = f"{path.name}.stale-{stamp or 'unstamped'}-{time.strftime('%Y%m%d%H%M%S')}"
    target, k = path.with_name(base), 0
    while target.exists():
        k += 1
        target = path.with_name(f"{base}-{k}")
    path.rename(target)
    return target


class Ladder:
    def __init__(self, model: PointModel, structure: Structure, specs: Sequence[RungSpec],
                 mitm_radius: int = 7, log: Callable[[str], None] | None = None) -> None:
        self.model = model
        self.structure = structure
        self.specs = list(specs)
        self.mitm_radius = mitm_radius
        self.log = log if log is not None else (lambda msg: print(msg, flush=True))
        self.alg = WordAlgebra(model)
        self.rungs: list[Rung] = []
        self.legacy: list[str] = []          # rungs loaded from cache files that carry no fingerprint
        self._ball: tuple | None = None

    # -- fingerprints ----------------------------------------------------------------
    def rung_fingerprints(self) -> list[dict]:
        """For every rung of the specification, the fingerprints of its candidate pool, its actions, its
        table (None for a learned rung) and the pool it hands to the next rung."""
        out: list[dict] = []
        pool = digest("pool0", CACHE_VERSION, puzzle_fingerprint(self.model.puzzle), self.mitm_radius)
        for i, spec in enumerate(self.specs):
            feats = [feature_key(fs) for fs in spec.features]
            actions = digest("actions", pool, i, feats, spec.max_actions, spec.max_cost, spec.check_sample,
                             spec.check_witnesses, spec.check_candidates, spec.max_bridges)
            table = digest("table", actions, spec.max_states) if spec.mode == "table" else None
            # The next pool uses this rung's actions only through the Schreier words of its table.
            nxt = digest("pool", pool, i, feats, spec.mode, (actions, spec.schreier_cost) if spec.mode == "table" else None)
            out.append({"pool": pool, "actions": actions, "table": table, "next_pool": nxt})
            pool = nxt
        return out

    def fingerprint(self) -> str:
        """Fingerprint of the built ladder as a solver: every rung's actions and table, and how far each
        network has been trained. Two solves with equal fingerprints use the same heuristics."""
        parts = []
        for rung in self.rungs:
            fp = rung.info.get("fingerprint", {})
            h = rung.heuristic
            parts.append((fp.get("actions"), fp.get("table"), None if h is None else (h.update, h.itr)))
        return digest("ladder", parts)

    # -- features ----------------------------------------------------------------
    def _features(self, spec: RungSpec) -> list[Feature]:
        return [self.structure.feature(*fs) for fs in spec.features]

    def _fixes(self, features: Sequence[Feature], tables: np.ndarray) -> np.ndarray:
        """Which permutations leave the goal value of every feature unchanged."""
        ok = np.ones(len(tables), dtype=bool)
        for f in features:
            ok &= (f.compute(tables) == f.goal).all(axis=-1)
        return ok

    # -- building ----------------------------------------------------------------
    def _primitive_ball(self):
        if self._ball is None:
            t0 = time.time()
            prim = ActionSet.primitives(self.model)
            tables, words, _ = ball(self.model, prim, self.mitm_radius)
            self._ball = (tables, words)
            self.log(f"primitive ball of radius {self.mitm_radius}: {len(tables):,} elements ({time.time() - t0:.1f} s)")
        return self._ball

    def _select_actions(self, spec: RungSpec, feats: list[Feature], pool: list[Word]) -> ActionSet:
        """Cheapest distinct moves of the rung, completed so that they generate what the pool does."""
        model = self.model
        ident_sig = _identity_signature(feats, model.n)
        by_sig: dict[bytes, Word] = {}
        for w in sorted(pool, key=len):
            if spec.max_cost is not None and len(w) > spec.max_cost:
                break
            for cand in (w, self.alg.invert(w)):
                sig = _signature(feats, model.compile(cand))
                if sig != ident_sig and sig not in by_sig:
                    by_sig[sig] = cand
        ordered = sorted(by_sig.values(), key=lambda w: (len(w), w))
        if not ordered:
            raise RuntimeError(f"rung {spec.name}: no candidate move changes its features")

        chosen = ordered[:spec.max_actions]
        rest = ordered[spec.max_actions:]
        # Generation check on the rung's own pieces. Every remaining candidate that the
        # chosen moves cannot already produce is added. The longest candidates are the
        # random Schreier generators, which generate the whole stabiliser with high
        # probability, so they are always tested.
        from sympy.combinatorics import Permutation, PermutationGroup

        def group_of(words: Sequence[Word]) -> "PermutationGroup":
            perms = _restricted_perms(model, feats, model.compile_many(words))
            return PermutationGroup([Permutation(list(p)) for p in dict.fromkeys(perms)])

        def outside(group: "PermutationGroup", w: Word) -> bool:
            perm = _restricted_perms(model, feats, model.compile(w)[None, :])[0]
            return not group.contains(Permutation(list(perm)))

        group = group_of(chosen)
        witnesses = rest[spec.check_sample:][-spec.check_witnesses:]
        added = 0
        if any(outside(group, w) for w in rest[:spec.check_sample] + witnesses):
            # The cheapest moves generate only a subgroup. One move from outside it would
            # restore generation, but then that one move is the only way between the
            # subgroup's cosets: random walks almost never take it, so a value network is
            # trained on a sliver of the rung, and a search has to find it again and again.
            # So every cheap candidate outside the subgroup is kept (up to max_bridges).
            capped = group
            capped_order = int(capped.order())
            bridges = [w for w in rest[:spec.check_candidates] if outside(capped, w)][:spec.max_bridges]
            chosen.extend(bridges)
            added += len(bridges)
            group = group_of(chosen)
            # then whatever is still missing, cheapest first
            for w in rest[:spec.check_candidates] + witnesses:
                if outside(group, w):
                    chosen.append(w)
                    added += 1
                    group = group_of(chosen)
            self.log(f"  the {spec.max_actions} cheapest moves reach only 1 / {int(group.order()) // capped_order:,} of the group; "
                     f"kept {len(bridges)} of the cheapest moves that leave that subgroup")
        order = int(group.order())
        # inverse closure (an inverse may have been cut off by the cap)
        have = {_signature(feats, model.compile(w)) for w in chosen}
        for w in list(chosen):
            inv = self.alg.invert(w)
            sig = _signature(feats, model.compile(inv))
            if sig not in have and sig != ident_sig:
                have.add(sig)
                chosen.append(inv)
        chosen.sort(key=lambda w: (len(w), w))
        actions = ActionSet.from_words(model, chosen)
        # inverses as moves of this rung (two different words can be the same move here)
        sig_index = {_signature(feats, actions.tables[a]): a for a in range(len(actions))}
        for a in range(len(actions)):
            inv_sig = _signature(feats, model.compile(self.alg.invert(actions.words[a])))
            actions.inverse[a] = sig_index[inv_sig]
        self.log(f"  actions: {len(actions)} of {len(ordered)} distinct moves ({added} added for generation), "
                 f"costs {int(actions.costs.min())}..{int(actions.costs.max())}; "
                 f"group on the rung's pieces: 2^{np.log2(float(order)):.2f}")
        return actions

    def build(self, cache_dir: str | Path | None = None, upto: int | None = None) -> "Ladder":
        cache = Path(cache_dir) if cache_dir is not None else None
        fps = self.rung_fingerprints()
        pool: list[Word] = [(m,) for m in range(self.model.num_moves)]
        fixed: list[Feature] = []                     # features of the rungs already built
        self.rungs = []
        self.legacy = []
        for i, spec in enumerate(self.specs[:upto]):
            rung, pool = self._build_rung(i, spec, pool, fixed, cache, fps[i])
            fixed = fixed + rung.features
            self.rungs.append(rung)
        return self

    def _build_rung(self, i: int, spec: RungSpec, pool: list[Word], fixed: list[Feature], cache: Path | None,
                    fp: dict) -> tuple[Rung, list[Word]]:
        """Build rung i from the candidates ``pool`` (words that keep ``fixed`` solved), reusing cache files
        whose fingerprints match ``fp``. Returns the rung and the candidate pool for the next rung."""
        model = self.model
        t0 = time.time()
        feats = self._features(spec)
        self.log(f"rung {i} [{spec.name}]  features {[f.name for f in feats]}  mode {spec.mode}")
        stem = f"rung{i:02d}_{spec.name}"
        words_file = cache / f"{stem}.json" if cache else None
        table_file = cache / f"{stem}.npz" if cache else None
        net_file = cache / f"{stem}.pt" if cache else None

        saved = json.loads(words_file.read_text()) if words_file is not None and words_file.exists() else None
        legacy = saved is not None and "fingerprint" not in saved
        stamp = (saved or {}).get("fingerprint") or {}
        actions: ActionSet | None = None
        next_pool: list[Word] | None = None
        if saved is not None and (legacy or stamp.get("actions") == fp["actions"]):
            actions = ActionSet.from_words(model, [tuple(w) for w in saved["words"]])
            actions.inverse = np.array(saved["inverse"], dtype=np.int64)
            self.log(f"  loaded {len(actions)} actions from {words_file.name}"
                     + (" (older cache file without a fingerprint: trusted as it is)" if legacy else ""))
        if saved is not None and (legacy or stamp.get("next_pool") == fp["next_pool"]):
            next_pool = [tuple(w) for w in saved["next_pool"]]
        if legacy:
            self.legacy.append(spec.name)
        if actions is None:
            if saved is not None:
                self.log(f"  {words_file.name} was built for other settings: set aside as {retire(words_file, stamp.get('actions')).name}")
                saved = None
            actions = self._select_actions(spec, feats, pool)
            self.log(f"  (selection took {time.time() - t0:.1f} s)")

        quotient = Quotient(feats, actions, name=spec.name)
        rung = Rung(spec=spec, features=feats, actions=actions, quotient=quotient)
        rung.info["fingerprint"] = dict(fp)
        rung.info["legacy"] = legacy

        if spec.mode == "learned":
            if net_file is not None and net_file.exists():
                from core.davi import RungValue

                net_stamp = RungValue.read_fingerprint(net_file)
                if net_stamp == fp["actions"] or (net_stamp is None and legacy):
                    rung.heuristic = RungValue.from_checkpoint(quotient, net_file)
                    rung.heuristic.fingerprint = fp["actions"]
                    self.log(f"  loaded value network from {net_file.name} ({rung.heuristic.update} updates)")
                else:
                    self.log(f"  {net_file.name} was trained for other actions: set aside as {retire(net_file, net_stamp).name}")
            if rung.heuristic is None:
                self.log("  no trained value network yet (scripts/train_rung.py or scripts/auto.py)")
        if spec.mode == "table":
            table = None
            if table_file is not None and table_file.exists():
                table_stamp = CosetTable.read_fingerprint(table_file)
                if table_stamp == fp["table"] or (table_stamp is None and legacy):
                    table = CosetTable.load(table_file)
                else:
                    self.log(f"  {table_file.name} was built for other actions: set aside as {retire(table_file, table_stamp).name}")
            if table is None:
                table = build_table_auto(quotient, max_states=spec.max_states)
                table.fingerprint = fp["table"]
                if table_file is not None and table.complete:
                    table.save(table_file)
            rung.table = table
            if not table.complete:
                raise RuntimeError(f"rung {spec.name}: more than {spec.max_states:,} cosets; split it or use mode='learned'")
            rung.info["cosets"] = len(table)
            rung.info["max_cost"] = int(table.dist.max())
            rung.info["mean_cost"] = float(table.dist.mean())
            self.log(f"  table: {len(table):,} cosets, worst {rung.info['max_cost']}, "
                     f"mean {rung.info['mean_cost']:.1f} moves")

        if next_pool is None:
            next_pool = self._next_pool(i, spec, pool, fixed + feats, rung)
        if words_file is not None and not legacy and (saved is None or stamp != {"actions": fp["actions"], "next_pool": fp["next_pool"]}):
            if words_file.exists():
                retire(words_file, stamp.get("actions"))
            words_file.parent.mkdir(parents=True, exist_ok=True)
            words_file.write_text(json.dumps({
                "name": spec.name, "fingerprint": {"actions": fp["actions"], "next_pool": fp["next_pool"]},
                "words": [list(w) for w in actions.words], "inverse": actions.inverse.tolist(),
                "next_pool": [list(w) for w in next_pool]}))
        rung.info["seconds"] = time.time() - t0
        self.log(f"  done in {rung.info['seconds']:.1f} s")
        return rung, next_pool

    def _next_pool(self, i: int, spec: RungSpec, pool: list[Word], fixed: list[Feature], rung: Rung) -> list[Word]:
        """Candidates for the next rung: every known word that keeps ``fixed`` solved."""
        model = self.model
        found: dict[bytes, Word] = {}

        def offer(words: Sequence[Word]) -> None:
            if not len(words):
                return
            tabs = model.compile_many(words)
            ok = self._fixes(fixed, tabs)
            for w, t, good in zip(words, tabs, ok):
                if good:
                    key = t.tobytes()
                    if key not in found or len(w) < len(found[key]):
                        found[key] = w

        offer([w for w in pool if len(w) > 0])
        b_tables, b_words = self._primitive_ball()
        mitm = collisions(model, b_tables, b_words, fixed)
        for key16, w in mitm.items():
            key = np.frombuffer(key16, dtype=np.int16).astype(np.int32).tobytes()
            if key not in found or len(w) < len(found[key]):
                found[key] = w
        n_mitm = len(mitm)
        n_sch = 0
        if rung.table is not None:
            sch = schreier_elements(model, rung.quotient, rung.table, max_cost=spec.schreier_cost,
                                    max_count=20_000, n_random=40, seed=i)
            n_sch = len(sch)
            offer(sch)
        found.pop(model.identity().tobytes(), None)
        next_pool = sorted(found.values(), key=lambda w: (len(w), w))
        self.log(f"  next-rung candidates: {len(next_pool):,} (meet in the middle {n_mitm:,}, Schreier {n_sch:,})")
        return next_pool

    # -- solving -----------------------------------------------------------------
    def descend(self, lab: np.ndarray, upto: int | None = None, beam: int | None = None,
                weight: float = 0.3, deadline: float | None = None,
                schedule: SearchSchedule | None = None) -> tuple[np.ndarray, list[Word], list[int]]:
        """Solve the first ``upto`` rungs (all of them by default) of a labelled state.

        Returns the labels afterwards, the words applied and the cost spent per rung. With
        ``upto=i`` the result projected on rung i is a state exactly as the solver meets it,
        which is what a learned rung has to be tested on.
        ``beam`` is the beam width used on the learned rungs (None: each rung's default), or
        ``schedule`` the full list of searches they try. ``deadline`` (a ``time.time()`` value) is shared
        by all rungs. Raises ``RungFailure`` naming the rung and the reason when one cannot be solved."""
        lab = np.asarray(lab, dtype=np.int32)
        pieces: list[Word] = []
        per_rung: list[int] = []
        for rung in self.rungs[:upto]:
            q = rung.quotient
            name = rung.spec.name
            vec = q.project(lab)
            if rung.table is not None:
                acts = table_solve(q, rung.table, vec)
                if acts is None:
                    raise RungFailure(name, "outside_table",
                                      f"rung {name}: state is outside the table (its actions do not generate enough)")
            elif rung.heuristic is not None:
                try:
                    acts = rung.heuristic.solve(vec, width=beam, weight=weight, deadline=deadline, schedule=schedule)
                except SearchTimeout:
                    raise RungFailure(name, "timeout", f"rung {name}: out of time before a solution was found") from None
                if acts is None:
                    raise RungFailure(name, "search_exhausted", f"rung {name}: search found no solution")
            else:
                raise RuntimeError(f"rung {name} has no table and no trained heuristic")
            cost = 0
            for a in acts:
                lab = lab[rung.actions.tables[a]]
                pieces.append(rung.actions.words[a])
                cost += int(rung.actions.costs[a])
            per_rung.append(cost)
        return lab, pieces, per_rung

    def solve_labels(self, lab: np.ndarray, beam: int | None = None, weight: float = 0.3,
                     deadline: float | None = None, schedule: SearchSchedule | None = None) -> tuple[Word, list[int]]:
        """Solve a labelled state. Returns the reduced primitive word (turns merged across rungs, and
        across commuting moves) and the cost spent per rung."""
        _, pieces, per_rung = self.descend(lab, beam=beam, weight=weight, deadline=deadline, schedule=schedule)
        return self.alg.simplify(self.alg.concat(*pieces)), per_rung

    def solve(self, state: np.ndarray, beam: int | None = None, weight: float = 0.3,
              time_limit: float | None = None, schedule: SearchSchedule | None = None) -> tuple[Word, list[int]]:
        """Solve a visible state of the underlying GatherPuzzle, within ``time_limit`` seconds if given.
        Raises ``RungFailure`` (a ``LookupError``) naming the rung that could not be solved and why."""
        deadline = None if time_limit is None else time.time() + time_limit
        return self.solve_labels(self.structure.decode(state), beam=beam, weight=weight, deadline=deadline,
                                 schedule=schedule)

    def ready(self) -> bool:
        """True when every rung of the specification is built and has a table or a trained network."""
        return len(self.rungs) == len(self.specs) and all(r.table is not None or r.heuristic is not None for r in self.rungs)
