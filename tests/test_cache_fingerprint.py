"""Cache files are reused only when they were built for the same puzzle, radius and rung settings."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import torch

from core.coset_table import CosetTable
from core.davi import DaviConfig, RungValue
from core.fingerprint import digest, feature_key, puzzle_fingerprint
from core.ladder import Ladder, RungSpec
from puzzles.toy.two_face_corners import LADDER

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _quiet(_msg: str) -> None:
    pass


def _build(toy, specs, cache=None, log=_quiet) -> Ladder:
    _, model, st = toy
    return Ladder(model, st, specs, mitm_radius=4, log=log).build(cache_dir=cache)


def _stale(path) -> list[str]:
    return sorted(p.name for p in path.iterdir() if ".stale-" in p.name)


def test_digest_is_stable_and_canonical(toy) -> None:
    puzzle, _, _ = toy
    assert digest(1, "a", (2, 3)) == digest(1, "a", [2, 3])
    assert digest(np.arange(3)) != digest(np.arange(3).astype(np.int8))
    assert feature_key([2, "full", [1, 2]]) == feature_key((2, "full", (2, 1))) == (2, "full", (1, 2))
    other = replace(puzzle, goal=puzzle.goal[::-1].copy())
    assert puzzle_fingerprint(puzzle) == puzzle_fingerprint(replace(puzzle, name="renamed"))
    assert puzzle_fingerprint(puzzle) != puzzle_fingerprint(other)


def test_matching_cache_is_reused_without_rebuilding(toy, tmp_path) -> None:
    _build(toy, LADDER, tmp_path)
    lines: list[str] = []
    again = _build(toy, LADDER, tmp_path, log=lines.append)
    assert sum("loaded" in line for line in lines) == 2
    assert not any("selection took" in line for line in lines)
    assert again.legacy == [] and _stale(tmp_path) == []
    saved = json.loads((tmp_path / "rung00_place.json").read_text())
    assert saved["fingerprint"]["actions"] == again.rungs[0].info["fingerprint"]["actions"]


def test_changed_feature_under_the_same_name_is_rebuilt(toy, tmp_path) -> None:
    """The bug this fixes: a rung renamed in place used to load the old rung's table."""
    first = _build(toy, LADDER, tmp_path)
    assert len(first.rungs[0].table) == 120
    other = [replace(LADDER[0], features=[(0, "full")])]
    rebuilt = _build(toy, other, tmp_path)
    assert len(rebuilt.rungs[0].table) == 29_160
    assert len(_build(toy, other).rungs[0].table) == 29_160                   # what a fresh build gives
    stale = _stale(tmp_path)
    assert any(n.startswith("rung00_place.json.stale-") for n in stale)        # set aside, not deleted
    assert any(n.startswith("rung00_place.npz.stale-") for n in stale)


def test_cache_from_another_radius_is_not_used(toy, tmp_path) -> None:
    _, model, st = toy
    Ladder(model, st, LADDER, mitm_radius=4, log=_quiet).build(cache_dir=tmp_path)
    lines: list[str] = []
    Ladder(model, st, LADDER, mitm_radius=3, log=lines.append).build(cache_dir=tmp_path)
    assert not any("loaded" in line for line in lines)


def test_files_without_a_fingerprint_are_trusted(toy, tmp_path) -> None:
    """Caches written before fingerprints existed keep working (the Maltese and Tangram networks)."""
    _build(toy, LADDER, tmp_path)
    for i, spec in enumerate(LADDER):
        stem = tmp_path / f"rung{i:02d}_{spec.name}"
        saved = json.loads(stem.with_suffix(".json").read_text())
        del saved["fingerprint"]
        stem.with_suffix(".json").write_text(json.dumps(saved))
        table = CosetTable.load(stem.with_suffix(".npz"))
        table.fingerprint = None
        table.save(stem.with_suffix(".npz"))
    lines: list[str] = []
    again = _build(toy, LADDER, tmp_path, log=lines.append)
    assert again.legacy == ["place", "twist"]
    assert not any("selection took" in line for line in lines) and _stale(tmp_path) == []
    assert "fingerprint" not in json.loads((tmp_path / "rung00_place.json").read_text())   # left as it was


def test_learned_rung_action_settings_leave_later_rungs_valid(toy) -> None:
    _, model, st = toy
    learned = [replace(LADDER[0], mode="learned"), LADDER[1]]
    more = [replace(learned[0], max_actions=8), LADDER[1]]
    a = Ladder(model, st, learned, mitm_radius=4, log=_quiet).rung_fingerprints()
    b = Ladder(model, st, more, mitm_radius=4, log=_quiet).rung_fingerprints()
    assert a[0]["actions"] != b[0]["actions"] and a[1] == b[1]
    # a table rung hands its Schreier words on, so its action settings do matter downstream
    table_more = [replace(LADDER[0], max_actions=8), LADDER[1]]
    c = Ladder(model, st, LADDER, mitm_radius=4, log=_quiet).rung_fingerprints()
    d = Ladder(model, st, table_more, mitm_radius=4, log=_quiet).rung_fingerprints()
    assert c[1]["pool"] != d[1]["pool"]


def test_network_trained_for_other_actions_is_set_aside(toy, tmp_path) -> None:
    specs = [RungSpec("twist", [(0, "full")], mode="learned", max_actions=4)]
    ladder = _build(toy, specs, tmp_path)
    rung = ladder.rungs[0]
    value = RungValue(rung.quotient, DaviConfig(res_dim=16, num_blocks=1), device=DEVICE)
    value.fingerprint = rung.info["fingerprint"]["actions"]
    value.save(tmp_path / "rung00_twist.pt")
    assert _build(toy, specs, tmp_path).rungs[0].heuristic is not None    # trained for these actions: loaded

    value.fingerprint = "0123456789abcdef"                                 # trained for some other action set
    value.save(tmp_path / "rung00_twist.pt")
    assert _build(toy, specs, tmp_path).rungs[0].heuristic is None
    assert any(n.startswith("rung00_twist.pt.stale-0123456789abcdef") for n in _stale(tmp_path))
