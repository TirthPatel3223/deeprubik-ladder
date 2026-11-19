"""Content fingerprints for cached artefacts.

Every file a build writes (a rung's actions, its table, its network) is stamped with a fingerprint of
everything that determined it. A cache file is reused only when its stamp matches what the current build
would produce, so changing a rung's features or settings can never silently reuse a stale file.
"""
from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np

from core.gather_engine import GatherPuzzle

# Bump when a change to the harvesting or table code changes what a cached file contains.
CACHE_VERSION = 1


def _feed(h: "hashlib._Hash", part: Any) -> None:
    if isinstance(part, np.ndarray):
        h.update(f"nd:{part.dtype.str}:{part.shape}:".encode())
        h.update(np.ascontiguousarray(part).tobytes())
    elif isinstance(part, (list, tuple)):
        h.update(f"seq:{len(part)}:".encode())
        for p in part:
            _feed(h, p)
    elif isinstance(part, dict):
        h.update(f"map:{len(part)}:".encode())
        for k in sorted(part, key=str):
            _feed(h, str(k))
            _feed(h, part[k])
    else:
        h.update(f"{type(part).__name__}:{part!r};".encode())


def digest(*parts: Any) -> str:
    """Short stable hash of numbers, strings, arrays and nested sequences or dicts of them."""
    h = hashlib.sha256()
    for part in parts:
        _feed(h, part)
    return h.hexdigest()[:16]


def puzzle_fingerprint(puzzle: GatherPuzzle) -> str:
    """Fingerprint of a puzzle's move tables and solved state (its name is not part of it)."""
    return digest("puzzle", puzzle.idx.astype(np.int64), puzzle.add.astype(np.int64),
                  puzzle.modulus.astype(np.int64), puzzle.goal.astype(np.int64),
                  puzzle.inverse.astype(np.int64), list(puzzle.move_names))


def feature_key(spec: Sequence) -> tuple:
    """Canonical form of a feature specification (orbit, kind) or (orbit, kind, mask), whether it came
    from Python code (tuples) or from JSON (lists)."""
    orbit, kind = int(spec[0]), str(spec[1])
    if len(spec) > 2 and spec[2] is not None:
        return (orbit, kind, tuple(sorted(int(c) for c in spec[2])))
    return (orbit, kind)
