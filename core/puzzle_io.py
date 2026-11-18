"""Puzzles read from files, so a new puzzle needs no code.

Gather tables (``.npz``, or ``.json`` with the same keys as lists): the arrays of a ``GatherPuzzle``.

    idx         (M, D) int   required   a move m gives next[i] = (state[idx[m, i]] + add[m, i]) mod modulus[i]
    goal        (D,)   int   required   the solved state, values 0..255
    add         (M, D) int   optional   default 0 everywhere (coordinates that are only moved, like colours)
    modulus     (D,)   int   optional   default 256; required wherever ``add`` is not 0
    move_names  (M,)   str   optional   default m0, m1, ...
    inverse     (M,)   int   optional   the move that undoes each move; checked if given, computed if not
    name               str   optional   default: the file name

Each row of ``idx`` must be a permutation (every move is legal in every state). A move whose inverse is
not among the moves gets its inverse appended, named with a prime (``X`` -> ``X'``), so the move set is
closed under inverses as the solver requires.

A 3D model (``puzzle-geometry/1`` JSON) is compiled by ``core/geometry.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.gather_engine import GatherPuzzle


class PuzzleFileError(ValueError):
    """The file does not describe a valid puzzle."""


def _undo(idx: np.ndarray, add: np.ndarray, modulus: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Gather table and addend of the move that undoes move m."""
    inv_idx = np.argsort(idx[m])
    inv_add = (-add[m][inv_idx]) % modulus
    return inv_idx, inv_add


def _undoes(idx: np.ndarray, add: np.ndarray, modulus: np.ndarray, a: int, b: int) -> bool:
    """Does move b, applied after move a, give back every state?"""
    after = idx[a][idx[b]]
    return bool(np.array_equal(after, np.arange(idx.shape[1]))
                and np.all((add[a][idx[b]] + add[b]) % modulus == 0))


def _prime(name: str, taken: set[str]) -> str:
    candidates = [name[:-1]] if name.endswith("'") else []
    candidates += [name + "'", name + "_inv"]
    for c in candidates:
        if c and c not in taken:
            return c
    k = 2
    while f"{name}_inv{k}" in taken:
        k += 1
    return f"{name}_inv{k}"


def complete_inverses(idx: np.ndarray, add: np.ndarray, modulus: np.ndarray,
                      names: list[str]) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, list[str]]:
    """The inverse of every move, appending the missing ones. Returns idx, add, names, inverse and the names
    of the appended moves."""
    idx, add, names = idx.copy(), add.copy(), list(names)
    keys = {(idx[m].tobytes(), add[m].tobytes()): m for m in range(len(names))}
    added: list[str] = []
    m = 0
    while m < len(names):
        inv_idx, inv_add = _undo(idx, add, modulus, m)
        key = (inv_idx.tobytes(), inv_add.tobytes())
        if key not in keys:
            name = _prime(names[m], set(names))
            idx = np.vstack([idx, inv_idx[None]])
            add = np.vstack([add, inv_add[None]])
            names.append(name)
            added.append(name)
            keys[key] = len(names) - 1
        m += 1
    inverse = np.array([keys[tuple(a.tobytes() for a in _undo(idx, add, modulus, m))] for m in range(len(names))],
                       dtype=np.intp)
    return idx, add, names, inverse, added


def puzzle_from_arrays(data: dict, default_name: str = "puzzle") -> tuple[GatherPuzzle, list[str]]:
    """Validate the arrays of a gather-table file and make the puzzle. Returns it and the names of any
    inverse moves that had to be appended."""
    if "idx" not in data or "goal" not in data:
        raise PuzzleFileError("a puzzle file needs at least 'idx' (one gather table per move) and 'goal'")
    idx = np.asarray(data["idx"], dtype=np.int64)
    goal = np.asarray(data["goal"], dtype=np.int64)
    if idx.ndim != 2 or goal.ndim != 1 or idx.shape[1] != goal.shape[0]:
        raise PuzzleFileError(f"'idx' must be (moves, D) and 'goal' (D,); got {idx.shape} and {goal.shape}")
    n_moves, d = idx.shape
    if n_moves == 0:
        raise PuzzleFileError("the puzzle has no moves")
    for m in range(n_moves):
        if not np.array_equal(np.sort(idx[m]), np.arange(d)):
            raise PuzzleFileError(f"move {m}: 'idx' row is not a permutation of 0..{d - 1} "
                                  "(every move must be legal in every state)")
    if goal.min() < 0 or goal.max() > 255:
        raise PuzzleFileError("'goal' values must be 0..255")
    add = np.zeros_like(idx) if data.get("add") is None else np.asarray(data["add"], dtype=np.int64)
    if add.shape != idx.shape:
        raise PuzzleFileError(f"'add' must have the shape of 'idx' {idx.shape}; got {add.shape}")
    if data.get("modulus") is None:
        if np.any(add != 0):
            raise PuzzleFileError("'modulus' is required when 'add' is not zero")
        modulus = np.full(d, 256, dtype=np.int64)
    else:
        modulus = np.asarray(data["modulus"], dtype=np.int64)
    if modulus.shape != (d,) or modulus.min() < 1 or modulus.max() > 256:
        raise PuzzleFileError("'modulus' must be (D,) with values 1..256")
    if np.any(goal >= modulus):
        raise PuzzleFileError("each 'goal' value must be smaller than its coordinate's modulus")
    names = [f"m{m}" for m in range(n_moves)] if data.get("move_names") is None else [str(x) for x in data["move_names"]]
    if len(names) != n_moves or len(set(names)) != n_moves:
        raise PuzzleFileError("'move_names' must give one distinct name per move")
    keys = {(idx[m].tobytes(), (add[m] % modulus).tobytes()) for m in range(n_moves)}
    if len(keys) != n_moves:
        raise PuzzleFileError("two moves have the same effect; give each move once")
    add = add % modulus

    if data.get("inverse") is not None:
        inverse = np.asarray(data["inverse"], dtype=np.intp)
        if inverse.shape != (n_moves,) or inverse.min() < 0 or inverse.max() >= n_moves:
            raise PuzzleFileError("'inverse' must give one move index per move")
        for m in range(n_moves):
            if not _undoes(idx, add, modulus, m, int(inverse[m])):
                raise PuzzleFileError(f"'inverse' says {names[inverse[m]]} undoes {names[m]}, but it does not")
        added: list[str] = []
    else:
        idx, add, names, inverse, added = complete_inverses(idx, add, modulus, names)
    puzzle = GatherPuzzle(
        name=str(data.get("name") or default_name), move_names=tuple(names), idx=idx.astype(np.intp),
        add=add.astype(np.int16), modulus=modulus.astype(np.int16), goal=goal.astype(np.uint8), inverse=inverse)
    return puzzle, added


def load_tables(path: str | Path) -> tuple[GatherPuzzle, list[str]]:
    """A puzzle from a ``.npz`` or ``.json`` gather-table file (see the module docstring)."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        for key in ("name",):
            if key in data:
                data[key] = str(data[key])
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
    else:
        raise PuzzleFileError(f"{path}: expected a .npz or .json gather-table file")
    return puzzle_from_arrays(data, default_name=path.stem)


def save_tables(puzzle: GatherPuzzle, path: str | Path) -> None:
    """Write a puzzle as a gather-table file that ``load_tables`` reads back unchanged."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(name=puzzle.name, move_names=list(puzzle.move_names), idx=puzzle.idx, add=puzzle.add,
                  modulus=puzzle.modulus, goal=puzzle.goal, inverse=puzzle.inverse)
    if path.suffix.lower() == ".npz":
        np.savez(path, **{k: np.asarray(v) for k, v in arrays.items()})
    elif path.suffix.lower() == ".json":
        path.write_text(json.dumps({k: (np.asarray(v).tolist() if not isinstance(v, str) else v)
                                    for k, v in arrays.items()}))
    else:
        raise PuzzleFileError(f"{path}: use a .npz or .json file name")


def load_geometry_puzzle(path: str | Path) -> GatherPuzzle:
    """A puzzle compiled from its 3D model (``puzzle-geometry/1``), with every check of ``core/geometry.py``."""
    from core.geometry import compile_geometry, load_geometry

    return compile_geometry(load_geometry(path)).puzzle
