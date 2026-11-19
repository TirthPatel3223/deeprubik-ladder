"""Every puzzle the scripts know: its move tables, ladders, notation and defaults.

    from puzzles.registry import get_puzzle
    entry = get_puzzle("tangram_extreme")
    puzzle = entry.build()
    specs = entry.ladders()["learned"]

A puzzle can also be given as a file, with no code (``--tables`` or ``--geometry``, see ``entry_from_args``):
its ladder is then designed automatically by ``scripts/auto.py``.

To add a puzzle permanently, write its ``GatherPuzzle`` (by hand, like the Maltese Gear Cube, or compiled
from a 3D model with ``core/geometry.py``, like the Tangram Cube) and, optionally, hand-written ladders, then
add an entry here.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from core.gather_engine import GatherPuzzle
from core.harvest import WordAlgebra
from core.notation import Notation, PlainNotation, face_turns, steps, turns

LADDER_ROOT = Path("data/ladders")


@dataclass(frozen=True)
class PuzzleEntry:
    name: str
    title: str
    build: Callable[[], GatherPuzzle]
    ladders: Callable[[], dict]                   # ladder name -> list of RungSpec (imported lazily)
    notation: Callable[[WordAlgebra], Notation]
    default_ladder: str = "learned"
    scramble_len: int = 200
    legacy_cache: bool = False                    # caches directly under data/ladders/ (the Maltese runs)
    geometry_variant: str | None = None           # Tangram: the variant in the 3D model's notation

    def cache_dir(self, ladder: str, radius: int) -> Path:
        leaf = f"{ladder}_r{radius}"
        return LADDER_ROOT / leaf if self.legacy_cache else LADDER_ROOT / self.name / leaf


_MALTESE_LEGEND = ("`U`, `L`, `F` are clockwise quarter turns and `'` means counter-clockwise. A number repeats "
                   "the turn, so `U4` is four quarter turns of U and `L'3` is three counter-clockwise quarter "
                   "turns of L.")

_FLIPS = "`F`, `R` and `L` are 180° flips about three edges; each is its own inverse."

_TANGRAM_LEGENDS = {
    "extreme": ("`U` and `D` turn the top and bottom corner layers 60° clockwise, as seen from that corner, and `'` "
                "means anticlockwise. A number repeats the turn, so `U3` is a half turn and `U2'` is 120° "
                "anticlockwise. " + _FLIPS + " This is the notation of the puzzle's simulator."),
    "original": ("`U` and `D` turn the top and bottom corner layers 120° clockwise, as seen from that corner, and "
                 "`'` means anticlockwise. " + _FLIPS + " This is the notation of the puzzle's simulator."),
}


def _maltese_notation(alg: WordAlgebra) -> Notation:
    return Notation(alg, prime_then_count=True, metrics=[("quarter turns", steps), ("face turns", face_turns)],
                    legend=_MALTESE_LEGEND)


def _tangram_notation(variant: str) -> Callable[[WordAlgebra], Notation]:
    def notation(alg: WordAlgebra) -> Notation:
        return Notation(alg, prime_then_count=False, metrics=[("steps", steps), ("turns", turns)],
                        legend=_TANGRAM_LEGENDS[variant])
    return notation


def _maltese_build() -> GatherPuzzle:
    from puzzles.maltese_gear.puzzle import build_maltese_gear_puzzle
    return build_maltese_gear_puzzle()


def _maltese_ladders() -> dict:
    from puzzles.maltese_gear.ladder import LADDERS
    return LADDERS


def _toy_build() -> GatherPuzzle:
    from puzzles.toy.two_face_corners import build_two_face_corners
    return build_two_face_corners()


def _toy_ladders() -> dict:
    from puzzles.toy.two_face_corners import LADDER
    return {"table": LADDER}


def _tangram_build(variant: str) -> Callable[[], GatherPuzzle]:
    def build() -> GatherPuzzle:
        from puzzles.tangram.puzzle import build_tangram_puzzle
        return build_tangram_puzzle(variant)
    return build


def _tangram_ladders(variant: str) -> Callable[[], dict]:
    def ladders() -> dict:
        from puzzles.tangram import ladder
        return ladder.LADDERS if variant == "extreme" else ladder.ORIGINAL_LADDERS
    return ladders


PUZZLES: dict[str, PuzzleEntry] = {
    "maltese": PuzzleEntry("maltese", "Maltese Gear Cube", _maltese_build, _maltese_ladders, _maltese_notation,
                           legacy_cache=True),
    "tangram_extreme": PuzzleEntry("tangram_extreme", "Tangram Cube Extreme", _tangram_build("extreme"),
                                   _tangram_ladders("extreme"), _tangram_notation("extreme"),
                                   geometry_variant="extreme"),
    "tangram_original": PuzzleEntry("tangram_original", "Tangram Cube", _tangram_build("original"),
                                    _tangram_ladders("original"), _tangram_notation("original"), default_ladder="table",
                                    scramble_len=100, geometry_variant="original"),
    "toy": PuzzleEntry("toy", "two-face corners (test puzzle)", _toy_build, _toy_ladders, _maltese_notation,
                       default_ladder="table", scramble_len=50),
}


def get_puzzle(name: str) -> PuzzleEntry:
    try:
        return PUZZLES[name]
    except KeyError:
        raise KeyError(f"unknown puzzle {name!r}; known puzzles: {', '.join(PUZZLES)}") from None


def file_entry(puzzle: GatherPuzzle, title: str | None = None) -> PuzzleEntry:
    """An entry for a puzzle read from a file: no hand-written ladders, notation by move name."""
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", puzzle.name).strip("_.") or "puzzle"
    return PuzzleEntry(name=name, title=title or puzzle.name, build=lambda: puzzle, ladders=lambda: {},
                       notation=PlainNotation, default_ladder="auto")


def add_puzzle_arguments(ap: argparse.ArgumentParser) -> None:
    """``--puzzle NAME`` for a registered puzzle, or ``--tables FILE`` / ``--geometry FILE`` for any other."""
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--puzzle", choices=sorted(PUZZLES), default=None, help="a registered puzzle (default: maltese)")
    group.add_argument("--tables", default=None, help="a puzzle given as gather tables, .npz or .json (core/puzzle_io.py)")
    group.add_argument("--geometry", default=None, help="a puzzle given as a 3D model, puzzle-geometry/1 JSON (core/geometry.py)")


def entry_from_args(args: argparse.Namespace, default: str = "maltese",
                    log: Callable[[str], None] = lambda msg: print(msg, flush=True)) -> PuzzleEntry:
    """The puzzle chosen by ``add_puzzle_arguments``."""
    from core.puzzle_io import load_geometry_puzzle, load_tables

    if getattr(args, "tables", None):
        puzzle, added = load_tables(args.tables)
        if added:
            log(f"{args.tables}: appended the inverse moves {', '.join(added)} (the move set must contain every inverse)")
        return file_entry(puzzle)
    if getattr(args, "geometry", None):
        return file_entry(load_geometry_puzzle(args.geometry))
    return get_puzzle(getattr(args, "puzzle", None) or default)
