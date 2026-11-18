"""Puzzles read from files: round trips, computed inverses, and refusal of invalid files."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

from core.fingerprint import puzzle_fingerprint
from core.harvest import WordAlgebra
from core.notation import PlainNotation
from core.perm_model import build_point_model
from core.puzzle_io import PuzzleFileError, load_geometry_puzzle, load_tables, puzzle_from_arrays, save_tables
from puzzles.registry import add_puzzle_arguments, entry_from_args, get_puzzle
from puzzles.tangram.puzzle import geometry_file


def _arrays(puzzle, *moves) -> dict:
    moves = list(moves) if moves else list(range(puzzle.num_moves))
    return dict(idx=puzzle.idx[moves], add=puzzle.add[moves], modulus=puzzle.modulus, goal=puzzle.goal,
                move_names=[puzzle.move_names[m] for m in moves])


@pytest.mark.parametrize("suffix", [".npz", ".json"])
def test_round_trip(maltese, tmp_path, suffix) -> None:
    puzzle, _, _ = maltese
    path = tmp_path / f"maltese{suffix}"
    save_tables(puzzle, path)
    again, added = load_tables(path)
    assert added == [] and again.name == puzzle.name
    assert puzzle_fingerprint(again) == puzzle_fingerprint(puzzle)


def test_inverses_are_computed_when_not_given(maltese) -> None:
    puzzle, _, _ = maltese
    again, added = puzzle_from_arrays(_arrays(puzzle))
    assert added == [] and np.array_equal(again.inverse, puzzle.inverse)


def test_missing_inverse_moves_are_appended(maltese) -> None:
    """Given only the clockwise turns U, L, F, the anticlockwise ones are added and named with a prime."""
    puzzle, _, _ = maltese
    u, l_, f = (puzzle.move_names.index(x) for x in ("U", "L", "F"))
    three, added = puzzle_from_arrays(_arrays(puzzle, u, l_, f))
    assert three.move_names == ("U", "L", "F", "U'", "L'", "F'") and added == ["U'", "L'", "F'"]
    rng = np.random.default_rng(0)
    state = three.apply_word(three.goal, rng.integers(0, 6, 40).tolist())
    for m in range(6):
        assert np.array_equal(three.apply_word(state, [m, int(three.inverse[m])]), state)
        assert np.array_equal(three.idx[m], puzzle.idx[puzzle.move_names.index(three.move_names[m])])


def test_self_inverse_moves_need_nothing_appended(tangram) -> None:
    puzzle, _, _ = tangram
    flips = [puzzle.move_names.index(x) for x in ("F", "R", "L")]
    sub, added = puzzle_from_arrays(_arrays(puzzle, *flips))
    assert added == [] and list(sub.inverse) == [0, 1, 2]


def test_invalid_files_are_refused(maltese) -> None:
    puzzle, _, _ = maltese
    bad = _arrays(puzzle)
    bad["idx"] = bad["idx"].copy()
    bad["idx"][0, 0] = bad["idx"][0, 1]
    with pytest.raises(PuzzleFileError, match="not a permutation"):
        puzzle_from_arrays(bad)
    with pytest.raises(PuzzleFileError, match="modulus"):
        puzzle_from_arrays({k: v for k, v in _arrays(puzzle).items() if k != "modulus"})
    with pytest.raises(PuzzleFileError, match="distinct name"):
        puzzle_from_arrays(_arrays(puzzle, 0, 0))
    with pytest.raises(PuzzleFileError, match="same effect"):
        puzzle_from_arrays(dict(_arrays(puzzle, 0, 0), move_names=["U", "U again"]))
    wrong = dict(_arrays(puzzle), inverse=list(range(puzzle.num_moves)))
    with pytest.raises(PuzzleFileError, match="does not"):
        puzzle_from_arrays(wrong)
    with pytest.raises(PuzzleFileError, match="at least"):
        puzzle_from_arrays({"goal": puzzle.goal})


def test_geometry_file_compiles_to_the_registered_puzzle() -> None:
    puzzle = load_geometry_puzzle(geometry_file("original"))
    assert puzzle_fingerprint(puzzle) == puzzle_fingerprint(get_puzzle("tangram_original").build())


def test_plain_notation_handles_any_move_names(maltese) -> None:
    puzzle, _, _ = maltese
    data = _arrays(puzzle)
    data["move_names"] = ["Up", "Up-1", "Left2", "Left2'", "F", "F*"]
    renamed, _ = puzzle_from_arrays(data)
    notation = PlainNotation(WordAlgebra(build_point_model(renamed)))
    word = notation.parse("Up Left2 Left2 F* Up-1")
    assert notation.format(word) == "Up Left2 Left2 F* Up-1"
    assert notation.format(notation.parse("Up Up-1 F")) == "F"                     # cancels like any word
    with pytest.raises(ValueError):
        notation.parse("Down")


def test_scripts_accept_a_puzzle_file(maltese, tmp_path) -> None:
    puzzle, _, _ = maltese
    path = tmp_path / "my puzzle.json"
    data = _arrays(puzzle, 0, 2, 4)
    data = {k: (np.asarray(v).tolist() if not isinstance(v, list) else v) for k, v in data.items()}
    path.write_text(json.dumps(data))
    ap = argparse.ArgumentParser()
    add_puzzle_arguments(ap)
    messages: list[str] = []
    entry = entry_from_args(ap.parse_args(["--tables", str(path)]), log=messages.append)
    assert entry.name == "my_puzzle" and entry.build().num_moves == 6 and entry.ladders() == {}
    assert messages and "U'" in messages[0]
    assert entry_from_args(ap.parse_args([])).name == "maltese"
