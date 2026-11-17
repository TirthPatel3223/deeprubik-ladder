"""Shared fixtures: the puzzles and their structure are built once per test session."""
from __future__ import annotations

import pytest

from core.perm_model import build_point_model
from core.structure import analyse
from puzzles.maltese_gear.puzzle import build_maltese_gear_puzzle
from puzzles.tangram.puzzle import build_tangram_puzzle
from puzzles.toy.two_face_corners import build_two_face_corners


@pytest.fixture(scope="session")
def maltese():
    puzzle = build_maltese_gear_puzzle()
    model = build_point_model(puzzle)
    return puzzle, model, analyse(model)


@pytest.fixture(scope="session")
def toy():
    puzzle = build_two_face_corners()
    model = build_point_model(puzzle)
    return puzzle, model, analyse(model)


@pytest.fixture(scope="session")
def tangram():
    """The Tangram Cube Extreme, compiled from its 3D model."""
    puzzle = build_tangram_puzzle("extreme")
    model = build_point_model(puzzle)
    return puzzle, model, analyse(model)


@pytest.fixture(scope="session")
def tangram_original():
    puzzle = build_tangram_puzzle("original")
    model = build_point_model(puzzle)
    return puzzle, model, analyse(model)
