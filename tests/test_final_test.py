"""Tests for the final consistent TEST runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_final_test.py"

SPEC = importlib.util.spec_from_file_location(
    "run_final_test",
    SCRIPT_PATH,
)

if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {SCRIPT_PATH}.")

MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_validate_matrix_accepts_float_matrix() -> None:
    matrix = np.ones((3, 2), dtype=np.float32)

    result = MODULE.validate_matrix(
        matrix,
        name="matrix",
        expected_rows=3,
        expected_dim=2,
    )

    assert result.shape == (3, 2)
    assert result.dtype == np.float32


def test_validate_matrix_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        MODULE.validate_matrix(
            np.ones((2, 2), dtype=np.float32),
            name="matrix",
            expected_rows=3,
            expected_dim=2,
        )


def test_compare_matrices_exact() -> None:
    matrix = np.asarray(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )

    comparison = MODULE.compare_matrices(
        matrix,
        matrix.copy(),
        atol=1e-6,
        rtol=1e-5,
    )

    assert comparison["validation_passed"] is True
    assert comparison["practical_allclose"] is True
    assert comparison["max_absolute_difference"] == 0.0
    assert comparison["minimum_cosine_similarity"] == pytest.approx(1.0)


def test_norm_summary() -> None:
    matrix = np.asarray(
        [[1.0, 0.0], [0.0, -1.0]],
        dtype=np.float32,
    )

    summary = MODULE.norm_summary(matrix)

    assert summary["minimum"] == pytest.approx(1.0)
    assert summary["mean"] == pytest.approx(1.0)
    assert summary["maximum"] == pytest.approx(1.0)
    assert summary["maximum_absolute_error_from_one"] == pytest.approx(0.0)
