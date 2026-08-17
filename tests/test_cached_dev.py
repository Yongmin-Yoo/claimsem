"""Tests for the cached DEV reproduction runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_cached_dev.py"

SPEC = importlib.util.spec_from_file_location(
    "run_cached_dev",
    SCRIPT_PATH,
)

if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {SCRIPT_PATH}.")

MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_compare_labels_exact() -> None:
    """Identical labels must produce an exact comparison."""
    labels = np.asarray(
        [0, 0, 1, 1, 2, 2],
        dtype=np.int64,
    )

    comparison = MODULE.compare_labels(
        labels,
        labels.copy(),
    )

    assert comparison["exact_label_match"] is True
    assert comparison["direct_label_match_rate"] == 1.0
    assert comparison["mismatch_count"] == 0
    assert comparison["adjusted_rand_index"] == 1.0
    assert comparison["clustering_nmi"] == 1.0


def test_compare_labels_detects_label_permutation() -> None:
    """A cluster-ID permutation changes labels but not the partition."""
    reference = np.asarray(
        [0, 0, 1, 1, 2, 2],
        dtype=np.int64,
    )
    candidate = np.asarray(
        [2, 2, 0, 0, 1, 1],
        dtype=np.int64,
    )

    comparison = MODULE.compare_labels(
        reference,
        candidate,
    )

    assert comparison["exact_label_match"] is False
    assert comparison["direct_label_match_rate"] == 0.0
    assert comparison["mismatch_count"] == 6
    assert comparison["adjusted_rand_index"] == 1.0
    assert comparison["clustering_nmi"] == 1.0


def test_compare_labels_rejects_shape_mismatch() -> None:
    """Prediction arrays must contain the same number of samples."""
    with pytest.raises(
        ValueError,
        match="shape mismatch",
    ):
        MODULE.compare_labels(
            np.asarray([0, 1, 2]),
            np.asarray([0, 1]),
        )


def test_load_saved_seed42(tmp_path: Path) -> None:
    """The helper must load a valid one-dimensional NPY file."""
    path = tmp_path / "predictions.npy"
    expected = np.asarray(
        [0, 1, 1, 2],
        dtype=np.int64,
    )

    np.save(path, expected)

    loaded = MODULE.load_saved_seed42(
        path,
        expected_rows=4,
    )

    assert loaded.dtype == np.int64
    assert np.array_equal(
        loaded,
        expected,
    )


def test_load_saved_seed42_flattens_column_vector(
    tmp_path: Path,
) -> None:
    """A legacy column vector must be normalized to one dimension."""
    path = tmp_path / "predictions.npy"
    expected = np.asarray(
        [0, 1, 2, 3],
        dtype=np.int64,
    )

    np.save(
        path,
        expected[:, None],
    )

    loaded = MODULE.load_saved_seed42(
        path,
        expected_rows=4,
    )

    assert loaded.shape == (4,)
    assert np.array_equal(
        loaded,
        expected,
    )


def test_load_saved_seed42_rejects_wrong_count(
    tmp_path: Path,
) -> None:
    """An unexpected number of predictions must be rejected."""
    path = tmp_path / "predictions.npy"

    np.save(
        path,
        np.asarray([0, 1, 2]),
    )

    with pytest.raises(
        ValueError,
        match="prediction shape",
    ):
        MODULE.load_saved_seed42(
            path,
            expected_rows=4,
        )
