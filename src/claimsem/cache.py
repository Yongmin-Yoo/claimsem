"""Cached representation loading and validation for ClaimSem."""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class CacheError(RuntimeError):
    """Raised when a ClaimSem cache is invalid or misaligned."""


@dataclass(frozen=True)
class CachedArrayInfo:
    """Metadata describing one cached representation matrix."""

    path: str
    key: str | None
    shape: tuple[int, ...]
    dtype: str
    finite: bool
    row_norm_min: float
    row_norm_mean: float
    row_norm_max: float

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible cache metadata."""
        result = asdict(self)
        result["shape"] = list(self.shape)
        return result


@dataclass(frozen=True)
class AlignmentReport:
    """Record and assignment alignment report."""

    n_records: int
    n_assignments: int
    patent_id_match: bool
    global_index_match: bool
    section_match: bool
    class_match: bool
    subclass_match: bool
    patent_id_mismatches: int
    section_mismatches: int
    class_mismatches: int
    subclass_mismatches: int

    @property
    def valid(self) -> bool:
        """Return whether all alignment checks passed."""
        return all(
            [
                self.n_records == self.n_assignments,
                self.patent_id_match,
                self.global_index_match,
                self.section_match,
                self.class_match,
                self.subclass_match,
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible alignment metadata."""
        result = asdict(self)
        result["valid"] = self.valid
        return result


def normalize_identifier(value: Any) -> str | None:
    """Normalize identifiers from pickle, NumPy, or CSV data."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        number = float(value)

        if number.is_integer():
            return str(int(number))

        return str(number)

    text = str(value).strip()

    if not text:
        return None

    if text.endswith(".0"):
        possible_integer = text[:-2]

        if possible_integer.lstrip("-").isdigit():
            return possible_integer

    return text


def normalize_label(value: Any) -> str | None:
    """Normalize CPC label values."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    return text or None


def validate_representation_matrix(
    matrix: np.ndarray,
    *,
    name: str,
    expected_rows: int | None = None,
    expected_dim: int | None = None,
    require_float32: bool = True,
    require_finite: bool = True,
) -> np.ndarray:
    """Validate a cached two-dimensional representation matrix."""
    array = np.asarray(matrix)

    if array.ndim != 2:
        raise CacheError(f"{name} must have two dimensions, got {array.shape}.")

    if array.shape[0] == 0 or array.shape[1] == 0:
        raise CacheError(f"{name} cannot contain an empty dimension: {array.shape}.")

    if expected_rows is not None and array.shape[0] != expected_rows:
        raise CacheError(
            f"{name} contains {array.shape[0]} rows, expected {expected_rows}."
        )

    if expected_dim is not None and array.shape[1] != expected_dim:
        raise CacheError(
            f"{name} has dimension {array.shape[1]}, expected {expected_dim}."
        )

    if not np.issubdtype(array.dtype, np.floating):
        raise CacheError(
            f"{name} must contain floating-point values, got {array.dtype}."
        )

    if require_float32 and array.dtype != np.float32:
        raise CacheError(f"{name} must have dtype float32, got {array.dtype}.")

    if require_finite and not np.all(np.isfinite(array)):
        raise CacheError(f"{name} contains non-finite values.")

    return array


def describe_cached_matrix(
    matrix: np.ndarray,
    *,
    path: str | Path,
    key: str | None = None,
) -> CachedArrayInfo:
    """Create a summary of one cached matrix."""
    array = validate_representation_matrix(
        matrix,
        name=key or Path(path).name,
        require_float32=False,
        require_finite=True,
    )

    row_norms = np.linalg.norm(
        array.astype(np.float64, copy=False),
        axis=1,
    )

    return CachedArrayInfo(
        path=str(Path(path).expanduser().resolve()),
        key=key,
        shape=tuple(int(value) for value in array.shape),
        dtype=str(array.dtype),
        finite=bool(np.all(np.isfinite(array))),
        row_norm_min=float(row_norms.min()),
        row_norm_mean=float(row_norms.mean()),
        row_norm_max=float(row_norms.max()),
    )


def list_npz_keys(path: str | Path) -> list[str]:
    """List arrays stored in an NPZ cache."""
    source = Path(path).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(f"NPZ cache not found: {source}")

    try:
        with np.load(source, allow_pickle=False) as archive:
            return list(archive.files)
    except Exception as exc:
        raise CacheError(f"Could not inspect NPZ cache: {source}") from exc


def load_npz_matrix(
    path: str | Path,
    key: str,
    *,
    expected_rows: int | None = None,
    expected_dim: int | None = None,
) -> np.ndarray:
    """Load and validate one numeric matrix from an NPZ cache."""
    source = Path(path).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(f"NPZ cache not found: {source}")

    try:
        with np.load(source, allow_pickle=False) as archive:
            if key not in archive.files:
                raise CacheError(
                    f"Key {key!r} was not found in {source}. "
                    f"Available keys: {archive.files}"
                )

            matrix = np.asarray(
                archive[key],
                dtype=np.float32,
            )
    except CacheError:
        raise
    except Exception as exc:
        raise CacheError(f"Could not load key {key!r} from {source}.") from exc

    return validate_representation_matrix(
        matrix,
        name=f"{source.name}:{key}",
        expected_rows=expected_rows,
        expected_dim=expected_dim,
        require_float32=True,
        require_finite=True,
    )


def load_npy_matrix(
    path: str | Path,
    *,
    expected_rows: int | None = None,
    expected_dim: int | None = None,
    mmap_mode: str | None = "r",
) -> np.ndarray:
    """Load and validate an NPY representation matrix."""
    source = Path(path).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(f"NPY cache not found: {source}")

    try:
        matrix = np.load(
            source,
            mmap_mode=mmap_mode,
            allow_pickle=False,
        )
    except Exception as exc:
        raise CacheError(f"Could not load NPY cache: {source}") from exc

    return validate_representation_matrix(
        matrix,
        name=source.name,
        expected_rows=expected_rows,
        expected_dim=expected_dim,
        require_float32=True,
        require_finite=True,
    )


def load_legacy_records(
    path: str | Path,
    *,
    expected_count: int | None = None,
) -> list[dict[str, Any]]:
    """Load trusted legacy patent records from pickle.

    Pickle files can execute arbitrary code. This function must only be used
    with trusted files created by the project owner.
    """
    source = Path(path).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(f"Legacy record file not found: {source}")

    try:
        with source.open("rb") as handle:
            records = pickle.load(handle)
    except Exception as exc:
        raise CacheError(f"Could not load trusted record file: {source}") from exc

    if not isinstance(records, list):
        raise CacheError(f"Legacy records must be a list, got {type(records)}.")

    if expected_count is not None and len(records) != expected_count:
        raise CacheError(
            f"Record file contains {len(records)} patents, expected {expected_count}."
        )

    required_fields = {
        "patent_id",
        "section",
        "class",
        "subclass",
    }

    patent_ids: list[str] = []

    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise CacheError(f"Record {index} is not a dictionary.")

        missing_fields = required_fields - set(record.keys())

        if missing_fields:
            raise CacheError(
                f"Record {index} is missing fields: {sorted(missing_fields)}."
            )

        patent_id = normalize_identifier(record.get("patent_id"))

        if patent_id is None:
            raise CacheError(f"Record {index} has no valid patent ID.")

        patent_ids.append(patent_id)

        for level in ["section", "class", "subclass"]:
            if normalize_label(record.get(level)) is None:
                raise CacheError(f"Record {index} has no CPC {level} label.")

    if len(set(patent_ids)) != len(patent_ids):
        raise CacheError("Legacy records contain duplicate patent IDs.")

    return records


def make_evaluation_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert legacy records to the format expected by metrics.py."""
    evaluation_records: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        patent_id = normalize_identifier(record.get("patent_id"))

        if patent_id is None:
            raise CacheError(f"Record {index} has no valid patent ID.")

        cpc = {
            level: normalize_label(record.get(level))
            for level in ["section", "class", "subclass"]
        }

        missing_levels = [level for level, value in cpc.items() if value is None]

        if missing_levels:
            raise CacheError(f"Record {index} is missing CPC levels: {missing_levels}.")

        evaluation_records.append(
            {
                "patent_id": patent_id,
                "cpc": cpc,
            }
        )

    return evaluation_records


def load_pca_model(
    path: str | Path,
    *,
    expected_input_dim: int = 768,
    expected_output_dim: int = 128,
) -> PCA:
    """Load and validate a trusted fitted PCA artifact."""
    source = Path(path).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(f"PCA artifact not found: {source}")

    try:
        artifact = joblib.load(source)
    except Exception as exc:
        raise CacheError(f"Could not load PCA artifact: {source}") from exc

    if isinstance(artifact, PCA):
        pca = artifact
    elif isinstance(artifact, Mapping) and isinstance(artifact.get("model"), PCA):
        pca = artifact["model"]
    else:
        raise CacheError(f"Unsupported PCA artifact type: {type(artifact)}.")

    if not hasattr(pca, "components_"):
        raise CacheError("The PCA artifact is not fitted.")

    if int(pca.n_features_in_) != expected_input_dim:
        raise CacheError(
            f"PCA expects {pca.n_features_in_} input dimensions, "
            f"expected {expected_input_dim}."
        )

    if int(pca.n_components_) != expected_output_dim:
        raise CacheError(
            f"PCA produces {pca.n_components_} dimensions, "
            f"expected {expected_output_dim}."
        )

    if pca.components_.shape != (
        expected_output_dim,
        expected_input_dim,
    ):
        raise CacheError(
            f"Unexpected PCA component matrix shape: {pca.components_.shape}."
        )

    return pca


def apply_pca_and_normalize(
    raw_matrix: np.ndarray,
    pca: PCA,
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Apply a fitted PCA model and row-wise L2 normalization."""
    raw = validate_representation_matrix(
        raw_matrix,
        name="raw_matrix",
        expected_dim=int(pca.n_features_in_),
        require_float32=False,
        require_finite=True,
    )

    transformed = pca.transform(raw).astype(
        np.float32,
        copy=False,
    )

    row_norms = np.linalg.norm(
        transformed,
        axis=1,
        keepdims=True,
    )

    if np.any(row_norms < epsilon):
        indices = np.flatnonzero(row_norms[:, 0] < epsilon).tolist()

        raise CacheError(
            f"PCA produced zero or near-zero rows at indices {indices[:20]}."
        )

    normalized = (transformed / np.maximum(row_norms, epsilon)).astype(
        np.float32, copy=False
    )

    if not np.all(np.isfinite(normalized)):
        raise CacheError("PCA and normalization produced non-finite values.")

    return normalized


def compare_cached_pca(
    raw_matrix: np.ndarray,
    cached_pca_matrix: np.ndarray,
    pca: PCA,
) -> dict[str, float | bool]:
    """Compare a cached normalized PCA matrix with recomputation."""
    recomputed = apply_pca_and_normalize(
        raw_matrix,
        pca,
    )

    cached = validate_representation_matrix(
        cached_pca_matrix,
        name="cached_pca_matrix",
        expected_rows=recomputed.shape[0],
        expected_dim=recomputed.shape[1],
        require_float32=True,
        require_finite=True,
    )

    absolute_error = np.abs(cached.astype(np.float64) - recomputed.astype(np.float64))

    max_error = float(absolute_error.max())
    mean_error = float(absolute_error.mean())

    return {
        "matches": bool(max_error < 1e-5),
        "max_absolute_error": max_error,
        "mean_absolute_error": mean_error,
    }


def validate_assignment_alignment(
    records: Sequence[Mapping[str, Any]],
    assignments_path: str | Path,
) -> AlignmentReport:
    """Validate record ordering against the legacy assignment CSV."""
    source = Path(assignments_path).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(f"Assignment CSV not found: {source}")

    assignments = pd.read_csv(source)

    required_columns = {
        "global_index",
        "patent_id",
        "section",
        "class",
        "subclass",
    }

    missing_columns = required_columns - set(assignments.columns)

    if missing_columns:
        raise CacheError(
            f"Assignment CSV is missing columns: {sorted(missing_columns)}."
        )

    n_records = len(records)
    n_assignments = len(assignments)

    record_ids = [normalize_identifier(record.get("patent_id")) for record in records]

    assignment_ids = [
        normalize_identifier(value) for value in assignments["patent_id"].tolist()
    ]

    id_comparison_length = min(
        n_records,
        n_assignments,
    )

    patent_id_mismatches = sum(
        record_ids[index] != assignment_ids[index]
        for index in range(id_comparison_length)
    )

    patent_id_mismatches += abs(n_records - n_assignments)

    patent_id_match = n_records == n_assignments and patent_id_mismatches == 0

    expected_indices = np.arange(
        n_assignments,
        dtype=np.int64,
    )

    actual_indices = assignments["global_index"].to_numpy(dtype=np.int64)

    global_index_match = bool(
        np.array_equal(
            expected_indices,
            actual_indices,
        )
    )

    level_matches: dict[str, bool] = {}
    level_mismatches: dict[str, int] = {}

    for level in ["section", "class", "subclass"]:
        record_labels = [normalize_label(record.get(level)) for record in records]

        assignment_labels = [
            normalize_label(value) for value in assignments[level].tolist()
        ]

        comparison_length = min(
            len(record_labels),
            len(assignment_labels),
        )

        mismatch_count = sum(
            record_labels[index] != assignment_labels[index]
            for index in range(comparison_length)
        )

        mismatch_count += abs(len(record_labels) - len(assignment_labels))

        level_mismatches[level] = mismatch_count
        level_matches[level] = mismatch_count == 0 and len(record_labels) == len(
            assignment_labels
        )

    report = AlignmentReport(
        n_records=n_records,
        n_assignments=n_assignments,
        patent_id_match=patent_id_match,
        global_index_match=global_index_match,
        section_match=level_matches["section"],
        class_match=level_matches["class"],
        subclass_match=level_matches["subclass"],
        patent_id_mismatches=patent_id_mismatches,
        section_mismatches=level_mismatches["section"],
        class_mismatches=level_mismatches["class"],
        subclass_mismatches=level_mismatches["subclass"],
    )

    if not report.valid:
        raise CacheError(f"Record and assignment alignment failed: {report.to_dict()}.")

    return report
