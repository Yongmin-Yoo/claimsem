"""Artifact saving and checksum utilities for ClaimSem."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


class ArtifactError(RuntimeError):
    """Raised when experiment artifacts cannot be saved."""


def ensure_directory(path: str | Path) -> Path:
    """Create an output directory if it does not exist."""
    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _json_compatible(value: Any) -> Any:
    """Recursively convert common Python objects to JSON-compatible values."""
    if is_dataclass(value):
        return _json_compatible(asdict(value))

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_json_compatible(item) for item in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_compatible(value.to_dict())

    return str(value)


def save_json(
    data: Any,
    path: str | Path,
    indent: int = 2,
) -> Path:
    """Atomically save data as UTF-8 JSON."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    serializable = _json_compatible(data)

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".tmp",
            prefix=f"{destination.name}.",
            dir=destination.parent,
            delete=False,
        ) as handle:
            json.dump(
                serializable,
                handle,
                indent=indent,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            temporary_path = Path(handle.name)

        os.replace(temporary_path, destination)

    except Exception as exc:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

        raise ArtifactError(
            f"Failed to save JSON artifact to {destination}."
        ) from exc

    return destination


def load_json(path: str | Path) -> Any:
    """Load a UTF-8 JSON artifact."""
    source = Path(path).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(f"JSON artifact not found: {source}")

    try:
        with source.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ArtifactError(
            f"Invalid JSON artifact: {source}"
        ) from exc


def save_numpy_array(
    array: np.ndarray,
    path: str | Path,
) -> Path:
    """Save one NumPy array."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    matrix = np.asarray(array)

    if not np.all(np.isfinite(matrix)):
        raise ArtifactError(
            f"Cannot save non-finite values to {destination}."
        )

    np.save(destination, matrix)

    if destination.suffix != ".npy":
        destination = destination.with_suffix(
            destination.suffix + ".npy"
        )

    return destination


def save_numpy_archive(
    path: str | Path,
    **arrays: np.ndarray,
) -> Path:
    """Save multiple named NumPy arrays in a compressed NPZ archive."""
    destination = Path(path).expanduser().resolve()

    if destination.suffix != ".npz":
        destination = destination.with_suffix(".npz")

    destination.parent.mkdir(parents=True, exist_ok=True)

    normalized: dict[str, np.ndarray] = {}

    for name, array in arrays.items():
        normalized_array = np.asarray(array)

        if not np.all(np.isfinite(normalized_array)):
            raise ArtifactError(
                f"Array {name!r} contains non-finite values."
            )

        normalized[name] = normalized_array

    np.savez_compressed(destination, **normalized)

    return destination


def sha256_file(
    path: str | Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Compute the SHA-256 checksum of a file."""
    source = Path(path).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(
            f"Cannot checksum missing file: {source}"
        )

    digest = hashlib.sha256()

    with source.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def file_record(path: str | Path) -> dict[str, Any]:
    """Return path, size, and SHA-256 information for one file."""
    source = Path(path).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(f"Artifact not found: {source}")

    return {
        "path": str(source),
        "size_bytes": int(source.stat().st_size),
        "sha256": sha256_file(source),
    }


def save_predictions_csv(
    records: Sequence[Mapping[str, Any]],
    clustering_results: Sequence[Any],
    path: str | Path,
) -> Path:
    """Save patent identifiers and predicted clusters for all seeds."""
    if not records:
        raise ArtifactError("At least one patent record is required.")

    if not clustering_results:
        raise ArtifactError(
            "At least one clustering result is required."
        )

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    patent_ids = [
        str(record.get("patent_id", index))
        for index, record in enumerate(records)
    ]

    result_columns: list[tuple[str, np.ndarray]] = []

    for result in clustering_results:
        if not hasattr(result, "labels"):
            raise ArtifactError(
                "Each clustering result must provide labels."
            )

        labels = np.asarray(result.labels)

        if labels.ndim != 1 or labels.shape[0] != len(records):
            raise ArtifactError(
                "A clustering result contains an invalid label array."
            )

        seed = getattr(result, "seed", len(result_columns))
        result_columns.append(
            (f"cluster_seed_{seed}", labels)
        )

    fieldnames = [
        "patent_id",
        "cpc_section",
        "cpc_class",
        "cpc_subclass",
    ] + [name for name, _ in result_columns]

    with destination.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for index, record in enumerate(records):
            cpc = record.get("cpc", {})

            row: dict[str, Any] = {
                "patent_id": patent_ids[index],
                "cpc_section": cpc.get("section", ""),
                "cpc_class": cpc.get("class", ""),
                "cpc_subclass": cpc.get("subclass", ""),
            }

            for column_name, labels in result_columns:
                row[column_name] = int(labels[index])

            writer.writerow(row)

    return destination


def save_clustering_archives(
    clustering_results: Sequence[Any],
    output_directory: str | Path,
) -> list[Path]:
    """Save labels, centroids, and cluster counts for every seed."""
    output_dir = ensure_directory(output_directory)
    saved_paths: list[Path] = []

    for result in clustering_results:
        required = (
            "labels",
            "centroids",
            "cluster_counts",
            "seed",
        )

        if not all(hasattr(result, name) for name in required):
            raise ArtifactError(
                "Clustering result is missing required attributes."
            )

        path = output_dir / (
            f"clustering_seed_{int(result.seed)}.npz"
        )

        saved_path = save_numpy_archive(
            path,
            labels=np.asarray(result.labels),
            centroids=np.asarray(result.centroids),
            cluster_counts=np.asarray(result.cluster_counts),
            objective=np.asarray(
                [float(result.objective)],
                dtype=np.float64,
            ),
            n_iterations=np.asarray(
                [int(result.n_iterations)],
                dtype=np.int64,
            ),
            converged=np.asarray(
                [bool(result.converged)],
                dtype=np.bool_,
            ),
        )

        saved_paths.append(saved_path)

    return saved_paths


def create_manifest(
    experiment_name: str,
    config: Mapping[str, Any],
    input_files: Sequence[str | Path] | None = None,
    output_files: Sequence[str | Path] | None = None,
    environment: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a reproducibility manifest."""
    manifest: dict[str, Any] = {
        "experiment_name": str(experiment_name),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "config": _json_compatible(config),
        "inputs": [],
        "outputs": [],
    }

    if input_files:
        manifest["inputs"] = [
            file_record(path)
            for path in input_files
            if Path(path).expanduser().exists()
        ]

    if output_files:
        manifest["outputs"] = [
            file_record(path)
            for path in output_files
            if Path(path).expanduser().exists()
        ]

    if environment is not None:
        manifest["environment"] = _json_compatible(environment)

    if extra is not None:
        manifest["extra"] = _json_compatible(extra)

    return manifest
