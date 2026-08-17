#!/usr/bin/env python3
"""Run the final consistent ClaimSem TEST experiment.

The runner loads consistent TEST patent features, applies the fixed PCA model
fitted on DEV, L2-normalizes the PCA output, validates the corresponding DEV
transformation, runs the legacy GPU spherical K-means backend for three seeds,
and evaluates the resulting clusters against CPC hierarchy labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA

from claimsem.cache import load_legacy_records, make_evaluation_records
from claimsem.clustering import run_multiple_seeds_legacy
from claimsem.metrics import evaluate_multiple_seeds, summarize_seed_evaluations
from claimsem.reduction import transform_and_normalize

DEFAULT_SEEDS = (17, 42, 73)
LABEL_LEVELS = ("section", "class", "subclass")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a fixed DEV PCA model to consistent TEST features and run "
            "the final three-seed ClaimSem clustering evaluation."
        )
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--pca-model", type=Path, required=True)
    parser.add_argument("--legacy-dev-raw", type=Path, default=None)
    parser.add_argument("--legacy-dev-pca", type=Path, required=True)
    parser.add_argument("--consistent-dev-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pool-key", default="root12_d010")
    parser.add_argument("--expected-test-rows", type=int, default=9881)
    parser.add_argument("--expected-dev-rows", type=int, default=9855)
    parser.add_argument("--input-dim", type=int, default=768)
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--n-clusters", type=int, default=30)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dev-atol", type=float, default=1e-6)
    parser.add_argument("--dev-rtol", type=float, default=1e-5)
    parser.add_argument("--require-dev-match", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def require_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()

    if not resolved.is_file():
        raise FileNotFoundError(f"{description} was not found: {resolved}")

    return resolved


def validate_matrix(
    matrix: np.ndarray,
    *,
    name: str,
    expected_rows: int,
    expected_dim: int,
) -> np.ndarray:
    array = np.asarray(matrix)

    if array.shape != (expected_rows, expected_dim):
        raise ValueError(
            f"{name} shape is {array.shape}; expected {(expected_rows, expected_dim)}."
        )

    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must contain numeric values.")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")

    return array.astype(np.float32, copy=False)


def load_feature_archive(
    path: Path,
    *,
    expected_rows: int,
    expected_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if "features" not in archive.files:
            raise KeyError(f"{path} does not contain 'features'. Keys: {archive.files}")

        if "patent_ids" not in archive.files:
            raise KeyError(
                f"{path} does not contain 'patent_ids'. Keys: {archive.files}"
            )

        features = np.asarray(archive["features"], dtype=np.float32)
        patent_ids = np.asarray(archive["patent_ids"], dtype=np.int64)

    features = validate_matrix(
        features,
        name=path.name,
        expected_rows=expected_rows,
        expected_dim=expected_dim,
    )

    if patent_ids.shape != (expected_rows,):
        raise ValueError(
            f"{path.name} patent ID shape is {patent_ids.shape}; "
            f"expected {(expected_rows,)}."
        )

    if np.unique(patent_ids).size != expected_rows:
        raise ValueError(f"{path.name} contains duplicate patent IDs.")

    return features, patent_ids


def load_pool_matrix(
    path: Path,
    *,
    key: str,
    expected_rows: int,
    expected_dim: int,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive.files:
            raise KeyError(
                f"Pool key {key!r} was not found in {path}. "
                f"Available keys: {archive.files}"
            )

        matrix = np.asarray(archive[key], dtype=np.float32)

    return validate_matrix(
        matrix,
        name=f"{path.name}:{key}",
        expected_rows=expected_rows,
        expected_dim=expected_dim,
    )


def load_pca_model(path: Path) -> PCA:
    payload = joblib.load(path)

    if isinstance(payload, PCA):
        pca = payload
    elif isinstance(payload, Mapping) and "model" in payload:
        pca = payload["model"]
    elif isinstance(payload, Mapping) and "pca" in payload:
        pca = payload["pca"]
    else:
        raise ValueError(f"Unsupported PCA artifact format: {path}")

    if not isinstance(pca, PCA):
        raise TypeError(f"The PCA artifact does not contain sklearn PCA: {path}")

    if not hasattr(pca, "components_") or not hasattr(pca, "mean_"):
        raise ValueError(f"The PCA model is not fitted: {path}")

    return pca


def compare_matrices(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if candidate.shape != reference.shape:
        raise ValueError(
            f"Matrix shape mismatch: {candidate.shape} versus {reference.shape}."
        )

    candidate64 = candidate.astype(np.float64)
    reference64 = reference.astype(np.float64)

    difference = np.abs(candidate64 - reference64)
    dots = np.sum(candidate64 * reference64, axis=1)
    denominators = np.linalg.norm(candidate64, axis=1) * np.linalg.norm(
        reference64, axis=1
    )
    cosine = dots / np.maximum(denominators, 1e-12)

    practical_allclose = bool(np.allclose(candidate, reference, atol=atol, rtol=rtol))

    return {
        "shape": list(candidate.shape),
        "max_absolute_difference": float(difference.max()),
        "mean_absolute_difference": float(difference.mean()),
        "p99_absolute_difference": float(np.quantile(difference, 0.99)),
        "minimum_cosine_similarity": float(cosine.min()),
        "mean_cosine_similarity": float(cosine.mean()),
        "practical_allclose": practical_allclose,
        "validation_passed": bool(
            practical_allclose and float(cosine.min()) >= 0.99999
        ),
    }


def record_patent_ids(
    records: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    values: list[int] = []

    for index, record in enumerate(records):
        if "patent_id" not in record:
            raise KeyError(f"Record {index} does not contain patent_id.")

        values.append(int(record["patent_id"]))

    return np.asarray(values, dtype=np.int64)


def norm_summary(matrix: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)

    return {
        "minimum": float(norms.min()),
        "mean": float(norms.mean()),
        "maximum": float(norms.max()),
        "maximum_absolute_error_from_one": float(np.max(np.abs(norms - 1.0))),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()

    paths = {
        "test_features": require_file(args.features, "TEST feature file"),
        "records": require_file(args.records, "TEST records"),
        "pca_model": require_file(args.pca_model, "PCA model"),
        "legacy_dev_pca": require_file(
            args.legacy_dev_pca,
            "Legacy DEV PCA cache",
        ),
        "consistent_dev_features": require_file(
            args.consistent_dev_features,
            "Consistent DEV feature file",
        ),
    }

    if args.legacy_dev_raw is not None:
        paths["legacy_dev_raw"] = require_file(
            args.legacy_dev_raw,
            "Legacy DEV raw cache",
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_dir = output_dir / "clusters"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. In Colab select "
            "Runtime > Change runtime type > T4 GPU."
        )

    print("=" * 88)
    print("FINAL CONSISTENT CLAIMSEM TEST")
    print("=" * 88)
    print("TEST features       :", paths["test_features"])
    print("TEST records        :", paths["records"])
    print("PCA model           :", paths["pca_model"])
    print("Legacy DEV PCA      :", paths["legacy_dev_pca"])
    print("Consistent DEV raw  :", paths["consistent_dev_features"])
    print("Output directory    :", output_dir)
    print("Seeds               :", args.seeds)
    print("Clusters            :", args.n_clusters)
    print("Device              :", args.device)

    if torch.cuda.is_available():
        print("GPU                 :", torch.cuda.get_device_name(0))

    pca = load_pca_model(paths["pca_model"])

    if tuple(pca.components_.shape) != (args.pca_dim, args.input_dim):
        raise ValueError(
            f"PCA component shape is {pca.components_.shape}; expected "
            f"{(args.pca_dim, args.input_dim)}."
        )

    consistent_dev_raw, consistent_dev_ids = load_feature_archive(
        paths["consistent_dev_features"],
        expected_rows=args.expected_dev_rows,
        expected_dim=args.input_dim,
    )

    legacy_dev_pca = load_pool_matrix(
        paths["legacy_dev_pca"],
        key=args.pool_key,
        expected_rows=args.expected_dev_rows,
        expected_dim=args.pca_dim,
    )

    transformed_consistent_dev = transform_and_normalize(
        consistent_dev_raw,
        pca,
    )

    consistent_dev_comparison = compare_matrices(
        transformed_consistent_dev,
        legacy_dev_pca,
        atol=args.dev_atol,
        rtol=args.dev_rtol,
    )

    legacy_dev_reproduction = None

    if "legacy_dev_raw" in paths:
        legacy_dev_raw = load_pool_matrix(
            paths["legacy_dev_raw"],
            key=args.pool_key,
            expected_rows=args.expected_dev_rows,
            expected_dim=args.input_dim,
        )

        transformed_legacy_dev = transform_and_normalize(
            legacy_dev_raw,
            pca,
        )

        legacy_dev_reproduction = compare_matrices(
            transformed_legacy_dev,
            legacy_dev_pca,
            atol=args.dev_atol,
            rtol=args.dev_rtol,
        )

    dev_validation_passed = bool(
        consistent_dev_comparison["validation_passed"]
        and (
            legacy_dev_reproduction is None
            or legacy_dev_reproduction["validation_passed"]
        )
    )

    print("\n" + "=" * 88)
    print("DEV PCA VALIDATION")
    print("=" * 88)
    print(
        "Consistent DEV vs legacy DEV PCA:",
        json.dumps(consistent_dev_comparison, indent=2),
    )

    if legacy_dev_reproduction is not None:
        print(
            "Legacy DEV PCA reproduction:",
            json.dumps(legacy_dev_reproduction, indent=2),
        )

    if args.require_dev_match and not dev_validation_passed:
        raise RuntimeError("Fixed DEV PCA validation failed.")

    test_raw, test_patent_ids = load_feature_archive(
        paths["test_features"],
        expected_rows=args.expected_test_rows,
        expected_dim=args.input_dim,
    )

    records = load_legacy_records(
        paths["records"],
        expected_count=args.expected_test_rows,
    )
    evaluation_records = make_evaluation_records(records)
    record_ids = record_patent_ids(records)

    patent_order_match = bool(np.array_equal(test_patent_ids, record_ids))

    if not patent_order_match:
        mismatch = np.flatnonzero(test_patent_ids != record_ids)
        raise RuntimeError(
            "TEST patent order does not match records. "
            f"First mismatches: {mismatch[:20].tolist()}"
        )

    test_pca = transform_and_normalize(test_raw, pca)
    test_pca = validate_matrix(
        test_pca,
        name="TEST PCA features",
        expected_rows=args.expected_test_rows,
        expected_dim=args.pca_dim,
    )

    test_norms = norm_summary(test_pca)

    if not np.allclose(
        np.linalg.norm(test_pca, axis=1),
        1.0,
        atol=1e-5,
    ):
        raise RuntimeError("TEST PCA features are not L2-normalized.")

    pca_output_path = output_dir / "test_pca128_features.npz"
    np.savez_compressed(
        pca_output_path,
        features=test_pca,
        patent_ids=test_patent_ids,
    )

    clustering_started = time.time()

    results = run_multiple_seeds_legacy(
        vectors=test_pca,
        n_clusters=args.n_clusters,
        seeds=args.seeds,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
        device=args.device,
    )

    clustering_seconds = time.time() - clustering_started

    evaluations = evaluate_multiple_seeds(
        records=evaluation_records,
        clustering_results=results,
        label_levels=LABEL_LEVELS,
    )

    summary = summarize_seed_evaluations(
        evaluations=evaluations,
        label_levels=LABEL_LEVELS,
    )

    evaluation_by_seed = {int(item["seed"]): item for item in evaluations}

    per_seed: list[dict[str, Any]] = []

    for result in results:
        evaluation = evaluation_by_seed[result.seed]

        row = {
            "seed": int(result.seed),
            "iterations": int(result.n_iterations),
            "converged": bool(result.converged),
            "active_clusters": int(result.n_active_clusters),
            "max_cluster_share": float(result.max_cluster_share),
            "mean_cosine_similarity": float(1.0 - result.objective),
            "mean_cosine_distance": float(result.objective),
            "mean_metrics": evaluation["mean"],
            "levels": evaluation["levels"],
        }
        per_seed.append(row)

        np.savez_compressed(
            cluster_dir / f"clustering_seed_{result.seed}.npz",
            labels=result.labels,
            centroids=result.centroids,
            cluster_counts=result.cluster_counts,
            objective=np.asarray([result.objective], dtype=np.float64),
            n_iterations=np.asarray([result.n_iterations], dtype=np.int64),
            converged=np.asarray([result.converged], dtype=np.bool_),
            seed=np.asarray([result.seed], dtype=np.int64),
        )

        print("\nSeed:", result.seed)
        print(json.dumps(row, indent=2))

    predictions_path = output_dir / "test_3seed_predictions.npz"
    np.savez_compressed(
        predictions_path,
        **{f"seed_{result.seed}": result.labels.astype(np.int64) for result in results},
    )

    evaluations_path = output_dir / "seed_evaluations.json"
    metrics_path = output_dir / "metrics_summary.json"

    save_json(evaluations_path, evaluations)
    save_json(metrics_path, summary)

    all_converged = all(result.converged for result in results)
    all_clusters_active = all(
        result.n_active_clusters == args.n_clusters for result in results
    )

    validation_passed = bool(
        dev_validation_passed
        and patent_order_match
        and all_converged
        and all_clusters_active
        and np.all(np.isfinite(test_pca))
        and test_norms["maximum_absolute_error_from_one"] <= 1e-5
    )

    report = {
        "experiment": "final_consistent_claimsem_test",
        "validation_passed": validation_passed,
        "dev_validation_passed": dev_validation_passed,
        "patent_order_match": patent_order_match,
        "pca_refitted_on_test": False,
        "pca_fit_split": "dev",
        "backend": "legacy_gpu_spherical_kmeans",
        "n_samples": int(test_pca.shape[0]),
        "raw_feature_dimension": int(test_raw.shape[1]),
        "feature_dimension": int(test_pca.shape[1]),
        "n_clusters": int(args.n_clusters),
        "seeds": [int(seed) for seed in args.seeds],
        "max_iter": int(args.max_iter),
        "tolerance": float(args.tolerance),
        "pool_key": args.pool_key,
        "all_converged": all_converged,
        "all_clusters_active": all_clusters_active,
        "test_pca_norms": test_norms,
        "consistent_dev_comparison": consistent_dev_comparison,
        "legacy_dev_reproduction": legacy_dev_reproduction,
        "consistent_dev_patent_count": int(consistent_dev_ids.size),
        "per_seed": per_seed,
        "metrics": summary,
        "runtime": {
            "clustering_seconds": float(clustering_seconds),
            "total_seconds": float(time.time() - started),
        },
        "environment": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "inputs": {key: str(value) for key, value in paths.items()},
        "outputs": {
            "pca_features": str(pca_output_path),
            "predictions": str(predictions_path),
            "seed_evaluations": str(evaluations_path),
            "metrics_summary": str(metrics_path),
        },
        "checksums": {
            "inputs": {key: sha256_file(value) for key, value in paths.items()},
            "outputs": {
                "pca_features": sha256_file(pca_output_path),
                "predictions": sha256_file(predictions_path),
                "seed_evaluations": sha256_file(evaluations_path),
                "metrics_summary": sha256_file(metrics_path),
            },
        },
    }

    report_path = output_dir / "final_test_report.json"
    save_json(report_path, report)

    print("\n" + "=" * 88)
    print("FINAL CONSISTENT CLAIMSEM TEST RESULT")
    print("=" * 88)
    print("Validation passed       :", validation_passed)
    print("DEV validation passed   :", dev_validation_passed)
    print("Patent order match      :", patent_order_match)
    print("PCA refitted on TEST    :", False)
    print("Samples                 :", test_pca.shape[0])
    print("Feature dimension       :", test_pca.shape[1])
    print("All seeds converged     :", all_converged)
    print("All clusters active     :", all_clusters_active)
    print("TEST PCA norm summary   :", test_norms)
    print("Mean metrics:")
    print(json.dumps(summary["mean"], indent=2))
    print("Mean cosine distance    :", summary.get("objective"))
    print("Maximum cluster share   :", summary.get("max_cluster_share"))
    print("Predictions             :", predictions_path)
    print("Metrics summary         :", metrics_path)
    print("Final report            :", report_path)

    if not validation_passed:
        raise RuntimeError("Final consistent TEST validation failed.")

    print("\nSUCCESS: Final consistent ClaimSem TEST evaluation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
