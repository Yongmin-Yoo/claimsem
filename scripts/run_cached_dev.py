#!/usr/bin/env python3
"""Reproduce the cached legacy ClaimSem DEV clustering experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from claimsem.cache import (
    describe_cached_matrix,
    list_npz_keys,
    load_legacy_records,
    load_npz_matrix,
    make_evaluation_records,
    validate_assignment_alignment,
)
from claimsem.clustering import run_multiple_seeds_legacy
from claimsem.metrics import evaluate_multiple_seeds, summarize_seed_evaluations

SEEDS = (17, 42, 73)
LABEL_LEVELS = ("section", "class", "subclass")
CACHE_KEY = "root12_d010"

EXPECTED_ROWS = 9855
EXPECTED_DIM = 128
N_CLUSTERS = 30

EXPECTED_SEED42_MEAN = {
    "nmi": 0.3704923641293494,
    "predicted_cluster_purity": 0.4186707255197017,
    "label_wise_inverse_purity": 0.34780991036698717,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce cached DEV clustering with the "
            "legacy Depth-OT GPU spherical K-means backend."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/content/drive/MyDrive/depth_ot_patent"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/cached_dev_reproduction"),
    )
    parser.add_argument(
        "--require-exact-seed42",
        action="store_true",
        help=(
            "Fail unless recomputed seed-42 labels exactly "
            "match the saved legacy DEV predictions."
        ),
    )
    parser.add_argument(
        "--metric-atol",
        type=float,
        default=1e-9,
        help=("Absolute tolerance for comparing metrics across scikit-learn versions."),
    )

    return parser.parse_args()


def find_candidates(
    root: Path,
    filename: str,
) -> list[Path]:
    candidates = list(root.rglob(filename))

    candidates.sort(
        key=lambda path: (
            "epoch016_gpu_claimwise_semantic_fusion_dev_search" in str(path),
            path.stat().st_mtime,
        ),
        reverse=True,
    )

    return candidates


def find_artifact(
    root: Path,
    filename: str,
) -> Path:
    candidates = find_candidates(
        root,
        filename,
    )

    if not candidates:
        raise FileNotFoundError(f"{filename} was not found under {root}.")

    if len(candidates) > 1:
        print(f"\nCandidates for {filename}:")

        for index, path in enumerate(candidates):
            print(f"  [{index}] {path}")

    selected = candidates[0]
    print(f"Selected {filename}: {selected}")

    return selected


def find_optional_artifact(
    root: Path,
    filename: str,
) -> Path | None:
    candidates = find_candidates(
        root,
        filename,
    )

    if not candidates:
        return None

    selected = candidates[0]
    print(f"Selected {filename}: {selected}")

    return selected


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def save_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def compare_labels(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    reference_array = np.asarray(
        reference,
        dtype=np.int64,
    )
    candidate_array = np.asarray(
        candidate,
        dtype=np.int64,
    )

    if reference_array.shape != candidate_array.shape:
        raise ValueError(
            "Prediction shape mismatch: "
            f"{reference_array.shape} versus "
            f"{candidate_array.shape}."
        )

    return {
        "exact_label_match": bool(
            np.array_equal(
                reference_array,
                candidate_array,
            )
        ),
        "direct_label_match_rate": float(np.mean(reference_array == candidate_array)),
        "mismatch_count": int(np.count_nonzero(reference_array != candidate_array)),
        "adjusted_rand_index": float(
            adjusted_rand_score(
                reference_array,
                candidate_array,
            )
        ),
        "clustering_nmi": float(
            normalized_mutual_info_score(
                reference_array,
                candidate_array,
            )
        ),
    }


def load_saved_seed42(
    path: Path,
    expected_rows: int,
) -> np.ndarray:
    labels = np.load(
        path,
        allow_pickle=False,
    )

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    if labels.ndim > 1:
        labels = labels.reshape(-1)

    if labels.shape != (expected_rows,):
        raise ValueError(
            f"Saved seed-42 prediction shape is "
            f"{labels.shape}; expected {(expected_rows,)}."
        )

    return labels


def main() -> None:
    args = parse_args()

    project_root = args.project_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not project_root.exists():
        raise FileNotFoundError(f"Project root not found: {project_root}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exact legacy DEV reproduction.")

    records_path = find_artifact(
        project_root,
        "dev_records.pkl",
    )
    pca_cache_path = find_artifact(
        project_root,
        "claimwise_semantic_pca128_all_pools.npz",
    )
    assignments_path = find_artifact(
        project_root,
        "selected_claimwise_fusion_dev_assignments.csv",
    )
    saved_seed42_path = find_optional_artifact(
        project_root,
        "selected_claimwise_fusion_dev_predictions.npy",
    )

    print("\n" + "=" * 88)
    print("CACHED DEV LEGACY REPRODUCTION")
    print("=" * 88)
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("Records:", records_path)
    print("PCA cache:", pca_cache_path)
    print("PCA key:", CACHE_KEY)
    print("Assignments:", assignments_path)
    print("Saved seed 42:", saved_seed42_path)
    print("Output:", output_dir)

    available_keys = list_npz_keys(pca_cache_path)

    if CACHE_KEY not in available_keys:
        raise KeyError(
            f"Cache key {CACHE_KEY!r} was not found. Available keys: {available_keys}"
        )

    records = load_legacy_records(
        records_path,
        expected_count=EXPECTED_ROWS,
    )

    evaluation_records = make_evaluation_records(records)

    features = load_npz_matrix(
        pca_cache_path,
        CACHE_KEY,
        expected_rows=EXPECTED_ROWS,
        expected_dim=EXPECTED_DIM,
    )

    cache_info = describe_cached_matrix(
        features,
        path=pca_cache_path,
        key=CACHE_KEY,
    )

    alignment = validate_assignment_alignment(
        records,
        assignments_path,
    )

    print("\n" + "=" * 88)
    print("INPUT VALIDATION")
    print("=" * 88)
    print("Feature shape:", features.shape)
    print("Feature dtype:", features.dtype)
    print(
        "Feature norms:",
        {
            "min": cache_info.row_norm_min,
            "mean": cache_info.row_norm_mean,
            "max": cache_info.row_norm_max,
        },
    )
    print("Alignment valid:", alignment.valid)

    if not np.allclose(
        np.linalg.norm(features, axis=1),
        1.0,
        atol=1e-5,
    ):
        raise RuntimeError("DEV cached PCA vectors are not L2-normalized.")

    results = run_multiple_seeds_legacy(
        vectors=features,
        n_clusters=N_CLUSTERS,
        seeds=SEEDS,
        max_iter=100,
        tolerance=1e-5,
        device="cuda",
    )

    evaluations = evaluate_multiple_seeds(
        records=evaluation_records,
        clustering_results=results,
        label_levels=LABEL_LEVELS,
    )

    summary = summarize_seed_evaluations(
        evaluations=evaluations,
        label_levels=LABEL_LEVELS,
    )

    result_by_seed = {result.seed: result for result in results}

    evaluation_by_seed = {
        int(evaluation["seed"]): evaluation for evaluation in evaluations
    }

    seed42_evaluation = evaluation_by_seed[42]
    seed42_metric_comparison: dict[str, Any] = {}

    for metric, expected in EXPECTED_SEED42_MEAN.items():
        actual = float(seed42_evaluation["mean"][metric])
        difference = actual - expected

        seed42_metric_comparison[metric] = {
            "expected": expected,
            "actual": actual,
            "difference": difference,
            "matches": bool(abs(difference) <= args.metric_atol),
        }

    seed42_metrics_match = all(
        item["matches"] for item in seed42_metric_comparison.values()
    )

    seed42_prediction_comparison = None

    if saved_seed42_path is not None:
        saved_seed42 = load_saved_seed42(
            saved_seed42_path,
            EXPECTED_ROWS,
        )

        seed42_prediction_comparison = compare_labels(
            saved_seed42,
            result_by_seed[42].labels,
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cluster_dir = output_dir / "clusters"
    cluster_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for result in results:
        np.savez_compressed(
            cluster_dir / f"clustering_seed_{result.seed}.npz",
            labels=result.labels,
            centroids=result.centroids,
            cluster_counts=result.cluster_counts,
            objective=np.asarray(
                [result.objective],
                dtype=np.float64,
            ),
            mean_cosine_similarity=np.asarray(
                [1.0 - result.objective],
                dtype=np.float64,
            ),
            n_iterations=np.asarray(
                [result.n_iterations],
                dtype=np.int64,
            ),
            converged=np.asarray(
                [result.converged],
                dtype=np.bool_,
            ),
            seed=np.asarray(
                [result.seed],
                dtype=np.int64,
            ),
        )

    np.savez_compressed(
        output_dir / "predictions.npz",
        **{f"seed_{result.seed}": result.labels for result in results},
    )

    save_json(
        output_dir / "seed_evaluations.json",
        evaluations,
    )
    save_json(
        output_dir / "metrics_summary.json",
        summary,
    )
    save_json(
        output_dir / "alignment_report.json",
        alignment.to_dict(),
    )
    save_json(
        output_dir / "cache_info.json",
        cache_info.to_dict(),
    )

    per_seed_runtime_summary = []

    for result in results:
        evaluation = evaluation_by_seed[result.seed]

        row = {
            "seed": result.seed,
            "iterations": result.n_iterations,
            "converged": result.converged,
            "active_clusters": result.n_active_clusters,
            "max_cluster_share": result.max_cluster_share,
            "mean_cosine_similarity": float(1.0 - result.objective),
            "mean_cosine_distance": result.objective,
            "mean_metrics": evaluation["mean"],
        }

        per_seed_runtime_summary.append(row)

        print("\nSeed:", result.seed)
        print(
            json.dumps(
                row,
                indent=2,
            )
        )

    report = {
        "experiment": ("cached_dev_legacy_reproduction"),
        "backend": "legacy_gpu",
        "split": "dev",
        "cache_key": CACHE_KEY,
        "n_samples": EXPECTED_ROWS,
        "feature_dimension": EXPECTED_DIM,
        "n_clusters": N_CLUSTERS,
        "seeds": list(SEEDS),
        "max_iter": 100,
        "tolerance": 1e-5,
        "objective_definition": ("mean_cosine_distance"),
        "alignment_valid": alignment.valid,
        "seed42_metrics_match": (seed42_metrics_match),
        "seed42_metric_comparison": (seed42_metric_comparison),
        "seed42_prediction_comparison": (seed42_prediction_comparison),
        "per_seed": (per_seed_runtime_summary),
        "metrics": summary,
        "cache_info": cache_info.to_dict(),
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        },
        "inputs": {
            "records": str(records_path),
            "pca_cache": str(pca_cache_path),
            "assignments": str(assignments_path),
            "saved_seed42_predictions": (
                str(saved_seed42_path) if saved_seed42_path is not None else None
            ),
        },
        "checksums": {
            "records_sha256": sha256_file(records_path),
            "pca_cache_sha256": sha256_file(pca_cache_path),
            "assignments_sha256": sha256_file(assignments_path),
            "saved_seed42_predictions_sha256": (
                sha256_file(saved_seed42_path)
                if saved_seed42_path is not None
                else None
            ),
        },
    }

    save_json(
        output_dir / "reproduction_report.json",
        report,
    )

    print("\n" + "=" * 88)
    print("FINAL DEV REPRODUCTION RESULT")
    print("=" * 88)
    print(
        "Alignment valid:",
        alignment.valid,
    )
    print(
        "Seed-42 metrics match:",
        seed42_metrics_match,
    )
    print(
        "Seed-42 metric comparison:",
    )
    print(
        json.dumps(
            seed42_metric_comparison,
            indent=2,
        )
    )

    if seed42_prediction_comparison is not None:
        print(
            "Seed-42 prediction comparison:",
        )
        print(
            json.dumps(
                seed42_prediction_comparison,
                indent=2,
            )
        )

    print("Multi-seed metrics:")
    print(
        json.dumps(
            summary["mean"],
            indent=2,
        )
    )
    print(
        "Report:",
        output_dir / "reproduction_report.json",
    )

    if not seed42_metrics_match:
        raise RuntimeError("Seed-42 DEV metric reproduction failed.")

    if args.require_exact_seed42:
        if seed42_prediction_comparison is None:
            raise RuntimeError("Saved seed-42 predictions were not found.")

        if not seed42_prediction_comparison["exact_label_match"]:
            raise RuntimeError(
                "Seed-42 DEV labels did not exactly match the saved legacy predictions."
            )

    print("Cached DEV legacy reproduction passed.")


if __name__ == "__main__":
    main()
