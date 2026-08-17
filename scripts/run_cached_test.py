#!/usr/bin/env python3
"""Reproduce the cached legacy ClaimSem TEST clustering experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from claimsem.cache import (
    load_legacy_records,
    load_npy_matrix,
    make_evaluation_records,
)
from claimsem.clustering import (
    run_multiple_seeds_legacy,
)
from claimsem.metrics import (
    evaluate_multiple_seeds,
    summarize_seed_evaluations,
)

SEEDS = (17, 42, 73)
LABEL_LEVELS = (
    "section",
    "class",
    "subclass",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce cached TEST spherical K-means "
            "with the legacy GPU backend."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(
            "/content/drive/MyDrive/depth_ot_patent"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/cached_test_reproduction"
        ),
    )
    parser.add_argument(
        "--require-exact",
        action="store_true",
        help=(
            "Fail unless all recomputed labels exactly "
            "match the saved legacy labels."
        ),
    )

    return parser.parse_args()


def find_artifact(
    root: Path,
    filename: str,
) -> Path:
    candidates = list(
        root.rglob(filename)
    )

    if not candidates:
        raise FileNotFoundError(
            f"{filename} was not found under {root}."
        )

    candidates.sort(
        key=lambda path: (
            "final_fixed_root12_d010_claimwise_test_evaluation"
            in str(path),
            path.stat().st_mtime,
        ),
        reverse=True,
    )

    return candidates[0]


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


def main() -> None:
    args = parse_args()

    project_root = (
        args.project_root.expanduser().resolve()
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for exact legacy reproduction."
        )

    pca_path = find_artifact(
        project_root,
        "test_root12_d010_pca128.npy",
    )
    predictions_path = find_artifact(
        project_root,
        "test_3seed_predictions.npz",
    )
    records_path = find_artifact(
        project_root,
        "test_records.pkl",
    )

    expected_predictions_path = (
        pca_path.parent.parent
        / "test_3seed_predictions.npz"
    )

    if expected_predictions_path.exists():
        predictions_path = (
            expected_predictions_path
        )

    print("=" * 88)
    print("CACHED TEST LEGACY REPRODUCTION")
    print("=" * 88)
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("PCA cache:", pca_path)
    print("Predictions:", predictions_path)
    print("Records:", records_path)
    print("Output:", output_dir)

    features = load_npy_matrix(
        pca_path,
        expected_rows=9881,
        expected_dim=128,
        mmap_mode=None,
    )

    records = load_legacy_records(
        records_path,
        expected_count=9881,
    )

    evaluation_records = (
        make_evaluation_records(records)
    )

    with np.load(
        predictions_path,
        allow_pickle=False,
    ) as archive:
        saved_predictions = {
            seed: np.asarray(
                archive[f"seed_{seed}"],
                dtype=np.int64,
            )
            for seed in SEEDS
        }

    results = run_multiple_seeds_legacy(
        vectors=features,
        n_clusters=30,
        seeds=SEEDS,
        max_iter=100,
        tolerance=1e-5,
        device="cuda",
    )

    comparisons: list[dict[str, Any]] = []

    for result in results:
        saved = saved_predictions[
            result.seed
        ]

        exact_match = bool(
            np.array_equal(
                result.labels,
                saved,
            )
        )

        direct_match_rate = float(
            np.mean(
                result.labels == saved
            )
        )

        ari = float(
            adjusted_rand_score(
                saved,
                result.labels,
            )
        )

        clustering_nmi = float(
            normalized_mutual_info_score(
                saved,
                result.labels,
            )
        )

        comparison = {
            "seed": result.seed,
            "exact_label_match": exact_match,
            "direct_label_match_rate": (
                direct_match_rate
            ),
            "mismatch_count": int(
                np.count_nonzero(
                    result.labels != saved
                )
            ),
            "adjusted_rand_index": ari,
            "clustering_nmi": clustering_nmi,
            "mean_cosine_similarity": float(
                1.0 - result.objective
            ),
            "mean_cosine_distance": float(
                result.objective
            ),
            "iterations": (
                result.n_iterations
            ),
            "converged": result.converged,
            "active_clusters": (
                result.n_active_clusters
            ),
        }

        comparisons.append(
            comparison
        )

        print("\nSeed:", result.seed)
        print(
            json.dumps(
                comparison,
                indent=2,
            )
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

    all_exact = all(
        comparison["exact_label_match"]
        for comparison in comparisons
    )

    all_partitions_equal = all(
        np.isclose(
            comparison[
                "adjusted_rand_index"
            ],
            1.0,
            atol=1e-12,
        )
        and np.isclose(
            comparison["clustering_nmi"],
            1.0,
            atol=1e-12,
        )
        for comparison in comparisons
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_dir
        / "recomputed_predictions.npz",
        **{
            f"seed_{result.seed}": (
                result.labels
            )
            for result in results
        },
    )

    save_json(
        output_dir
        / "seed_evaluations.json",
        evaluations,
    )

    save_json(
        output_dir
        / "metrics_summary.json",
        summary,
    )

    report = {
        "experiment": (
            "cached_test_legacy_reproduction"
        ),
        "backend": "legacy_gpu",
        "split": "test",
        "n_samples": int(
            features.shape[0]
        ),
        "feature_dimension": int(
            features.shape[1]
        ),
        "n_clusters": 30,
        "seeds": list(SEEDS),
        "max_iter": 100,
        "tolerance": 1e-5,
        "objective_definition": (
            "mean_cosine_distance"
        ),
        "all_exact_label_matches": (
            all_exact
        ),
        "all_partitions_equal": (
            all_partitions_equal
        ),
        "comparisons": comparisons,
        "metrics": summary,
        "environment": {
            "gpu": (
                torch.cuda.get_device_name(0)
            ),
            "torch_version": (
                torch.__version__
            ),
            "cuda_version": (
                torch.version.cuda
            ),
            "tf32_matmul": bool(
                torch.backends.cuda
                .matmul.allow_tf32
            ),
        },
        "inputs": {
            "pca_features": str(
                pca_path
            ),
            "saved_predictions": str(
                predictions_path
            ),
            "records": str(
                records_path
            ),
        },
        "checksums": {
            "pca_features_sha256": (
                sha256_file(pca_path)
            ),
            "saved_predictions_sha256": (
                sha256_file(
                    predictions_path
                )
            ),
            "records_sha256": (
                sha256_file(records_path)
            ),
        },
    }

    save_json(
        output_dir
        / "reproduction_report.json",
        report,
    )

    print("\n" + "=" * 88)
    print("FINAL RESULT")
    print("=" * 88)
    print(
        "All exact label matches:",
        all_exact,
    )
    print(
        "All partitions equal:",
        all_partitions_equal,
    )
    print(
        "Mean metrics:",
        summary["mean"],
    )
    print(
        "Report:",
        output_dir
        / "reproduction_report.json",
    )

    if (
        args.require_exact
        and not all_exact
    ):
        raise RuntimeError(
            "Exact legacy reproduction failed."
        )

    if all_exact:
        print(
            "Cached TEST legacy reproduction passed."
        )


if __name__ == "__main__":
    main()
