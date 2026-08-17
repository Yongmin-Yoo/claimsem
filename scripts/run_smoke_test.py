#!/usr/bin/env python
"""Run the complete ClaimSem synthetic smoke test."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _find_project_root() -> Path:
    """Return the repository root from the script location."""
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = _find_project_root()
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from claimsem.artifacts import (  # noqa: E402
    create_manifest,
    ensure_directory,
    save_clustering_archives,
    save_json,
    save_numpy_array,
    save_predictions_csv,
)
from claimsem.clustering import run_multiple_seeds  # noqa: E402
from claimsem.config import load_config  # noqa: E402
from claimsem.data import load_records, summarize_records  # noqa: E402
from claimsem.metrics import (  # noqa: E402
    evaluate_multiple_seeds,
    summarize_seed_evaluations,
)
from claimsem.pooling import pool_records  # noqa: E402
from claimsem.reduction import fit_transform_pca, save_pca  # noqa: E402
from claimsem.reproducibility import (  # noqa: E402
    collect_environment_info,
    set_global_seed,
)


class SmokeTestError(RuntimeError):
    """Raised when the smoke-test configuration is unsupported."""


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the complete ClaimSem synthetic smoke test."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "smoke_test.json",
        help="Path to the smoke-test JSON configuration.",
    )

    return parser.parse_args()


def prepare_output_directory(
    output_root: str | Path,
    *,
    overwrite: bool,
    resume: bool,
) -> Path:
    """Prepare the smoke-test output directory."""
    output_dir = Path(output_root).expanduser().resolve()

    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        elif not resume and any(output_dir.iterdir()):
            raise SmokeTestError(
                f"Output directory is not empty: {output_dir}. "
                "Enable runtime.overwrite or runtime.resume."
            )

    return ensure_directory(output_dir)


def load_smoke_records(
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Load and normalize the configured synthetic patent records."""
    paths = config["paths"]
    data_config = config["data"]

    return load_records(
        paths["records"],
        data_format=data_config.get("format"),
        patent_id_field=data_config.get(
            "patent_id_field",
            "patent_id",
        ),
        claims_field=data_config.get(
            "claims_field",
            "claims",
        ),
        claim_id_field=data_config.get(
            "claim_id_field",
            "claim_id",
        ),
        claim_text_field=data_config.get(
            "claim_text_field",
            "text",
        ),
        parent_ids_field=data_config.get(
            "parent_ids_field",
            "parent_ids",
        ),
        cpc_field=data_config.get(
            "cpc_field",
            "cpc",
        ),
        remove_invalid_references=data_config.get(
            "remove_invalid_references",
            False,
        ),
        require_acyclic_graph=data_config.get(
            "require_acyclic_graph",
            True,
        ),
    )


def generate_synthetic_claim_embeddings(
    records: list[dict[str, Any]],
    *,
    embedding_dim: int,
    seed: int,
    dtype: str = "float32",
) -> list[np.ndarray]:
    """Generate deterministic synthetic claim embeddings."""
    if embedding_dim <= 0:
        raise SmokeTestError("Synthetic embedding dimension must be positive.")

    supported_dtypes = {
        "float32": np.float32,
        "float64": np.float64,
    }

    if dtype not in supported_dtypes:
        raise SmokeTestError(f"Unsupported synthetic feature dtype: {dtype}")

    rng = np.random.default_rng(seed)
    numpy_dtype = supported_dtypes[dtype]

    embeddings: list[np.ndarray] = []

    for record in records:
        n_claims = len(record["claims"])

        if n_claims == 0:
            raise SmokeTestError(f"Patent {record['patent_id']!r} has no claims.")

        matrix = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(n_claims, embedding_dim),
        ).astype(numpy_dtype)

        embeddings.append(matrix)

    return embeddings


def verify_pipeline_outputs(
    records: list[dict[str, Any]],
    patent_vectors: np.ndarray,
    reduced_vectors: np.ndarray,
    clustering_results: list[Any],
    *,
    embedding_dim: int,
    pca_dim: int,
    n_clusters: int,
) -> None:
    """Check important smoke-test invariants."""
    n_patents = len(records)

    if patent_vectors.shape != (n_patents, embedding_dim):
        raise SmokeTestError(
            f"Unexpected pooled representation shape: {patent_vectors.shape}"
        )

    if reduced_vectors.shape != (n_patents, pca_dim):
        raise SmokeTestError(
            f"Unexpected reduced representation shape: {reduced_vectors.shape}"
        )

    if not np.all(np.isfinite(patent_vectors)):
        raise SmokeTestError("Pooled patent representations contain non-finite values.")

    if not np.all(np.isfinite(reduced_vectors)):
        raise SmokeTestError(
            "Reduced patent representations contain non-finite values."
        )

    norms = np.linalg.norm(reduced_vectors, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-5):
        raise SmokeTestError("Reduced patent representations are not L2-normalized.")

    for result in clustering_results:
        if result.labels.shape != (n_patents,):
            raise SmokeTestError(
                f"Invalid label shape for seed {result.seed}: {result.labels.shape}"
            )

        if result.centroids.shape != (n_clusters, pca_dim):
            raise SmokeTestError(
                f"Invalid centroid shape for seed {result.seed}: "
                f"{result.centroids.shape}"
            )

        if result.cluster_counts.sum() != n_patents:
            raise SmokeTestError(
                f"Cluster counts do not sum to {n_patents} for seed {result.seed}."
            )

        if result.n_active_clusters != n_clusters:
            raise SmokeTestError(
                f"Seed {result.seed} contains "
                f"{result.n_active_clusters} active clusters, "
                f"expected {n_clusters}."
            )


def run_smoke_test(config_path: str | Path) -> dict[str, Any]:
    """Execute the complete synthetic ClaimSem pipeline."""
    config = load_config(config_path)

    if config["features"].get("mode") != "synthetic":
        raise SmokeTestError("run_smoke_test.py requires features.mode='synthetic'.")

    set_global_seed(
        int(config["global_seed"]),
        deterministic=False,
    )

    runtime = config["runtime"]

    output_dir = prepare_output_directory(
        config["paths"]["output_root"],
        overwrite=bool(runtime.get("overwrite", False)),
        resume=bool(runtime.get("resume", False)),
    )

    print("=" * 72)
    print("ClaimSem synthetic smoke test")
    print("=" * 72)
    print("Configuration:", config["_metadata"]["config_path"])
    print("Records:", config["paths"]["records"])
    print("Output:", output_dir)

    print("\n[1/8] Loading and validating patent records")

    records = load_smoke_records(config)
    dataset_summary = summarize_records(records)

    print("Patents:", dataset_summary.n_patents)
    print("Claims:", dataset_summary.n_claims)
    print("Edges:", dataset_summary.n_edges)
    print("Roots:", dataset_summary.n_roots)
    print("Maximum depth:", dataset_summary.max_depth)

    print("\n[2/8] Generating deterministic synthetic embeddings")

    feature_config = config["features"]
    embedding_dim = int(feature_config["embedding_dim"])

    claim_embeddings = generate_synthetic_claim_embeddings(
        records=records,
        embedding_dim=embedding_dim,
        seed=int(
            feature_config.get(
                "synthetic_seed",
                config["global_seed"],
            )
        ),
        dtype=str(feature_config.get("dtype", "float32")),
    )

    total_embedding_rows = sum(matrix.shape[0] for matrix in claim_embeddings)

    if total_embedding_rows != dataset_summary.n_claims:
        raise SmokeTestError("Synthetic embedding count does not match claim count.")

    print("Claim embeddings:", total_embedding_rows)
    print("Embedding dimension:", embedding_dim)

    print("\n[3/8] Applying root and depth-aware pooling")

    pooling_config = config["pooling"]

    patent_vectors = pool_records(
        records=records,
        claim_embeddings=claim_embeddings,
        root_weight=float(pooling_config["root_weight"]),
        depth_decay=float(pooling_config["depth_decay"]),
        claim_selection="all",
    )

    print("Pooled shape:", patent_vectors.shape)

    print("\n[4/8] Fitting PCA and applying L2 normalization")

    pca_config = config["pca"]

    if not pca_config.get("enabled", True):
        raise SmokeTestError("The current smoke test requires PCA to be enabled.")

    pca_dim = int(pca_config["output_dim"])

    pca, reduced_vectors = fit_transform_pca(
        vectors=patent_vectors,
        output_dim=pca_dim,
        random_state=int(pca_config.get("random_state", 42)),
        whiten=bool(pca_config.get("whiten", False)),
        svd_solver=str(pca_config.get("svd_solver", "auto")),
        apply_l2_normalization=True,
    )

    print("Reduced shape:", reduced_vectors.shape)
    print(
        "Explained variance ratio sum:",
        float(pca.explained_variance_ratio_.sum()),
    )

    print("\n[5/8] Running multi-seed spherical K-means")

    clustering_config = config["clustering"]
    n_clusters = int(clustering_config["n_clusters"])
    seeds = [int(seed) for seed in clustering_config["seeds"]]

    clustering_results = run_multiple_seeds(
        vectors=reduced_vectors,
        n_clusters=n_clusters,
        seeds=seeds,
        max_iter=int(clustering_config["max_iter"]),
        tolerance=float(clustering_config["tolerance"]),
        device=str(clustering_config.get("device", "auto")),
    )

    for result in clustering_results:
        print(
            f"Seed {result.seed}: "
            f"objective={result.objective:.6f}, "
            f"active={result.n_active_clusters}, "
            f"iterations={result.n_iterations}, "
            f"converged={result.converged}"
        )

    verify_pipeline_outputs(
        records=records,
        patent_vectors=patent_vectors,
        reduced_vectors=reduced_vectors,
        clustering_results=clustering_results,
        embedding_dim=embedding_dim,
        pca_dim=pca_dim,
        n_clusters=n_clusters,
    )

    print("\n[6/8] Evaluating CPC alignment")

    label_levels = config["evaluation"]["label_levels"]

    evaluations = evaluate_multiple_seeds(
        records=records,
        clustering_results=clustering_results,
        label_levels=label_levels,
    )

    metrics_summary = summarize_seed_evaluations(
        evaluations=evaluations,
        label_levels=label_levels,
    )

    mean_metrics = metrics_summary["mean"]

    print(
        "Mean NMI:",
        f"{mean_metrics['nmi']['mean']:.6f}",
        "+/-",
        f"{mean_metrics['nmi']['std']:.6f}",
    )
    print(
        "Mean Pur_p:",
        f"{mean_metrics['predicted_cluster_purity']['mean']:.6f}",
        "+/-",
        f"{mean_metrics['predicted_cluster_purity']['std']:.6f}",
    )
    print(
        "Mean Pur_a:",
        f"{mean_metrics['label_wise_inverse_purity']['mean']:.6f}",
        "+/-",
        f"{mean_metrics['label_wise_inverse_purity']['std']:.6f}",
    )

    print("\n[7/8] Saving artifacts")

    saved_files: list[Path] = []

    config_snapshot_path = save_json(
        config,
        output_dir / "config_snapshot.json",
    )
    saved_files.append(config_snapshot_path)

    dataset_summary_path = save_json(
        dataset_summary.to_dict(),
        output_dir / "dataset_summary.json",
    )
    saved_files.append(dataset_summary_path)

    patent_vectors_path = save_numpy_array(
        patent_vectors,
        output_dir / "patent_vectors.npy",
    )
    saved_files.append(patent_vectors_path)

    reduced_vectors_path = save_numpy_array(
        reduced_vectors,
        output_dir / "reduced_vectors.npy",
    )
    saved_files.append(reduced_vectors_path)

    pca_path = save_pca(
        pca=pca,
        path=output_dir / f"pca_dim{pca_dim}.joblib",
        metadata={
            "fit_split": pca_config.get(
                "fit_split",
                "smoke_test",
            ),
            "input_dim": embedding_dim,
            "output_dim": pca_dim,
            "random_state": int(pca_config.get("random_state", 42)),
        },
    )
    saved_files.append(pca_path)

    evaluations_path = save_json(
        evaluations,
        output_dir / "seed_evaluations.json",
    )
    saved_files.append(evaluations_path)

    metrics_path = save_json(
        metrics_summary,
        output_dir / "metrics_summary.json",
    )
    saved_files.append(metrics_path)

    predictions_path = save_predictions_csv(
        records=records,
        clustering_results=clustering_results,
        path=output_dir / "predictions.csv",
    )
    saved_files.append(predictions_path)

    cluster_paths = save_clustering_archives(
        clustering_results=clustering_results,
        output_directory=output_dir / "clusters",
    )
    saved_files.extend(cluster_paths)

    run_summary = {
        "experiment_name": config["experiment_name"],
        "status": "passed",
        "scientific_result": False,
        "purpose": "software smoke test",
        "dataset": dataset_summary.to_dict(),
        "representation_shapes": {
            "pooled": list(patent_vectors.shape),
            "reduced": list(reduced_vectors.shape),
        },
        "seeds": seeds,
        "metrics": metrics_summary,
    }

    run_summary_path = save_json(
        run_summary,
        output_dir / "run_summary.json",
    )
    saved_files.append(run_summary_path)

    print("\n[8/8] Creating reproducibility manifest")

    environment = collect_environment_info(project_root=PROJECT_ROOT)

    manifest = create_manifest(
        experiment_name=config["experiment_name"],
        config=config,
        input_files=[config["paths"]["records"]],
        output_files=saved_files,
        environment=environment,
        extra={
            "status": "passed",
            "purpose": "software smoke test",
            "scientific_result": False,
        },
    )

    manifest_path = save_json(
        manifest,
        output_dir / "manifest.json",
    )

    print("\n" + "=" * 72)
    print("Smoke test passed")
    print("=" * 72)
    print("Manifest:", manifest_path)
    print("Output directory:", output_dir)
    print(
        "Important: synthetic smoke-test metrics are not scientific experiment results."
    )

    return {
        "config": config,
        "records": records,
        "dataset_summary": dataset_summary,
        "patent_vectors": patent_vectors,
        "reduced_vectors": reduced_vectors,
        "clustering_results": clustering_results,
        "evaluations": evaluations,
        "metrics_summary": metrics_summary,
        "manifest_path": manifest_path,
    }


def main() -> None:
    """Command-line entry point."""
    arguments = parse_arguments()
    run_smoke_test(arguments.config)


if __name__ == "__main__":
    main()
