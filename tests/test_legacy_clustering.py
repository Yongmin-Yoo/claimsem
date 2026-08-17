"""Tests for the legacy Depth-OT GPU spherical K-means backend."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from claimsem.clustering import (
    ClusteringError,
    legacy_gpu_spherical_kmeans,
    run_multiple_seeds_legacy,
)


def make_vectors(
    n_samples: int = 36,
    n_features: int = 8,
) -> np.ndarray:
    """Create deterministic nonzero synthetic vectors."""
    rng = np.random.default_rng(42)

    vectors = rng.normal(
        size=(n_samples, n_features)
    ).astype(np.float32)

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    return (
        vectors
        / np.maximum(norms, 1e-12)
    ).astype(np.float32)


def test_legacy_backend_requires_cuda() -> None:
    """The reproduction backend must reject a CPU device."""
    vectors = make_vectors()

    with pytest.raises(
        ClusteringError,
        match="requires CUDA",
    ):
        legacy_gpu_spherical_kmeans(
            vectors=vectors,
            n_clusters=3,
            seed=42,
            device="cpu",
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Legacy reproduction requires CUDA.",
)
def test_legacy_backend_is_seed_reproducible() -> None:
    """Repeated runs with the same seed must be identical."""
    vectors = make_vectors()

    first = legacy_gpu_spherical_kmeans(
        vectors=vectors,
        n_clusters=4,
        seed=17,
        max_iter=50,
        tolerance=1e-6,
        device="cuda",
    )

    second = legacy_gpu_spherical_kmeans(
        vectors=vectors,
        n_clusters=4,
        seed=17,
        max_iter=50,
        tolerance=1e-6,
        device="cuda",
    )

    assert np.array_equal(
        first.labels,
        second.labels,
    )
    assert np.allclose(
        first.centroids,
        second.centroids,
        atol=0.0,
        rtol=0.0,
    )
    assert np.array_equal(
        first.cluster_counts,
        second.cluster_counts,
    )
    assert first.objective == second.objective
    assert first.n_iterations == second.n_iterations

    assert first.labels.shape == (
        vectors.shape[0],
    )
    assert first.centroids.shape == (
        4,
        vectors.shape[1],
    )
    assert first.cluster_counts.shape == (4,)
    assert int(first.cluster_counts.sum()) == (
        vectors.shape[0]
    )
    assert first.n_active_clusters == 4
    assert np.isfinite(first.objective)

    centroid_norms = np.linalg.norm(
        first.centroids,
        axis=1,
    )

    assert np.allclose(
        centroid_norms,
        1.0,
        atol=1e-5,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Legacy reproduction requires CUDA.",
)
def test_legacy_multiple_seed_runner() -> None:
    """The multi-seed wrapper must preserve seed order."""
    vectors = make_vectors()

    results = run_multiple_seeds_legacy(
        vectors=vectors,
        n_clusters=4,
        seeds=[17, 42, 73],
        max_iter=50,
        tolerance=1e-6,
        device="cuda",
    )

    assert len(results) == 3
    assert [
        result.seed
        for result in results
    ] == [17, 42, 73]

    for result in results:
        assert result.labels.shape == (
            vectors.shape[0],
        )
        assert result.centroids.shape == (
            4,
            vectors.shape[1],
        )
        assert result.cluster_counts.sum() == (
            vectors.shape[0]
        )
        assert result.n_active_clusters == 4


def test_legacy_multiple_seed_runner_rejects_duplicates() -> None:
    """Duplicate seeds must be rejected before GPU execution."""
    vectors = make_vectors()

    with pytest.raises(
        ClusteringError,
        match="must be unique",
    ):
        run_multiple_seeds_legacy(
            vectors=vectors,
            n_clusters=4,
            seeds=[42, 42],
            device="auto",
        )
