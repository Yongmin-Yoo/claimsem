"""Spherical K-means clustering for ClaimSem."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F


class ClusteringError(ValueError):
    """Raised when spherical K-means clustering cannot be performed."""


@dataclass
class SphericalKMeansResult:
    """Result of one spherical K-means run."""

    labels: np.ndarray
    centroids: np.ndarray
    cluster_counts: np.ndarray
    objective: float
    n_iterations: int
    converged: bool
    seed: int

    @property
    def n_clusters(self) -> int:
        """Return the configured number of clusters."""
        return int(self.centroids.shape[0])

    @property
    def n_active_clusters(self) -> int:
        """Return the number of clusters containing at least one sample."""
        return int(np.count_nonzero(self.cluster_counts))

    @property
    def max_cluster_share(self) -> float:
        """Return the fraction of samples in the largest cluster."""
        total = int(self.cluster_counts.sum())

        if total == 0:
            return 0.0

        return float(self.cluster_counts.max() / total)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable clustering summary."""
        return {
            "seed": self.seed,
            "n_clusters": self.n_clusters,
            "n_active_clusters": self.n_active_clusters,
            "objective": self.objective,
            "n_iterations": self.n_iterations,
            "converged": self.converged,
            "max_cluster_share": self.max_cluster_share,
            "cluster_counts": self.cluster_counts.tolist(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the complete result as a dictionary."""
        result = asdict(self)
        result["labels"] = self.labels.tolist()
        result["centroids"] = self.centroids.tolist()
        result["cluster_counts"] = self.cluster_counts.tolist()
        result["n_clusters"] = self.n_clusters
        result["n_active_clusters"] = self.n_active_clusters
        result["max_cluster_share"] = self.max_cluster_share
        return result


def _validate_vectors(vectors: np.ndarray) -> np.ndarray:
    """Validate a patent representation matrix."""
    matrix = np.asarray(vectors)

    if matrix.ndim != 2:
        raise ClusteringError(
            "vectors must have shape (n_samples, n_features), "
            f"got {matrix.shape}."
        )

    if matrix.shape[0] == 0:
        raise ClusteringError("At least one patent vector is required.")

    if matrix.shape[1] == 0:
        raise ClusteringError("Patent vectors must have at least one feature.")

    if not np.issubdtype(matrix.dtype, np.number):
        raise ClusteringError("Patent vectors must contain numeric values.")

    if not np.all(np.isfinite(matrix)):
        raise ClusteringError("Patent vectors contain non-finite values.")

    row_norms = np.linalg.norm(matrix, axis=1)

    if np.any(row_norms <= 1e-12):
        indices = np.flatnonzero(row_norms <= 1e-12).tolist()
        raise ClusteringError(
            "Zero or near-zero patent vectors cannot be clustered. "
            f"Affected row indices: {indices[:20]}."
        )

    normalized = matrix / row_norms[:, None]

    return normalized.astype(np.float32, copy=False)


def _resolve_device(requested: str | torch.device = "auto") -> torch.device:
    """Resolve the requested PyTorch device."""
    if isinstance(requested, torch.device):
        device = requested
    else:
        value = str(requested).lower()

        if value == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(value)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise ClusteringError("CUDA was requested but is not available.")

    if device.type == "mps":
        available = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
        if not available:
            raise ClusteringError("MPS was requested but is not available.")

    return device


def _kmeans_plus_plus(
    vectors: np.ndarray,
    n_clusters: int,
    seed: int,
) -> np.ndarray:
    """Initialize normalized centroids using cosine K-means++."""
    rng = np.random.default_rng(seed)
    n_samples = vectors.shape[0]

    first_index = int(rng.integers(0, n_samples))
    selected_indices = [first_index]
    selected_mask = np.zeros(n_samples, dtype=bool)
    selected_mask[first_index] = True

    closest_distance = 1.0 - np.clip(
        vectors @ vectors[first_index],
        -1.0,
        1.0,
    )
    closest_distance = np.maximum(closest_distance, 0.0)

    while len(selected_indices) < n_clusters:
        probabilities = closest_distance.copy()
        probabilities[selected_mask] = 0.0

        probability_sum = float(probabilities.sum())

        if probability_sum <= 1e-12:
            candidates = np.flatnonzero(~selected_mask)

            if candidates.size == 0:
                break

            next_index = int(rng.choice(candidates))
        else:
            probabilities /= probability_sum
            next_index = int(rng.choice(n_samples, p=probabilities))

            if selected_mask[next_index]:
                candidates = np.flatnonzero(~selected_mask)

                if candidates.size == 0:
                    break

                next_index = int(rng.choice(candidates))

        selected_indices.append(next_index)
        selected_mask[next_index] = True

        new_distance = 1.0 - np.clip(
            vectors @ vectors[next_index],
            -1.0,
            1.0,
        )
        new_distance = np.maximum(new_distance, 0.0)
        closest_distance = np.minimum(closest_distance, new_distance)

    if len(selected_indices) != n_clusters:
        raise ClusteringError(
            "K-means++ initialization did not produce the requested "
            f"{n_clusters} centroids."
        )

    centroids = vectors[np.asarray(selected_indices, dtype=np.int64)]
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.maximum(centroid_norms, 1e-12)

    return centroids.astype(np.float32, copy=False)


def spherical_kmeans(
    vectors: np.ndarray,
    n_clusters: int,
    seed: int = 42,
    max_iter: int = 300,
    tolerance: float = 1e-6,
    device: str | torch.device = "auto",
) -> SphericalKMeansResult:
    """Cluster L2-normalized patent vectors using spherical K-means.

    Cluster assignment maximizes cosine similarity. Cluster centroids are
    recomputed as normalized means of their assigned patent vectors.
    """
    matrix = _validate_vectors(vectors)
    n_samples, n_features = matrix.shape

    if not isinstance(n_clusters, int) or isinstance(n_clusters, bool):
        raise ClusteringError("n_clusters must be an integer.")

    if n_clusters <= 1:
        raise ClusteringError("n_clusters must be greater than one.")

    if n_clusters > n_samples:
        raise ClusteringError(
            f"n_clusters={n_clusters} exceeds n_samples={n_samples}."
        )

    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ClusteringError("max_iter must be a positive integer.")

    if not np.isfinite(tolerance) or tolerance < 0:
        raise ClusteringError(
            "tolerance must be a non-negative finite number."
        )

    resolved_device = _resolve_device(device)

    initial_centroids = _kmeans_plus_plus(
        vectors=matrix,
        n_clusters=n_clusters,
        seed=int(seed),
    )

    tensor = torch.as_tensor(
        matrix,
        dtype=torch.float32,
        device=resolved_device,
    )
    centroids = torch.as_tensor(
        initial_centroids,
        dtype=torch.float32,
        device=resolved_device,
    )

    centroids = F.normalize(centroids, p=2, dim=1)

    previous_labels: torch.Tensor | None = None
    converged = False
    n_iterations = 0

    for iteration in range(1, max_iter + 1):
        similarities = tensor @ centroids.T
        labels = torch.argmax(similarities, dim=1)

        sums = torch.zeros(
            (n_clusters, n_features),
            dtype=torch.float32,
            device=resolved_device,
        )
        sums.index_add_(0, labels, tensor)

        counts = torch.bincount(
            labels,
            minlength=n_clusters,
        )

        new_centroids = torch.zeros_like(centroids)
        nonempty_mask = counts > 0

        if torch.any(nonempty_mask):
            new_centroids[nonempty_mask] = (
                sums[nonempty_mask]
                / counts[nonempty_mask].unsqueeze(1)
            )

        empty_indices = torch.nonzero(
            ~nonempty_mask,
            as_tuple=False,
        ).flatten()

        if empty_indices.numel() > 0:
            assigned_similarities = similarities.gather(
                1,
                labels.unsqueeze(1),
            ).squeeze(1)

            candidate_order = torch.argsort(
                assigned_similarities,
                descending=False,
            )

            used_candidates: set[int] = set()
            candidate_position = 0

            for empty_cluster in empty_indices.tolist():
                while candidate_position < candidate_order.numel():
                    candidate_index = int(
                        candidate_order[candidate_position].item()
                    )
                    candidate_position += 1

                    if candidate_index not in used_candidates:
                        used_candidates.add(candidate_index)
                        break
                else:
                    candidate_index = int(
                        empty_cluster % n_samples
                    )

                new_centroids[empty_cluster] = tensor[candidate_index]

        new_centroids = F.normalize(
            new_centroids,
            p=2,
            dim=1,
        )

        centroid_shift = torch.max(
            torch.linalg.vector_norm(
                new_centroids - centroids,
                dim=1,
            )
        ).item()

        labels_unchanged = (
            previous_labels is not None
            and torch.equal(labels, previous_labels)
        )

        centroids = new_centroids
        previous_labels = labels.clone()
        n_iterations = iteration

        if labels_unchanged or centroid_shift <= tolerance:
            converged = True
            break

    final_similarities = tensor @ centroids.T
    final_labels = torch.argmax(final_similarities, dim=1)
    assigned_similarities = final_similarities.gather(
        1,
        final_labels.unsqueeze(1),
    ).squeeze(1)

    objective = torch.mean(
        1.0 - assigned_similarities
    ).item()

    cluster_counts = torch.bincount(
        final_labels,
        minlength=n_clusters,
    )

    return SphericalKMeansResult(
        labels=final_labels.detach().cpu().numpy().astype(np.int64),
        centroids=centroids.detach().cpu().numpy().astype(np.float32),
        cluster_counts=cluster_counts.detach().cpu().numpy().astype(
            np.int64
        ),
        objective=float(objective),
        n_iterations=int(n_iterations),
        converged=bool(converged),
        seed=int(seed),
    )


def run_multiple_seeds(
    vectors: np.ndarray,
    n_clusters: int,
    seeds: Sequence[int] = (17, 42, 73),
    max_iter: int = 300,
    tolerance: float = 1e-6,
    device: str | torch.device = "auto",
) -> list[SphericalKMeansResult]:
    """Run spherical K-means independently for multiple seeds."""
    if not seeds:
        raise ClusteringError("At least one clustering seed is required.")

    normalized_seeds = [int(seed) for seed in seeds]

    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ClusteringError("Clustering seeds must be unique.")

    return [
        spherical_kmeans(
            vectors=vectors,
            n_clusters=n_clusters,
            seed=seed,
            max_iter=max_iter,
            tolerance=tolerance,
            device=device,
        )
        for seed in normalized_seeds
    ]


def assign_to_centroids(
    vectors: np.ndarray,
    centroids: np.ndarray,
    device: str | torch.device = "auto",
) -> np.ndarray:
    """Assign patent vectors to existing spherical centroids."""
    matrix = _validate_vectors(vectors)
    centroid_matrix = _validate_vectors(centroids)

    if matrix.shape[1] != centroid_matrix.shape[1]:
        raise ClusteringError(
            f"Vector dimension {matrix.shape[1]} does not match "
            f"centroid dimension {centroid_matrix.shape[1]}."
        )

    resolved_device = _resolve_device(device)

    tensor = torch.as_tensor(
        matrix,
        dtype=torch.float32,
        device=resolved_device,
    )
    centroid_tensor = torch.as_tensor(
        centroid_matrix,
        dtype=torch.float32,
        device=resolved_device,
    )

    similarities = tensor @ centroid_tensor.T
    labels = torch.argmax(similarities, dim=1)

    return labels.detach().cpu().numpy().astype(np.int64)

# BEGIN LEGACY DEPTH-OT GPU SPHERICAL K-MEANS


def _validate_legacy_vectors(
    vectors: np.ndarray,
) -> np.ndarray:
    """Validate vectors without pre-normalizing them on the CPU.

    The legacy implementation converts the original float32 matrix directly
    to a device tensor and performs L2 normalization on that device. Avoiding
    CPU normalization is necessary for exact reproduction.
    """
    matrix = np.asarray(vectors)

    if matrix.ndim != 2:
        raise ClusteringError(
            "vectors must have shape (n_samples, n_features), "
            f"got {matrix.shape}."
        )

    if matrix.shape[0] == 0:
        raise ClusteringError(
            "At least one patent vector is required."
        )

    if matrix.shape[1] == 0:
        raise ClusteringError(
            "Patent vectors must have at least one feature."
        )

    if not np.issubdtype(matrix.dtype, np.number):
        raise ClusteringError(
            "Patent vectors must contain numeric values."
        )

    if not np.all(np.isfinite(matrix)):
        raise ClusteringError(
            "Patent vectors contain non-finite values."
        )

    row_norms = np.linalg.norm(
        matrix.astype(np.float64, copy=False),
        axis=1,
    )

    if np.any(row_norms <= 1e-12):
        indices = np.flatnonzero(
            row_norms <= 1e-12
        ).tolist()

        raise ClusteringError(
            "Zero or near-zero patent vectors cannot be clustered. "
            f"Affected row indices: {indices[:20]}."
        )

    return matrix.astype(
        np.float32,
        copy=False,
    )


@torch.no_grad()
def legacy_gpu_spherical_kmeans(
    vectors: np.ndarray,
    n_clusters: int,
    seed: int = 42,
    max_iter: int = 100,
    tolerance: float = 1e-5,
    device: str | torch.device = "auto",
) -> SphericalKMeansResult:
    """Reproduce the Depth-OT legacy GPU spherical K-means.

    This backend preserves the implementation used by
    ``proposed_model_section_11.py``:

    * device-side float32 L2 normalization;
    * device-specific ``torch.Generator``;
    * cosine-distance K-means++ sampling with ``torch.multinomial``;
    * empty-cluster replacement using the least-similar samples;
    * convergence based on the change in mean cosine similarity;
    * final label reassignment after the last centroid update.

    ``SphericalKMeansResult.objective`` remains consistent with the public
    ClaimSem API and stores mean cosine distance, i.e.
    ``1 - mean_cosine_similarity``.
    """
    matrix = _validate_legacy_vectors(
        vectors
    )
    n_samples, n_features = matrix.shape

    if (
        not isinstance(n_clusters, int)
        or isinstance(n_clusters, bool)
    ):
        raise ClusteringError(
            "n_clusters must be an integer."
        )

    if n_clusters <= 1:
        raise ClusteringError(
            "n_clusters must be greater than one."
        )

    if n_clusters > n_samples:
        raise ClusteringError(
            f"n_clusters={n_clusters} exceeds "
            f"n_samples={n_samples}."
        )

    if (
        not isinstance(max_iter, int)
        or isinstance(max_iter, bool)
        or max_iter <= 0
    ):
        raise ClusteringError(
            "max_iter must be a positive integer."
        )

    if (
        not np.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ClusteringError(
            "tolerance must be a non-negative "
            "finite number."
        )

    resolved_device = _resolve_device(
        device
    )

    if resolved_device.type != "cuda":
        raise ClusteringError(
            "legacy_gpu_spherical_kmeans requires CUDA "
            "for exact Depth-OT reproduction."
        )

    # Match the original Colab/T4 execution configuration.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    x = torch.as_tensor(
        matrix,
        dtype=torch.float32,
        device=resolved_device,
    )

    x = F.normalize(
        x,
        p=2,
        dim=1,
    )

    generator = torch.Generator(
        device=resolved_device,
    )
    generator.manual_seed(
        int(seed)
    )

    first_index = int(
        torch.randint(
            0,
            n_samples,
            (1,),
            generator=generator,
            device=resolved_device,
        ).item()
    )

    centroid_indices = [
        first_index
    ]

    closest_distance = (
        1.0
        - torch.matmul(
            x,
            x[first_index].unsqueeze(1),
        ).squeeze(1)
    )

    # Preserve the original GPU K-means++ procedure exactly.
    for _ in range(1, n_clusters):
        probabilities = torch.clamp(
            closest_distance,
            min=1e-8,
        )

        probabilities = (
            probabilities
            / probabilities.sum()
        )

        next_index = int(
            torch.multinomial(
                probabilities,
                num_samples=1,
                generator=generator,
            ).item()
        )

        centroid_indices.append(
            next_index
        )

        new_distance = (
            1.0
            - torch.matmul(
                x,
                x[next_index].unsqueeze(1),
            ).squeeze(1)
        )

        closest_distance = torch.minimum(
            closest_distance,
            new_distance,
        )

    centroids = x[
        torch.tensor(
            centroid_indices,
            dtype=torch.long,
            device=resolved_device,
        )
    ].clone()

    centroids = F.normalize(
        centroids,
        p=2,
        dim=1,
    )

    previous_similarity: float | None = None
    converged = False
    n_iterations = 0

    for iteration in range(max_iter):
        similarities = torch.matmul(
            x,
            centroids.T,
        )

        best_similarity, labels = (
            similarities.max(dim=1)
        )

        new_centroids = torch.zeros(
            (n_clusters, n_features),
            dtype=torch.float32,
            device=resolved_device,
        )

        new_centroids.index_add_(
            0,
            labels,
            x,
        )

        counts = torch.bincount(
            labels,
            minlength=n_clusters,
        ).float()

        empty_clusters = torch.where(
            counts == 0
        )[0]

        if len(empty_clusters) > 0:
            difficult_points = torch.argsort(
                best_similarity
            )[:len(empty_clusters)]

            new_centroids[
                empty_clusters
            ] = x[difficult_points]

            counts[
                empty_clusters
            ] = 1.0

        new_centroids = (
            new_centroids
            / counts.unsqueeze(1)
        )

        new_centroids = F.normalize(
            new_centroids,
            p=2,
            dim=1,
        )

        current_similarity = float(
            best_similarity.mean().item()
        )

        centroids = new_centroids
        n_iterations = iteration + 1

        if previous_similarity is not None:
            if (
                abs(
                    current_similarity
                    - previous_similarity
                )
                < tolerance
            ):
                converged = True
                break

        previous_similarity = (
            current_similarity
        )

    final_similarities = torch.matmul(
        x,
        centroids.T,
    )

    final_values, final_labels = (
        final_similarities.max(dim=1)
    )

    mean_similarity = float(
        final_values.mean().item()
    )

    # Public ClaimSem objective: lower cosine distance is better.
    objective = float(
        1.0 - mean_similarity
    )

    cluster_counts = torch.bincount(
        final_labels,
        minlength=n_clusters,
    )

    return SphericalKMeansResult(
        labels=(
            final_labels.detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        ),
        centroids=(
            centroids.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        ),
        cluster_counts=(
            cluster_counts.detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        ),
        objective=objective,
        n_iterations=int(n_iterations),
        converged=bool(converged),
        seed=int(seed),
    )


def run_multiple_seeds_legacy(
    vectors: np.ndarray,
    n_clusters: int,
    seeds: Sequence[int] = (17, 42, 73),
    max_iter: int = 100,
    tolerance: float = 1e-5,
    device: str | torch.device = "auto",
) -> list[SphericalKMeansResult]:
    """Run the legacy GPU backend for multiple independent seeds."""
    if not seeds:
        raise ClusteringError(
            "At least one clustering seed is required."
        )

    normalized_seeds = [
        int(seed)
        for seed in seeds
    ]

    if (
        len(set(normalized_seeds))
        != len(normalized_seeds)
    ):
        raise ClusteringError(
            "Clustering seeds must be unique."
        )

    return [
        legacy_gpu_spherical_kmeans(
            vectors=vectors,
            n_clusters=n_clusters,
            seed=seed,
            max_iter=max_iter,
            tolerance=tolerance,
            device=device,
        )
        for seed in normalized_seeds
    ]


# END LEGACY DEPTH-OT GPU SPHERICAL K-MEANS
