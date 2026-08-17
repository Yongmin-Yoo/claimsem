"""Clustering evaluation metrics for ClaimSem."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix


class MetricsError(ValueError):
    """Raised when clustering metrics cannot be computed."""


DEFAULT_LABEL_LEVELS = ("section", "class", "subclass")
DEFAULT_METRICS = (
    "nmi",
    "predicted_cluster_purity",
    "label_wise_inverse_purity",
)


def _normalize_label_array(
    labels: Sequence[Any] | np.ndarray,
    name: str,
) -> np.ndarray:
    """Validate labels and convert them to a one-dimensional string array."""
    array = np.asarray(labels, dtype=object)

    if array.ndim != 1:
        raise MetricsError(f"{name} must be one-dimensional, got shape {array.shape}.")

    if array.size == 0:
        raise MetricsError(f"{name} must contain at least one label.")

    normalized: list[str] = []

    for index, value in enumerate(array.tolist()):
        if value is None:
            raise MetricsError(f"{name} contains a missing label at index {index}.")

        text = str(value).strip()

        if not text:
            raise MetricsError(f"{name} contains an empty label at index {index}.")

        normalized.append(text)

    return np.asarray(normalized, dtype=str)


def _validate_label_pair(
    true_labels: Sequence[Any] | np.ndarray,
    predicted_labels: Sequence[Any] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate true and predicted clustering labels."""
    true_array = _normalize_label_array(
        true_labels,
        name="true_labels",
    )
    predicted_array = _normalize_label_array(
        predicted_labels,
        name="predicted_labels",
    )

    if true_array.shape[0] != predicted_array.shape[0]:
        raise MetricsError(
            "True and predicted label counts do not match: "
            f"{true_array.shape[0]} versus {predicted_array.shape[0]}."
        )

    return true_array, predicted_array


def nmi_score(
    true_labels: Sequence[Any] | np.ndarray,
    predicted_labels: Sequence[Any] | np.ndarray,
    average_method: str = "arithmetic",
) -> float:
    """Compute normalized mutual information."""
    true_array, predicted_array = _validate_label_pair(
        true_labels,
        predicted_labels,
    )

    supported_methods = {
        "min",
        "geometric",
        "arithmetic",
        "max",
    }

    if average_method not in supported_methods:
        raise MetricsError(
            f"Unsupported NMI average_method {average_method!r}. "
            f"Expected one of {sorted(supported_methods)}."
        )

    score = normalized_mutual_info_score(
        true_array,
        predicted_array,
        average_method=average_method,
    )

    return float(score)


def predicted_cluster_purity(
    true_labels: Sequence[Any] | np.ndarray,
    predicted_labels: Sequence[Any] | np.ndarray,
) -> float:
    """Compute predicted-cluster purity.

    For each predicted cluster, this metric counts the most frequent true
    label. The counts are summed and divided by the total number of samples.
    """
    true_array, predicted_array = _validate_label_pair(
        true_labels,
        predicted_labels,
    )

    table = contingency_matrix(
        true_array,
        predicted_array,
        sparse=False,
    )

    if table.size == 0:
        raise MetricsError("The contingency matrix is empty.")

    score = table.max(axis=0).sum() / table.sum()

    return float(score)


def label_wise_inverse_purity(
    true_labels: Sequence[Any] | np.ndarray,
    predicted_labels: Sequence[Any] | np.ndarray,
) -> float:
    """Compute label-wise inverse purity.

    For each true label, this metric counts the largest number of samples
    assigned to a single predicted cluster. The counts are summed and divided
    by the total number of samples.
    """
    true_array, predicted_array = _validate_label_pair(
        true_labels,
        predicted_labels,
    )

    table = contingency_matrix(
        true_array,
        predicted_array,
        sparse=False,
    )

    if table.size == 0:
        raise MetricsError("The contingency matrix is empty.")

    score = table.max(axis=1).sum() / table.sum()

    return float(score)


def evaluate_clustering(
    true_labels: Sequence[Any] | np.ndarray,
    predicted_labels: Sequence[Any] | np.ndarray,
) -> dict[str, float]:
    """Compute all ClaimSem clustering metrics for one label level."""
    return {
        "nmi": nmi_score(
            true_labels=true_labels,
            predicted_labels=predicted_labels,
        ),
        "predicted_cluster_purity": predicted_cluster_purity(
            true_labels=true_labels,
            predicted_labels=predicted_labels,
        ),
        "label_wise_inverse_purity": label_wise_inverse_purity(
            true_labels=true_labels,
            predicted_labels=predicted_labels,
        ),
    }


def extract_cpc_labels(
    records: Sequence[Mapping[str, Any]],
    level: str,
) -> np.ndarray:
    """Extract one CPC label level from normalized patent records."""
    if level not in DEFAULT_LABEL_LEVELS:
        raise MetricsError(
            f"Unsupported CPC level {level!r}. Expected one of {DEFAULT_LABEL_LEVELS}."
        )

    labels: list[str] = []

    for index, record in enumerate(records):
        patent_id = str(record.get("patent_id", index))
        cpc = record.get("cpc")

        if not isinstance(cpc, Mapping):
            raise MetricsError(
                f"Patent {patent_id!r} does not contain a valid CPC mapping."
            )

        value = cpc.get(level)

        if value is None or not str(value).strip():
            raise MetricsError(f"Patent {patent_id!r} has no CPC {level} label.")

        labels.append(str(value).strip())

    return np.asarray(labels, dtype=str)


def evaluate_cpc_levels(
    records: Sequence[Mapping[str, Any]],
    predicted_labels: Sequence[Any] | np.ndarray,
    label_levels: Sequence[str] = DEFAULT_LABEL_LEVELS,
    seed: int | None = None,
) -> dict[str, Any]:
    """Evaluate clustering against multiple CPC hierarchy levels."""
    if not records:
        raise MetricsError("At least one patent record is required.")

    predicted_array = _normalize_label_array(
        predicted_labels,
        name="predicted_labels",
    )

    if len(records) != predicted_array.shape[0]:
        raise MetricsError(
            f"Received {len(records)} records and "
            f"{predicted_array.shape[0]} predicted labels."
        )

    normalized_levels = [str(level) for level in label_levels]

    if not normalized_levels:
        raise MetricsError("At least one CPC label level is required.")

    if len(set(normalized_levels)) != len(normalized_levels):
        raise MetricsError("CPC label levels must be unique.")

    level_results: dict[str, dict[str, float]] = {}

    for level in normalized_levels:
        true_labels = extract_cpc_labels(records, level)

        level_results[level] = evaluate_clustering(
            true_labels=true_labels,
            predicted_labels=predicted_array,
        )

    mean_results = {
        metric: float(
            np.mean([level_results[level][metric] for level in normalized_levels])
        )
        for metric in DEFAULT_METRICS
    }

    result: dict[str, Any] = {
        "levels": level_results,
        "mean": mean_results,
        "n_samples": int(len(records)),
        "n_predicted_clusters": int(np.unique(predicted_array).size),
    }

    if seed is not None:
        result["seed"] = int(seed)

    return result


def evaluate_multiple_seeds(
    records: Sequence[Mapping[str, Any]],
    clustering_results: Sequence[Any],
    label_levels: Sequence[str] = DEFAULT_LABEL_LEVELS,
) -> list[dict[str, Any]]:
    """Evaluate multiple spherical K-means results."""
    if not clustering_results:
        raise MetricsError("At least one clustering result is required.")

    evaluations: list[dict[str, Any]] = []

    for result in clustering_results:
        if not hasattr(result, "labels"):
            raise MetricsError(
                "Each clustering result must provide a labels attribute."
            )

        seed = getattr(result, "seed", None)

        evaluation = evaluate_cpc_levels(
            records=records,
            predicted_labels=result.labels,
            label_levels=label_levels,
            seed=seed,
        )

        if hasattr(result, "objective"):
            evaluation["objective"] = float(result.objective)

        if hasattr(result, "n_active_clusters"):
            evaluation["n_active_clusters"] = int(result.n_active_clusters)

        if hasattr(result, "max_cluster_share"):
            evaluation["max_cluster_share"] = float(result.max_cluster_share)

        evaluations.append(evaluation)

    return evaluations


def _mean_and_std(
    values: Sequence[float],
) -> dict[str, float]:
    """Return population mean and standard deviation."""
    array = np.asarray(values, dtype=np.float64)

    if array.ndim != 1 or array.size == 0:
        raise MetricsError("At least one scalar value is required for aggregation.")

    if not np.all(np.isfinite(array)):
        raise MetricsError("Cannot aggregate non-finite metric values.")

    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
    }


def summarize_seed_evaluations(
    evaluations: Sequence[Mapping[str, Any]],
    label_levels: Sequence[str] = DEFAULT_LABEL_LEVELS,
) -> dict[str, Any]:
    """Aggregate CPC metrics across clustering seeds."""
    if not evaluations:
        raise MetricsError("At least one seed evaluation is required.")

    normalized_levels = [str(level) for level in label_levels]

    level_summary: dict[str, dict[str, dict[str, float]]] = {}

    for level in normalized_levels:
        level_summary[level] = {}

        for metric in DEFAULT_METRICS:
            values = [
                float(evaluation["levels"][level][metric]) for evaluation in evaluations
            ]

            level_summary[level][metric] = _mean_and_std(values)

    mean_summary: dict[str, dict[str, float]] = {}

    for metric in DEFAULT_METRICS:
        values = [float(evaluation["mean"][metric]) for evaluation in evaluations]

        mean_summary[metric] = _mean_and_std(values)

    summary: dict[str, Any] = {
        "n_seeds": int(len(evaluations)),
        "seeds": [evaluation.get("seed") for evaluation in evaluations],
        "levels": level_summary,
        "mean": mean_summary,
    }

    if all("objective" in evaluation for evaluation in evaluations):
        summary["objective"] = _mean_and_std(
            [float(evaluation["objective"]) for evaluation in evaluations]
        )

    if all("max_cluster_share" in evaluation for evaluation in evaluations):
        summary["max_cluster_share"] = _mean_and_std(
            [float(evaluation["max_cluster_share"]) for evaluation in evaluations]
        )

    if all("n_active_clusters" in evaluation for evaluation in evaluations):
        active_counts = [
            int(evaluation["n_active_clusters"]) for evaluation in evaluations
        ]

        summary["n_active_clusters"] = {
            "minimum": int(min(active_counts)),
            "maximum": int(max(active_counts)),
            "values": active_counts,
        }

    return summary
