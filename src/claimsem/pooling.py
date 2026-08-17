"""Root and depth-aware claim pooling for ClaimSem."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


class PoolingError(ValueError):
    """Raised when claim embeddings cannot be pooled."""


def claim_weights(
    depths: Sequence[int] | np.ndarray,
    root_weight: float = 12.0,
    depth_decay: float = 0.1,
) -> np.ndarray:
    """Compute root and depth-aware claim weights.

    The weight of claim c is

        w_c = alpha^[d(c)=0] * exp(-lambda * d(c)),

    where alpha is ``root_weight`` and lambda is ``depth_decay``.

    Parameters
    ----------
    depths:
        Non-negative claim dependency depths.
    root_weight:
        Multiplicative weight assigned to root claims.
    depth_decay:
        Exponential decay applied according to dependency depth.

    Returns
    -------
    numpy.ndarray
        One positive floating-point weight per claim.
    """
    depth_array = np.asarray(depths, dtype=np.float64)

    if depth_array.ndim != 1:
        raise PoolingError(
            f"depths must be one-dimensional, got shape {depth_array.shape}."
        )

    if depth_array.size == 0:
        raise PoolingError("At least one claim depth is required.")

    if not np.all(np.isfinite(depth_array)):
        raise PoolingError("Claim depths must be finite.")

    if np.any(depth_array < 0):
        raise PoolingError("Claim depths must be non-negative.")

    if not np.all(depth_array == np.floor(depth_array)):
        raise PoolingError("Claim depths must be integers.")

    if not np.isfinite(root_weight) or root_weight <= 0:
        raise PoolingError("root_weight must be a positive finite number.")

    if not np.isfinite(depth_decay) or depth_decay < 0:
        raise PoolingError("depth_decay must be a non-negative finite number.")

    weights = np.exp(-float(depth_decay) * depth_array)
    weights[depth_array == 0] *= float(root_weight)

    if not np.all(np.isfinite(weights)):
        raise PoolingError("Non-finite claim weights were produced.")

    if np.any(weights <= 0):
        raise PoolingError("All claim weights must be positive.")

    return weights


def weighted_mean_pool(
    claim_embeddings: np.ndarray,
    weights: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Compute the weighted mean of claim embeddings.

    Parameters
    ----------
    claim_embeddings:
        Matrix with shape ``(n_claims, embedding_dim)``.
    weights:
        Vector with shape ``(n_claims,)``.

    Returns
    -------
    numpy.ndarray
        Patent representation with shape ``(embedding_dim,)``.
    """
    embeddings = np.asarray(claim_embeddings)
    weight_array = np.asarray(weights, dtype=np.float64)

    if embeddings.ndim != 2:
        raise PoolingError(
            "claim_embeddings must have shape "
            f"(n_claims, embedding_dim), got {embeddings.shape}."
        )

    if embeddings.shape[0] == 0:
        raise PoolingError("At least one claim embedding is required.")

    if embeddings.shape[1] == 0:
        raise PoolingError("Embedding dimension must be greater than zero.")

    if weight_array.ndim != 1:
        raise PoolingError(
            f"weights must be one-dimensional, got shape {weight_array.shape}."
        )

    if embeddings.shape[0] != weight_array.shape[0]:
        raise PoolingError(
            "Claim and weight counts do not match: "
            f"{embeddings.shape[0]} embeddings versus "
            f"{weight_array.shape[0]} weights."
        )

    if not np.issubdtype(embeddings.dtype, np.number):
        raise PoolingError("claim_embeddings must contain numeric values.")

    if not np.all(np.isfinite(embeddings)):
        raise PoolingError("claim_embeddings contain non-finite values.")

    if not np.all(np.isfinite(weight_array)):
        raise PoolingError("weights contain non-finite values.")

    if np.any(weight_array < 0):
        raise PoolingError("weights must be non-negative.")

    weight_sum = float(weight_array.sum())

    if weight_sum <= 0:
        raise PoolingError("The sum of claim weights must be positive.")

    pooled = np.average(
        embeddings.astype(np.float64, copy=False),
        axis=0,
        weights=weight_array,
    )

    return pooled.astype(np.float32, copy=False)


def _select_claim_indices(
    claims: Sequence[Mapping[str, Any]],
    claim_selection: str,
) -> np.ndarray:
    """Return claim indices for a pooling or ablation configuration."""
    if not claims:
        raise PoolingError("A patent must contain at least one claim.")

    if claim_selection == "all":
        return np.arange(len(claims), dtype=np.int64)

    if claim_selection == "root_only":
        indices = [
            index
            for index, claim in enumerate(claims)
            if int(claim.get("depth", -1)) == 0
        ]

        if not indices:
            raise PoolingError("No root claims were found.")

        return np.asarray(indices, dtype=np.int64)

    if claim_selection == "first_only":
        return np.asarray([0], dtype=np.int64)

    raise PoolingError(
        "Unsupported claim_selection value: "
        f"{claim_selection!r}. Expected 'all', 'root_only', or 'first_only'."
    )


def pool_patent(
    record: Mapping[str, Any],
    claim_embeddings: np.ndarray,
    root_weight: float = 12.0,
    depth_decay: float = 0.1,
    claim_selection: str = "all",
) -> np.ndarray:
    """Create one ClaimSem patent representation.

    The order of rows in ``claim_embeddings`` must match the order of claims
    in ``record["claims"]``.
    """
    claims = record.get("claims")

    if not isinstance(claims, Sequence) or isinstance(claims, (str, bytes)):
        raise PoolingError("record['claims'] must be a sequence.")

    embeddings = np.asarray(claim_embeddings)

    if embeddings.ndim != 2:
        raise PoolingError(
            "claim_embeddings must have shape "
            f"(n_claims, embedding_dim), got {embeddings.shape}."
        )

    if len(claims) != embeddings.shape[0]:
        patent_id = record.get("patent_id", "<unknown>")
        raise PoolingError(
            f"Patent {patent_id!r} contains {len(claims)} claims, "
            f"but {embeddings.shape[0]} embeddings were provided."
        )

    indices = _select_claim_indices(claims, claim_selection)
    selected_embeddings = embeddings[indices]

    selected_depths = np.asarray(
        [int(claims[index]["depth"]) for index in indices],
        dtype=np.int64,
    )

    weights = claim_weights(
        depths=selected_depths,
        root_weight=root_weight,
        depth_decay=depth_decay,
    )

    return weighted_mean_pool(selected_embeddings, weights)


def pool_records(
    records: Sequence[Mapping[str, Any]],
    claim_embeddings: Sequence[np.ndarray],
    root_weight: float = 12.0,
    depth_decay: float = 0.1,
    claim_selection: str = "all",
) -> np.ndarray:
    """Pool claim embeddings for multiple patent records.

    Parameters
    ----------
    records:
        Normalized patent records.
    claim_embeddings:
        Sequence containing one claim-embedding matrix per patent.
    root_weight:
        Root-claim emphasis parameter alpha.
    depth_decay:
        Depth-decay parameter lambda.
    claim_selection:
        ``all``, ``root_only``, or ``first_only``.

    Returns
    -------
    numpy.ndarray
        Patent representation matrix with shape
        ``(n_patents, embedding_dim)``.
    """
    if len(records) != len(claim_embeddings):
        raise PoolingError(
            "Record and embedding collection sizes do not match: "
            f"{len(records)} records versus "
            f"{len(claim_embeddings)} embedding matrices."
        )

    if not records:
        raise PoolingError("At least one patent record is required.")

    pooled_vectors = [
        pool_patent(
            record=record,
            claim_embeddings=embeddings,
            root_weight=root_weight,
            depth_decay=depth_decay,
            claim_selection=claim_selection,
        )
        for record, embeddings in zip(records, claim_embeddings, strict=True)
    ]

    dimensions = {vector.shape[0] for vector in pooled_vectors}

    if len(dimensions) != 1:
        raise PoolingError(
            "All pooled patent representations must have the same dimension."
        )

    return np.stack(pooled_vectors, axis=0).astype(np.float32, copy=False)


def pool_from_claim_id_mapping(
    record: Mapping[str, Any],
    embedding_by_claim_id: Mapping[str, np.ndarray],
    root_weight: float = 12.0,
    depth_decay: float = 0.1,
    claim_selection: str = "all",
) -> np.ndarray:
    """Pool a patent when embeddings are indexed by claim identifier."""
    claims = record.get("claims")

    if not isinstance(claims, Sequence) or isinstance(claims, (str, bytes)):
        raise PoolingError("record['claims'] must be a sequence.")

    ordered_embeddings: list[np.ndarray] = []

    for claim in claims:
        claim_id = str(claim["claim_id"])

        if claim_id not in embedding_by_claim_id:
            patent_id = record.get("patent_id", "<unknown>")
            raise PoolingError(
                f"Embedding for claim {claim_id!r} is missing "
                f"from patent {patent_id!r}."
            )

        ordered_embeddings.append(
            np.asarray(embedding_by_claim_id[claim_id], dtype=np.float32)
        )

    try:
        embedding_matrix = np.stack(ordered_embeddings, axis=0)
    except ValueError as exc:
        raise PoolingError(
            "Claim embeddings do not have a consistent dimension."
        ) from exc

    return pool_patent(
        record=record,
        claim_embeddings=embedding_matrix,
        root_weight=root_weight,
        depth_decay=depth_decay,
        claim_selection=claim_selection,
    )
