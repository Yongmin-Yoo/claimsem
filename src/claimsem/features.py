"""Feature construction utilities for ClaimSem.

This module converts token-level or claim-level representations into
patent-level feature vectors. It keeps the consistent ClaimSem preprocessing
path separate from the historical legacy TEST preprocessing path.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from claimsem.pooling import PoolingError, claim_weights, pool_patent


class FeaturesError(ValueError):
    """Raised when ClaimSem features cannot be constructed."""


CONSISTENT_CLAIMSEM = "consistent_claimsem"
LEGACY_TEST = "legacy_test"
SUPPORTED_PREPROCESSING_MODES = {
    CONSISTENT_CLAIMSEM,
    LEGACY_TEST,
}


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for patent-level feature construction.

    ``consistent_claimsem`` applies:

        claim L2 -> weighted patent pooling -> patent L2

    ``legacy_test`` reproduces the historical TEST preprocessing:

        raw claim vectors -> weighted patent pooling

    PCA transformation and PCA-output normalization are performed separately
    by ``claimsem.reduction``.
    """

    root_weight: float = 12.0
    depth_decay: float = 0.1
    preprocessing_mode: str = CONSISTENT_CLAIMSEM
    claim_selection: str = "all"
    eps: float = 1e-12

    def __post_init__(self) -> None:
        mode = str(self.preprocessing_mode).strip().lower()

        if mode not in SUPPORTED_PREPROCESSING_MODES:
            raise FeaturesError(
                "Unsupported preprocessing_mode: "
                f"{self.preprocessing_mode!r}. Expected one of "
                f"{sorted(SUPPORTED_PREPROCESSING_MODES)}."
            )

        object.__setattr__(self, "preprocessing_mode", mode)

        if self.claim_selection not in {"all", "root_only", "first_only"}:
            raise FeaturesError(
                "claim_selection must be 'all', 'root_only', or 'first_only'."
            )

        if not np.isfinite(self.eps) or self.eps <= 0:
            raise FeaturesError("eps must be a positive finite number.")

        try:
            claim_weights(
                [0],
                root_weight=self.root_weight,
                depth_decay=self.depth_decay,
            )
        except PoolingError as exc:
            raise FeaturesError(str(exc)) from exc

    @property
    def claim_l2_before_pooling(self) -> bool:
        """Whether claim vectors are normalized before patent pooling."""
        return self.preprocessing_mode == CONSISTENT_CLAIMSEM

    @property
    def patent_l2_after_pooling(self) -> bool:
        """Whether patent vectors are normalized before PCA."""
        return self.preprocessing_mode == CONSISTENT_CLAIMSEM

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable configuration metadata."""
        result = asdict(self)
        result["claim_l2_before_pooling"] = self.claim_l2_before_pooling
        result["patent_l2_after_pooling"] = self.patent_l2_after_pooling
        return result


def _as_numeric_array(
    values: Any,
    *,
    name: str,
    ndim: int | tuple[int, ...],
) -> np.ndarray:
    """Validate and return a finite numeric NumPy array."""
    array = np.asarray(values)

    expected_ndim = (ndim,) if isinstance(ndim, int) else ndim

    if array.ndim not in expected_ndim:
        raise FeaturesError(
            f"{name} must have ndim in {expected_ndim}, got shape {array.shape}."
        )

    if not np.issubdtype(array.dtype, np.number):
        raise FeaturesError(f"{name} must contain numeric values.")

    if array.size == 0:
        raise FeaturesError(f"{name} cannot be empty.")

    if not np.all(np.isfinite(array)):
        raise FeaturesError(f"{name} contains NaN or infinite values.")

    return array


def l2_normalize(
    vectors: np.ndarray,
    *,
    eps: float = 1e-12,
    name: str = "vectors",
) -> np.ndarray:
    """L2-normalize one vector or a matrix of row vectors."""
    if not np.isfinite(eps) or eps <= 0:
        raise FeaturesError("eps must be a positive finite number.")

    array = _as_numeric_array(
        vectors,
        name=name,
        ndim=(1, 2),
    ).astype(np.float32, copy=False)

    norms = np.linalg.norm(array, axis=-1, keepdims=True)

    if np.any(norms <= eps):
        if array.ndim == 1:
            raise FeaturesError(f"{name} has zero or near-zero norm.")

        bad_rows = np.flatnonzero(norms.reshape(-1) <= eps)
        preview = ", ".join(str(index) for index in bad_rows[:10])
        raise FeaturesError(
            f"{name} contains zero or near-zero rows at indices: {preview}."
        )

    normalized = array / np.maximum(norms, np.float32(eps))

    if not np.all(np.isfinite(normalized)):
        raise FeaturesError(f"L2 normalization produced non-finite values for {name}.")

    return normalized.astype(np.float32, copy=False)


def masked_mean_pool(
    token_embeddings: np.ndarray,
    attention_mask: np.ndarray,
) -> np.ndarray:
    """Masked-mean pool token representations.

    Parameters
    ----------
    token_embeddings:
        Shape ``(sequence_length, hidden_dim)`` or
        ``(batch_size, sequence_length, hidden_dim)``.
    attention_mask:
        Shape ``(sequence_length,)`` or
        ``(batch_size, sequence_length)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(hidden_dim,)`` or ``(batch_size, hidden_dim)``.
    """
    embeddings = _as_numeric_array(
        token_embeddings,
        name="token_embeddings",
        ndim=(2, 3),
    ).astype(np.float32, copy=False)

    mask = _as_numeric_array(
        attention_mask,
        name="attention_mask",
        ndim=(1, 2),
    ).astype(np.float32, copy=False)

    expected_mask_shape = embeddings.shape[:-1]

    if mask.shape != expected_mask_shape:
        raise FeaturesError(
            "attention_mask shape does not match token embeddings: "
            f"expected {expected_mask_shape}, got {mask.shape}."
        )

    if np.any(mask < 0):
        raise FeaturesError("attention_mask cannot contain negative values.")

    token_counts = mask.sum(axis=-1, keepdims=True)

    if np.any(token_counts <= 0):
        bad_rows = np.flatnonzero(token_counts.reshape(-1) <= 0)
        preview = ", ".join(str(index) for index in bad_rows[:10])
        raise FeaturesError(
            "Every sequence must contain at least one unmasked token. "
            f"Invalid sequence indices: {preview}."
        )

    pooled = (embeddings * mask[..., np.newaxis]).sum(axis=-2) / token_counts

    if not np.all(np.isfinite(pooled)):
        raise FeaturesError("Masked mean pooling produced non-finite values.")

    return pooled.astype(np.float32, copy=False)


def prepare_claim_embeddings(
    claim_embeddings: np.ndarray,
    *,
    config: FeatureConfig | None = None,
) -> np.ndarray:
    """Validate claim vectors and apply mode-specific preprocessing."""
    resolved_config = config or FeatureConfig()

    embeddings = _as_numeric_array(
        claim_embeddings,
        name="claim_embeddings",
        ndim=2,
    ).astype(np.float32, copy=False)

    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise FeaturesError(
            "claim_embeddings must contain at least one claim and dimension."
        )

    if resolved_config.claim_l2_before_pooling:
        return l2_normalize(
            embeddings,
            eps=resolved_config.eps,
            name="claim_embeddings",
        )

    return embeddings.copy()


def build_patent_feature(
    record: Mapping[str, Any],
    claim_embeddings: np.ndarray,
    *,
    config: FeatureConfig | None = None,
) -> np.ndarray:
    """Construct one patent-level vector from ordered claim vectors."""
    resolved_config = config or FeatureConfig()
    prepared = prepare_claim_embeddings(
        claim_embeddings,
        config=resolved_config,
    )

    try:
        pooled = pool_patent(
            record=record,
            claim_embeddings=prepared,
            root_weight=resolved_config.root_weight,
            depth_decay=resolved_config.depth_decay,
            claim_selection=resolved_config.claim_selection,
        )
    except (KeyError, PoolingError, TypeError, ValueError) as exc:
        patent_id = record.get("patent_id", "<unknown>")
        raise FeaturesError(
            f"Could not construct feature for patent {patent_id!r}: {exc}"
        ) from exc

    pooled = np.asarray(pooled, dtype=np.float32)

    if pooled.ndim != 1 or pooled.size == 0:
        raise FeaturesError(f"Patent pooling returned invalid shape {pooled.shape}.")

    if not np.all(np.isfinite(pooled)):
        raise FeaturesError("Patent pooling produced non-finite values.")

    if resolved_config.patent_l2_after_pooling:
        pooled = l2_normalize(
            pooled,
            eps=resolved_config.eps,
            name="pooled patent vector",
        )

    return pooled.astype(np.float32, copy=False)


def build_patent_features(
    records: Sequence[Mapping[str, Any]],
    claim_embeddings: Sequence[np.ndarray],
    *,
    config: FeatureConfig | None = None,
) -> np.ndarray:
    """Construct patent features in record order."""
    if len(records) != len(claim_embeddings):
        raise FeaturesError(
            "Record and embedding counts do not match: "
            f"{len(records)} records versus "
            f"{len(claim_embeddings)} embedding collections."
        )

    if not records:
        raise FeaturesError("At least one patent record is required.")

    resolved_config = config or FeatureConfig()

    features = [
        build_patent_feature(
            record,
            embeddings,
            config=resolved_config,
        )
        for record, embeddings in zip(
            records,
            claim_embeddings,
            strict=True,
        )
    ]

    dimensions = {feature.shape[0] for feature in features}

    if len(dimensions) != 1:
        raise FeaturesError("All patent features must have the same dimension.")

    matrix = np.stack(features).astype(np.float32, copy=False)

    if not np.all(np.isfinite(matrix)):
        raise FeaturesError("Patent feature matrix contains non-finite values.")

    return matrix


class PatentFeatureAccumulator:
    """Streaming patent-feature accumulator with resumable state.

    Claims may arrive from multiple embedding shards. Each call to ``add``
    updates the weighted sum for one patent. ``save`` and ``load`` support
    interruption-safe partial processing.
    """

    def __init__(
        self,
        n_patents: int,
        hidden_dim: int,
        *,
        config: FeatureConfig | None = None,
    ) -> None:
        if isinstance(n_patents, bool) or int(n_patents) <= 0:
            raise FeaturesError("n_patents must be a positive integer.")

        if isinstance(hidden_dim, bool) or int(hidden_dim) <= 0:
            raise FeaturesError("hidden_dim must be a positive integer.")

        self.n_patents = int(n_patents)
        self.hidden_dim = int(hidden_dim)
        self.config = config or FeatureConfig()

        self.sums = np.zeros(
            (self.n_patents, self.hidden_dim),
            dtype=np.float64,
        )
        self.weight_sums = np.zeros(self.n_patents, dtype=np.float64)
        self.claim_counts = np.zeros(self.n_patents, dtype=np.int64)

    def add(
        self,
        patent_index: int,
        claim_embedding: np.ndarray,
        *,
        depth: int,
    ) -> None:
        """Add one claim vector to its patent accumulator."""
        if isinstance(patent_index, bool):
            raise FeaturesError("patent_index must be an integer.")

        index = int(patent_index)

        if index < 0 or index >= self.n_patents:
            raise FeaturesError(
                f"patent_index {index} is outside [0, {self.n_patents})."
            )

        if isinstance(depth, bool):
            raise FeaturesError("depth must be a non-negative integer.")

        normalized_depth = int(depth)

        if normalized_depth < 0 or normalized_depth != depth:
            raise FeaturesError("depth must be a non-negative integer.")

        vector = _as_numeric_array(
            claim_embedding,
            name="claim_embedding",
            ndim=1,
        ).astype(np.float32, copy=False)

        if vector.shape[0] != self.hidden_dim:
            raise FeaturesError(
                f"Expected hidden dimension {self.hidden_dim}, got {vector.shape[0]}."
            )

        if self.config.claim_l2_before_pooling:
            vector = l2_normalize(
                vector,
                eps=self.config.eps,
                name="claim_embedding",
            )

        weight = float(
            claim_weights(
                [normalized_depth],
                root_weight=self.config.root_weight,
                depth_decay=self.config.depth_decay,
            )[0]
        )

        self.sums[index] += vector.astype(np.float64, copy=False) * weight
        self.weight_sums[index] += weight
        self.claim_counts[index] += 1

    def finalize(self) -> np.ndarray:
        """Return the completed patent feature matrix."""
        missing = np.flatnonzero(self.claim_counts == 0)

        if missing.size:
            preview = ", ".join(str(index) for index in missing[:10])
            raise FeaturesError(
                f"Some patents have no accumulated claims. Patent indices: {preview}."
            )

        if np.any(self.weight_sums <= 0):
            raise FeaturesError("Every patent must have a positive accumulated weight.")

        features = (self.sums / self.weight_sums[:, np.newaxis]).astype(np.float32)

        if self.config.patent_l2_after_pooling:
            features = l2_normalize(
                features,
                eps=self.config.eps,
                name="accumulated patent features",
            )

        return features.astype(np.float32, copy=False)

    def summary(self) -> dict[str, Any]:
        """Return JSON-serializable accumulation statistics."""
        completed = int(np.count_nonzero(self.claim_counts))
        return {
            "n_patents": self.n_patents,
            "hidden_dim": self.hidden_dim,
            "completed_patents": completed,
            "missing_patents": self.n_patents - completed,
            "total_claims": int(self.claim_counts.sum()),
            "config": self.config.to_dict(),
        }

    def save(
        self,
        path: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Save partial accumulation state as a compressed NPZ file."""
        output_path = Path(path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload_metadata = dict(metadata or {})
        payload_metadata["summary"] = self.summary()

        temporary_path = output_path.with_name(f"{output_path.name}.tmp.npz")

        np.savez_compressed(
            temporary_path,
            sums=self.sums,
            weight_sums=self.weight_sums,
            claim_counts=self.claim_counts,
            config_json=np.asarray(json.dumps(self.config.to_dict())),
            metadata_json=np.asarray(json.dumps(payload_metadata, default=str)),
        )

        temporary_path.replace(output_path)
        return output_path

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> tuple["PatentFeatureAccumulator", dict[str, Any]]:
        """Load a previously saved accumulator state."""
        input_path = Path(path).expanduser().resolve()

        if not input_path.exists():
            raise FileNotFoundError(f"Accumulator state not found: {input_path}")

        with np.load(input_path, allow_pickle=False) as data:
            required = {
                "sums",
                "weight_sums",
                "claim_counts",
                "config_json",
                "metadata_json",
            }
            missing = required.difference(data.files)

            if missing:
                raise FeaturesError(
                    f"Accumulator state is missing arrays: {sorted(missing)}."
                )

            sums = np.asarray(data["sums"], dtype=np.float64)
            weight_sums = np.asarray(
                data["weight_sums"],
                dtype=np.float64,
            )
            claim_counts = np.asarray(
                data["claim_counts"],
                dtype=np.int64,
            )
            config_data = json.loads(str(data["config_json"].item()))
            metadata = json.loads(str(data["metadata_json"].item()))

        if sums.ndim != 2:
            raise FeaturesError(
                f"Saved sums must be two-dimensional, got {sums.shape}."
            )

        n_patents, hidden_dim = sums.shape

        if weight_sums.shape != (n_patents,):
            raise FeaturesError("Saved weight_sums shape is invalid.")

        if claim_counts.shape != (n_patents,):
            raise FeaturesError("Saved claim_counts shape is invalid.")

        config = FeatureConfig(
            root_weight=config_data["root_weight"],
            depth_decay=config_data["depth_decay"],
            preprocessing_mode=config_data["preprocessing_mode"],
            claim_selection=config_data["claim_selection"],
            eps=config_data["eps"],
        )

        accumulator = cls(
            n_patents=n_patents,
            hidden_dim=hidden_dim,
            config=config,
        )
        accumulator.sums[...] = sums
        accumulator.weight_sums[...] = weight_sums
        accumulator.claim_counts[...] = claim_counts

        return accumulator, metadata
