"""PCA reduction and L2 normalization utilities for ClaimSem."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


class ReductionError(ValueError):
    """Raised when dimensionality reduction cannot be performed."""


def _validate_matrix(
    vectors: np.ndarray,
    name: str = "vectors",
) -> np.ndarray:
    """Validate and return a two-dimensional floating-point matrix."""
    matrix = np.asarray(vectors)

    if matrix.ndim != 2:
        raise ReductionError(
            f"{name} must have shape (n_samples, n_features), "
            f"got {matrix.shape}."
        )

    if matrix.shape[0] == 0:
        raise ReductionError(f"{name} must contain at least one sample.")

    if matrix.shape[1] == 0:
        raise ReductionError(f"{name} must contain at least one feature.")

    if not np.issubdtype(matrix.dtype, np.number):
        raise ReductionError(f"{name} must contain numeric values.")

    if not np.all(np.isfinite(matrix)):
        raise ReductionError(f"{name} contains non-finite values.")

    return matrix.astype(np.float32, copy=False)


def fit_pca(
    vectors: np.ndarray,
    output_dim: int = 128,
    random_state: int = 42,
    whiten: bool = False,
    svd_solver: str = "auto",
) -> PCA:
    """Fit PCA on development patent representations.

    Parameters
    ----------
    vectors:
        Patent representation matrix with shape
        ``(n_patents, embedding_dim)``.
    output_dim:
        Number of principal components.
    random_state:
        Random state used by randomized PCA solvers.
    whiten:
        Whether to whiten PCA outputs.
    svd_solver:
        Scikit-learn PCA solver.

    Returns
    -------
    sklearn.decomposition.PCA
        Fitted PCA estimator.
    """
    matrix = _validate_matrix(vectors)

    if not isinstance(output_dim, int) or isinstance(output_dim, bool):
        raise ReductionError("output_dim must be an integer.")

    if output_dim <= 0:
        raise ReductionError("output_dim must be greater than zero.")

    maximum_dim = min(matrix.shape[0], matrix.shape[1])

    if output_dim > maximum_dim:
        raise ReductionError(
            f"output_dim={output_dim} exceeds the maximum valid PCA "
            f"dimension {maximum_dim} for matrix shape {matrix.shape}."
        )

    supported_solvers = {
        "auto",
        "full",
        "arpack",
        "randomized",
        "covariance_eigh",
    }

    if svd_solver not in supported_solvers:
        raise ReductionError(
            f"Unsupported PCA solver {svd_solver!r}. "
            f"Expected one of {sorted(supported_solvers)}."
        )

    pca = PCA(
        n_components=output_dim,
        whiten=bool(whiten),
        svd_solver=svd_solver,
        random_state=int(random_state),
    )

    pca.fit(matrix)

    return pca


def transform_pca(
    vectors: np.ndarray,
    pca: PCA,
) -> np.ndarray:
    """Transform patent representations with a fitted PCA model."""
    matrix = _validate_matrix(vectors)

    if not isinstance(pca, PCA):
        raise ReductionError("pca must be a fitted sklearn PCA instance.")

    if not hasattr(pca, "components_"):
        raise ReductionError("The PCA model has not been fitted.")

    expected_features = int(pca.n_features_in_)

    if matrix.shape[1] != expected_features:
        raise ReductionError(
            f"The PCA model expects {expected_features} input features, "
            f"but received {matrix.shape[1]}."
        )

    transformed = pca.transform(matrix)

    if not np.all(np.isfinite(transformed)):
        raise ReductionError("PCA transformation produced non-finite values.")

    return transformed.astype(np.float32, copy=False)


def l2_normalize(
    vectors: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """L2-normalize patent vectors row by row."""
    matrix = _validate_matrix(vectors)

    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ReductionError("epsilon must be a positive finite number.")

    row_norms = np.linalg.norm(matrix, axis=1)

    if np.any(row_norms < epsilon):
        zero_indices = np.flatnonzero(row_norms < epsilon).tolist()
        raise ReductionError(
            "Cannot L2-normalize zero or near-zero vectors. "
            f"Affected row indices: {zero_indices[:20]}."
        )

    normalized = normalize(
        matrix,
        norm="l2",
        axis=1,
        copy=True,
    )

    if not np.all(np.isfinite(normalized)):
        raise ReductionError("L2 normalization produced non-finite values.")

    return normalized.astype(np.float32, copy=False)


def fit_transform_pca(
    vectors: np.ndarray,
    output_dim: int = 128,
    random_state: int = 42,
    whiten: bool = False,
    svd_solver: str = "auto",
    apply_l2_normalization: bool = True,
) -> tuple[PCA, np.ndarray]:
    """Fit PCA, transform the fitting data, and optionally L2-normalize."""
    matrix = _validate_matrix(vectors)

    pca = fit_pca(
        vectors=matrix,
        output_dim=output_dim,
        random_state=random_state,
        whiten=whiten,
        svd_solver=svd_solver,
    )

    transformed = transform_pca(matrix, pca)

    if apply_l2_normalization:
        transformed = l2_normalize(transformed)

    return pca, transformed


def transform_and_normalize(
    vectors: np.ndarray,
    pca: PCA,
) -> np.ndarray:
    """Apply a fitted PCA model and then L2-normalize the outputs."""
    transformed = transform_pca(vectors, pca)
    return l2_normalize(transformed)


def save_pca(
    pca: PCA,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a fitted PCA model and optional metadata."""
    if not isinstance(pca, PCA):
        raise ReductionError("pca must be a fitted sklearn PCA instance.")

    if not hasattr(pca, "components_"):
        raise ReductionError("Cannot save an unfitted PCA model.")

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": pca,
        "metadata": metadata or {},
    }

    joblib.dump(payload, destination)

    return destination


def load_pca(
    path: str | Path,
) -> tuple[PCA, dict[str, Any]]:
    """Load a saved PCA model and its metadata."""
    source = Path(path).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(f"PCA model file not found: {source}")

    payload = joblib.load(source)

    if isinstance(payload, PCA):
        pca = payload
        metadata: dict[str, Any] = {}
    elif isinstance(payload, dict) and "model" in payload:
        pca = payload["model"]
        metadata = payload.get("metadata", {})
    else:
        raise ReductionError(
            f"Unsupported PCA artifact format in {source}."
        )

    if not isinstance(pca, PCA):
        raise ReductionError(
            f"The artifact at {source} does not contain a PCA model."
        )

    if not hasattr(pca, "components_"):
        raise ReductionError(
            f"The PCA model loaded from {source} is not fitted."
        )

    if not isinstance(metadata, dict):
        raise ReductionError("PCA metadata must be a dictionary.")

    return pca, metadata
