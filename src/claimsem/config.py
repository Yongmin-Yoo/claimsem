"""Configuration loading and validation for ClaimSem."""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(ValueError):
    """Raised when a ClaimSem configuration is invalid."""


def _expand_environment_string(
    value: str,
    *,
    strict: bool,
) -> str:
    """Expand ${VARIABLE} expressions in a string."""

    missing_variables: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        variable_name = match.group(1)
        variable_value = os.environ.get(variable_name)

        if variable_value is None:
            missing_variables.add(variable_name)
            return match.group(0)

        return variable_value

    expanded = _ENV_PATTERN.sub(replace, value)

    if strict and missing_variables:
        missing = ", ".join(sorted(missing_variables))
        raise ConfigError(
            f"The following environment variables are not defined: {missing}"
        )

    return os.path.expanduser(expanded)


def _expand_environment_values(
    value: Any,
    *,
    strict: bool,
) -> Any:
    """Recursively expand environment variables in a configuration."""

    if isinstance(value, str):
        return _expand_environment_string(value, strict=strict)

    if isinstance(value, list):
        return [_expand_environment_values(item, strict=strict) for item in value]

    if isinstance(value, tuple):
        return tuple(_expand_environment_values(item, strict=strict) for item in value)

    if isinstance(value, dict):
        return {
            key: _expand_environment_values(item, strict=strict)
            for key, item in value.items()
        }

    return value


def _resolve_path(
    value: str,
    *,
    project_root: Path,
) -> str:
    """Resolve a local path relative to the repository root."""

    if _ENV_PATTERN.search(value):
        return value

    path = Path(value).expanduser()

    if not path.is_absolute():
        path = project_root / path

    return str(path.resolve())


def _resolve_config_paths(
    config: dict[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Resolve entries in the top-level paths section."""

    paths = config.get("paths")

    if not isinstance(paths, dict):
        return config

    resolved_paths: dict[str, Any] = {}

    for key, value in paths.items():
        if isinstance(value, str):
            resolved_paths[key] = _resolve_path(
                value,
                project_root=project_root,
            )
        else:
            resolved_paths[key] = value

    config["paths"] = resolved_paths
    return config


def _require_mapping(
    config: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    """Return a required mapping section."""

    value = config.get(key)

    if not isinstance(value, Mapping):
        raise ConfigError(
            f"Configuration section '{key}' is required and must be an object."
        )

    return value


def _require_positive_number(
    mapping: Mapping[str, Any],
    key: str,
    *,
    allow_zero: bool = False,
) -> float:
    """Validate a numeric configuration field."""

    value = mapping.get(key)

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"Configuration value '{key}' must be numeric.")

    number = float(value)

    if allow_zero:
        valid = number >= 0
    else:
        valid = number > 0

    if not valid:
        comparison = "non-negative" if allow_zero else "positive"
        raise ConfigError(f"Configuration value '{key}' must be {comparison}.")

    return number


def validate_config(
    config: Mapping[str, Any],
) -> None:
    """Validate common ClaimSem configuration fields.

    The validator accepts the final, development-search, and smoke-test
    configuration schemas. Experiment-specific validation can be added
    by the corresponding execution module.
    """

    if not isinstance(config, Mapping):
        raise ConfigError("The configuration must be a JSON object.")

    schema_version = config.get("schema_version")

    if not isinstance(schema_version, str) or not schema_version.strip():
        raise ConfigError("'schema_version' must be a non-empty string.")

    experiment_name = config.get("experiment_name")

    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ConfigError("'experiment_name' must be a non-empty string.")

    global_seed = config.get("global_seed")

    if isinstance(global_seed, bool) or not isinstance(global_seed, int):
        raise ConfigError("'global_seed' must be an integer.")

    if global_seed < 0:
        raise ConfigError("'global_seed' must be non-negative.")

    _require_mapping(config, "paths")
    data = _require_mapping(config, "data")
    features = _require_mapping(config, "features")
    pca = _require_mapping(config, "pca")
    clustering = _require_mapping(config, "clustering")
    evaluation = _require_mapping(config, "evaluation")
    _require_mapping(config, "runtime")

    embedding_dim = features.get("embedding_dim")

    if (
        isinstance(embedding_dim, bool)
        or not isinstance(embedding_dim, int)
        or embedding_dim <= 0
    ):
        raise ConfigError("'features.embedding_dim' must be a positive integer.")

    if not isinstance(
        data.get("validate_dependencies"),
        bool,
    ):
        raise ConfigError("'data.validate_dependencies' must be Boolean.")

    pca_enabled = pca.get("enabled")

    if not isinstance(pca_enabled, bool):
        raise ConfigError("'pca.enabled' must be Boolean.")

    if pca_enabled:
        output_dim = pca.get("output_dim")

        if (
            isinstance(output_dim, bool)
            or not isinstance(output_dim, int)
            or output_dim <= 0
        ):
            raise ConfigError("'pca.output_dim' must be a positive integer.")

        if output_dim > embedding_dim:
            raise ConfigError(
                "'pca.output_dim' cannot exceed 'features.embedding_dim'."
            )

    method = clustering.get("method")

    if method != "spherical_kmeans":
        raise ConfigError("'clustering.method' must be 'spherical_kmeans'.")

    seeds = clustering.get("seeds")

    if (
        not isinstance(seeds, list)
        or not seeds
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in seeds
        )
    ):
        raise ConfigError(
            "'clustering.seeds' must be a non-empty list of non-negative integers."
        )

    n_clusters = clustering.get("n_clusters")

    if n_clusters is not None:
        if (
            isinstance(n_clusters, bool)
            or not isinstance(n_clusters, int)
            or n_clusters <= 1
        ):
            raise ConfigError(
                "'clustering.n_clusters' must be an integer greater than 1."
            )

    max_iter = clustering.get("max_iter")

    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
        raise ConfigError("'clustering.max_iter' must be a positive integer.")

    _require_positive_number(
        clustering,
        "tolerance",
        allow_zero=False,
    )

    label_levels = evaluation.get("label_levels")

    if not isinstance(label_levels, list) or not label_levels:
        raise ConfigError("'evaluation.label_levels' must be a non-empty list.")

    valid_levels = {"section", "class", "subclass"}
    unknown_levels = set(label_levels) - valid_levels

    if unknown_levels:
        unknown = ", ".join(sorted(unknown_levels))
        raise ConfigError(f"Unsupported CPC label levels: {unknown}")

    pooling = config.get("pooling")

    if pooling is not None:
        if not isinstance(pooling, Mapping):
            raise ConfigError("'pooling' must be a JSON object.")

        _require_positive_number(
            pooling,
            "root_weight",
            allow_zero=False,
        )
        _require_positive_number(
            pooling,
            "depth_decay",
            allow_zero=True,
        )


def load_config(
    path: str | Path,
    *,
    strict_environment: bool = True,
    resolve_paths: bool = True,
    validate: bool = True,
) -> dict[str, Any]:
    """Load a ClaimSem JSON configuration.

    Parameters
    ----------
    path:
        Path to a JSON configuration file.
    strict_environment:
        Raise an error when a referenced environment variable is missing.
    resolve_paths:
        Resolve entries in the top-level paths section relative to the
        repository root.
    validate:
        Validate common configuration fields after loading.

    Returns
    -------
    dict
        Expanded and validated configuration.
    """

    config_path = Path(path).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if not config_path.is_file():
        raise ConfigError(f"Configuration path is not a file: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as file:
            loaded = json.load(file)
    except json.JSONDecodeError as error:
        raise ConfigError(f"Invalid JSON in {config_path}: {error}") from error

    if not isinstance(loaded, dict):
        raise ConfigError("The top-level configuration value must be a JSON object.")

    config = deepcopy(loaded)
    config = _expand_environment_values(
        config,
        strict=strict_environment,
    )

    project_root = config_path.parent.parent

    if resolve_paths:
        config = _resolve_config_paths(
            config,
            project_root=project_root,
        )

    config["_metadata"] = {
        "config_path": str(config_path),
        "project_root": str(project_root.resolve()),
    }

    if validate:
        validate_config(config)

    return config


def save_config(
    config: Mapping[str, Any],
    path: str | Path,
    *,
    include_metadata: bool = False,
) -> Path:
    """Save a configuration as formatted JSON."""

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = deepcopy(dict(config))

    if not include_metadata:
        serializable.pop("_metadata", None)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            serializable,
            file,
            indent=2,
            ensure_ascii=False,
            sort_keys=False,
        )
        file.write("\n")

    return output_path
