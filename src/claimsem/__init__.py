"""ClaimSem: Structure-aware claim aggregation for patent clustering."""

from claimsem.config import (
    ConfigError,
    load_config,
    validate_config,
)
from claimsem.reproducibility import (
    collect_environment_info,
    set_global_seed,
)

__version__ = "0.1.0"
__author__ = "Yongmin Yoo"

__all__ = [
    "__version__",
    "__author__",
    "ConfigError",
    "load_config",
    "validate_config",
    "set_global_seed",
    "collect_environment_info",
]
