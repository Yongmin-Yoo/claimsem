"""Reproducibility and environment utilities for ClaimSem."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_global_seed(
    seed: int,
    *,
    deterministic: bool = False,
) -> None:
    """Seed Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed:
        Non-negative random seed.
    deterministic:
        Request deterministic PyTorch algorithms. This can reduce
        performance and may raise an error for unsupported operations.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("The random seed must be an integer.")

    if seed < 0:
        raise ValueError("The random seed must be non-negative.")

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True


def select_device(
    requested: str = "auto",
) -> torch.device:
    """Select a PyTorch device."""

    normalized = requested.strip().lower()

    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")

        return torch.device("cuda")

    if normalized == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested, but no MPS device is available.")

        return torch.device("mps")

    if normalized == "cpu":
        return torch.device("cpu")

    raise ValueError("Unsupported device request. Use 'auto', 'cuda', 'mps', or 'cpu'.")


def get_git_commit(
    project_root: str | Path | None = None,
) -> str | None:
    """Return the current Git commit hash when available."""

    working_directory = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path.cwd()
    )

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=working_directory,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return None

    commit = result.stdout.strip()
    return commit or None


def get_git_status(
    project_root: str | Path | None = None,
) -> str | None:
    """Return a concise Git working-tree status."""

    working_directory = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path.cwd()
    )

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=working_directory,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return None

    return "clean" if not result.stdout.strip() else "modified"


def _cuda_information() -> dict[str, Any]:
    """Collect CUDA and GPU information."""

    information: dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "device_count": 0,
        "devices": [],
    }

    if not torch.cuda.is_available():
        return information

    information["device_count"] = torch.cuda.device_count()

    devices: list[dict[str, Any]] = []

    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)

        devices.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "compute_capability": (f"{properties.major}.{properties.minor}"),
            }
        )

    information["devices"] = devices
    return information


def collect_environment_info(
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Collect software, hardware, and Git metadata."""

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": {
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "cuda": _cuda_information(),
        "git": {
            "commit": get_git_commit(project_root),
            "working_tree": get_git_status(project_root),
        },
    }


def save_environment_info(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> Path:
    """Collect and save environment information as JSON."""

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    information = collect_environment_info(project_root=project_root)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            information,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")

    return output_path
