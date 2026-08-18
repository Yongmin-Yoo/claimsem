"""Tests for committed ClaimSem DEV ablation results."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def load_verifier():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts/verify_dev_ablation_results.py"

    spec = importlib.util.spec_from_file_location(
        "verify_dev_ablation_results",
        path,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verified_dev_ablation_results():
    module = load_verifier()
    root = Path(__file__).resolve().parents[1]
    report = module.verify(root)

    assert report["status"] == "passed"
    assert report["table5_rows"] == 24
    assert report["table6_rows"] == 5
    assert report["best_pool"] == "root12_d020"
