"""Tests for the feature preparation runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from claimsem.features import FeatureConfig

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/prepare_features.py"

SPEC = importlib.util.spec_from_file_location(
    "prepare_features_script",
    SCRIPT_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None

MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _records() -> list[dict]:
    return [
        {
            "patent_id": 101,
            "claims": {
                1: "root claim",
                2: "dependent claim",
            },
            "depth": {
                1: 0,
                2: 1,
            },
        },
        {
            "patent_id": 202,
            "claims": {
                1: "second root claim",
            },
            "depth": {
                1: 0,
            },
        },
    ]


def _write_shard(path: Path) -> None:
    shard = {
        "patent_ids": [101, 202],
        "entries": [
            {
                "patent_id": 101,
                "claim_id": 1,
                "token_embeddings": torch.tensor(
                    [[1.0, 0.0], [1.0, 0.0]],
                    dtype=torch.float16,
                ),
                "num_tokens": 2,
                "truncated": False,
            },
            {
                "patent_id": 101,
                "claim_id": 2,
                "token_embeddings": torch.tensor(
                    [[0.0, 1.0]],
                    dtype=torch.float16,
                ),
                "num_tokens": 1,
                "truncated": True,
            },
            {
                "patent_id": 202,
                "claim_id": 1,
                "token_embeddings": torch.tensor(
                    [[1.0, 1.0]],
                    dtype=torch.float16,
                ),
                "num_tokens": 1,
                "truncated": False,
            },
        ],
    }
    torch.save(shard, path)


def test_build_record_maps() -> None:
    patent_ids, patent_to_index, depth_lookup = MODULE.build_record_maps(_records())

    np.testing.assert_array_equal(
        patent_ids,
        np.asarray([101, 202], dtype=np.int64),
    )
    assert patent_to_index == {101: 0, 202: 1}
    assert depth_lookup[(101, 1)] == 0
    assert depth_lookup[(101, 2)] == 1
    assert depth_lookup[(202, 1)] == 0


def test_process_shard(tmp_path: Path) -> None:
    shard_path = tmp_path / "shard_00000.pt"
    _write_shard(shard_path)

    _, _, depth_lookup = MODULE.build_record_maps(_records())
    config = FeatureConfig(
        root_weight=12.0,
        depth_decay=0.1,
    )

    result = MODULE.process_shard(
        shard_path,
        depth_lookup=depth_lookup,
        config=config,
        hidden_dim=2,
        local_cache_dir=tmp_path / "local",
    )

    assert result["features"].shape == (2, 2)
    assert result["features"].dtype == np.float32
    assert result["claim_count"] == 3
    assert result["truncated_count"] == 1
    assert result["missing_depth_count"] == 0

    np.testing.assert_allclose(
        np.linalg.norm(result["features"], axis=1),
        np.ones(2),
        atol=1e-6,
    )


def test_partial_round_trip(tmp_path: Path) -> None:
    shard_path = tmp_path / "shard_00000.pt"
    partial_path = tmp_path / "partial.npz"
    _write_shard(shard_path)

    _, _, depth_lookup = MODULE.build_record_maps(_records())
    config = FeatureConfig()

    result = MODULE.process_shard(
        shard_path,
        depth_lookup=depth_lookup,
        config=config,
        hidden_dim=2,
    )

    MODULE.save_partial(
        partial_path,
        result,
        config=config,
    )
    restored = MODULE.load_partial(
        partial_path,
        config=config,
        source_path=shard_path,
        hidden_dim=2,
    )

    np.testing.assert_array_equal(
        restored["patent_ids"],
        result["patent_ids"],
    )
    np.testing.assert_allclose(
        restored["features"],
        result["features"],
    )
    assert restored["claim_count"] == 3


def test_merge_partials_preserves_global_order(
    tmp_path: Path,
) -> None:
    shard_path = tmp_path / "shard_00000.pt"
    _write_shard(shard_path)

    _, patent_to_index, depth_lookup = MODULE.build_record_maps(_records())
    config = FeatureConfig()

    result = MODULE.process_shard(
        shard_path,
        depth_lookup=depth_lookup,
        config=config,
        hidden_dim=2,
    )

    features, statistics = MODULE.merge_partials(
        [result],
        patent_to_index=patent_to_index,
        n_patents=2,
        hidden_dim=2,
    )

    assert features.shape == (2, 2)
    assert statistics["claim_count"] == 3
    assert statistics["truncated_count"] == 1


def test_missing_depth_is_rejected(tmp_path: Path) -> None:
    shard_path = tmp_path / "shard_00000.pt"
    _write_shard(shard_path)

    config = FeatureConfig()

    with pytest.raises(MODULE.FeaturePreparationError):
        MODULE.process_shard(
            shard_path,
            depth_lookup={(101, 1): 0, (202, 1): 0},
            config=config,
            hidden_dim=2,
        )


def test_reference_comparison(tmp_path: Path) -> None:
    features = np.asarray(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    reference_path = tmp_path / "reference.npz"
    np.savez(
        reference_path,
        root12_d010=features.copy(),
    )

    report = MODULE.compare_reference(
        features,
        reference_path,
        reference_key="root12_d010",
    )

    assert report["exact_array_equal"]
    assert report["practical_allclose"]
    assert report["validation_passed"]
    assert report["minimum_cosine_similarity"] == pytest.approx(1.0)
