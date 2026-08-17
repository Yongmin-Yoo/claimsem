"""Tests for ClaimSem feature construction."""

from __future__ import annotations

import numpy as np
import pytest

from claimsem.features import (
    CONSISTENT_CLAIMSEM,
    LEGACY_TEST,
    FeatureConfig,
    FeaturesError,
    PatentFeatureAccumulator,
    build_patent_feature,
    build_patent_features,
    l2_normalize,
    masked_mean_pool,
)


def _record(patent_id: str = "P1") -> dict:
    return {
        "patent_id": patent_id,
        "claims": [
            {
                "claim_id": f"{patent_id}-1",
                "depth": 0,
                "is_root": True,
            },
            {
                "claim_id": f"{patent_id}-2",
                "depth": 1,
                "is_root": False,
            },
        ],
    }


def test_masked_mean_pool_single_sequence() -> None:
    tokens = np.asarray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [100.0, 200.0],
        ],
        dtype=np.float32,
    )
    mask = np.asarray([1, 1, 0], dtype=np.int64)

    pooled = masked_mean_pool(tokens, mask)

    np.testing.assert_allclose(
        pooled,
        np.asarray([2.0, 3.0], dtype=np.float32),
    )
    assert pooled.dtype == np.float32


def test_masked_mean_pool_batch() -> None:
    tokens = np.asarray(
        [
            [[1.0, 0.0], [3.0, 2.0]],
            [[2.0, 4.0], [10.0, 20.0]],
        ],
        dtype=np.float32,
    )
    mask = np.asarray(
        [
            [1, 1],
            [1, 0],
        ],
        dtype=np.int64,
    )

    pooled = masked_mean_pool(tokens, mask)

    expected = np.asarray(
        [
            [2.0, 1.0],
            [2.0, 4.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(pooled, expected)


def test_masked_mean_pool_rejects_empty_mask() -> None:
    tokens = np.ones((2, 3), dtype=np.float32)
    mask = np.zeros(2, dtype=np.int64)

    with pytest.raises(FeaturesError):
        masked_mean_pool(tokens, mask)


def test_l2_normalize_rejects_zero_rows() -> None:
    vectors = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(FeaturesError):
        l2_normalize(vectors)


def test_consistent_and_legacy_modes_are_separate() -> None:
    record = _record()
    embeddings = np.asarray(
        [
            [10.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=np.float32,
    )

    consistent = build_patent_feature(
        record,
        embeddings,
        config=FeatureConfig(
            root_weight=1.0,
            depth_decay=0.0,
            preprocessing_mode=CONSISTENT_CLAIMSEM,
        ),
    )
    legacy = build_patent_feature(
        record,
        embeddings,
        config=FeatureConfig(
            root_weight=1.0,
            depth_decay=0.0,
            preprocessing_mode=LEGACY_TEST,
        ),
    )

    np.testing.assert_allclose(
        np.linalg.norm(consistent),
        1.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        consistent,
        np.asarray([1.0, 1.0], dtype=np.float32) / np.sqrt(2.0),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        legacy,
        np.asarray([5.0, 1.0], dtype=np.float32),
        atol=1e-6,
    )
    assert not np.allclose(consistent, legacy)


def test_build_patent_features_preserves_record_order() -> None:
    records = [_record("P1"), _record("P2")]
    embeddings = [
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.asarray([[0.0, 2.0], [3.0, 0.0]], dtype=np.float32),
    ]

    features = build_patent_features(
        records,
        embeddings,
        config=FeatureConfig(
            root_weight=2.0,
            depth_decay=0.1,
        ),
    )

    assert features.shape == (2, 2)
    assert features.dtype == np.float32
    np.testing.assert_allclose(
        np.linalg.norm(features, axis=1),
        np.ones(2),
        atol=1e-6,
    )


def test_streaming_accumulator_matches_direct_pooling() -> None:
    record = _record()
    embeddings = np.asarray(
        [
            [3.0, 4.0],
            [2.0, 1.0],
        ],
        dtype=np.float32,
    )
    config = FeatureConfig(
        root_weight=12.0,
        depth_decay=0.1,
    )

    direct = build_patent_feature(
        record,
        embeddings,
        config=config,
    )

    accumulator = PatentFeatureAccumulator(
        n_patents=1,
        hidden_dim=2,
        config=config,
    )
    accumulator.add(0, embeddings[0], depth=0)
    accumulator.add(0, embeddings[1], depth=1)

    streamed = accumulator.finalize()[0]

    np.testing.assert_allclose(
        streamed,
        direct,
        rtol=1e-6,
        atol=1e-6,
    )


def test_accumulator_save_and_resume(tmp_path) -> None:
    config = FeatureConfig()
    accumulator = PatentFeatureAccumulator(
        n_patents=1,
        hidden_dim=3,
        config=config,
    )
    accumulator.add(
        0,
        np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        depth=0,
    )

    state_path = tmp_path / "partial_state.npz"
    accumulator.save(
        state_path,
        metadata={"shard_index": 7},
    )

    restored, metadata = PatentFeatureAccumulator.load(state_path)

    assert metadata["shard_index"] == 7
    assert restored.summary()["total_claims"] == 1
    np.testing.assert_allclose(
        restored.finalize(),
        accumulator.finalize(),
    )


def test_accumulator_rejects_missing_patents() -> None:
    accumulator = PatentFeatureAccumulator(
        n_patents=2,
        hidden_dim=2,
    )
    accumulator.add(
        0,
        np.asarray([1.0, 1.0], dtype=np.float32),
        depth=0,
    )

    with pytest.raises(FeaturesError):
        accumulator.finalize()


def test_feature_config_rejects_unknown_mode() -> None:
    with pytest.raises(FeaturesError):
        FeatureConfig(preprocessing_mode="mixed_unknown_mode")
