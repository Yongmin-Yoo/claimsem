from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "additional_clustering_baselines"


def read_mean_nmi(path: Path) -> float:
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    level_values: dict[str, list[float]] = {}

    for row in rows:
        level_values.setdefault(row["level"], []).append(float(row["nmi"]))

    means = [sum(values) / len(values) for values in level_values.values()]
    return sum(means) / len(means)


def test_additional_clustering_baseline_results() -> None:
    expected = {
        "tfidf_50k/tfidf_50k_spherical_kmeans_metrics.csv": 0.301134,
        "tfidf_lsa256/tfidf_lsa256_metrics.csv": 0.303779,
        "patentsberta_agglomerative/"
        "patentsberta_uniform_agglomerative_metrics.csv": 0.269942,
        "minilm_agglomerative/minilm_agglomerative_metrics.csv": 0.219877,
    }

    for relative_path, expected_mean in expected.items():
        actual_mean = read_mean_nmi(RESULT_ROOT / relative_path)
        assert abs(actual_mean - expected_mean) < 1e-5
