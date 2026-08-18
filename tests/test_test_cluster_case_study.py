from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "test_case_study"


def test_test_cluster_case_study_results() -> None:
    table2 = pd.read_csv(RESULT_ROOT / "table2_cluster_composition.csv")
    table3 = pd.read_csv(RESULT_ROOT / "table3_centroid_patents.csv")

    with (RESULT_ROOT / "table2_3_case_study.json").open(
        "r", encoding="utf-8"
    ) as handle:
        report = json.load(handle)

    assert report["data"]["n_patents"] == 9881
    assert report["data"]["n_sections"] == 9
    assert report["data"]["n_classes"] == 121
    assert report["data"]["n_subclasses"] == 466
    assert report["configuration"]["case_study_seed"] == 42
    assert report["clustering"]["active_clusters"] == 30
    assert report["clustering"]["converged"] is True
    assert report["clustering"]["centroid_reassignment_mismatches"] == 0

    selected = report["selected_clusters"]
    assert [item["cluster_display"] for item in selected] == [
        "C03", "C01", "C04"
    ]
    assert [item["size"] for item in selected] == [340, 222, 335]
    assert [item["dominant_subclass"] for item in selected] == [
        "H01L", "G06Q", "F16B"
    ]

    assert len(table2) == 9
    assert len(table3) == 9

    assert table3["patent_id"].astype(str).tolist() == [
        "10516034",
        "10170334",
        "10522686",
        "10255631",
        "10382586",
        "10198765",
        "10208780",
        "10443229",
        "10344483",
    ]

    for _, group in table3.groupby("cluster", sort=False):
        similarities = group["cosine_similarity"].to_numpy()
        assert (similarities[:-1] >= similarities[1:]).all()
