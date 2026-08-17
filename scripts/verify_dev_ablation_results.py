#!/usr/bin/env python3
"""Verify committed ClaimSem DEV Tables 5 and 6."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ANCHOR_NMI = 0.368289425533
BEST_NMI = 0.368636867127


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def verify(root: Path) -> dict[str, object]:
    result_dir = root / "results/dev_ablations"

    table5_csv = result_dir / "table5_sensitivity.csv"
    table5_json = result_dir / "table5_sensitivity.json"
    table6_csv = result_dir / "table6_cluster_robustness.csv"
    table6_json = result_dir / "table6_cluster_robustness.json"

    for path in [table5_csv, table5_json, table6_csv, table6_json]:
        if not path.is_file():
            raise FileNotFoundError(path)

    rows5 = load_csv(table5_csv)
    rows6 = load_csv(table6_csv)

    if len(rows5) != 24:
        raise AssertionError(f"Expected 24 Table 5 rows, got {len(rows5)}")

    pairs = {(float(row["alpha"]), float(row["lambda"])) for row in rows5}

    if len(pairs) != 24:
        raise AssertionError("Table 5 does not contain 24 unique pairs.")

    anchor = [
        row
        for row in rows5
        if float(row["alpha"]) == 12.0 and float(row["lambda"]) == 0.1
    ]

    if len(anchor) != 1:
        raise AssertionError("Missing or duplicate Table 5 anchor.")

    if abs(float(anchor[0]["mean_nmi"]) - ANCHOR_NMI) >= 1e-10:
        raise AssertionError("Table 5 anchor NMI mismatch.")

    best = max(rows5, key=lambda row: float(row["mean_nmi"]))

    if best["pool"] != "root12_d020":
        raise AssertionError(f"Unexpected best Table 5 pool: {best['pool']}")

    if abs(float(best["mean_nmi"]) - BEST_NMI) >= 1e-10:
        raise AssertionError("Table 5 best NMI mismatch.")

    if [int(row["k"]) for row in rows6] != [20, 25, 30, 35, 40]:
        raise AssertionError("Unexpected Table 6 cluster counts.")

    if [int(row["active_clusters"]) for row in rows6] != [20, 25, 30, 35, 40]:
        raise AssertionError("Inactive clusters detected in Table 6.")

    payload5 = json.loads(table5_json.read_text(encoding="utf-8"))
    payload6 = json.loads(table6_json.read_text(encoding="utf-8"))

    if payload5["pca_refitted_per_variant"]:
        raise AssertionError("Verified Table 5 must use frozen DEV PCA.")

    if not payload5["all_converged"]:
        raise AssertionError("Table 5 convergence check failed.")

    if not payload6["all_converged"]:
        raise AssertionError("Table 6 convergence check failed.")

    return {
        "table5_rows": len(rows5),
        "table6_rows": len(rows6),
        "anchor_mean_nmi": float(anchor[0]["mean_nmi"]),
        "best_pool": best["pool"],
        "best_mean_nmi": float(best["mean_nmi"]),
        "status": "passed",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()

    report = verify(args.repo_root.resolve())
    print(json.dumps(report, indent=2))
    print("SUCCESS: Committed DEV ablation results verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
