# ============================================================
# AUTOMATIC TABLES 2–3 CASE-STUDY GENERATOR
# Full ClaimSem TEST partition, seed 42
# ============================================================

import hashlib
import json
import math
import pickle
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

# ------------------------------------------------------------
# 1. Paths and fixed protocol
# ------------------------------------------------------------

TEST_RECORDS_PATH = Path(
    "/content/drive/MyDrive/depth_ot_patent/data/processed/test_records.pkl"
)

RUN_ROOT = Path(
    "/content/drive/MyDrive/claimsem_artifacts/final_test_consistent_root12_d010_runner"
)

PREDICTION_PATH = RUN_ROOT / "clusters/clustering_seed_42.npz"

FEATURE_PATH = RUN_ROOT / "test_pca128_features.npz"

# Optional independent copy used only for consistency checking.
REFERENCE_PREDICTION_PATH = Path(
    "/content/drive/MyDrive/claimsem_artifacts/"
    "final_test_consistent_root12_d010/"
    "clusters/clustering_seed_42.npz"
)

OUTPUT_ROOT = Path(
    "/content/drive/MyDrive/claimsem_artifacts/test_case_study_tables_2_3"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

EXPECTED_N_PATENTS = 9881
EXPECTED_DIMENSION = 128
EXPECTED_CLUSTERS = 30
EXPECTED_SEED = 42
EXPECTED_SUBCLASSES = 466

MINIMUM_CLUSTER_SIZE = max(
    50,
    math.ceil(0.005 * EXPECTED_N_PATENTS),
)

TOP_SUBCLASSES_PER_CLUSTER = 3
NEAREST_PATENTS_PER_CLUSTER = 3

assert MINIMUM_CLUSTER_SIZE == 50

for required_path in [
    TEST_RECORDS_PATH,
    PREDICTION_PATH,
    FEATURE_PATH,
]:
    assert required_path.exists(), f"Required file not found: {required_path}"

print("=" * 110)
print("TABLES 2–3 CASE-STUDY PREFLIGHT")
print("=" * 110)
print("Records             :", TEST_RECORDS_PATH)
print("Prediction          :", PREDICTION_PATH)
print("PCA features        :", FEATURE_PATH)
print("Output              :", OUTPUT_ROOT)
print("Selection seed      :", EXPECTED_SEED)
print("Minimum cluster size:", MINIMUM_CLUSTER_SIZE)


# ------------------------------------------------------------
# 2. Utility functions
# ------------------------------------------------------------


def scalar_text(value):
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            value = value.item()
        elif value.size == 1:
            value = value.reshape(-1)[0]
        else:
            return None

    if isinstance(value, np.generic):
        value = value.item()

    text = str(value).strip()

    if not text or text.lower() in {
        "none",
        "nan",
        "null",
    }:
        return None

    return text


def canonical_patent_id(value):
    text = scalar_text(value)

    if text is None:
        return ""

    text = text.upper().strip()

    # Normalize only superficial formatting.
    text = re.sub(r"\s+", "", text)
    text = text.replace("-", "")

    if text.endswith(".0"):
        text = text[:-2]

    return text


def parse_cpc_code(value):
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple, set)):
        for item in value:
            parsed = parse_cpc_code(item)

            if parsed is not None:
                return parsed

        return None

    text = scalar_text(value)

    if text is None:
        return None

    text = text.upper()

    match = re.search(
        r"([A-HY])\s*([0-9]{2})\s*([A-Z])",
        text,
    )

    if match is None:
        return None

    section = match.group(1)
    cpc_class = match.group(1) + match.group(2)
    subclass = match.group(1) + match.group(2) + match.group(3)

    return {
        "section": section,
        "class": cpc_class,
        "subclass": subclass,
    }


def normalize_cpc(record):
    section = scalar_text(record.get("section"))
    cpc_class = scalar_text(record.get("class"))
    subclass = scalar_text(record.get("subclass"))

    if section and cpc_class and subclass:
        return {
            "section": section.upper(),
            "class": cpc_class.upper(),
            "subclass": subclass.upper(),
        }

    for field in (
        "cpc",
        "cpc_labels",
    ):
        value = record.get(field)

        if isinstance(value, dict):
            section = scalar_text(value.get("section"))
            cpc_class = scalar_text(value.get("class"))
            subclass = scalar_text(value.get("subclass"))

            if section and cpc_class and subclass:
                return {
                    "section": section.upper(),
                    "class": cpc_class.upper(),
                    "subclass": subclass.upper(),
                }

            for nested_field in (
                "code",
                "cpc_code",
                "symbol",
            ):
                parsed = parse_cpc_code(value.get(nested_field))

                if parsed is not None:
                    return parsed

    for field in (
        "cpc_code",
        "cpc_codes",
        "cpc_symbol",
        "classification",
    ):
        parsed = parse_cpc_code(record.get(field))

        if parsed is not None:
            return parsed

    raise ValueError(f"No usable CPC label found for patent {record.get('patent_id')}")


def l2_normalize(matrix):
    matrix = np.asarray(
        matrix,
        dtype=np.float32,
    )

    norms = np.linalg.norm(
        matrix.astype(np.float64),
        axis=1,
        keepdims=True,
    )

    if np.any(norms <= 1e-12):
        bad = np.flatnonzero(norms.reshape(-1) <= 1e-12)

        raise ValueError(f"Zero vectors at indices {bad[:20].tolist()}")

    return (matrix / np.maximum(norms, 1e-12)).astype(np.float32)


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def latex_escape(value):
    text = str(value)

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }

    for source, target in replacements.items():
        text = text.replace(
            source,
            target,
        )

    return text


def display_patent_id(value):
    text = str(value).strip()

    if re.fullmatch(r"\d+", text):
        return f"US~{text}"

    return latex_escape(text)


def display_cluster_id(cluster_id):
    # Internal labels are zero-based; paper labels are one-based.
    return f"C{int(cluster_id) + 1:02d}"


# ------------------------------------------------------------
# 3. Load and normalize TEST records
# ------------------------------------------------------------

with open(
    TEST_RECORDS_PATH,
    "rb",
) as handle:
    raw_records = pickle.load(handle)

assert len(raw_records) == EXPECTED_N_PATENTS, (
    len(raw_records),
    EXPECTED_N_PATENTS,
)

records = []
record_patent_ids = []
sections = []
classes = []
subclasses = []

for record_index, raw_record in enumerate(raw_records):
    normalized_record = dict(raw_record)

    normalized_cpc = normalize_cpc(raw_record)

    normalized_record["cpc"] = normalized_cpc

    patent_id = scalar_text(raw_record.get("patent_id"))

    if patent_id is None:
        raise ValueError(f"Missing patent ID at record index {record_index}")

    records.append(normalized_record)
    record_patent_ids.append(patent_id)
    sections.append(normalized_cpc["section"])
    classes.append(normalized_cpc["class"])
    subclasses.append(normalized_cpc["subclass"])

record_patent_ids = np.asarray(
    record_patent_ids,
    dtype=str,
)
sections = np.asarray(
    sections,
    dtype=str,
)
classes = np.asarray(
    classes,
    dtype=str,
)
subclasses = np.asarray(
    subclasses,
    dtype=str,
)

assert np.unique(sections).size == 9
assert np.unique(classes).size == 121
assert np.unique(subclasses).size == EXPECTED_SUBCLASSES

print("\nRecords loaded         :", len(records))
print("Sections               :", np.unique(sections).size)
print("Classes                :", np.unique(classes).size)
print("Subclasses             :", np.unique(subclasses).size)
print("First patent ID        :", record_patent_ids[0])
print("First CPC subclass     :", subclasses[0])


# ------------------------------------------------------------
# 4. Load PCA features and verify patent order
# ------------------------------------------------------------

with np.load(
    FEATURE_PATH,
    allow_pickle=False,
) as archive:
    features = archive["features"].astype(
        np.float32,
        copy=True,
    )

    feature_patent_ids = archive["patent_ids"].astype(str)

assert features.shape == (
    EXPECTED_N_PATENTS,
    EXPECTED_DIMENSION,
), features.shape

assert feature_patent_ids.shape == (EXPECTED_N_PATENTS,)

record_ids_canonical = np.asarray(
    [canonical_patent_id(value) for value in record_patent_ids],
    dtype=str,
)

feature_ids_canonical = np.asarray(
    [canonical_patent_id(value) for value in feature_patent_ids],
    dtype=str,
)

if not np.array_equal(
    record_ids_canonical,
    feature_ids_canonical,
):
    mismatches = np.flatnonzero(record_ids_canonical != feature_ids_canonical)

    mismatch_preview = [
        {
            "index": int(index),
            "record": record_patent_ids[index],
            "feature": feature_patent_ids[index],
        }
        for index in mismatches[:20]
    ]

    raise RuntimeError(
        "Feature/patent order mismatch:\n"
        + json.dumps(
            mismatch_preview,
            indent=2,
        )
    )

features = l2_normalize(features)

feature_norms = np.linalg.norm(
    features.astype(np.float64),
    axis=1,
)

print("\nFeature shape          :", features.shape)
print(
    f"Feature norm range     : [{feature_norms.min():.8f}, {feature_norms.max():.8f}]"
)
print("Patent order aligned   : True")


# ------------------------------------------------------------
# 5. Load seed-42 clustering result
# ------------------------------------------------------------

with np.load(
    PREDICTION_PATH,
    allow_pickle=False,
) as archive:
    labels = archive["labels"].astype(
        np.int64,
        copy=True,
    )

    centroids = archive["centroids"].astype(
        np.float32,
        copy=True,
    )

    saved_cluster_counts = archive["cluster_counts"].astype(
        np.int64,
        copy=True,
    )

    saved_seed = int(np.asarray(archive["seed"]).item())

    saved_converged = bool(np.asarray(archive["converged"]).item())

    saved_iterations = int(np.asarray(archive["n_iterations"]).item())

    saved_objective = float(np.asarray(archive["objective"]).item())

assert labels.shape == (EXPECTED_N_PATENTS,)
assert centroids.shape == (
    EXPECTED_CLUSTERS,
    EXPECTED_DIMENSION,
)
assert saved_cluster_counts.shape == (EXPECTED_CLUSTERS,)
assert saved_seed == EXPECTED_SEED
assert saved_converged

unique_clusters = np.unique(labels)

assert unique_clusters.size == (EXPECTED_CLUSTERS), unique_clusters

computed_cluster_counts = np.bincount(
    labels,
    minlength=EXPECTED_CLUSTERS,
)

assert np.array_equal(
    computed_cluster_counts,
    saved_cluster_counts,
), (
    computed_cluster_counts,
    saved_cluster_counts,
)

centroids = l2_normalize(centroids)

similarity_matrix = features @ centroids.T

reassigned_labels = np.argmax(
    similarity_matrix,
    axis=1,
).astype(np.int64)

assignment_mismatch_count = int(np.count_nonzero(reassigned_labels != labels))

if assignment_mismatch_count != 0:
    raise RuntimeError(
        "Saved labels do not match nearest saved centroids. "
        f"Mismatched patents: "
        f"{assignment_mismatch_count}"
    )

print("\nPrediction shape       :", labels.shape)
print("Centroid shape         :", centroids.shape)
print("Seed                   :", saved_seed)
print("Converged              :", saved_converged)
print("Iterations             :", saved_iterations)
print("Objective              :", f"{saved_objective:.8f}")
print("Active clusters        :", unique_clusters.size)
print("Centroid reassignment  : exact match")


# ------------------------------------------------------------
# 6. Optional duplicate-artifact verification
# ------------------------------------------------------------

reference_identical = None

if REFERENCE_PREDICTION_PATH.exists():
    with np.load(
        REFERENCE_PREDICTION_PATH,
        allow_pickle=False,
    ) as archive:
        reference_labels = archive["labels"].astype(
            np.int64,
            copy=False,
        )

    reference_identical = bool(
        np.array_equal(
            labels,
            reference_labels,
        )
    )

    print(
        "Independent prediction copy identical:",
        reference_identical,
    )

    if not reference_identical:
        raise RuntimeError(
            "Runner and non-runner seed-42 predictions are not identical."
        )


# ------------------------------------------------------------
# 7. Seed-42 metric verification
# ------------------------------------------------------------

section_nmi = float(
    normalized_mutual_info_score(
        sections,
        labels,
        average_method="arithmetic",
    )
)

class_nmi = float(
    normalized_mutual_info_score(
        classes,
        labels,
        average_method="arithmetic",
    )
)

subclass_nmi = float(
    normalized_mutual_info_score(
        subclasses,
        labels,
        average_method="arithmetic",
    )
)

mean_nmi = float(
    np.mean(
        [
            section_nmi,
            class_nmi,
            subclass_nmi,
        ]
    )
)

print("\nSeed-42 section NMI    :", f"{section_nmi:.12f}")
print("Seed-42 class NMI      :", f"{class_nmi:.12f}")
print("Seed-42 subclass NMI   :", f"{subclass_nmi:.12f}")
print("Seed-42 mean NMI       :", f"{mean_nmi:.12f}")


# ------------------------------------------------------------
# 8. Compute cluster-level subclass composition
# ------------------------------------------------------------

cluster_statistics = []

for cluster_id in range(EXPECTED_CLUSTERS):
    member_indices = np.flatnonzero(labels == cluster_id)

    cluster_size = int(member_indices.size)

    if cluster_size == 0:
        raise RuntimeError(f"Cluster {cluster_id} is empty.")

    subclass_counter = Counter(subclasses[member_indices].tolist())

    ranked_subclasses = sorted(
        subclass_counter.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    dominant_subclass = ranked_subclasses[0][0]
    dominant_count = int(ranked_subclasses[0][1])
    purity = dominant_count / cluster_size

    cluster_statistics.append(
        {
            "cluster_id_zero_based": int(cluster_id),
            "cluster_display": (display_cluster_id(cluster_id)),
            "size": cluster_size,
            "subclass_purity": float(purity),
            "dominant_subclass": (dominant_subclass),
            "dominant_count": (dominant_count),
            "eligible": bool(cluster_size >= MINIMUM_CLUSTER_SIZE),
            "ranked_subclasses": [
                {
                    "subclass": subclass,
                    "count": int(count),
                    "share": float(count / cluster_size),
                }
                for subclass, count in ranked_subclasses
            ],
        }
    )

eligible_clusters = [item for item in cluster_statistics if item["eligible"]]

assert len(eligible_clusters) >= 3

high_ranked_clusters = sorted(
    eligible_clusters,
    key=lambda item: (
        -item["subclass_purity"],
        -item["size"],
        item["cluster_id_zero_based"],
    ),
)

low_ranked_clusters = sorted(
    eligible_clusters,
    key=lambda item: (
        item["subclass_purity"],
        -item["size"],
        item["cluster_id_zero_based"],
    ),
)

selected_high = high_ranked_clusters[:2]

selected_high_ids = {item["cluster_id_zero_based"] for item in selected_high}

selected_low = next(
    item
    for item in low_ranked_clusters
    if item["cluster_id_zero_based"] not in selected_high_ids
)

selected_clusters = [
    {
        **selected_high[0],
        "selection_role": ("highest_purity"),
    },
    {
        **selected_high[1],
        "selection_role": ("second_highest_purity"),
    },
    {
        **selected_low,
        "selection_role": ("lowest_purity"),
    },
]

print("\n" + "=" * 110)
print("AUTOMATIC CLUSTER SELECTION")
print("=" * 110)

for selected in selected_clusters:
    print(
        f"{selected['selection_role']:23s} | "
        f"{selected['cluster_display']} | "
        f"internal={selected['cluster_id_zero_based']} | "
        f"size={selected['size']:4d} | "
        f"purity={selected['subclass_purity']:.6f} | "
        f"dominant={selected['dominant_subclass']}"
    )


# ------------------------------------------------------------
# 9. Build Table 2
# ------------------------------------------------------------

table2_rows = []

for selected in selected_clusters:
    top_subclasses = selected["ranked_subclasses"][:TOP_SUBCLASSES_PER_CLUSTER]

    for subclass_rank, item in enumerate(
        top_subclasses,
        start=1,
    ):
        table2_rows.append(
            {
                "selection_role": selected["selection_role"],
                "cluster": selected["cluster_display"],
                "cluster_id_zero_based": selected["cluster_id_zero_based"],
                "cluster_size": selected["size"],
                "cluster_subclass_purity": selected["subclass_purity"],
                "subclass_rank": subclass_rank,
                "cpc_subclass": item["subclass"],
                "subclass_count": item["count"],
                "share": item["share"],
            }
        )

table2_df = pd.DataFrame(table2_rows)


# ------------------------------------------------------------
# 10. Retrieve centroid-nearest patents for Table 3
# ------------------------------------------------------------

table3_rows = []

for selected in selected_clusters:
    cluster_id = selected["cluster_id_zero_based"]

    member_indices = np.flatnonzero(labels == cluster_id)

    member_similarities = similarity_matrix[
        member_indices,
        cluster_id,
    ]

    # Primary key: descending similarity.
    # Tie-breaker: original record index.
    ordering = np.lexsort(
        (
            member_indices,
            -member_similarities,
        )
    )

    selected_positions = ordering[:NEAREST_PATENTS_PER_CLUSTER]

    for patent_rank, position in enumerate(
        selected_positions,
        start=1,
    ):
        global_index = int(member_indices[position])

        table3_rows.append(
            {
                "selection_role": selected["selection_role"],
                "cluster": selected["cluster_display"],
                "cluster_id_zero_based": cluster_id,
                "patent_rank": patent_rank,
                "record_index": global_index,
                "patent_id": record_patent_ids[global_index],
                "cpc_section": sections[global_index],
                "cpc_class": classes[global_index],
                "cpc_subclass": subclasses[global_index],
                "cosine_similarity": float(
                    similarity_matrix[
                        global_index,
                        cluster_id,
                    ]
                ),
            }
        )

table3_df = pd.DataFrame(table3_rows)

# Verify descending similarity within each selected cluster.
for cluster_name, group in table3_df.groupby(
    "cluster",
    sort=False,
):
    values = group["cosine_similarity"].to_numpy()

    assert np.all(values[:-1] >= values[1:]), cluster_name


# ------------------------------------------------------------
# 11. Save CSV
# ------------------------------------------------------------

TABLE2_CSV = OUTPUT_ROOT / "table2_cluster_composition.csv"

TABLE3_CSV = OUTPUT_ROOT / "table3_centroid_patents.csv"

table2_df.to_csv(
    TABLE2_CSV,
    index=False,
)

table3_df.to_csv(
    TABLE3_CSV,
    index=False,
)


# ------------------------------------------------------------
# 12. Create Table 2 LaTeX
# ------------------------------------------------------------

table2_lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\small",
    r"\caption{CPC subclass composition of the automatically selected \textsc{ClaimSem} clusters in the seed-42 test partition. The first two clusters have the highest subclass purity among clusters with at least 50 patents, while the third has the lowest purity among the same eligible set. Share is the fraction of patents in the cluster assigned to the indicated subclass. Cluster identifiers are displayed using one-based numbering.}",
    r"\label{tab:cluster_composition}",
    r"\setlength{\tabcolsep}{4pt}",
    r"\renewcommand{\arraystretch}{1.12}",
    r"\begin{tabular}{@{}lrrlr@{}}",
    r"\toprule",
    r"\textbf{Cluster} & \textbf{Size} & \textbf{Purity} & \textbf{CPC subclass} & \textbf{Share} \\",
    r"\midrule",
]

for cluster_index, selected in enumerate(selected_clusters):
    rows = table2_df[table2_df["cluster"] == selected["cluster_display"]]

    for local_index, row in enumerate(rows.itertuples(index=False)):
        if local_index == 0:
            cluster_cell = row.cluster
            size_cell = str(row.cluster_size)
            purity_cell = f"{row.cluster_subclass_purity:.3f}"
        else:
            cluster_cell = ""
            size_cell = ""
            purity_cell = ""

        table2_lines.append(
            f"{cluster_cell} & "
            f"{size_cell} & "
            f"{purity_cell} & "
            f"{latex_escape(row.cpc_subclass)} & "
            f"{row.share:.3f} \\\\"
        )

    if cluster_index < (len(selected_clusters) - 1):
        table2_lines.append(r"\addlinespace")

table2_lines.extend(
    [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
)

TABLE2_TEX = OUTPUT_ROOT / "table2_cluster_composition.tex"

TABLE2_TEX.write_text(
    "\n".join(table2_lines) + "\n",
    encoding="utf-8",
)


# ------------------------------------------------------------
# 13. Create Table 3 LaTeX
# ------------------------------------------------------------

table3_lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\small",
    r"\caption{Patents closest to the centroids of the automatically selected \textsc{ClaimSem} clusters in the seed-42 test partition. Similarity is cosine similarity between each patent representation and its assigned cluster centroid in the normalized 128-dimensional PCA space used for spherical K-means clustering.}",
    r"\label{tab:centroid_patents}",
    r"\setlength{\tabcolsep}{4pt}",
    r"\renewcommand{\arraystretch}{1.12}",
    r"\begin{tabular}{@{}lllr@{}}",
    r"\toprule",
    r"\textbf{Cluster} & \textbf{Patent ID} & \textbf{CPC subclass} & \textbf{Similarity} \\",
    r"\midrule",
]

for cluster_index, selected in enumerate(selected_clusters):
    rows = table3_df[table3_df["cluster"] == selected["cluster_display"]]

    for local_index, row in enumerate(rows.itertuples(index=False)):
        cluster_cell = row.cluster if local_index == 0 else ""

        table3_lines.append(
            f"{cluster_cell} & "
            f"{display_patent_id(row.patent_id)} & "
            f"{latex_escape(row.cpc_subclass)} & "
            f"{row.cosine_similarity:.4f} \\\\"
        )

    if cluster_index < (len(selected_clusters) - 1):
        table3_lines.append(r"\addlinespace")

table3_lines.extend(
    [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
)

TABLE3_TEX = OUTPUT_ROOT / "table3_centroid_patents.tex"

TABLE3_TEX.write_text(
    "\n".join(table3_lines) + "\n",
    encoding="utf-8",
)


# ------------------------------------------------------------
# 14. Save selected centroids and JSON report
# ------------------------------------------------------------

SELECTED_CENTROIDS_PATH = OUTPUT_ROOT / "selected_cluster_centroids.npz"

selected_internal_ids = np.asarray(
    [item["cluster_id_zero_based"] for item in selected_clusters],
    dtype=np.int64,
)

np.savez_compressed(
    SELECTED_CENTROIDS_PATH,
    cluster_ids_zero_based=selected_internal_ids,
    cluster_ids_display=np.asarray(
        [item["cluster_display"] for item in selected_clusters],
        dtype=str,
    ),
    selection_roles=np.asarray(
        [item["selection_role"] for item in selected_clusters],
        dtype=str,
    ),
    centroids=centroids[selected_internal_ids],
)

REPORT_JSON = OUTPUT_ROOT / "table2_3_case_study.json"

report = {
    "schema_version": "1.0",
    "split": "test",
    "method": "full_claimsem",
    "configuration": {
        "root_weight": 12.0,
        "depth_decay": 0.1,
        "pca_dimension": 128,
        "n_clusters": 30,
        "case_study_seed": 42,
    },
    "data": {
        "n_patents": EXPECTED_N_PATENTS,
        "n_sections": int(np.unique(sections).size),
        "n_classes": int(np.unique(classes).size),
        "n_subclasses": int(np.unique(subclasses).size),
    },
    "selection_protocol": {
        "minimum_cluster_size": (MINIMUM_CLUSTER_SIZE),
        "high_purity_clusters": 2,
        "low_purity_clusters": 1,
        "purity_label_level": ("CPC subclass"),
        "high_purity_tie_breaking": [
            "larger cluster size",
            "lower internal cluster ID",
        ],
        "low_purity_tie_breaking": [
            "larger cluster size",
            "lower internal cluster ID",
        ],
        "manual_filtering": False,
        "manual_text_inspection_before_selection": False,
    },
    "seed_42_metrics": {
        "section_nmi": section_nmi,
        "class_nmi": class_nmi,
        "subclass_nmi": subclass_nmi,
        "mean_nmi": mean_nmi,
    },
    "clustering": {
        "seed": saved_seed,
        "converged": saved_converged,
        "n_iterations": saved_iterations,
        "objective": saved_objective,
        "active_clusters": int(unique_clusters.size),
        "centroid_reassignment_mismatches": (assignment_mismatch_count),
    },
    "selected_clusters": [
        {
            "selection_role": item["selection_role"],
            "cluster_id_zero_based": item["cluster_id_zero_based"],
            "cluster_display": item["cluster_display"],
            "size": item["size"],
            "subclass_purity": item["subclass_purity"],
            "dominant_subclass": item["dominant_subclass"],
            "top_subclasses": item["ranked_subclasses"][:TOP_SUBCLASSES_PER_CLUSTER],
        }
        for item in selected_clusters
    ],
    "table2": table2_df.to_dict(orient="records"),
    "table3": table3_df.to_dict(orient="records"),
}

with open(
    REPORT_JSON,
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        report,
        handle,
        indent=2,
        ensure_ascii=False,
    )


# ------------------------------------------------------------
# 15. Manifest and checksums
# ------------------------------------------------------------

generated_files = [
    TABLE2_CSV,
    TABLE2_TEX,
    TABLE3_CSV,
    TABLE3_TEX,
    REPORT_JSON,
    SELECTED_CENTROIDS_PATH,
]

MANIFEST_PATH = OUTPUT_ROOT / "table2_3_manifest.json"

manifest = {
    "created_at_unix": time.time(),
    "source_artifacts": {
        "records": {
            "path": str(TEST_RECORDS_PATH),
            "sha256": sha256_file(TEST_RECORDS_PATH),
        },
        "predictions": {
            "path": str(PREDICTION_PATH),
            "sha256": sha256_file(PREDICTION_PATH),
        },
        "pca_features": {
            "path": str(FEATURE_PATH),
            "sha256": sha256_file(FEATURE_PATH),
        },
    },
    "verification": {
        "n_patents": EXPECTED_N_PATENTS,
        "feature_shape": list(features.shape),
        "prediction_shape": list(labels.shape),
        "centroid_shape": list(centroids.shape),
        "patent_order_aligned": True,
        "seed": saved_seed,
        "converged": saved_converged,
        "active_clusters": int(unique_clusters.size),
        "centroid_reassignment_exact": (assignment_mismatch_count == 0),
        "independent_prediction_copy_identical": (reference_identical),
    },
    "generated_files": {},
}

for generated_path in generated_files:
    manifest["generated_files"][generated_path.name] = {
        "path": str(generated_path),
        "sha256": sha256_file(generated_path),
        "bytes": generated_path.stat().st_size,
    }

with open(
    MANIFEST_PATH,
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        manifest,
        handle,
        indent=2,
        ensure_ascii=False,
    )

CHECKSUM_PATH = OUTPUT_ROOT / "checksums.sha256"

checksum_lines = []

for path in [
    *generated_files,
    MANIFEST_PATH,
]:
    checksum_lines.append(f"{sha256_file(path)}  {path.name}")

CHECKSUM_PATH.write_text(
    "\n".join(checksum_lines) + "\n",
    encoding="utf-8",
)


# ------------------------------------------------------------
# 16. Final output
# ------------------------------------------------------------

print("\n" + "=" * 110)
print("TABLE 2 — CLUSTER COMPOSITION")
print("=" * 110)

print(
    table2_df[
        [
            "selection_role",
            "cluster",
            "cluster_size",
            "cluster_subclass_purity",
            "subclass_rank",
            "cpc_subclass",
            "subclass_count",
            "share",
        ]
    ].to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

print("\n" + "=" * 110)
print("TABLE 3 — CENTROID-NEAREST PATENTS")
print("=" * 110)

print(
    table3_df[
        [
            "selection_role",
            "cluster",
            "patent_rank",
            "patent_id",
            "cpc_subclass",
            "cosine_similarity",
        ]
    ].to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

print("\n" + "=" * 110)
print("GENERATED TABLE 2 LATEX")
print("=" * 110)
print(TABLE2_TEX.read_text(encoding="utf-8"))

print("=" * 110)
print("GENERATED TABLE 3 LATEX")
print("=" * 110)
print(TABLE3_TEX.read_text(encoding="utf-8"))

print("=" * 110)
print("TABLES 2–3 FINAL VERIFICATION")
print("=" * 110)
print("TEST patents                  :", len(records))
print("Features                      :", features.shape)
print("Predictions                   :", labels.shape)
print("Centroids                     :", centroids.shape)
print("Seed                          :", saved_seed)
print("Converged                     :", saved_converged)
print("Active clusters               :", unique_clusters.size)
print("Patent order aligned          : True")
print("Centroid reassignment exact   :", assignment_mismatch_count == 0)
print("Minimum eligible cluster size :", MINIMUM_CLUSTER_SIZE)
print(
    "Selected clusters             :",
    [item["cluster_display"] for item in selected_clusters],
)
print("Table 2 CSV                   :", TABLE2_CSV)
print("Table 2 LaTeX                 :", TABLE2_TEX)
print("Table 3 CSV                   :", TABLE3_CSV)
print("Table 3 LaTeX                 :", TABLE3_TEX)
print("Report JSON                   :", REPORT_JSON)
print("Manifest                      :", MANIFEST_PATH)
print("Checksums                     :", CHECKSUM_PATH)
print(
    "\nSUCCESS: Tables 2–3 were generated without manual cluster or patent selection."
)
