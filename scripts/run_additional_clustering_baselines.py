# ruff: noqa

"""Reproduce additional lexical and agglomerative baselines.

This script expects the original processed USPTO record files and cached
embedding artifacts at the paths configured below. CPC labels are used only
for final evaluation.
"""

import pickle
from pathlib import Path

DATA_ROOT = Path("/content/drive/MyDrive/depth_ot_patent/data/processed")

PATHS = {
    "train": DATA_ROOT / "train_records.pkl",
    "dev": DATA_ROOT / "dev_records.pkl",
    "test": DATA_ROOT / "test_records.pkl",
}

records = {}

for split, path in PATHS.items():
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        records[split] = pickle.load(handle)

if len(records["train"]) != 49_599:
    raise RuntimeError("Unexpected TRAIN record count")
if len(records["dev"]) != 9_855:
    raise RuntimeError("Unexpected DEV record count")
if len(records["test"]) != 9_881:
    raise RuntimeError("Unexpected TEST record count")


# Cell 3 — Build ordered patent documents and labels


def build_document(record):
    claims = record["claims"]
    ordered_ids = sorted(claims, key=lambda x: int(x))
    texts = [str(claims[cid]).strip() for cid in ordered_ids]
    texts = [text for text in texts if text]
    assert texts, f"No claims: {record['patent_id']}"
    return "\n".join(texts)


documents = {}
labels = {}

for split in ("train", "dev", "test"):
    documents[split] = [build_document(r) for r in records[split]]
    labels[split] = {
        "section": [str(r["section"]) for r in records[split]],
        "class": [str(r["class"]) for r in records[split]],
        "subclass": [str(r["subclass"]) for r in records[split]],
    }

    assert len(documents[split]) == len(records[split])
    assert all(documents[split])
    assert all(labels[split]["subclass"])

    print(
        f"{split:5s}: documents={len(documents[split]):,}, "
        f"mean characters={sum(map(len, documents[split])) / len(documents[split]):,.1f}"
    )

print("\nFirst TEST patent:", records["test"][0]["patent_id"])
print("First TEST labels:", {k: v[0] for k, v in labels["test"].items()})
print("First document preview:", documents["test"][0][:500])
print("\nSUCCESS: Ordered patent documents prepared.")


# Cell 4 — Fit TRAIN-only TF-IDF-50K and transform TEST

import json
import joblib
import numpy as np
from pathlib import Path
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

OUTPUT_DIR = Path(
    "/content/drive/MyDrive/claimsem_artifacts/"
    "additional_clustering_baselines/tfidf_spherical_kmeans"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VECTORIZER_PATH = OUTPUT_DIR / "tfidf_50k_vectorizer.joblib"
TEST_MATRIX_PATH = OUTPUT_DIR / "test_tfidf_50k.npz"
METADATA_PATH = OUTPUT_DIR / "tfidf_50k_metadata.json"

vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    stop_words="english",
    token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.5,
    max_features=50_000,
    sublinear_tf=True,
    norm="l2",
    dtype=np.float32,
)

print("Fitting TF-IDF on TRAIN only...")
vectorizer.fit(documents["train"])

print("Transforming TEST...")
X_test_tfidf = vectorizer.transform(documents["test"]).tocsr()

assert X_test_tfidf.shape[0] == 9_881
assert X_test_tfidf.shape[1] <= 50_000
assert np.isfinite(X_test_tfidf.data).all()

norms = np.sqrt(X_test_tfidf.multiply(X_test_tfidf).sum(axis=1)).A1
assert np.allclose(norms, 1.0, atol=1e-5)
assert np.all(X_test_tfidf.getnnz(axis=1) > 0)

joblib.dump(vectorizer, VECTORIZER_PATH, compress=3)
save_npz(TEST_MATRIX_PATH, X_test_tfidf, compressed=True)

metadata = {
    "baseline": "TF-IDF-50K + spherical K-means",
    "fit_split": "train",
    "train_documents": len(documents["train"]),
    "test_documents": len(documents["test"]),
    "shape": list(X_test_tfidf.shape),
    "dtype": str(X_test_tfidf.dtype),
    "nnz": int(X_test_tfidf.nnz),
    "vocabulary_size": len(vectorizer.vocabulary_),
    "parameters": {
        "ngram_range": [1, 2],
        "min_df": 5,
        "max_df": 0.5,
        "max_features": 50_000,
        "sublinear_tf": True,
        "norm": "l2",
        "stop_words": "english",
    },
}

METADATA_PATH.write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

print("\nTF-IDF shape:", X_test_tfidf.shape)
print("Vocabulary size:", len(vectorizer.vocabulary_))
print("Non-zero values:", f"{X_test_tfidf.nnz:,}")
print("Norm range:", float(norms.min()), float(norms.max()))
print("Matrix:", TEST_MATRIX_PATH)
print("Vectorizer:", VECTORIZER_PATH)
print("\nSUCCESS: TF-IDF baseline features prepared.")


# Cell 5 — Sparse spherical K-means and CPC evaluation

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix

N_CLUSTERS = 30
SEEDS = [17, 42, 73]
MAX_ITER = 100
TOL = 1e-5

RESULT_JSON = OUTPUT_DIR / "tfidf_50k_spherical_kmeans_results.json"
RESULT_CSV = OUTPUT_DIR / "tfidf_50k_spherical_kmeans_metrics.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "tfidf_50k_spherical_kmeans_predictions.npz"


def spherical_kmeans_plus_plus(X, n_clusters, rng):
    n_samples = X.shape[0]
    centers = np.zeros((n_clusters, X.shape[1]), dtype=np.float32)

    first = int(rng.integers(n_samples))
    centers[0] = X[first].toarray().ravel()

    closest_similarity = np.asarray(X @ centers[0]).ravel()

    for k in range(1, n_clusters):
        distances = np.clip(1.0 - closest_similarity, 0.0, None)
        total = distances.sum()

        if total <= 0:
            index = int(rng.integers(n_samples))
        else:
            index = int(rng.choice(n_samples, p=distances / total))

        centers[k] = X[index].toarray().ravel()
        similarity = np.asarray(X @ centers[k]).ravel()
        closest_similarity = np.maximum(closest_similarity, similarity)

    return centers


def sparse_spherical_kmeans(X, n_clusters, seed, max_iter=100, tol=1e-5):
    X = X.tocsr().astype(np.float32)
    rng = np.random.default_rng(seed)
    centers = spherical_kmeans_plus_plus(X, n_clusters, rng)

    previous_objective = None
    previous_labels = None
    converged = False

    for iteration in range(1, max_iter + 1):
        similarities = np.asarray(X @ centers.T)
        labels_pred = similarities.argmax(axis=1).astype(np.int32)
        best_similarity = similarities[np.arange(X.shape[0]), labels_pred]
        objective = float(best_similarity.mean())

        if previous_labels is not None and np.array_equal(labels_pred, previous_labels):
            converged = True
            break

        if (
            previous_objective is not None
            and abs(objective - previous_objective) <= tol
        ):
            converged = True
            break

        membership = sparse.csr_matrix(
            (
                np.ones(X.shape[0], dtype=np.float32),
                (labels_pred, np.arange(X.shape[0])),
            ),
            shape=(n_clusters, X.shape[0]),
        )

        centers = (membership @ X).toarray().astype(np.float32)
        center_norms = np.linalg.norm(centers, axis=1)

        empty_clusters = np.flatnonzero(center_norms == 0)

        for cluster_id in empty_clusters:
            replacement = int(np.argmin(best_similarity))
            centers[cluster_id] = X[replacement].toarray().ravel()
            center_norms[cluster_id] = np.linalg.norm(centers[cluster_id])

        centers /= np.maximum(center_norms[:, None], 1e-12)

        previous_labels = labels_pred.copy()
        previous_objective = objective

    similarities = np.asarray(X @ centers.T)
    labels_pred = similarities.argmax(axis=1).astype(np.int32)
    objective = float(similarities[np.arange(X.shape[0]), labels_pred].mean())

    return {
        "labels": labels_pred,
        "centers": centers,
        "objective": objective,
        "iterations": iteration,
        "converged": converged,
        "active_clusters": int(np.unique(labels_pred).size),
    }


def evaluate_partition(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred, sparse=False)
    n = cm.sum()

    predicted_cluster_purity = float(cm.max(axis=0).sum() / n)
    inverse_label_purity = float(cm.max(axis=1).sum() / n)
    nmi = float(
        normalized_mutual_info_score(
            y_true,
            y_pred,
            average_method="arithmetic",
        )
    )

    return {
        "predicted_cluster_purity": predicted_cluster_purity,
        "inverse_label_purity": inverse_label_purity,
        "nmi": nmi,
    }


all_runs = []
prediction_arrays = {}

for seed in SEEDS:
    print(f"\nRunning seed {seed}...")

    run = sparse_spherical_kmeans(
        X_test_tfidf,
        n_clusters=N_CLUSTERS,
        seed=seed,
        max_iter=MAX_ITER,
        tol=TOL,
    )

    assert run["active_clusters"] == N_CLUSTERS

    metrics = {
        level: evaluate_partition(labels["test"][level], run["labels"])
        for level in ("section", "class", "subclass")
    }

    mean_nmi = float(np.mean([metrics[level]["nmi"] for level in metrics]))

    all_runs.append(
        {
            "seed": seed,
            "converged": run["converged"],
            "iterations": run["iterations"],
            "active_clusters": run["active_clusters"],
            "objective": run["objective"],
            "mean_nmi": mean_nmi,
            "metrics": metrics,
        }
    )

    prediction_arrays[f"seed_{seed}"] = run["labels"]

    print(
        f"iterations={run['iterations']}, "
        f"converged={run['converged']}, "
        f"active={run['active_clusters']}, "
        f"objective={run['objective']:.6f}, "
        f"mean NMI={mean_nmi:.6f}"
    )


rows = []

for run in all_runs:
    for level in ("section", "class", "subclass"):
        rows.append(
            {
                "seed": run["seed"],
                "level": level,
                **run["metrics"][level],
            }
        )

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(RESULT_CSV, index=False)

summary = metrics_df.groupby("level", sort=False)[
    ["predicted_cluster_purity", "inverse_label_purity", "nmi"]
].agg(["mean", "std"])

np.savez_compressed(PREDICTIONS_PATH, **prediction_arrays)

result = {
    "baseline": "TF-IDF-50K + spherical K-means",
    "n_test": 9881,
    "n_features": int(X_test_tfidf.shape[1]),
    "n_clusters": N_CLUSTERS,
    "seeds": SEEDS,
    "max_iter": MAX_ITER,
    "tolerance": TOL,
    "runs": all_runs,
}

RESULT_JSON.write_text(
    json.dumps(result, indent=2),
    encoding="utf-8",
)

print("\n" + "=" * 80)
print("FINAL TF-IDF + SPHERICAL K-MEANS RESULTS")
print("=" * 80)
print(summary.to_string())

mean_nmi = metrics_df.groupby("seed")["nmi"].mean()

print("\nMean NMI by seed:")
print(mean_nmi.to_string())

print("\nOverall mean NMI:", f"{mean_nmi.mean():.6f}")
print("Overall SD:", f"{mean_nmi.std(ddof=0):.6f}")
print("\nResults:", RESULT_JSON)
print("Metrics:", RESULT_CSV)
print("Predictions:", PREDICTIONS_PATH)


# ============================================================
# PatentSBERTa-V2 Uniform Pooling + Agglomerative Clustering
# ============================================================

import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix

# ------------------------------------------------------------
# 1. Paths and configuration
# ------------------------------------------------------------

EXPECTED_N = 9_881
N_CLUSTERS = 30

SEARCH_ROOT = Path("/content/drive/MyDrive/claimsem_artifacts/test_embedding_baselines")
TEST_RECORDS_PATH = Path(
    "/content/drive/MyDrive/depth_ot_patent/data/processed/test_records.pkl"
)
OUTPUT_DIR = Path(
    "/content/drive/MyDrive/claimsem_artifacts/"
    "additional_clustering_baselines/"
    "patentsberta_uniform_agglomerative"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_PATH = OUTPUT_DIR / "patentsberta_uniform_agglomerative_predictions.npz"
METRICS_CSV = OUTPUT_DIR / "patentsberta_uniform_agglomerative_metrics.csv"
RESULT_JSON = OUTPUT_DIR / "patentsberta_uniform_agglomerative_results.json"
LATEX_PATH = OUTPUT_DIR / "patentsberta_uniform_agglomerative_row.tex"


# ------------------------------------------------------------
# 2. Load TEST records independently
# ------------------------------------------------------------

assert TEST_RECORDS_PATH.exists(), TEST_RECORDS_PATH

with TEST_RECORDS_PATH.open("rb") as f:
    test_records = pickle.load(f)

assert len(test_records) == EXPECTED_N

test_patent_ids = np.asarray([str(record["patent_id"]) for record in test_records])

test_labels = {
    "section": np.asarray([str(record["section"]) for record in test_records]),
    "class": np.asarray([str(record["class"]) for record in test_records]),
    "subclass": np.asarray([str(record["subclass"]) for record in test_records]),
}

assert len(np.unique(test_labels["section"])) == 9
assert len(np.unique(test_labels["class"])) == 121
assert len(np.unique(test_labels["subclass"])) == 466

print("TEST records:", len(test_records))
print("First patent:", test_patent_ids[0])
print(
    "CPC categories:",
    len(np.unique(test_labels["section"])),
    len(np.unique(test_labels["class"])),
    len(np.unique(test_labels["subclass"])),
)


# ------------------------------------------------------------
# 3. Locate the uniform-pooling PCA-128 artifact
# ------------------------------------------------------------

assert SEARCH_ROOT.exists(), f"Search root not found: {SEARCH_ROOT}"

candidates = []

for path in SEARCH_ROOT.rglob("*.npz"):
    text = str(path).lower()

    if "prediction" in text or "cluster" in text:
        continue

    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = list(archive.files)

            for key in keys:
                arr = archive[key]

                if arr.shape == (EXPECTED_N, 128):
                    score = 0
                    combined = f"{text} {key.lower()}"

                    for token, points in [
                        ("patentsberta", 10),
                        ("patent_sberta", 10),
                        ("uniform", 10),
                        ("pooling", 5),
                        ("pca128", 8),
                        ("pca_128", 8),
                        ("feature", 4),
                        ("embedding", 3),
                        ("test", 3),
                    ]:
                        if token in combined:
                            score += points

                    candidates.append(
                        {
                            "score": score,
                            "path": path,
                            "key": key,
                            "keys": keys,
                        }
                    )

    except Exception as exc:
        warnings.warn(f"Could not inspect {path}: {exc}")

assert candidates, (
    f"No (9881, 128) PatentSBERTa feature artifact was found under {SEARCH_ROOT}"
)

candidates.sort(key=lambda x: (-x["score"], str(x["path"]), x["key"]))

print("\nFeature candidates:")
for candidate in candidates[:10]:
    print(
        f"score={candidate['score']:2d} | "
        f"key={candidate['key']} | "
        f"path={candidate['path']}"
    )

selected = candidates[0]
FEATURE_PATH = selected["path"]
FEATURE_KEY = selected["key"]

selected_text = f"{FEATURE_PATH} {FEATURE_KEY}".lower()

assert "uniform" in selected_text, (
    "Top candidate does not explicitly contain 'uniform'. "
    "Do not continue with a ROOTS feature artifact."
)
assert "patentsberta" in selected_text or "patent_sberta" in selected_text, (
    "Selected artifact is not clearly identified as PatentSBERTa."
)

print("\nSelected feature artifact:")
print("Path:", FEATURE_PATH)
print("Key :", FEATURE_KEY)


# ------------------------------------------------------------
# 4. Load and validate PCA-128 features and patent ordering
# ------------------------------------------------------------

with np.load(FEATURE_PATH, allow_pickle=False) as archive:
    X = np.asarray(archive[FEATURE_KEY], dtype=np.float32)

    id_key = next(
        (
            key
            for key in archive.files
            if key.lower()
            in {
                "patent_ids",
                "patent_id",
                "ids",
                "document_ids",
                "doc_ids",
            }
        ),
        None,
    )

    stored_ids = np.asarray(archive[id_key]).astype(str) if id_key is not None else None

assert X.shape == (EXPECTED_N, 128)
assert np.isfinite(X).all()

original_norms = np.linalg.norm(X, axis=1)
assert np.all(original_norms > 0)

# Ensure exact unit normalization for cosine clustering
X = X / np.maximum(original_norms[:, None], 1e-12)
normalized_norms = np.linalg.norm(X, axis=1)

assert np.allclose(normalized_norms, 1.0, atol=1e-5)

if stored_ids is not None:
    assert stored_ids.shape[0] == EXPECTED_N
    assert np.array_equal(stored_ids, test_patent_ids), (
        "Feature/patent ordering mismatch."
    )
    order_status = "verified from stored patent IDs"
else:
    order_status = (
        "no patent_ids key in archive; using the frozen TEST order "
        "from the existing baseline pipeline"
    )

print("\nFeature shape:", X.shape)
print("Feature dtype:", X.dtype)
print(
    "Normalized norm range:",
    float(normalized_norms.min()),
    float(normalized_norms.max()),
)
print("Patent order:", order_status)


# ------------------------------------------------------------
# 5. Run deterministic cosine-average agglomerative clustering
# ------------------------------------------------------------

print("\nRunning agglomerative clustering...")
print("This step may take several minutes and may not print progress.")

start_time = time.time()

try:
    model = AgglomerativeClustering(
        n_clusters=N_CLUSTERS,
        metric="cosine",
        linkage="average",
        compute_full_tree=True,
    )
except TypeError:
    # Compatibility with older scikit-learn
    model = AgglomerativeClustering(
        n_clusters=N_CLUSTERS,
        affinity="cosine",
        linkage="average",
        compute_full_tree=True,
    )

predictions = model.fit_predict(X).astype(np.int32)
elapsed_seconds = time.time() - start_time

active_clusters = int(np.unique(predictions).size)
cluster_sizes = np.bincount(predictions, minlength=N_CLUSTERS)

assert predictions.shape == (EXPECTED_N,)
assert active_clusters == N_CLUSTERS
assert cluster_sizes.sum() == EXPECTED_N
assert np.all(cluster_sizes > 0)

print(f"Finished in {elapsed_seconds / 60:.2f} minutes")
print("Active clusters:", active_clusters)
print(
    "Cluster size range:",
    int(cluster_sizes.min()),
    int(cluster_sizes.max()),
)


# ------------------------------------------------------------
# 6. Evaluate CPC alignment
# ------------------------------------------------------------


def evaluate_partition(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred, sparse=False)
    n = int(cm.sum())

    return {
        "predicted_cluster_purity": float(cm.max(axis=0).sum() / n),
        "inverse_label_purity": float(cm.max(axis=1).sum() / n),
        "nmi": float(
            normalized_mutual_info_score(
                y_true,
                y_pred,
                average_method="arithmetic",
            )
        ),
    }


metrics = {
    level: evaluate_partition(test_labels[level], predictions)
    for level in ("section", "class", "subclass")
}

mean_nmi = float(np.mean([metrics[level]["nmi"] for level in metrics]))

rows = []

for level in ("section", "class", "subclass"):
    rows.append(
        {
            "model": "PatentSBERTa-V2 + agglomerative clustering",
            "level": level,
            **metrics[level],
        }
    )

metrics_df = pd.DataFrame(rows)


# ------------------------------------------------------------
# 7. Save artifacts
# ------------------------------------------------------------

np.savez_compressed(
    PREDICTION_PATH,
    labels=predictions,
    patent_ids=test_patent_ids,
    cluster_sizes=cluster_sizes,
)

metrics_df.to_csv(METRICS_CSV, index=False)

result = {
    "baseline": "PatentSBERTa-V2 uniform pooling + agglomerative clustering",
    "representation": {
        "source_path": str(FEATURE_PATH),
        "source_key": FEATURE_KEY,
        "shape": list(X.shape),
        "normalized": True,
        "patent_order": order_status,
    },
    "clustering": {
        "algorithm": "AgglomerativeClustering",
        "n_clusters": N_CLUSTERS,
        "metric": "cosine",
        "linkage": "average",
        "deterministic": True,
        "active_clusters": active_clusters,
        "cluster_size_min": int(cluster_sizes.min()),
        "cluster_size_max": int(cluster_sizes.max()),
        "elapsed_seconds": elapsed_seconds,
    },
    "metrics": metrics,
    "mean_nmi": mean_nmi,
}

RESULT_JSON.write_text(
    json.dumps(result, indent=2),
    encoding="utf-8",
)

sec = metrics["section"]
cls = metrics["class"]
sub = metrics["subclass"]

latex_row = (
    "PatentSBERTa-V2 + agglomerative clustering\n"
    f"& {sec['predicted_cluster_purity']:.4f} "
    f"& {sec['inverse_label_purity']:.4f} "
    f"& {sec['nmi']:.4f}\n"
    f"& {cls['predicted_cluster_purity']:.4f} "
    f"& {cls['inverse_label_purity']:.4f} "
    f"& {cls['nmi']:.4f}\n"
    f"& {sub['predicted_cluster_purity']:.4f} "
    f"& {sub['inverse_label_purity']:.4f} "
    f"& {sub['nmi']:.4f} \\\\\n"
)

LATEX_PATH.write_text(latex_row, encoding="utf-8")


# ------------------------------------------------------------
# 8. Final report
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("FINAL PATENTSBERTA-V2 + AGGLOMERATIVE RESULTS")
print("=" * 80)

for level in ("section", "class", "subclass"):
    values = metrics[level]
    print(
        f"{level:8s} | "
        f"Pur_p={values['predicted_cluster_purity']:.6f} | "
        f"Pur_a={values['inverse_label_purity']:.6f} | "
        f"NMI={values['nmi']:.6f}"
    )

print(f"\nMean NMI: {mean_nmi:.6f}")
print("\nLaTeX row:")
print(latex_row)

print("Predictions:", PREDICTION_PATH)
print("Metrics    :", METRICS_CSV)
print("Results    :", RESULT_JSON)
print("LaTeX      :", LATEX_PATH)
print("\nSUCCESS: PatentSBERTa-V2 agglomerative baseline completed.")


# ============================================================
# MiniLM Document Embeddings + Agglomerative Clustering
# ============================================================

import gc
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix

gc.collect()

EXPECTED_N = 9_881
N_CLUSTERS = 30

SEARCH_ROOT = Path("/content/drive/MyDrive/claimsem_artifacts/test_embedding_baselines")
TEST_RECORDS_PATH = Path(
    "/content/drive/MyDrive/depth_ot_patent/data/processed/test_records.pkl"
)
OUTPUT_DIR = Path(
    "/content/drive/MyDrive/claimsem_artifacts/"
    "additional_clustering_baselines/minilm_agglomerative"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_PATH = OUTPUT_DIR / "minilm_agglomerative_predictions.npz"
METRICS_PATH = OUTPUT_DIR / "minilm_agglomerative_metrics.csv"
RESULT_PATH = OUTPUT_DIR / "minilm_agglomerative_results.json"
LATEX_PATH = OUTPUT_DIR / "minilm_agglomerative_row.tex"

# ------------------------------------------------------------
# Load TEST labels
# ------------------------------------------------------------

with TEST_RECORDS_PATH.open("rb") as f:
    test_records_local = pickle.load(f)

assert len(test_records_local) == EXPECTED_N

patent_ids = np.asarray([str(record["patent_id"]) for record in test_records_local])

cpc_labels = {
    "section": np.asarray([str(record["section"]) for record in test_records_local]),
    "class": np.asarray([str(record["class"]) for record in test_records_local]),
    "subclass": np.asarray([str(record["subclass"]) for record in test_records_local]),
}

assert len(np.unique(cpc_labels["section"])) == 9
assert len(np.unique(cpc_labels["class"])) == 121
assert len(np.unique(cpc_labels["subclass"])) == 466

# ------------------------------------------------------------
# Locate cached MiniLM TEST document embeddings
# ------------------------------------------------------------

candidate_paths = sorted(
    {
        *SEARCH_ROOT.rglob("*minilm*.npz"),
        *SEARCH_ROOT.rglob("*MiniLM*.npz"),
    }
)

feature_candidates = []

for path in candidate_paths:
    path_text = str(path).lower()

    if "prediction" in path_text or "cluster" in path_text:
        continue

    try:
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                array = archive[key]

                if (
                    len(array.shape) == 2
                    and array.shape[0] == EXPECTED_N
                    and array.shape[1] in {384, 768}
                ):
                    score = 0
                    combined = f"{path_text} {key.lower()}"

                    if "minilm" in combined:
                        score += 10
                    if "document" in combined:
                        score += 8
                    if "embedding" in combined:
                        score += 6
                    if "bertopic" in combined:
                        score += 4
                    if "test" in combined:
                        score += 3
                    if array.shape[1] == 384:
                        score += 5

                    feature_candidates.append(
                        {
                            "score": score,
                            "path": path,
                            "key": key,
                            "shape": array.shape,
                            "keys": list(archive.files),
                        }
                    )

    except Exception as exc:
        print("Skipped:", path, exc)

assert feature_candidates, (
    f"No MiniLM TEST embedding artifact found under {SEARCH_ROOT}"
)

feature_candidates.sort(
    key=lambda item: (-item["score"], str(item["path"]), item["key"])
)

print("MiniLM candidates:")
for item in feature_candidates[:10]:
    print(
        f"score={item['score']:2d} | shape={item['shape']} | "
        f"key={item['key']} | {item['path']}"
    )

selected = feature_candidates[0]
FEATURE_PATH = selected["path"]
FEATURE_KEY = selected["key"]

assert "minilm" in f"{FEATURE_PATH} {FEATURE_KEY}".lower()

print("\nSelected:")
print("Path:", FEATURE_PATH)
print("Key :", FEATURE_KEY)

# ------------------------------------------------------------
# Load and normalize embeddings
# ------------------------------------------------------------

with np.load(FEATURE_PATH, allow_pickle=False) as archive:
    X_minilm = np.asarray(archive[FEATURE_KEY], dtype=np.float32)

    id_key = next(
        (
            key
            for key in archive.files
            if key.lower()
            in {
                "patent_ids",
                "patent_id",
                "ids",
                "document_ids",
                "doc_ids",
            }
        ),
        None,
    )

    stored_ids = np.asarray(archive[id_key]).astype(str) if id_key is not None else None

assert X_minilm.shape[0] == EXPECTED_N
assert X_minilm.ndim == 2
assert np.isfinite(X_minilm).all()

norms = np.linalg.norm(X_minilm, axis=1)
assert np.all(norms > 0)

X_minilm /= np.maximum(norms[:, None], 1e-12)

if stored_ids is not None:
    assert np.array_equal(stored_ids, patent_ids)
    order_status = "verified from stored patent IDs"
else:
    order_status = "frozen TEST order; no patent_ids key in archive"

print("\nFeature shape:", X_minilm.shape)
print("Patent order:", order_status)
print(
    "Norm range:",
    float(np.linalg.norm(X_minilm, axis=1).min()),
    float(np.linalg.norm(X_minilm, axis=1).max()),
)

# ------------------------------------------------------------
# Agglomerative clustering
# ------------------------------------------------------------

print("\nRunning cosine-average agglomerative clustering...")
start = time.time()

try:
    clusterer = AgglomerativeClustering(
        n_clusters=N_CLUSTERS,
        metric="cosine",
        linkage="average",
        compute_full_tree=True,
    )
except TypeError:
    clusterer = AgglomerativeClustering(
        n_clusters=N_CLUSTERS,
        affinity="cosine",
        linkage="average",
        compute_full_tree=True,
    )

predictions = clusterer.fit_predict(X_minilm).astype(np.int32)
elapsed = time.time() - start

cluster_sizes = np.bincount(
    predictions,
    minlength=N_CLUSTERS,
)
active_clusters = int(np.count_nonzero(cluster_sizes))
max_cluster_share = float(cluster_sizes.max() / EXPECTED_N)

assert predictions.shape == (EXPECTED_N,)
assert active_clusters == N_CLUSTERS

# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------


def evaluate(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred, sparse=False)
    n = cm.sum()

    return {
        "predicted_cluster_purity": float(cm.max(axis=0).sum() / n),
        "inverse_label_purity": float(cm.max(axis=1).sum() / n),
        "nmi": float(
            normalized_mutual_info_score(
                y_true,
                y_pred,
                average_method="arithmetic",
            )
        ),
    }


metrics = {
    level: evaluate(cpc_labels[level], predictions)
    for level in ("section", "class", "subclass")
}

mean_nmi = float(
    np.mean([metrics[level]["nmi"] for level in ("section", "class", "subclass")])
)

metrics_df = pd.DataFrame(
    [
        {
            "model": "MiniLM + agglomerative clustering",
            "level": level,
            **metrics[level],
        }
        for level in ("section", "class", "subclass")
    ]
)

metrics_df.to_csv(METRICS_PATH, index=False)

np.savez_compressed(
    PREDICTION_PATH,
    labels=predictions,
    patent_ids=patent_ids,
    cluster_sizes=cluster_sizes,
)

result = {
    "baseline": "MiniLM document embeddings + agglomerative clustering",
    "feature_path": str(FEATURE_PATH),
    "feature_key": FEATURE_KEY,
    "feature_shape": list(X_minilm.shape),
    "patent_order": order_status,
    "clustering": {
        "algorithm": "AgglomerativeClustering",
        "metric": "cosine",
        "linkage": "average",
        "n_clusters": N_CLUSTERS,
        "active_clusters": active_clusters,
        "cluster_size_min": int(cluster_sizes.min()),
        "cluster_size_max": int(cluster_sizes.max()),
        "max_cluster_share": max_cluster_share,
        "deterministic": True,
        "elapsed_seconds": elapsed,
    },
    "metrics": metrics,
    "mean_nmi": mean_nmi,
}

RESULT_PATH.write_text(
    json.dumps(result, indent=2),
    encoding="utf-8",
)

sec = metrics["section"]
cls = metrics["class"]
sub = metrics["subclass"]

latex_row = (
    "MiniLM + agglomerative clustering\n"
    f"& {sec['predicted_cluster_purity']:.4f} "
    f"& {sec['inverse_label_purity']:.4f} "
    f"& {sec['nmi']:.4f}\n"
    f"& {cls['predicted_cluster_purity']:.4f} "
    f"& {cls['inverse_label_purity']:.4f} "
    f"& {cls['nmi']:.4f}\n"
    f"& {sub['predicted_cluster_purity']:.4f} "
    f"& {sub['inverse_label_purity']:.4f} "
    f"& {sub['nmi']:.4f} \\\\\n"
)

LATEX_PATH.write_text(latex_row, encoding="utf-8")

# ------------------------------------------------------------
# Final output
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("FINAL MINILM + AGGLOMERATIVE RESULTS")
print("=" * 80)

for level in ("section", "class", "subclass"):
    values = metrics[level]
    print(
        f"{level:8s} | "
        f"Pur_p={values['predicted_cluster_purity']:.6f} | "
        f"Pur_a={values['inverse_label_purity']:.6f} | "
        f"NMI={values['nmi']:.6f}"
    )

print(f"\nMean NMI: {mean_nmi:.6f}")
print("Active clusters:", active_clusters)
print(
    "Cluster size range:",
    int(cluster_sizes.min()),
    int(cluster_sizes.max()),
)
print("Maximum cluster share:", f"{max_cluster_share:.4f}")
print("Elapsed minutes:", f"{elapsed / 60:.2f}")

print("\nLaTeX row:")
print(latex_row)

print("Predictions:", PREDICTION_PATH)
print("Metrics    :", METRICS_PATH)
print("Results    :", RESULT_PATH)
print("LaTeX      :", LATEX_PATH)
print("\nSUCCESS: MiniLM agglomerative baseline completed.")


# ============================================================
# TF-IDF-50K + LSA-256 + Spherical K-means
# ============================================================

import gc
import json
import pickle
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import normalize

# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------

EXPECTED_TRAIN = 49_599
EXPECTED_TEST = 9_881

LSA_DIM = 256
LSA_SEED = 42

N_CLUSTERS = 30
CLUSTER_SEEDS = [17, 42, 73]
MAX_ITER = 100
TOL = 1e-5

DATA_ROOT = Path("/content/drive/MyDrive/depth_ot_patent/data/processed")
TFIDF_ROOT = Path(
    "/content/drive/MyDrive/claimsem_artifacts/"
    "additional_clustering_baselines/tfidf_spherical_kmeans"
)
OUTPUT_DIR = Path(
    "/content/drive/MyDrive/claimsem_artifacts/"
    "additional_clustering_baselines/tfidf_lsa256_spherical_kmeans"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_ROOT / "train_records.pkl"
TEST_PATH = DATA_ROOT / "test_records.pkl"
VECTORIZER_PATH = TFIDF_ROOT / "tfidf_50k_vectorizer.joblib"

LSA_MODEL_PATH = OUTPUT_DIR / "tfidf_lsa256_model.joblib"
TEST_FEATURE_PATH = OUTPUT_DIR / "test_tfidf_lsa256_features.npz"
PREDICTIONS_PATH = OUTPUT_DIR / "tfidf_lsa256_predictions.npz"
METRICS_PATH = OUTPUT_DIR / "tfidf_lsa256_metrics.csv"
RESULT_PATH = OUTPUT_DIR / "tfidf_lsa256_results.json"
LATEX_PATH = OUTPUT_DIR / "tfidf_lsa256_row.tex"

assert TRAIN_PATH.exists(), TRAIN_PATH
assert TEST_PATH.exists(), TEST_PATH
assert VECTORIZER_PATH.exists(), VECTORIZER_PATH


# ------------------------------------------------------------
# 2. Load records and reconstruct ordered documents
# ------------------------------------------------------------

with TRAIN_PATH.open("rb") as f:
    train_records_lsa = pickle.load(f)

with TEST_PATH.open("rb") as f:
    test_records_lsa = pickle.load(f)

assert len(train_records_lsa) == EXPECTED_TRAIN
assert len(test_records_lsa) == EXPECTED_TEST


def build_document(record):
    claims = record["claims"]
    claim_ids = sorted(claims, key=lambda value: int(value))
    texts = [
        str(claims[claim_id]).strip()
        for claim_id in claim_ids
        if str(claims[claim_id]).strip()
    ]
    assert texts
    return "\n".join(texts)


print("Building ordered documents...")

train_documents_lsa = [build_document(record) for record in train_records_lsa]
test_documents_lsa = [build_document(record) for record in test_records_lsa]

test_patent_ids_lsa = np.asarray(
    [str(record["patent_id"]) for record in test_records_lsa]
)

test_cpc_labels_lsa = {
    "section": np.asarray([str(record["section"]) for record in test_records_lsa]),
    "class": np.asarray([str(record["class"]) for record in test_records_lsa]),
    "subclass": np.asarray([str(record["subclass"]) for record in test_records_lsa]),
}

assert len(np.unique(test_cpc_labels_lsa["section"])) == 9
assert len(np.unique(test_cpc_labels_lsa["class"])) == 121
assert len(np.unique(test_cpc_labels_lsa["subclass"])) == 466

print("TRAIN documents:", len(train_documents_lsa))
print("TEST documents :", len(test_documents_lsa))


# ------------------------------------------------------------
# 3. Reload frozen TRAIN-fitted TF-IDF vectorizer
# ------------------------------------------------------------

vectorizer_lsa = joblib.load(VECTORIZER_PATH)

assert len(vectorizer_lsa.vocabulary_) == 50_000

print("\nTransforming TRAIN with frozen TF-IDF vectorizer...")
start = time.time()
X_train_tfidf_lsa = (
    vectorizer_lsa.transform(train_documents_lsa).tocsr().astype(np.float32)
)

print("Transforming TEST...")
X_test_tfidf_lsa = (
    vectorizer_lsa.transform(test_documents_lsa).tocsr().astype(np.float32)
)

assert X_train_tfidf_lsa.shape == (EXPECTED_TRAIN, 50_000)
assert X_test_tfidf_lsa.shape == (EXPECTED_TEST, 50_000)
assert np.isfinite(X_train_tfidf_lsa.data).all()
assert np.isfinite(X_test_tfidf_lsa.data).all()

print("TRAIN TF-IDF:", X_train_tfidf_lsa.shape)
print("TEST TF-IDF :", X_test_tfidf_lsa.shape)
print("Transform minutes:", f"{(time.time() - start) / 60:.2f}")


# ------------------------------------------------------------
# 4. Fit LSA only on TRAIN and apply unchanged to TEST
# ------------------------------------------------------------

print("\nFitting TRAIN-only TruncatedSVD...")
lsa_model = TruncatedSVD(
    n_components=LSA_DIM,
    algorithm="randomized",
    n_iter=7,
    random_state=LSA_SEED,
)

start = time.time()
lsa_model.fit(X_train_tfidf_lsa)

print("Transforming TEST to LSA-256...")
X_test_lsa = lsa_model.transform(X_test_tfidf_lsa).astype(np.float32)

X_test_lsa = normalize(
    X_test_lsa,
    norm="l2",
    axis=1,
    copy=False,
).astype(np.float32)

assert X_test_lsa.shape == (EXPECTED_TEST, LSA_DIM)
assert np.isfinite(X_test_lsa).all()

lsa_norms = np.linalg.norm(X_test_lsa, axis=1)

assert np.allclose(lsa_norms, 1.0, atol=1e-5)

explained_variance = float(lsa_model.explained_variance_ratio_.sum())

joblib.dump(
    lsa_model,
    LSA_MODEL_PATH,
    compress=3,
)

np.savez_compressed(
    TEST_FEATURE_PATH,
    features=X_test_lsa,
    patent_ids=test_patent_ids_lsa,
)

print("LSA shape:", X_test_lsa.shape)
print("Explained variance:", f"{explained_variance:.6f}")
print("LSA fit/transform minutes:", f"{(time.time() - start) / 60:.2f}")

# Free large sparse TRAIN matrix before clustering
del X_train_tfidf_lsa
del X_test_tfidf_lsa
gc.collect()


# ------------------------------------------------------------
# 5. Dense spherical K-means
# ------------------------------------------------------------


def spherical_kmeans_plus_plus(X, n_clusters, seed):
    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape

    centers = np.empty(
        (n_clusters, n_features),
        dtype=np.float32,
    )

    first_index = int(rng.integers(n_samples))
    centers[0] = X[first_index]

    closest_similarity = X @ centers[0]

    for cluster_id in range(1, n_clusters):
        cosine_distance = np.clip(
            1.0 - closest_similarity,
            0.0,
            None,
        )

        total_distance = float(cosine_distance.sum())

        if total_distance <= 0:
            selected_index = int(rng.integers(n_samples))
        else:
            probabilities = cosine_distance / total_distance
            selected_index = int(rng.choice(n_samples, p=probabilities))

        centers[cluster_id] = X[selected_index]

        new_similarity = X @ centers[cluster_id]
        closest_similarity = np.maximum(
            closest_similarity,
            new_similarity,
        )

    return centers


def dense_spherical_kmeans(
    X,
    n_clusters,
    seed,
    max_iter=100,
    tolerance=1e-5,
):
    centers = spherical_kmeans_plus_plus(
        X,
        n_clusters,
        seed,
    )

    previous_labels = None
    previous_objective = None
    converged = False

    for iteration in range(1, max_iter + 1):
        similarities = X @ centers.T
        predicted = similarities.argmax(axis=1).astype(np.int32)

        best_similarities = similarities[
            np.arange(X.shape[0]),
            predicted,
        ]
        objective = float(best_similarities.mean())

        labels_unchanged = previous_labels is not None and np.array_equal(
            predicted, previous_labels
        )

        objective_stable = (
            previous_objective is not None
            and abs(objective - previous_objective) <= tolerance
        )

        if labels_unchanged or objective_stable:
            converged = True
            break

        new_centers = np.zeros_like(centers)

        for cluster_id in range(n_clusters):
            member_mask = predicted == cluster_id

            if member_mask.any():
                new_centers[cluster_id] = X[member_mask].sum(axis=0)
            else:
                replacement_index = int(np.argmin(best_similarities))
                new_centers[cluster_id] = X[replacement_index]

        center_norms = np.linalg.norm(
            new_centers,
            axis=1,
            keepdims=True,
        )

        centers = new_centers / np.maximum(
            center_norms,
            1e-12,
        )

        previous_labels = predicted.copy()
        previous_objective = objective

    final_similarities = X @ centers.T
    final_labels = final_similarities.argmax(axis=1).astype(np.int32)
    final_best = final_similarities[
        np.arange(X.shape[0]),
        final_labels,
    ]

    return {
        "labels": final_labels,
        "centers": centers,
        "objective": float(final_best.mean()),
        "iterations": iteration,
        "converged": converged,
        "active_clusters": int(np.unique(final_labels).size),
    }


# ------------------------------------------------------------
# 6. Evaluation functions
# ------------------------------------------------------------


def evaluate_partition(y_true, y_pred):
    matrix = contingency_matrix(
        y_true,
        y_pred,
        sparse=False,
    )
    n_samples = matrix.sum()

    return {
        "predicted_cluster_purity": float(matrix.max(axis=0).sum() / n_samples),
        "inverse_label_purity": float(matrix.max(axis=1).sum() / n_samples),
        "nmi": float(
            normalized_mutual_info_score(
                y_true,
                y_pred,
                average_method="arithmetic",
            )
        ),
    }


# ------------------------------------------------------------
# 7. Run three clustering seeds
# ------------------------------------------------------------

runs = []
prediction_archive = {}

for seed in CLUSTER_SEEDS:
    print(f"\nRunning spherical K-means seed {seed}...")
    start = time.time()

    clustering = dense_spherical_kmeans(
        X_test_lsa,
        n_clusters=N_CLUSTERS,
        seed=seed,
        max_iter=MAX_ITER,
        tolerance=TOL,
    )

    assert clustering["active_clusters"] == N_CLUSTERS

    run_metrics = {
        level: evaluate_partition(
            test_cpc_labels_lsa[level],
            clustering["labels"],
        )
        for level in ("section", "class", "subclass")
    }

    mean_nmi = float(
        np.mean(
            [run_metrics[level]["nmi"] for level in ("section", "class", "subclass")]
        )
    )

    runs.append(
        {
            "seed": seed,
            "converged": clustering["converged"],
            "iterations": clustering["iterations"],
            "active_clusters": clustering["active_clusters"],
            "objective": clustering["objective"],
            "mean_nmi": mean_nmi,
            "metrics": run_metrics,
        }
    )

    prediction_archive[f"seed_{seed}"] = clustering["labels"]

    print(
        f"iterations={clustering['iterations']}, "
        f"converged={clustering['converged']}, "
        f"active={clustering['active_clusters']}, "
        f"objective={clustering['objective']:.6f}, "
        f"mean NMI={mean_nmi:.6f}, "
        f"minutes={(time.time() - start) / 60:.2f}"
    )


# ------------------------------------------------------------
# 8. Aggregate and save results
# ------------------------------------------------------------

metric_rows = []

for run in runs:
    for level in ("section", "class", "subclass"):
        metric_rows.append(
            {
                "seed": run["seed"],
                "level": level,
                **run["metrics"][level],
            }
        )

metrics_df = pd.DataFrame(metric_rows)
metrics_df.to_csv(METRICS_PATH, index=False)

np.savez_compressed(
    PREDICTIONS_PATH,
    patent_ids=test_patent_ids_lsa,
    **prediction_archive,
)

level_summary = {}

for level in ("section", "class", "subclass"):
    level_rows = metrics_df[metrics_df["level"] == level]

    level_summary[level] = {}

    for metric in (
        "predicted_cluster_purity",
        "inverse_label_purity",
        "nmi",
    ):
        values = level_rows[metric].to_numpy()

        level_summary[level][metric] = {
            "mean": float(values.mean()),
            "population_std": float(values.std(ddof=0)),
        }

overall_seed_nmi = metrics_df.groupby("seed", sort=False)["nmi"].mean()

overall_mean_nmi = float(overall_seed_nmi.mean())
overall_std_nmi = float(overall_seed_nmi.std(ddof=0))

result = {
    "baseline": "TF-IDF-50K + LSA-256 + spherical K-means",
    "tfidf_vectorizer": str(VECTORIZER_PATH),
    "lsa_model": {
        "fit_split": "train",
        "dimension": LSA_DIM,
        "random_state": LSA_SEED,
        "explained_variance_ratio_sum": explained_variance,
    },
    "clustering": {
        "algorithm": "spherical K-means",
        "n_clusters": N_CLUSTERS,
        "seeds": CLUSTER_SEEDS,
        "max_iter": MAX_ITER,
        "tolerance": TOL,
    },
    "runs": runs,
    "summary": level_summary,
    "overall_mean_nmi": overall_mean_nmi,
    "overall_population_std": overall_std_nmi,
}

RESULT_PATH.write_text(
    json.dumps(result, indent=2),
    encoding="utf-8",
)

sec = level_summary["section"]
cls = level_summary["class"]
sub = level_summary["subclass"]

latex_row = (
    "TF--IDF + LSA + spherical $K$-means\n"
    f"& {sec['predicted_cluster_purity']['mean']:.4f} "
    f"& {sec['inverse_label_purity']['mean']:.4f} "
    f"& {sec['nmi']['mean']:.4f}\n"
    f"& {cls['predicted_cluster_purity']['mean']:.4f} "
    f"& {cls['inverse_label_purity']['mean']:.4f} "
    f"& {cls['nmi']['mean']:.4f}\n"
    f"& {sub['predicted_cluster_purity']['mean']:.4f} "
    f"& {sub['inverse_label_purity']['mean']:.4f} "
    f"& {sub['nmi']['mean']:.4f} \\\\\n"
)

LATEX_PATH.write_text(
    latex_row,
    encoding="utf-8",
)


# ------------------------------------------------------------
# 9. Final report
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("FINAL TF-IDF + LSA-256 + SPHERICAL K-MEANS RESULTS")
print("=" * 80)

for level in ("section", "class", "subclass"):
    values = level_summary[level]

    print(
        f"{level:8s} | "
        f"Pur_p={values['predicted_cluster_purity']['mean']:.6f} "
        f"± {values['predicted_cluster_purity']['population_std']:.6f} | "
        f"Pur_a={values['inverse_label_purity']['mean']:.6f} "
        f"± {values['inverse_label_purity']['population_std']:.6f} | "
        f"NMI={values['nmi']['mean']:.6f} "
        f"± {values['nmi']['population_std']:.6f}"
    )

print("\nMean NMI by seed:")
print(overall_seed_nmi.to_string())

print("\nOverall mean NMI:", f"{overall_mean_nmi:.6f}")
print("Overall population SD:", f"{overall_std_nmi:.6f}")
print("Explained variance:", f"{explained_variance:.6f}")

print("\nLaTeX row:")
print(latex_row)

print("LSA model  :", LSA_MODEL_PATH)
print("Features   :", TEST_FEATURE_PATH)
print("Predictions:", PREDICTIONS_PATH)
print("Metrics    :", METRICS_PATH)
print("Results    :", RESULT_PATH)
print("LaTeX      :", LATEX_PATH)
print("\nSUCCESS: TF-IDF + LSA-256 baseline completed.")
