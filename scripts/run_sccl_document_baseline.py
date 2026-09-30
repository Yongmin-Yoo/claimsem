# ruff: noqa
# Auto-exported from the executed Google Colab experiment.
# Raw patent data and credentials are not included.

# ============================================================
# SCCL COMPLETE PIPELINE
# Initial embeddings -> center initialization -> 3 training seeds
# -> native SCCL evaluation -> matched spherical K-means
# -> CSV / JSON / LaTeX / predictions
# ============================================================

import gc
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from transformers import AutoModel
from tqdm.auto import tqdm

# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

N_DOCUMENTS = 9_881
MAX_LENGTH = 512
N_CLUSTERS = 30

TRAINING_SEEDS = [17, 42, 73]
CLUSTERING_SEEDS = [17, 42, 73]

TRAIN_STEPS = 1_000
BATCH_SIZE = 64
ENCODE_BATCH_SIZE = 192

ENCODER_LR = 1e-5
HEAD_LR = 1e-3
TEMPERATURE = 0.5
ETA = 10.0
CLUSTER_LOSS_WEIGHT = 0.5
ALPHA = 1.0

CHECKPOINT_EVERY = 250
LOG_EVERY = 50

DEVICE = torch.device("cuda")
BF16_ENABLED = torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if BF16_ENABLED else torch.float16

assert torch.cuda.is_available()
assert "L4" in torch.cuda.get_device_name(0)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

INITIAL_EMBEDDINGS_PATH = CACHE_DIR / "initial_minilm_document_embeddings.npy"
FINAL_EMBEDDING_DIR = RESULT_DIR / "final_embeddings"
PREDICTION_DIR = RESULT_DIR / "predictions"

FINAL_EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

NATIVE_METRICS_PATH = RESULT_DIR / "sccl_native_metrics.csv"
SPHERICAL_METRICS_PATH = RESULT_DIR / "sccl_spherical_metrics.csv"
RESULT_JSON_PATH = RESULT_DIR / "sccl_results.json"
LATEX_PATH = RESULT_DIR / "sccl_rows.tex"

print("GPU:", torch.cuda.get_device_name(0))
print("BF16:", BF16_ENABLED)
print("Training seeds:", TRAINING_SEEDS)
print("Training steps per seed:", TRAIN_STEPS)
print("Batch size:", BATCH_SIZE)


# ------------------------------------------------------------
# 2. Reload token cache and label metadata
# ------------------------------------------------------------

input_ids_np = np.load(INPUT_IDS_PATH)
attention_mask_np = np.load(ATTENTION_MASK_PATH)

assert input_ids_np.shape == (N_DOCUMENTS, MAX_LENGTH)
assert attention_mask_np.shape == (N_DOCUMENTS, MAX_LENGTH)

input_ids_cpu = torch.from_numpy(input_ids_np.astype(np.int64, copy=False))
attention_mask_cpu = torch.from_numpy(attention_mask_np.astype(np.int64, copy=False))

metadata = np.load(METADATA_PATH)

patent_ids = metadata["patent_ids"].astype(str)

evaluation_labels = {
    "section": metadata["section"].astype(str),
    "class": metadata["class_labels"].astype(str),
    "subclass": metadata["subclass"].astype(str),
}

assert len(np.unique(evaluation_labels["section"])) == 9
assert len(np.unique(evaluation_labels["class"])) == 121
assert len(np.unique(evaluation_labels["subclass"])) == 466

print("Token cache and metadata loaded.")


# ------------------------------------------------------------
# 3. Utility functions
# ------------------------------------------------------------


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)

    summed = torch.sum(last_hidden_state * mask, dim=1)
    denominator = torch.clamp(mask.sum(dim=1), min=1e-9)

    return summed / denominator


def load_encoder():
    return AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
        attn_implementation="sdpa",
        torch_dtype=AMP_DTYPE,
    )


@torch.inference_mode()
def encode_all_documents(encoder, batch_size=ENCODE_BATCH_SIZE):
    encoder.eval()

    hidden_size = encoder.config.hidden_size
    embeddings = np.empty(
        (N_DOCUMENTS, hidden_size),
        dtype=np.float32,
    )

    for start in tqdm(
        range(0, N_DOCUMENTS, batch_size),
        desc="Encoding",
        leave=False,
    ):
        stop = min(start + batch_size, N_DOCUMENTS)

        ids = input_ids_cpu[start:stop].to(
            DEVICE,
            non_blocking=True,
        )
        mask = attention_mask_cpu[start:stop].to(
            DEVICE,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=AMP_DTYPE,
            enabled=True,
        ):
            output = encoder(
                input_ids=ids,
                attention_mask=mask,
            )
            pooled = mean_pool(
                output.last_hidden_state,
                mask,
            )

        embeddings[start:stop] = pooled.float().cpu().numpy()

    return embeddings


def evaluate_partition(y_true, y_pred):
    matrix = contingency_matrix(
        y_true,
        y_pred,
        sparse=False,
    )
    n = matrix.sum()

    return {
        "predicted_cluster_purity": float(matrix.max(axis=0).sum() / n),
        "inverse_label_purity": float(matrix.max(axis=1).sum() / n),
        "nmi": float(
            normalized_mutual_info_score(
                y_true,
                y_pred,
                average_method="arithmetic",
            )
        ),
    }


def evaluate_predictions(predictions):
    return {
        level: evaluate_partition(
            evaluation_labels[level],
            predictions,
        )
        for level in ("section", "class", "subclass")
    }


def mean_nmi(metrics):
    return float(
        np.mean(
            [
                metrics["section"]["nmi"],
                metrics["class"]["nmi"],
                metrics["subclass"]["nmi"],
            ]
        )
    )


def atomic_torch_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


# ------------------------------------------------------------
# 4. Official SCCL-style model
# ------------------------------------------------------------


class SCCLModel(nn.Module):
    def __init__(self, encoder, initial_centers, alpha=1.0):
        super().__init__()

        self.encoder = encoder
        self.alpha = alpha
        hidden_size = encoder.config.hidden_size

        self.contrast_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 128),
        )

        self.cluster_centers = nn.Parameter(
            torch.as_tensor(
                initial_centers,
                dtype=torch.float32,
            )
        )

    def encode(self, input_ids, attention_mask):
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return mean_pool(
            output.last_hidden_state,
            attention_mask,
        )

    def project(self, embeddings):
        return F.normalize(
            self.contrast_head(embeddings),
            dim=1,
        )

    def cluster_probabilities(self, embeddings):
        embeddings = embeddings.float()
        centers = self.cluster_centers.float()

        embedding_norm = torch.sum(
            embeddings * embeddings,
            dim=1,
            keepdim=True,
        )

        center_norm = torch.sum(
            centers * centers,
            dim=1,
        ).unsqueeze(0)

        squared_distance = torch.clamp(
            embedding_norm + center_norm - 2.0 * embeddings @ centers.T,
            min=0.0,
        )

        numerator = 1.0 / (1.0 + squared_distance / self.alpha)

        numerator = numerator ** ((self.alpha + 1.0) / 2.0)

        return numerator / torch.clamp(
            numerator.sum(dim=1, keepdim=True),
            min=1e-9,
        )


def pair_contrastive_loss(features_1, features_2, temperature):
    batch_size = features_1.shape[0]

    features = torch.cat(
        [features_1, features_2],
        dim=0,
    ).float()

    logits = features @ features.T
    logits = logits / temperature

    diagonal = torch.eye(
        2 * batch_size,
        dtype=torch.bool,
        device=features.device,
    )

    logits = logits.masked_fill(
        diagonal,
        torch.finfo(logits.dtype).min,
    )

    positive_indices = (
        torch.arange(
            2 * batch_size,
            device=features.device,
        )
        + batch_size
    ) % (2 * batch_size)

    return F.cross_entropy(
        logits,
        positive_indices,
    )


def target_distribution(probabilities):
    weight = probabilities**2 / torch.clamp(
        probabilities.sum(dim=0, keepdim=True),
        min=1e-9,
    )

    return weight / torch.clamp(
        weight.sum(dim=1, keepdim=True),
        min=1e-9,
    )


# ------------------------------------------------------------
# 5. Initial pretrained embeddings
# ------------------------------------------------------------

if INITIAL_EMBEDDINGS_PATH.exists():
    initial_embeddings = np.load(INITIAL_EMBEDDINGS_PATH)

    if initial_embeddings.shape != (N_DOCUMENTS, 384):
        INITIAL_EMBEDDINGS_PATH.unlink()
        initial_embeddings = None
    else:
        print("Reusing initial MiniLM embeddings.")

else:
    initial_embeddings = None

if initial_embeddings is None:
    print("\nGenerating initial pretrained embeddings...")

    set_seed(42)
    initial_encoder = load_encoder().to(DEVICE)

    initial_embeddings = encode_all_documents(initial_encoder)

    assert initial_embeddings.shape == (
        N_DOCUMENTS,
        384,
    )
    assert np.isfinite(initial_embeddings).all()

    np.save(
        INITIAL_EMBEDDINGS_PATH,
        initial_embeddings.astype(np.float32),
    )

    del initial_encoder
    torch.cuda.empty_cache()
    gc.collect()

print("Initial embeddings:", initial_embeddings.shape)


# ------------------------------------------------------------
# 6. Spherical K-means for matched downstream evaluation
# ------------------------------------------------------------


def spherical_kmeans_plus_plus(
    features,
    n_clusters,
    seed,
):
    rng = np.random.default_rng(seed)
    n_samples, n_features = features.shape

    centers = np.empty(
        (n_clusters, n_features),
        dtype=np.float32,
    )

    first = int(rng.integers(n_samples))
    centers[0] = features[first]

    closest_similarity = features @ centers[0]

    for cluster_id in range(1, n_clusters):
        distance = np.clip(
            1.0 - closest_similarity,
            0.0,
            None,
        )

        total = float(distance.sum())

        if total <= 0:
            index = int(rng.integers(n_samples))
        else:
            index = int(
                rng.choice(
                    n_samples,
                    p=distance / total,
                )
            )

        centers[cluster_id] = features[index]

        similarity = features @ centers[cluster_id]
        closest_similarity = np.maximum(
            closest_similarity,
            similarity,
        )

    return centers


def spherical_kmeans(
    features,
    n_clusters,
    seed,
    max_iter=100,
    tolerance=1e-5,
):
    norms = np.linalg.norm(
        features,
        axis=1,
        keepdims=True,
    )

    normalized = (features / np.maximum(norms, 1e-12)).astype(np.float32)

    centers = spherical_kmeans_plus_plus(
        normalized,
        n_clusters,
        seed,
    )

    previous_labels = None
    previous_objective = None
    converged = False

    for iteration in range(1, max_iter + 1):
        similarities = normalized @ centers.T
        labels = similarities.argmax(axis=1).astype(np.int32)

        best = similarities[
            np.arange(normalized.shape[0]),
            labels,
        ]

        objective = float(best.mean())

        if previous_labels is not None and np.array_equal(labels, previous_labels):
            converged = True
            break

        if (
            previous_objective is not None
            and abs(objective - previous_objective) <= tolerance
        ):
            converged = True
            break

        new_centers = np.zeros_like(centers)

        for cluster_id in range(n_clusters):
            members = labels == cluster_id

            if members.any():
                new_centers[cluster_id] = normalized[members].sum(axis=0)
            else:
                replacement = int(np.argmin(best))
                new_centers[cluster_id] = normalized[replacement]

        center_norms = np.linalg.norm(
            new_centers,
            axis=1,
            keepdims=True,
        )

        centers = new_centers / np.maximum(
            center_norms,
            1e-12,
        )

        previous_labels = labels.copy()
        previous_objective = objective

    similarities = normalized @ centers.T
    labels = similarities.argmax(axis=1).astype(np.int32)

    return {
        "labels": labels,
        "iterations": iteration,
        "converged": converged,
        "active_clusters": int(np.unique(labels).size),
        "objective": float(
            similarities[
                np.arange(normalized.shape[0]),
                labels,
            ].mean()
        ),
    }


# ------------------------------------------------------------
# 7. Train SCCL for each training seed
# ------------------------------------------------------------

native_runs = []
spherical_runs = []

experiment_start = time.time()

for training_seed in TRAINING_SEEDS:
    print("\n" + "=" * 80)
    print(f"SCCL TRAINING SEED {training_seed}")
    print("=" * 80)

    set_seed(training_seed)
    torch.cuda.empty_cache()
    gc.collect()

    seed_dir = CHECKPOINT_DIR / f"seed_{training_seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = seed_dir / "latest.pt"
    final_embedding_path = (
        FINAL_EMBEDDING_DIR / f"sccl_seed_{training_seed}_embeddings.npz"
    )
    prediction_path = PREDICTION_DIR / f"sccl_seed_{training_seed}_predictions.npz"

    # Deterministic K-means initialization
    initial_kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        n_init=20,
        max_iter=300,
        random_state=training_seed,
        algorithm="lloyd",
    )

    initial_kmeans.fit(initial_embeddings)
    initial_centers = initial_kmeans.cluster_centers_.astype(np.float32)

    encoder = load_encoder()
    model = SCCLModel(
        encoder=encoder,
        initial_centers=initial_centers,
        alpha=ALPHA,
    ).to(DEVICE)

    optimizer_kwargs = {}

    if "fused" in torch.optim.Adam.__init__.__code__.co_varnames:
        optimizer_kwargs["fused"] = True

    optimizer = torch.optim.Adam(
        [
            {
                "params": model.encoder.parameters(),
                "lr": ENCODER_LR,
            },
            {
                "params": model.contrast_head.parameters(),
                "lr": HEAD_LR,
            },
            {
                "params": [model.cluster_centers],
                "lr": HEAD_LR,
            },
        ],
        **optimizer_kwargs,
    )

    rng = np.random.default_rng(training_seed)
    order = rng.permutation(N_DOCUMENTS)
    position = 0
    start_step = 0

    if checkpoint_path.exists():
        print("Loading checkpoint:", checkpoint_path)

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        start_step = int(checkpoint["step"])
        order = checkpoint["order"]
        position = int(checkpoint["position"])
        rng.bit_generator.state = checkpoint["numpy_rng_state"]

        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

        print(f"Resuming from step {start_step}")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    rolling_total = 0.0
    rolling_contrast = 0.0
    rolling_cluster = 0.0
    rolling_count = 0
    seed_start = time.time()

    for step in range(start_step + 1, TRAIN_STEPS + 1):
        if position + BATCH_SIZE > N_DOCUMENTS:
            order = rng.permutation(N_DOCUMENTS)
            position = 0

        batch_indices = order[position : position + BATCH_SIZE]
        position += BATCH_SIZE

        batch_ids = input_ids_cpu[batch_indices].to(
            DEVICE,
            non_blocking=True,
        )

        batch_mask = attention_mask_cpu[batch_indices].to(
            DEVICE,
            non_blocking=True,
        )

        # Two virtual dropout views in one encoder pass
        doubled_ids = torch.cat(
            [batch_ids, batch_ids],
            dim=0,
        )
        doubled_mask = torch.cat(
            [batch_mask, batch_mask],
            dim=0,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=AMP_DTYPE,
            enabled=True,
        ):
            doubled_embeddings = model.encode(
                doubled_ids,
                doubled_mask,
            )

            embeddings_1, embeddings_2 = doubled_embeddings.chunk(2, dim=0)

            features_1 = model.project(embeddings_1)
            features_2 = model.project(embeddings_2)

            contrast_loss = pair_contrastive_loss(
                features_1,
                features_2,
                TEMPERATURE,
            )

            probabilities = model.cluster_probabilities(embeddings_1)

            target = target_distribution(probabilities).detach()

            cluster_loss = F.kl_div(
                torch.log(
                    torch.clamp(
                        probabilities,
                        min=1e-8,
                    )
                ),
                target,
                reduction="batchmean",
            )

            total_loss = ETA * contrast_loss + CLUSTER_LOSS_WEIGHT * cluster_loss

        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite loss at seed={training_seed}, step={step}")

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        rolling_total += float(total_loss.detach())
        rolling_contrast += float(contrast_loss.detach())
        rolling_cluster += float(cluster_loss.detach())
        rolling_count += 1

        if step % LOG_EVERY == 0 or step == 1:
            elapsed = time.time() - seed_start
            completed = step - start_step
            rate = completed / max(elapsed, 1e-9)
            remaining = (TRAIN_STEPS - step) / max(rate, 1e-9)

            print(
                f"seed={training_seed} "
                f"step={step:4d}/{TRAIN_STEPS} | "
                f"loss={rolling_total / rolling_count:.4f} | "
                f"contrast={rolling_contrast / rolling_count:.4f} | "
                f"cluster={rolling_cluster / rolling_count:.4f} | "
                f"ETA={remaining / 60:.1f} min | "
                f"GPU={torch.cuda.max_memory_allocated() / 2**30:.1f} GB"
            )

            rolling_total = 0.0
            rolling_contrast = 0.0
            rolling_cluster = 0.0
            rolling_count = 0

        if step % CHECKPOINT_EVERY == 0 or step == TRAIN_STEPS:
            atomic_torch_save(
                {
                    "seed": training_seed,
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "order": order,
                    "position": position,
                    "numpy_rng_state": rng.bit_generator.state,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all(),
                    "configuration": {
                        "model_id": MODEL_ID,
                        "model_revision": MODEL_REVISION,
                        "max_length": MAX_LENGTH,
                        "n_clusters": N_CLUSTERS,
                        "training_steps": TRAIN_STEPS,
                        "batch_size": BATCH_SIZE,
                        "encoder_lr": ENCODER_LR,
                        "head_lr": HEAD_LR,
                        "temperature": TEMPERATURE,
                        "eta": ETA,
                        "cluster_loss_weight": CLUSTER_LOSS_WEIGHT,
                        "alpha": ALPHA,
                    },
                },
                checkpoint_path,
            )

            print("Checkpoint saved:", checkpoint_path)

    # --------------------------------------------------------
    # Final embeddings
    # --------------------------------------------------------

    print(f"\nEncoding final SCCL representations: seed {training_seed}")

    final_embeddings = encode_all_documents(
        model.encoder,
        batch_size=ENCODE_BATCH_SIZE,
    )

    assert final_embeddings.shape == (
        N_DOCUMENTS,
        384,
    )
    assert np.isfinite(final_embeddings).all()

    np.savez_compressed(
        final_embedding_path,
        embeddings=final_embeddings,
        patent_ids=patent_ids,
    )

    # --------------------------------------------------------
    # Native SCCL cluster-head predictions
    # --------------------------------------------------------

    centers = model.cluster_centers.detach().float().cpu().numpy()

    embedding_norm = np.sum(
        final_embeddings**2,
        axis=1,
        keepdims=True,
    )
    center_norm = np.sum(
        centers**2,
        axis=1,
        keepdims=True,
    ).T

    squared_distance = np.maximum(
        embedding_norm + center_norm - 2.0 * final_embeddings @ centers.T,
        0.0,
    )

    native_predictions = squared_distance.argmin(axis=1).astype(np.int32)

    native_metrics = evaluate_predictions(native_predictions)
    native_mean_nmi = mean_nmi(native_metrics)

    native_active = int(np.unique(native_predictions).size)

    native_runs.append(
        {
            "training_seed": training_seed,
            "active_clusters": native_active,
            "mean_nmi": native_mean_nmi,
            "metrics": native_metrics,
        }
    )

    print(f"Native SCCL: active={native_active}, mean NMI={native_mean_nmi:.6f}")

    # --------------------------------------------------------
    # Matched spherical K-means on SCCL representations
    # --------------------------------------------------------

    spherical_prediction_arrays = {}

    for clustering_seed in CLUSTERING_SEEDS:
        clustering = spherical_kmeans(
            final_embeddings,
            n_clusters=N_CLUSTERS,
            seed=clustering_seed,
            max_iter=100,
            tolerance=1e-5,
        )

        spherical_metrics = evaluate_predictions(clustering["labels"])
        spherical_mean_nmi = mean_nmi(spherical_metrics)

        spherical_runs.append(
            {
                "training_seed": training_seed,
                "clustering_seed": clustering_seed,
                "active_clusters": clustering["active_clusters"],
                "converged": clustering["converged"],
                "iterations": clustering["iterations"],
                "objective": clustering["objective"],
                "mean_nmi": spherical_mean_nmi,
                "metrics": spherical_metrics,
            }
        )

        spherical_prediction_arrays[f"spherical_seed_{clustering_seed}"] = clustering[
            "labels"
        ]

        print(
            f"Spherical seed={clustering_seed}: "
            f"active={clustering['active_clusters']}, "
            f"converged={clustering['converged']}, "
            f"mean NMI={spherical_mean_nmi:.6f}"
        )

    np.savez_compressed(
        prediction_path,
        patent_ids=patent_ids,
        native_labels=native_predictions,
        **spherical_prediction_arrays,
    )

    del model, optimizer, encoder
    del final_embeddings, centers
    torch.cuda.empty_cache()
    gc.collect()


# ------------------------------------------------------------
# 8. Aggregate results
# ------------------------------------------------------------


def runs_to_dataframe(runs, run_type):
    rows = []

    for run in runs:
        for level in ("section", "class", "subclass"):
            row = {
                "run_type": run_type,
                "training_seed": run["training_seed"],
                "level": level,
                "active_clusters": run["active_clusters"],
                **run["metrics"][level],
            }

            if "clustering_seed" in run:
                row["clustering_seed"] = run["clustering_seed"]

            rows.append(row)

    return pd.DataFrame(rows)


native_df = runs_to_dataframe(
    native_runs,
    "native_sccl_head",
)

spherical_df = runs_to_dataframe(
    spherical_runs,
    "matched_spherical_kmeans",
)

native_df.to_csv(
    NATIVE_METRICS_PATH,
    index=False,
    lineterminator="\n",
)

spherical_df.to_csv(
    SPHERICAL_METRICS_PATH,
    index=False,
    lineterminator="\n",
)


def summarize_dataframe(frame):
    summary = {}

    for level in ("section", "class", "subclass"):
        subset = frame[frame["level"] == level]

        summary[level] = {}

        for metric in (
            "predicted_cluster_purity",
            "inverse_label_purity",
            "nmi",
        ):
            values = subset[metric].to_numpy()

            summary[level][metric] = {
                "mean": float(values.mean()),
                "population_std": float(values.std(ddof=0)),
            }

    return summary


native_summary = summarize_dataframe(native_df)
spherical_summary = summarize_dataframe(spherical_df)

native_seed_means = np.asarray([run["mean_nmi"] for run in native_runs])

spherical_run_means = np.asarray([run["mean_nmi"] for run in spherical_runs])

native_overall = float(native_seed_means.mean())
native_overall_std = float(native_seed_means.std(ddof=0))

spherical_overall = float(spherical_run_means.mean())
spherical_overall_std = float(spherical_run_means.std(ddof=0))

result = {
    "method": "SCCL modern faithful adaptation",
    "official_repository": ("https://github.com/amazon-science/sccl"),
    "input": {
        "documents": N_DOCUMENTS,
        "serialization": ("all claims concatenated in claim-ID order"),
        "max_length": MAX_LENGTH,
        "fraction_at_token_limit": float(
            json.loads(TOKEN_STATS_PATH.read_text(encoding="utf-8"))[
                "fraction_at_512_token_limit"
            ]
        ),
    },
    "encoder": {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "hidden_size": 384,
    },
    "training": {
        "transductive": True,
        "labels_used": False,
        "training_seeds": TRAINING_SEEDS,
        "steps_per_seed": TRAIN_STEPS,
        "batch_size": BATCH_SIZE,
        "encoder_lr": ENCODER_LR,
        "head_lr": HEAD_LR,
        "temperature": TEMPERATURE,
        "eta": ETA,
        "cluster_loss_weight": CLUSTER_LOSS_WEIGHT,
        "alpha": ALPHA,
        "mixed_precision": str(AMP_DTYPE),
    },
    "native_sccl": {
        "runs": native_runs,
        "summary": native_summary,
        "overall_mean_nmi": native_overall,
        "overall_population_std": native_overall_std,
    },
    "matched_spherical_kmeans": {
        "runs": spherical_runs,
        "summary": spherical_summary,
        "overall_mean_nmi": spherical_overall,
        "overall_population_std": spherical_overall_std,
    },
    "elapsed_seconds": time.time() - experiment_start,
}

RESULT_JSON_PATH.write_text(
    json.dumps(result, indent=2),
    encoding="utf-8",
)


# ------------------------------------------------------------
# 9. Generate LaTeX rows
# ------------------------------------------------------------


def make_latex_row(name, summary):
    sec = summary["section"]
    cls = summary["class"]
    sub = summary["subclass"]

    return (
        f"{name}\n"
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


native_latex = make_latex_row(
    "SCCL (native clustering head)",
    native_summary,
)

spherical_latex = make_latex_row(
    "SCCL + spherical $K$-means",
    spherical_summary,
)

LATEX_PATH.write_text(
    native_latex + "\n" + spherical_latex,
    encoding="utf-8",
)


# ------------------------------------------------------------
# 10. Final report
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("FINAL SCCL RESULTS")
print("=" * 90)

print("\nA. NATIVE SCCL CLUSTERING HEAD")

for level in ("section", "class", "subclass"):
    values = native_summary[level]

    print(
        f"{level:8s} | "
        f"Pur_p={values['predicted_cluster_purity']['mean']:.6f} "
        f"± {values['predicted_cluster_purity']['population_std']:.6f} | "
        f"Pur_a={values['inverse_label_purity']['mean']:.6f} "
        f"± {values['inverse_label_purity']['population_std']:.6f} | "
        f"NMI={values['nmi']['mean']:.6f} "
        f"± {values['nmi']['population_std']:.6f}"
    )

print(
    "\nNative overall mean NMI:",
    f"{native_overall:.6f} ± {native_overall_std:.6f}",
)
print(
    "Native active clusters:",
    [run["active_clusters"] for run in native_runs],
)

print("\nB. SCCL REPRESENTATIONS + SPHERICAL K-MEANS")

for level in ("section", "class", "subclass"):
    values = spherical_summary[level]

    print(
        f"{level:8s} | "
        f"Pur_p={values['predicted_cluster_purity']['mean']:.6f} "
        f"± {values['predicted_cluster_purity']['population_std']:.6f} | "
        f"Pur_a={values['inverse_label_purity']['mean']:.6f} "
        f"± {values['inverse_label_purity']['population_std']:.6f} | "
        f"NMI={values['nmi']['mean']:.6f} "
        f"± {values['nmi']['population_std']:.6f}"
    )

print(
    "\nSpherical overall mean NMI:",
    f"{spherical_overall:.6f} ± {spherical_overall_std:.6f}",
)

print("\nLaTeX rows:")
print(native_latex)
print(spherical_latex)

print("Native metrics   :", NATIVE_METRICS_PATH)
print("Spherical metrics:", SPHERICAL_METRICS_PATH)
print("Result JSON      :", RESULT_JSON_PATH)
print("LaTeX            :", LATEX_PATH)
print("Checkpoints      :", CHECKPOINT_DIR)
print(
    "Total elapsed:",
    f"{(time.time() - experiment_start) / 60:.1f} minutes",
)
print("\nSUCCESS: SCCL experiment completed.")
