# ruff: noqa
# Auto-exported from the executed Google Colab experiment.
# Raw patent data and credentials are not included.

# ============================================================
# LeBoT – L4-optimized single-cell pipeline
# Runs automatically after the preceding SCCL cell finishes.
#
# Pipeline:
#   MiniLM embeddings → representative BoT initialization
#   → Qwen3-0.6B similarity selection
#   → two iterative BoT refinements
#   → K-means (K=30, seeds 17/42/73)
#   → CPC purity/NMI evaluation
#   → checkpoint/results/LaTeX saving
#
# Official implementation:
# https://github.com/tom192180/BoT_vector
# ============================================================

import os
import sys
import gc
import json
import time
import pickle
import random
import subprocess
from pathlib import Path
from collections import Counter

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ------------------------------------------------------------
# 0. Install/verify packages
# ------------------------------------------------------------
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "transformers==4.57.1",
        "accelerate>=1.10.0",
        "scikit-learn>=1.7.0",
        "pandas>=2.2",
        "tqdm>=4.66",
        "safetensors>=0.4.5",
    ],
    check=True,
)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import normalize

assert torch.cuda.is_available(), "GPU가 없습니다. Colab L4 GPU를 선택하세요."
DEVICE = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("=" * 80)
print("LEBOT L4 PIPELINE")
print("=" * 80)
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)

# Release SCCL model memory while preserving ordinary variables.
for _name in list(globals()):
    try:
        _obj = globals()[_name]
        if isinstance(_obj, torch.nn.Module):
            del globals()[_name]
    except Exception:
        pass

gc.collect()
torch.cuda.empty_cache()

# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------
N_DOCUMENTS = 9_881
N_CLUSTERS = 30
CLUSTER_SEEDS = [17, 42, 73]
GLOBAL_SEED = 42

EMBEDDER_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

LLM_ID = "Qwen/Qwen3-0.6B"
LLM_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"

BOT_DIM = 1_024
CANDIDATE_COUNT = 30
DENSE_POOL_SIZE = 64
MAX_REFINEMENT_ITERS = 2
CONVERGENCE_THRESHOLD = 0.99

EMBED_BATCH_SIZE = 256
LLM_BATCH_SIZE = 24
MAX_PROMPT_TOKENS = 1_536
MAX_NEW_TOKENS = 24
CHECKPOINT_EVERY_BATCHES = 20

TEST_RECORDS_PATH = Path(
    "/content/drive/MyDrive/depth_ot_patent/data/processed/test_records.pkl"
)
SCCL_CACHE_DIR = Path(
    "/content/drive/MyDrive/claimsem_artifacts/sccl_document_clustering/cache"
)
INPUT_IDS_PATH = SCCL_CACHE_DIR / "sccl_input_ids_512.npy"
ATTENTION_MASK_PATH = SCCL_CACHE_DIR / "sccl_attention_mask_512.npy"

OUTPUT_ROOT = Path(
    "/content/drive/MyDrive/claimsem_artifacts/lebot_document_clustering"
)
CACHE_DIR = OUTPUT_ROOT / "cache"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
RESULT_DIR = OUTPUT_ROOT / "results"

for directory in (OUTPUT_ROOT, CACHE_DIR, CHECKPOINT_DIR, RESULT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

EMBEDDING_PATH = CACHE_DIR / "minilm_embeddings.npy"
SNIPPET_PATH = CACHE_DIR / "patent_snippets.json"
REPRESENTATIVE_PATH = CACHE_DIR / "representative_indices.npy"
INITIAL_CANDIDATES_PATH = CACHE_DIR / "initial_representative_candidates.npy"
DENSE_NEIGHBORS_PATH = CACHE_DIR / "dense_neighbors.npy"

INITIAL_BOT_PATH = CHECKPOINT_DIR / "initial_bot.npy"
INITIAL_STATE_PATH = CHECKPOINT_DIR / "initial_state.json"
ACTIVE_PATH = CHECKPOINT_DIR / "active_mask.npy"

RESULT_CSV = RESULT_DIR / "lebot_qwen06b_metrics.csv"
RESULT_JSON = RESULT_DIR / "lebot_qwen06b_results.json"
PREDICTIONS_PATH = RESULT_DIR / "lebot_qwen06b_predictions.npz"
LATEX_PATH = RESULT_DIR / "lebot_qwen06b_row.tex"
PROTOCOL_PATH = OUTPUT_ROOT / "protocol.json"

random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)

# ------------------------------------------------------------
# 2. Load records, labels and SCCL token cache
# ------------------------------------------------------------
assert TEST_RECORDS_PATH.exists(), TEST_RECORDS_PATH
assert INPUT_IDS_PATH.exists(), INPUT_IDS_PATH
assert ATTENTION_MASK_PATH.exists(), ATTENTION_MASK_PATH

with TEST_RECORDS_PATH.open("rb") as f:
    test_records = pickle.load(f)

assert len(test_records) == N_DOCUMENTS

patent_ids = np.asarray(
    [str(record["patent_id"]) for record in test_records],
    dtype=str,
)
labels = {
    "section": np.asarray(
        [str(record["section"]) for record in test_records], dtype=str
    ),
    "class": np.asarray([str(record["class"]) for record in test_records], dtype=str),
    "subclass": np.asarray(
        [str(record["subclass"]) for record in test_records], dtype=str
    ),
}

input_ids = np.load(INPUT_IDS_PATH, mmap_mode="r")
attention_mask = np.load(ATTENTION_MASK_PATH, mmap_mode="r")

assert input_ids.shape == (N_DOCUMENTS, 512)
assert attention_mask.shape == (N_DOCUMENTS, 512)

print("\nRecords:", len(test_records))
print(
    "CPC categories:",
    len(np.unique(labels["section"])),
    len(np.unique(labels["class"])),
    len(np.unique(labels["subclass"])),
)

# ------------------------------------------------------------
# 3. Create short prompt snippets
# ------------------------------------------------------------
embed_tokenizer = AutoTokenizer.from_pretrained(
    EMBEDDER_ID,
    revision=EMBEDDER_REVISION,
    use_fast=True,
)

if SNIPPET_PATH.exists():
    snippets = json.loads(SNIPPET_PATH.read_text(encoding="utf-8"))
    assert len(snippets) == N_DOCUMENTS
    print("Reused snippets:", SNIPPET_PATH)
else:
    snippets = []
    for start in tqdm(
        range(0, N_DOCUMENTS, 512),
        desc="Decoding patent snippets",
    ):
        end = min(start + 512, N_DOCUMENTS)

        # Approximately the beginning of the first independent claim.
        short_ids = []
        for row, mask in zip(
            input_ids[start:end],
            attention_mask[start:end],
        ):
            valid = row[np.asarray(mask).astype(bool)]
            short_ids.append(valid[:96].tolist())

        decoded = embed_tokenizer.batch_decode(
            short_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        snippets.extend([" ".join(text.split()) for text in decoded])

    SNIPPET_PATH.write_text(
        json.dumps(snippets, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Saved snippets:", SNIPPET_PATH)

assert len(snippets) == N_DOCUMENTS
assert all(snippets)

# ------------------------------------------------------------
# 4. Generate or load frozen MiniLM embeddings
# ------------------------------------------------------------
if EMBEDDING_PATH.exists():
    embeddings = np.load(EMBEDDING_PATH)
    assert embeddings.shape == (N_DOCUMENTS, 384)
    embeddings = normalize(embeddings.astype(np.float32), axis=1)
    print("Reused MiniLM embeddings:", EMBEDDING_PATH)
else:
    print("\nLoading MiniLM encoder...")
    encoder = AutoModel.from_pretrained(
        EMBEDDER_ID,
        revision=EMBEDDER_REVISION,
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)
    encoder.eval()

    all_embeddings = []

    with torch.inference_mode():
        for start in tqdm(
            range(0, N_DOCUMENTS, EMBED_BATCH_SIZE),
            desc="MiniLM encoding",
        ):
            end = min(start + EMBED_BATCH_SIZE, N_DOCUMENTS)

            ids = torch.as_tensor(
                np.asarray(input_ids[start:end], dtype=np.int64),
                device=DEVICE,
            )
            mask = torch.as_tensor(
                np.asarray(attention_mask[start:end], dtype=np.int64),
                device=DEVICE,
            )

            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = encoder(
                    input_ids=ids,
                    attention_mask=mask,
                    return_dict=True,
                ).last_hidden_state

                expanded_mask = mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * expanded_mask).sum(dim=1) / expanded_mask.sum(
                    dim=1
                ).clamp_min(1.0)
                pooled = F.normalize(pooled.float(), dim=1)

            all_embeddings.append(pooled.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    np.save(EMBEDDING_PATH, embeddings)

    del encoder, all_embeddings
    gc.collect()
    torch.cuda.empty_cache()

    print("Saved MiniLM embeddings:", EMBEDDING_PATH)

assert embeddings.shape == (N_DOCUMENTS, 384)
assert np.isfinite(embeddings).all()

# ------------------------------------------------------------
# 5. Select 1,024 representative patents
#    MiniBatchKMeans is used instead of O(N²) agglomerative
#    selection to optimize the 9,881-document L4 experiment.
# ------------------------------------------------------------
if REPRESENTATIVE_PATH.exists():
    representative_indices = np.load(REPRESENTATIVE_PATH)
    assert representative_indices.shape == (BOT_DIM,)
    print("Reused representatives:", REPRESENTATIVE_PATH)
else:
    print("\nSelecting representative patents...")
    representative_model = MiniBatchKMeans(
        n_clusters=BOT_DIM,
        random_state=GLOBAL_SEED,
        batch_size=2_048,
        n_init=3,
        max_iter=200,
        reassignment_ratio=0.01,
    )
    representative_labels = representative_model.fit_predict(embeddings)
    centers = normalize(
        representative_model.cluster_centers_.astype(np.float32),
        axis=1,
    )

    representative_indices = np.empty(BOT_DIM, dtype=np.int32)

    for cluster_id in range(BOT_DIM):
        members = np.flatnonzero(representative_labels == cluster_id)

        if len(members) == 0:
            used = set(representative_indices[:cluster_id].tolist())
            available = np.asarray(
                [i for i in range(N_DOCUMENTS) if i not in used],
                dtype=np.int32,
            )
            scores = embeddings[available] @ centers[cluster_id]
            representative_indices[cluster_id] = available[np.argmax(scores)]
        else:
            scores = embeddings[members] @ centers[cluster_id]
            representative_indices[cluster_id] = members[np.argmax(scores)]

    assert len(np.unique(representative_indices)) == BOT_DIM
    np.save(REPRESENTATIVE_PATH, representative_indices)
    print("Saved representatives:", REPRESENTATIVE_PATH)

representative_mask = np.zeros(N_DOCUMENTS, dtype=bool)
representative_mask[representative_indices] = True

# ------------------------------------------------------------
# 6. Initial representative candidates
# ------------------------------------------------------------
if INITIAL_CANDIDATES_PATH.exists():
    initial_candidates = np.load(INITIAL_CANDIDATES_PATH)
    assert initial_candidates.shape == (N_DOCUMENTS, CANDIDATE_COUNT)
    print("Reused initial candidates:", INITIAL_CANDIDATES_PATH)
else:
    representative_embeddings = embeddings[representative_indices]
    initial_candidates = np.empty(
        (N_DOCUMENTS, CANDIDATE_COUNT),
        dtype=np.int32,
    )

    emb_gpu = torch.from_numpy(embeddings).to(DEVICE)
    rep_gpu = torch.from_numpy(representative_embeddings).to(DEVICE)

    for start in tqdm(
        range(0, N_DOCUMENTS, 512),
        desc="Retrieving representative candidates",
    ):
        end = min(start + 512, N_DOCUMENTS)
        similarities = emb_gpu[start:end] @ rep_gpu.T
        local = torch.topk(
            similarities,
            k=CANDIDATE_COUNT,
            dim=1,
        ).indices
        initial_candidates[start:end] = representative_indices[local.cpu().numpy()]

    del emb_gpu, rep_gpu, similarities
    torch.cuda.empty_cache()

    np.save(INITIAL_CANDIDATES_PATH, initial_candidates)
    print("Saved initial candidates:", INITIAL_CANDIDATES_PATH)

# ------------------------------------------------------------
# 7. Dense neighbour pool for iterative refinement
# ------------------------------------------------------------
if DENSE_NEIGHBORS_PATH.exists():
    dense_neighbors = np.load(DENSE_NEIGHBORS_PATH)
    assert dense_neighbors.shape == (N_DOCUMENTS, DENSE_POOL_SIZE)
    print("Reused dense neighbours:", DENSE_NEIGHBORS_PATH)
else:
    dense_neighbors = np.empty(
        (N_DOCUMENTS, DENSE_POOL_SIZE),
        dtype=np.int32,
    )
    all_gpu = torch.from_numpy(embeddings).to(DEVICE)

    for start in tqdm(
        range(0, N_DOCUMENTS, 256),
        desc="Retrieving dense neighbour pool",
    ):
        end = min(start + 256, N_DOCUMENTS)
        similarities = all_gpu[start:end] @ all_gpu.T

        rows = torch.arange(end - start, device=DEVICE)
        columns = torch.arange(start, end, device=DEVICE)
        similarities[rows, columns] = -float("inf")

        nearest = torch.topk(
            similarities,
            k=DENSE_POOL_SIZE,
            dim=1,
        ).indices

        dense_neighbors[start:end] = nearest.cpu().numpy()

    del all_gpu, similarities
    torch.cuda.empty_cache()

    np.save(DENSE_NEIGHBORS_PATH, dense_neighbors)
    print("Saved dense neighbours:", DENSE_NEIGHBORS_PATH)

# ------------------------------------------------------------
# 8. Load Qwen3-0.6B
# ------------------------------------------------------------
print("\nLoading Qwen3-0.6B...")
llm_tokenizer = AutoTokenizer.from_pretrained(
    LLM_ID,
    revision=LLM_REVISION,
    use_fast=True,
    padding_side="left",
)
if llm_tokenizer.pad_token_id is None:
    llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id

llm = AutoModelForCausalLM.from_pretrained(
    LLM_ID,
    revision=LLM_REVISION,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to(DEVICE)
llm.eval()

print(
    "Qwen memory allocated:",
    f"{torch.cuda.memory_allocated() / 2**30:.2f} GB",
)


# ------------------------------------------------------------
# 9. Prompt/generation helpers
# ------------------------------------------------------------
def compact_text(text, max_words):
    words = " ".join(str(text).split()).split()
    return " ".join(words[:max_words])


def build_prompt(target_index, candidate_indices):
    target = compact_text(snippets[target_index], 64)

    candidate_lines = []
    for number, candidate_index in enumerate(candidate_indices, start=1):
        candidate = compact_text(snippets[int(candidate_index)], 24)
        candidate_lines.append(f"{number}. {candidate}")

    candidates = "\n".join(candidate_lines)

    instruction = f"""Determine which candidate patent claims concern the same technical topic as the target patent claim.

Target:
{target}

Candidates:
{candidates}

Return only candidate numbers separated by commas.
Return NONE if no candidate concerns the same technical topic.
Answer:"""

    messages = [{"role": "user", "content": instruction}]

    return llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def parse_selection(text, candidate_count):
    import re

    cleaned = text.strip().lower()

    if "none" in cleaned:
        return []

    numbers = []
    for match in re.findall(r"\d+", cleaned):
        value = int(match)
        if 1 <= value <= candidate_count:
            index = value - 1
            if index not in numbers:
                numbers.append(index)

    return numbers


@torch.inference_mode()
def generate_selections(target_indices, candidate_matrix):
    prompts = [
        build_prompt(int(target), candidate_matrix[row])
        for row, target in enumerate(target_indices)
    ]

    encoded = llm_tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_PROMPT_TOKENS,
    )
    encoded = {key: value.to(DEVICE) for key, value in encoded.items()}

    input_length = encoded["input_ids"].shape[1]

    generated = llm.generate(
        **encoded,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
        pad_token_id=llm_tokenizer.pad_token_id,
        eos_token_id=llm_tokenizer.eos_token_id,
    )

    continuations = generated[:, input_length:]
    decoded = llm_tokenizer.batch_decode(
        continuations,
        skip_special_tokens=True,
    )

    return [parse_selection(text, candidate_matrix.shape[1]) for text in decoded]


def save_state(path, payload):
    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


# ------------------------------------------------------------
# 10. Initial LeBoT construction with resume
# ------------------------------------------------------------
if INITIAL_BOT_PATH.exists() and INITIAL_STATE_PATH.exists():
    bot_vectors = np.load(INITIAL_BOT_PATH)
    initial_state = json.loads(INITIAL_STATE_PATH.read_text(encoding="utf-8"))
    initial_start = int(initial_state.get("next_position", 0))
    print("\nResuming initial BoT construction at:", initial_start)
else:
    bot_vectors = np.zeros(
        (N_DOCUMENTS, BOT_DIM),
        dtype=np.float32,
    )
    bot_vectors[
        representative_indices,
        np.arange(BOT_DIM),
    ] = 1.0

    initial_start = 0
    np.save(INITIAL_BOT_PATH, bot_vectors)
    save_state(
        INITIAL_STATE_PATH,
        {"next_position": 0, "complete": False},
    )

non_representatives = np.flatnonzero(~representative_mask)

if initial_start < len(non_representatives):
    print("\nInitial LeBoT construction...")

    batch_counter = 0

    for position in tqdm(
        range(initial_start, len(non_representatives), LLM_BATCH_SIZE),
        desc="LeBoT initial pass",
    ):
        target_indices = non_representatives[position : position + LLM_BATCH_SIZE]
        candidate_matrix = initial_candidates[target_indices]
        selections = generate_selections(target_indices, candidate_matrix)

        for row, target_index in enumerate(target_indices):
            selected_positions = selections[row]

            # Fixed-dimensional scalable adaptation:
            # if Qwen returns NONE, use the closest representative.
            if not selected_positions:
                selected_positions = [0]

            selected_patents = candidate_matrix[row, selected_positions]
            selected_columns = [
                int(np.where(representative_indices == patent)[0][0])
                for patent in selected_patents
            ]

            vector = np.zeros(BOT_DIM, dtype=np.float32)
            vector[selected_columns] = 1.0
            vector /= np.linalg.norm(vector).clip(min=1e-12)
            bot_vectors[int(target_index)] = vector

        batch_counter += 1
        next_position = min(
            position + len(target_indices),
            len(non_representatives),
        )

        if batch_counter % CHECKPOINT_EVERY_BATCHES == 0 or next_position == len(
            non_representatives
        ):
            np.save(INITIAL_BOT_PATH, bot_vectors)
            save_state(
                INITIAL_STATE_PATH,
                {
                    "next_position": next_position,
                    "complete": next_position == len(non_representatives),
                },
            )

    print("Initial BoT construction completed.")
else:
    print("\nInitial BoT construction already completed.")

bot_vectors = normalize(bot_vectors, axis=1).astype(np.float32)

# ------------------------------------------------------------
# 11. Iterative refinement with per-iteration resume
# ------------------------------------------------------------
if ACTIVE_PATH.exists():
    active_mask = np.load(ACTIVE_PATH).astype(bool)
else:
    active_mask = np.ones(N_DOCUMENTS, dtype=bool)

for iteration in range(1, MAX_REFINEMENT_ITERS + 1):
    completed_path = CHECKPOINT_DIR / f"refinement_{iteration}_complete.npy"
    work_path = CHECKPOINT_DIR / f"refinement_{iteration}_work.npy"
    state_path = CHECKPOINT_DIR / f"refinement_{iteration}_state.json"

    if completed_path.exists():
        bot_vectors = np.load(completed_path).astype(np.float32)
        print(f"Reused completed refinement {iteration}.")
        continue

    old_vectors = bot_vectors.copy()

    # Retrieve 30 candidates by reranking the 64 dense neighbours
    # using dense similarity + current BoT similarity.
    candidate_matrix = np.empty(
        (N_DOCUMENTS, CANDIDATE_COUNT),
        dtype=np.int32,
    )

    for start in tqdm(
        range(0, N_DOCUMENTS, 256),
        desc=f"Reranking candidates iter {iteration}",
    ):
        end = min(start + 256, N_DOCUMENTS)
        rows = np.arange(start, end)
        pool = dense_neighbors[start:end]

        dense_scores = np.einsum(
            "bd,bkd->bk",
            embeddings[start:end],
            embeddings[pool],
            optimize=True,
        )
        bot_scores = np.einsum(
            "bd,bkd->bk",
            old_vectors[start:end],
            old_vectors[pool],
            optimize=True,
        )
        combined = 0.5 * dense_scores + 0.5 * bot_scores

        local = np.argpartition(
            -combined,
            kth=CANDIDATE_COUNT - 1,
            axis=1,
        )[:, :CANDIDATE_COUNT]

        local_scores = np.take_along_axis(combined, local, axis=1)
        order = np.argsort(-local_scores, axis=1)
        local = np.take_along_axis(local, order, axis=1)
        candidate_matrix[start:end] = np.take_along_axis(pool, local, axis=1)

    active_indices = np.flatnonzero(active_mask)

    if work_path.exists() and state_path.exists():
        new_vectors = np.load(work_path).astype(np.float32)
        refine_state = json.loads(state_path.read_text(encoding="utf-8"))
        refine_start = int(refine_state.get("next_position", 0))
        print(
            f"Resuming refinement {iteration} at {refine_start}/{len(active_indices)}"
        )
    else:
        new_vectors = old_vectors.copy()
        refine_start = 0

    batch_counter = 0

    for position in tqdm(
        range(refine_start, len(active_indices), LLM_BATCH_SIZE),
        desc=f"LeBoT refinement {iteration}",
    ):
        target_indices = active_indices[position : position + LLM_BATCH_SIZE]
        candidates = candidate_matrix[target_indices]
        selections = generate_selections(target_indices, candidates)

        for row, target_index in enumerate(target_indices):
            selected_positions = selections[row]

            if selected_positions:
                selected_indices = candidates[row, selected_positions]
                vectors_to_average = np.vstack(
                    [
                        old_vectors[int(target_index)][None, :],
                        old_vectors[selected_indices],
                    ]
                )
                updated = vectors_to_average.mean(axis=0)
                norm = np.linalg.norm(updated)

                if norm > 0:
                    updated /= norm

                new_vectors[int(target_index)] = updated
            else:
                new_vectors[int(target_index)] = old_vectors[int(target_index)]

        batch_counter += 1
        next_position = min(
            position + len(target_indices),
            len(active_indices),
        )

        if batch_counter % CHECKPOINT_EVERY_BATCHES == 0 or next_position == len(
            active_indices
        ):
            np.save(work_path, new_vectors)
            save_state(
                state_path,
                {
                    "iteration": iteration,
                    "next_position": next_position,
                    "active_at_start": int(len(active_indices)),
                },
            )

    new_vectors = normalize(new_vectors, axis=1).astype(np.float32)

    cosine_changes = np.sum(old_vectors * new_vectors, axis=1)
    active_mask = cosine_changes <= CONVERGENCE_THRESHOLD
    bot_vectors = new_vectors

    np.save(completed_path, bot_vectors)
    np.save(ACTIVE_PATH, active_mask)

    if work_path.exists():
        work_path.unlink()
    if state_path.exists():
        state_path.unlink()

    print(
        f"Refinement {iteration}: "
        f"{np.sum(~active_mask):,} converged, "
        f"{np.sum(active_mask):,} still active"
    )

    if not active_mask.any():
        print("All documents converged.")
        break

FINAL_EMBEDDINGS_PATH = RESULT_DIR / "lebot_qwen06b_bot_vectors.npy"
np.save(FINAL_EMBEDDINGS_PATH, bot_vectors.astype(np.float32))

# Release LLM before clustering.
del llm
gc.collect()
torch.cuda.empty_cache()


# ------------------------------------------------------------
# 12. Evaluation helpers
# ------------------------------------------------------------
def predicted_cluster_purity(y_true, y_pred):
    total = 0

    for cluster_id in np.unique(y_pred):
        values = y_true[y_pred == cluster_id]
        total += Counter(values.tolist()).most_common(1)[0][1]

    return total / len(y_true)


def inverse_label_purity(y_true, y_pred):
    total = 0

    for label in np.unique(y_true):
        values = y_pred[y_true == label]
        total += Counter(values.tolist()).most_common(1)[0][1]

    return total / len(y_true)


def evaluate_partition(y_true, y_pred):
    return {
        "predicted_cluster_purity": float(predicted_cluster_purity(y_true, y_pred)),
        "inverse_label_purity": float(inverse_label_purity(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
    }


# ------------------------------------------------------------
# 13. K-means evaluation
# ------------------------------------------------------------
metric_rows = []
prediction_arrays = {}

print("\nRunning K-means evaluation...")

for seed in CLUSTER_SEEDS:
    clusterer = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=seed,
        n_init=10,
        max_iter=300,
        tol=1e-4,
        algorithm="lloyd",
    )
    predictions = clusterer.fit_predict(bot_vectors).astype(np.int32)
    prediction_arrays[f"seed_{seed}"] = predictions

    assert len(np.unique(predictions)) == N_CLUSTERS

    for level in ("section", "class", "subclass"):
        scores = evaluate_partition(labels[level], predictions)
        metric_rows.append(
            {
                "seed": seed,
                "level": level,
                **scores,
            }
        )

metrics_df = pd.DataFrame(metric_rows)
metrics_df.to_csv(RESULT_CSV, index=False, lineterminator="\n")
np.savez_compressed(PREDICTIONS_PATH, **prediction_arrays)

summary = {}

for level in ("section", "class", "subclass"):
    subset = metrics_df[metrics_df["level"] == level]
    summary[level] = {}

    for metric in (
        "predicted_cluster_purity",
        "inverse_label_purity",
        "nmi",
    ):
        summary[level][metric] = {
            "mean": float(subset[metric].mean()),
            "std": float(subset[metric].std(ddof=0)),
        }

mean_nmi_by_seed = metrics_df.groupby("seed")["nmi"].mean().to_dict()
overall_mean_nmi = float(np.mean(list(mean_nmi_by_seed.values())))
overall_std_nmi = float(np.std(list(mean_nmi_by_seed.values()), ddof=0))

latex_row = (
    "LeBoT (Qwen3-0.6B; L4 adaptation)"
    f" & {summary['section']['predicted_cluster_purity']['mean']:.4f}"
    f" & {summary['section']['inverse_label_purity']['mean']:.4f}"
    f" & {summary['section']['nmi']['mean']:.4f}"
    f" & {summary['class']['predicted_cluster_purity']['mean']:.4f}"
    f" & {summary['class']['inverse_label_purity']['mean']:.4f}"
    f" & {summary['class']['nmi']['mean']:.4f}"
    f" & {summary['subclass']['predicted_cluster_purity']['mean']:.4f}"
    f" & {summary['subclass']['inverse_label_purity']['mean']:.4f}"
    f" & {summary['subclass']['nmi']['mean']:.4f}"
    r" \\"
)

LATEX_PATH.write_text(latex_row + "\n", encoding="utf-8")

protocol = {
    "method": "LeBoT scalable L4 adaptation",
    "official_repository": "https://github.com/tom192180/BoT_vector",
    "paper": "LLMs Enable Bag-of-Texts Representations for Short-Text Clustering",
    "split": "test",
    "transductive_unsupervised_inference": True,
    "labels_used_during_representation_construction": False,
    "n_documents": N_DOCUMENTS,
    "n_clusters": N_CLUSTERS,
    "input_policy": (
        "All claims were tokenized in claim-ID order; short leading snippets "
        "from the 512-token document input were used in LLM prompts."
    ),
    "embedder": EMBEDDER_ID,
    "embedder_revision": EMBEDDER_REVISION,
    "llm": LLM_ID,
    "llm_revision": LLM_REVISION,
    "bot_dimension": BOT_DIM,
    "candidate_count": CANDIDATE_COUNT,
    "dense_candidate_pool": DENSE_POOL_SIZE,
    "maximum_refinement_iterations": MAX_REFINEMENT_ITERS,
    "convergence_threshold": CONVERGENCE_THRESHOLD,
    "representative_selection": (
        "MiniBatchKMeans medoids; scalable replacement for O(N^2) "
        "agglomerative representative selection"
    ),
    "none_policy": (
        "During fixed-dimensional initialization, NONE falls back to the "
        "nearest representative instead of creating a new dimension."
    ),
    "evaluation_seeds": CLUSTER_SEEDS,
    "raw_text_saved": False,
}

PROTOCOL_PATH.write_text(
    json.dumps(protocol, indent=2),
    encoding="utf-8",
)

result_payload = {
    "protocol": protocol,
    "summary": summary,
    "mean_nmi_by_seed": {
        str(key): float(value) for key, value in mean_nmi_by_seed.items()
    },
    "overall_mean_nmi": overall_mean_nmi,
    "overall_population_std_nmi": overall_std_nmi,
    "remaining_active_documents": int(active_mask.sum()),
    "latex_row": latex_row,
    "files": {
        "metrics_csv": str(RESULT_CSV),
        "results_json": str(RESULT_JSON),
        "predictions": str(PREDICTIONS_PATH),
        "bot_vectors": str(FINAL_EMBEDDINGS_PATH),
        "latex": str(LATEX_PATH),
        "protocol": str(PROTOCOL_PATH),
    },
}

RESULT_JSON.write_text(
    json.dumps(result_payload, indent=2),
    encoding="utf-8",
)

# ------------------------------------------------------------
# 14. Final report
# ------------------------------------------------------------
print("\n" + "=" * 80)
print("FINAL LEBOT RESULTS")
print("=" * 80)

for level in ("section", "class", "subclass"):
    values = summary[level]
    print(
        f"{level.capitalize():8s}: "
        f"Pur_p={values['predicted_cluster_purity']['mean']:.4f}"
        f" ± {values['predicted_cluster_purity']['std']:.4f}, "
        f"Pur_a={values['inverse_label_purity']['mean']:.4f}"
        f" ± {values['inverse_label_purity']['std']:.4f}, "
        f"NMI={values['nmi']['mean']:.4f}"
        f" ± {values['nmi']['std']:.4f}"
    )

print("\nMean NMI by seed:")
for seed, value in mean_nmi_by_seed.items():
    print(f"  {seed}: {value:.6f}")

print(f"Overall mean NMI: {overall_mean_nmi:.6f}")
print(f"Population SD:    {overall_std_nmi:.6f}")
print(f"Remaining active: {int(active_mask.sum()):,}")

print("\nLaTeX row:")
print(latex_row)

print("\nFiles:")
print("Metrics:     ", RESULT_CSV)
print("Results:     ", RESULT_JSON)
print("Predictions: ", PREDICTIONS_PATH)
print("BoT vectors: ", FINAL_EMBEDDINGS_PATH)
print("LaTeX:       ", LATEX_PATH)
print("Protocol:    ", PROTOCOL_PATH)
print("Checkpoints: ", CHECKPOINT_DIR)
print("\nSUCCESS: LeBoT L4 adaptation completed.")
