# ruff: noqa
# Auto-exported from the executed Google Colab experiment.
# Raw patent data and credentials are not included.

# ============================================================
# CELL 3 — Tokenize and cache SCCL inputs
# ============================================================

import json
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

MAX_LENGTH = 512
TOKENIZE_BATCH_SIZE = 256
N_DOCUMENTS = 9_881

INPUT_IDS_PATH = CACHE_DIR / "sccl_input_ids_512.npy"
ATTENTION_MASK_PATH = CACHE_DIR / "sccl_attention_mask_512.npy"
TOKEN_STATS_PATH = CACHE_DIR / "sccl_tokenization_stats.json"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    use_fast=True,
    trust_remote_code=False,
)

assert tokenizer.is_fast
assert tokenizer.pad_token_id is not None

start_time = time.time()

# ------------------------------------------------------------
# Reuse valid cache or generate it
# ------------------------------------------------------------

reuse_cache = False

if INPUT_IDS_PATH.exists() and ATTENTION_MASK_PATH.exists():
    cached_ids = np.load(INPUT_IDS_PATH, mmap_mode="r")
    cached_mask = np.load(ATTENTION_MASK_PATH, mmap_mode="r")

    reuse_cache = cached_ids.shape == (
        N_DOCUMENTS,
        MAX_LENGTH,
    ) and cached_mask.shape == (N_DOCUMENTS, MAX_LENGTH)

    del cached_ids, cached_mask

if reuse_cache:
    print("Reusing existing token cache.")

else:
    print("Generating 512-token cache...")

    input_ids_cache = np.lib.format.open_memmap(
        INPUT_IDS_PATH,
        mode="w+",
        dtype=np.int32,
        shape=(N_DOCUMENTS, MAX_LENGTH),
    )

    attention_mask_cache = np.lib.format.open_memmap(
        ATTENTION_MASK_PATH,
        mode="w+",
        dtype=np.uint8,
        shape=(N_DOCUMENTS, MAX_LENGTH),
    )

    for start in tqdm(
        range(0, N_DOCUMENTS, TOKENIZE_BATCH_SIZE),
        desc="Tokenizing patents",
    ):
        stop = min(start + TOKENIZE_BATCH_SIZE, N_DOCUMENTS)
        batch_texts = sccl_documents[start:stop]

        encoded = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="np",
        )

        input_ids_cache[start:stop] = encoded["input_ids"].astype(
            np.int32,
            copy=False,
        )
        attention_mask_cache[start:stop] = encoded["attention_mask"].astype(
            np.uint8, copy=False
        )

    input_ids_cache.flush()
    attention_mask_cache.flush()

    del input_ids_cache, attention_mask_cache

# ------------------------------------------------------------
# Validate cache and calculate length statistics
# ------------------------------------------------------------

input_ids_cache = np.load(INPUT_IDS_PATH, mmap_mode="r")
attention_mask_cache = np.load(ATTENTION_MASK_PATH, mmap_mode="r")

assert input_ids_cache.shape == (N_DOCUMENTS, MAX_LENGTH)
assert attention_mask_cache.shape == (N_DOCUMENTS, MAX_LENGTH)
assert input_ids_cache.dtype == np.int32
assert attention_mask_cache.dtype == np.uint8
assert np.isfinite(input_ids_cache).all()

token_lengths = np.asarray(
    attention_mask_cache.sum(axis=1),
    dtype=np.int32,
)

assert token_lengths.min() > 0
assert token_lengths.max() <= MAX_LENGTH

at_limit = token_lengths == MAX_LENGTH
at_limit_count = int(at_limit.sum())
at_limit_fraction = float(at_limit.mean())

stats = {
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "n_documents": N_DOCUMENTS,
    "max_length": MAX_LENGTH,
    "padding": "max_length",
    "truncation": True,
    "token_length_min": int(token_lengths.min()),
    "token_length_mean_after_truncation": float(token_lengths.mean()),
    "token_length_median_after_truncation": float(np.median(token_lengths)),
    "token_length_max": int(token_lengths.max()),
    "documents_at_512_token_limit": at_limit_count,
    "fraction_at_512_token_limit": at_limit_fraction,
    "cache_reused": reuse_cache,
    "elapsed_seconds": time.time() - start_time,
}

TOKEN_STATS_PATH.write_text(
    json.dumps(stats, indent=2),
    encoding="utf-8",
)

# Update protocol
protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
protocol["tokenization"] = stats
PROTOCOL_PATH.write_text(
    json.dumps(protocol, indent=2),
    encoding="utf-8",
)

print("\nInput IDs:", input_ids_cache.shape, input_ids_cache.dtype)
print(
    "Attention mask:",
    attention_mask_cache.shape,
    attention_mask_cache.dtype,
)
print(
    "Token length min/mean/median/max:",
    int(token_lengths.min()),
    f"{token_lengths.mean():.1f}",
    f"{np.median(token_lengths):.1f}",
    int(token_lengths.max()),
)
print(
    "Documents at 512-token limit:",
    f"{at_limit_count:,}/{N_DOCUMENTS:,}",
    f"({100 * at_limit_fraction:.2f}%)",
)
print("Cache reused:", reuse_cache)
print("Elapsed minutes:", f"{(time.time() - start_time) / 60:.2f}")
print("Input IDs:", INPUT_IDS_PATH)
print("Mask:", ATTENTION_MASK_PATH)
print("Stats:", TOKEN_STATS_PATH)
print("\nSUCCESS: SCCL token cache prepared.")
