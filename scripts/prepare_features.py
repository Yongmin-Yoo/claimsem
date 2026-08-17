"""Prepare ClaimSem patent features from token-embedding shards."""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from claimsem.features import (
    CONSISTENT_CLAIMSEM,
    LEGACY_TEST,
    FeatureConfig,
    PatentFeatureAccumulator,
)


class FeaturePreparationError(RuntimeError):
    """Raised when token shards cannot be converted to patent features."""


def load_records(path: str | Path) -> list[Mapping[str, Any]]:
    """Load the legacy patent-record pickle."""
    record_path = Path(path).expanduser().resolve()

    if not record_path.exists():
        raise FileNotFoundError(f"Record file not found: {record_path}")

    with record_path.open("rb") as file:
        loaded = pickle.load(file)

    if isinstance(loaded, Mapping):
        for key in ("records", "data", "items", "patents"):
            candidate = loaded.get(key)

            if isinstance(candidate, Sequence) and not isinstance(
                candidate, (str, bytes, bytearray)
            ):
                loaded = candidate
                break

    if not (
        isinstance(loaded, Sequence) and not isinstance(loaded, (str, bytes, bytearray))
    ):
        raise FeaturePreparationError(
            f"Unsupported record collection type: {type(loaded)}"
        )

    records = list(loaded)

    if not records:
        raise FeaturePreparationError("Record collection is empty.")

    if not all(isinstance(record, Mapping) for record in records):
        raise FeaturePreparationError("Every patent record must be a mapping.")

    return records


def build_record_maps(
    records: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, dict[int, int], dict[tuple[int, int], int]]:
    """Build patent order and the exact legacy claim-depth lookup."""
    patent_ids: list[int] = []
    patent_to_index: dict[int, int] = {}
    depth_lookup: dict[tuple[int, int], int] = {}

    for index, record in enumerate(records):
        if "patent_id" not in record:
            raise FeaturePreparationError(f"Record {index} is missing patent_id.")

        patent_id = int(record["patent_id"])

        if patent_id in patent_to_index:
            raise FeaturePreparationError(f"Duplicate patent identifier: {patent_id}")

        patent_ids.append(patent_id)
        patent_to_index[patent_id] = index

        depth_mapping = record.get("depth", {})

        if not isinstance(depth_mapping, Mapping):
            raise FeaturePreparationError(
                f"Patent {patent_id} has invalid depth mapping."
            )

        for claim_id, depth in depth_mapping.items():
            key = (patent_id, int(claim_id))

            if key in depth_lookup:
                raise FeaturePreparationError(f"Duplicate claim-depth entry: {key}")

            normalized_depth = int(depth)

            if normalized_depth < 0:
                raise FeaturePreparationError(
                    f"Negative depth for claim {key}: {normalized_depth}"
                )

            depth_lookup[key] = normalized_depth

    return (
        np.asarray(patent_ids, dtype=np.int64),
        patent_to_index,
        depth_lookup,
    )


def _load_torch_shard(path: Path) -> Mapping[str, Any]:
    """Load one torch shard with explicit pickle compatibility."""
    try:
        shard = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        shard = torch.load(path, map_location="cpu")

    if not isinstance(shard, Mapping):
        raise FeaturePreparationError(
            f"Shard must contain a mapping, got {type(shard)}."
        )

    required = {"patent_ids", "entries"}
    missing = required.difference(shard)

    if missing:
        raise FeaturePreparationError(
            f"Shard {path.name} is missing keys: {sorted(missing)}"
        )

    return shard


def _claim_mean_from_entry(
    entry: Mapping[str, Any],
    *,
    hidden_dim: int,
) -> np.ndarray:
    """Convert one token-level shard entry to a float32 claim mean."""
    if "token_embeddings" not in entry:
        raise FeaturePreparationError("Shard entry is missing token_embeddings.")

    token_embeddings = entry["token_embeddings"]

    if torch.is_tensor(token_embeddings):
        tokens = token_embeddings.detach().cpu().to(torch.float32)
    else:
        tokens = torch.as_tensor(
            token_embeddings,
            dtype=torch.float32,
        )

    if tokens.ndim == 3:
        if tokens.shape[0] != 1:
            raise FeaturePreparationError(
                f"Unsupported token shape: {tuple(tokens.shape)}"
            )

        tokens = tokens[0]

    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)

    if tokens.ndim != 2:
        raise FeaturePreparationError(
            f"Token embeddings must be two-dimensional, got {tuple(tokens.shape)}."
        )

    if int(tokens.shape[1]) != hidden_dim:
        raise FeaturePreparationError(
            f"Expected hidden dimension {hidden_dim}, got {tokens.shape[1]}."
        )

    available_tokens = int(tokens.shape[0])
    num_tokens = min(
        int(entry.get("num_tokens", available_tokens)),
        available_tokens,
    )

    if num_tokens <= 0:
        raise FeaturePreparationError("A claim entry contains no usable tokens.")

    claim_mean = tokens[:num_tokens].mean(dim=0).numpy().astype(np.float32, copy=False)

    if claim_mean.shape != (hidden_dim,):
        raise FeaturePreparationError(f"Invalid claim mean shape: {claim_mean.shape}")

    if not np.all(np.isfinite(claim_mean)):
        raise FeaturePreparationError("Claim mean contains non-finite values.")

    return claim_mean


def process_shard(
    shard_path: str | Path,
    *,
    depth_lookup: Mapping[tuple[int, int], int],
    config: FeatureConfig,
    hidden_dim: int = 768,
    local_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Process one token shard into patent-level features."""
    source_path = Path(shard_path).expanduser().resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"Shard not found: {source_path}")

    load_path = source_path
    copied_locally = False

    if local_cache_dir is not None:
        local_dir = Path(local_cache_dir).expanduser().resolve()
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / source_path.name

        if (
            not local_path.exists()
            or local_path.stat().st_size != source_path.stat().st_size
        ):
            temporary_path = local_path.with_name(f"{local_path.name}.tmp")
            temporary_path.unlink(missing_ok=True)
            shutil.copyfile(source_path, temporary_path)

            if temporary_path.stat().st_size != source_path.stat().st_size:
                raise OSError(f"Incomplete local copy for {source_path.name}.")

            temporary_path.replace(local_path)

        load_path = local_path
        copied_locally = True

    try:
        shard = _load_torch_shard(load_path)

        patent_ids = np.asarray(
            [int(value) for value in shard["patent_ids"]],
            dtype=np.int64,
        )
        entries = list(shard["entries"])

        if patent_ids.ndim != 1 or patent_ids.size == 0:
            raise FeaturePreparationError(
                f"Invalid patent ID array in {source_path.name}."
            )

        if np.unique(patent_ids).size != patent_ids.size:
            raise FeaturePreparationError(f"Duplicate patents in {source_path.name}.")

        local_index = {
            int(patent_id): index for index, patent_id in enumerate(patent_ids)
        }

        accumulator = PatentFeatureAccumulator(
            n_patents=len(patent_ids),
            hidden_dim=hidden_dim,
            config=config,
        )

        truncated_count = 0
        missing_depth_keys: list[tuple[int, int]] = []

        for entry_index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise FeaturePreparationError(f"Entry {entry_index} is not a mapping.")

            patent_id = int(entry["patent_id"])
            claim_id = int(entry["claim_id"])

            if patent_id not in local_index:
                raise FeaturePreparationError(
                    f"Entry patent {patent_id} is absent from "
                    f"{source_path.name} patent_ids."
                )

            depth_key = (patent_id, claim_id)
            depth = depth_lookup.get(depth_key)

            if depth is None:
                missing_depth_keys.append(depth_key)
                continue

            claim_mean = _claim_mean_from_entry(
                entry,
                hidden_dim=hidden_dim,
            )

            accumulator.add(
                patent_index=local_index[patent_id],
                claim_embedding=claim_mean,
                depth=int(depth),
            )

            truncated_count += int(bool(entry.get("truncated", False)))

        if missing_depth_keys:
            raise FeaturePreparationError(
                f"{source_path.name} contains "
                f"{len(missing_depth_keys)} claims without depth. "
                f"Preview: {missing_depth_keys[:10]}"
            )

        features = accumulator.finalize()

        if features.shape != (len(patent_ids), hidden_dim):
            raise FeaturePreparationError(f"Invalid output shape: {features.shape}")

        return {
            "patent_ids": patent_ids,
            "features": features,
            "claim_count": len(entries),
            "truncated_count": truncated_count,
            "missing_depth_count": len(missing_depth_keys),
            "source_name": source_path.name,
            "source_size": source_path.stat().st_size,
        }

    finally:
        if copied_locally:
            try:
                load_path.unlink(missing_ok=True)
            except OSError:
                pass

        gc.collect()


def _config_signature(config: FeatureConfig) -> str:
    return json.dumps(
        config.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )


def save_partial(
    path: str | Path,
    result: Mapping[str, Any],
    *,
    config: FeatureConfig,
) -> Path:
    """Atomically save one processed shard."""
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_name(f"{output_path.name}.tmp.npz")

    np.savez_compressed(
        temporary_path,
        patent_ids=np.asarray(
            result["patent_ids"],
            dtype=np.int64,
        ),
        features=np.asarray(
            result["features"],
            dtype=np.float32,
        ),
        claim_count=np.asarray(
            [int(result["claim_count"])],
            dtype=np.int64,
        ),
        truncated_count=np.asarray(
            [int(result["truncated_count"])],
            dtype=np.int64,
        ),
        missing_depth_count=np.asarray(
            [int(result["missing_depth_count"])],
            dtype=np.int64,
        ),
        source_name=np.asarray(str(result["source_name"])),
        source_size=np.asarray(
            [int(result["source_size"])],
            dtype=np.int64,
        ),
        config_json=np.asarray(_config_signature(config)),
    )

    temporary_path.replace(output_path)
    return output_path


def load_partial(
    path: str | Path,
    *,
    config: FeatureConfig,
    source_path: str | Path,
    hidden_dim: int,
) -> dict[str, Any]:
    """Load and validate a resumable partial file."""
    partial_path = Path(path).expanduser().resolve()
    source = Path(source_path).expanduser().resolve()

    with np.load(partial_path, allow_pickle=False) as archive:
        required = {
            "patent_ids",
            "features",
            "claim_count",
            "truncated_count",
            "missing_depth_count",
            "source_name",
            "source_size",
            "config_json",
        }
        missing = required.difference(archive.files)

        if missing:
            raise FeaturePreparationError(
                f"Partial {partial_path.name} is missing keys: {sorted(missing)}"
            )

        patent_ids = archive["patent_ids"].astype(np.int64)
        features = archive["features"].astype(np.float32)
        claim_count = int(archive["claim_count"][0])
        truncated_count = int(archive["truncated_count"][0])
        missing_depth_count = int(archive["missing_depth_count"][0])
        source_name = str(archive["source_name"].item())
        source_size = int(archive["source_size"][0])
        config_json = str(archive["config_json"].item())

    if source_name != source.name:
        raise FeaturePreparationError(
            f"Partial source mismatch: {source_name} != {source.name}"
        )

    if source_size != source.stat().st_size:
        raise FeaturePreparationError(f"Source size changed for {source.name}.")

    if config_json != _config_signature(config):
        raise FeaturePreparationError(
            f"Configuration mismatch for {partial_path.name}."
        )

    if features.shape != (len(patent_ids), hidden_dim):
        raise FeaturePreparationError(
            f"Invalid partial feature shape: {features.shape}"
        )

    if not np.all(np.isfinite(features)):
        raise FeaturePreparationError(f"Non-finite features in {partial_path.name}.")

    return {
        "patent_ids": patent_ids,
        "features": features,
        "claim_count": claim_count,
        "truncated_count": truncated_count,
        "missing_depth_count": missing_depth_count,
        "source_name": source_name,
        "source_size": source_size,
    }


def merge_partials(
    partial_results: Sequence[Mapping[str, Any]],
    *,
    patent_to_index: Mapping[int, int],
    n_patents: int,
    hidden_dim: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Merge shard outputs into global record order."""
    features = np.zeros(
        (n_patents, hidden_dim),
        dtype=np.float32,
    )
    observed = np.zeros(n_patents, dtype=bool)

    total_claims = 0
    total_truncated = 0
    total_missing_depth = 0

    for partial in partial_results:
        patent_ids = np.asarray(
            partial["patent_ids"],
            dtype=np.int64,
        )
        partial_features = np.asarray(
            partial["features"],
            dtype=np.float32,
        )

        global_indices = []

        for patent_id in patent_ids:
            normalized_id = int(patent_id)

            if normalized_id not in patent_to_index:
                raise FeaturePreparationError(
                    f"Unknown patent in partial: {normalized_id}"
                )

            global_indices.append(patent_to_index[normalized_id])

        indices = np.asarray(global_indices, dtype=np.int64)

        if observed[indices].any():
            duplicates = indices[observed[indices]]
            raise FeaturePreparationError(
                f"Duplicate global patent indices: {duplicates[:10].tolist()}"
            )

        features[indices] = partial_features
        observed[indices] = True

        total_claims += int(partial["claim_count"])
        total_truncated += int(partial["truncated_count"])
        total_missing_depth += int(partial["missing_depth_count"])

    if not observed.all():
        missing = np.flatnonzero(~observed)
        raise FeaturePreparationError(
            f"{missing.size} patents were not observed. "
            f"Preview: {missing[:10].tolist()}"
        )

    return features, {
        "claim_count": total_claims,
        "truncated_count": total_truncated,
        "missing_depth_count": total_missing_depth,
    }


def compare_reference(
    features: np.ndarray,
    reference_path: str | Path,
    *,
    reference_key: str,
) -> dict[str, Any]:
    """Compare generated features with a legacy NPZ cache."""
    path = Path(reference_path).expanduser().resolve()

    with np.load(path, allow_pickle=False) as archive:
        if reference_key not in archive.files:
            raise FeaturePreparationError(
                f"Reference key {reference_key!r} not found. Keys: {archive.files}"
            )

        reference = archive[reference_key].astype(np.float32)

    if reference.shape != features.shape:
        raise FeaturePreparationError(
            f"Reference shape {reference.shape} does not match "
            f"generated shape {features.shape}."
        )

    generated_64 = features.astype(np.float64)
    reference_64 = reference.astype(np.float64)

    absolute_difference = np.abs(generated_64 - reference_64)

    dots = np.sum(generated_64 * reference_64, axis=1)
    norm_products = np.linalg.norm(generated_64, axis=1) * np.linalg.norm(
        reference_64, axis=1
    )
    cosine = dots / np.maximum(norm_products, 1e-12)

    max_abs = float(absolute_difference.max())
    practical_allclose = bool(
        np.allclose(
            features,
            reference,
            rtol=1e-5,
            atol=1e-6,
        )
    )

    return {
        "reference_path": str(path),
        "reference_key": reference_key,
        "max_absolute_difference": max_abs,
        "mean_absolute_difference": float(absolute_difference.mean()),
        "minimum_cosine_similarity": float(cosine.min()),
        "mean_cosine_similarity": float(cosine.mean()),
        "exact_array_equal": bool(np.array_equal(features, reference)),
        "practical_allclose": practical_allclose,
        "validation_passed": bool(
            practical_allclose and max_abs <= 1e-5 and float(cosine.min()) >= 0.99999
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Prepare ClaimSem patent features from token shards.")
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/content/drive/MyDrive/depth_ot_patent"),
    )
    parser.add_argument(
        "--split",
        choices=("dev", "test", "train"),
        default="dev",
    )
    parser.add_argument(
        "--records-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--local-cache-dir",
        type=Path,
        default=Path("/content/claimsem_feature_shard_cache"),
    )
    parser.add_argument(
        "--preprocessing-mode",
        choices=(CONSISTENT_CLAIMSEM, LEGACY_TEST),
        default=CONSISTENT_CLAIMSEM,
    )
    parser.add_argument(
        "--root-weight",
        type=float,
        default=12.0,
    )
    parser.add_argument(
        "--depth-decay",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--expected-shards",
        type=int,
        default=99,
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--reference-cache",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--reference-key",
        default="root12_d010",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--require-reference-match",
        action="store_true",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    project_root = args.project_root.expanduser().resolve()

    records_path = (
        args.records_path.expanduser().resolve()
        if args.records_path is not None
        else project_root / f"data/processed/{args.split}_records.pkl"
    )
    shard_dir = (
        args.shard_dir.expanduser().resolve()
        if args.shard_dir is not None
        else project_root / f"data/processed/token_features/full/{args.split}"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else Path("artifacts/prepared_features") / args.split
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = output_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)

    config = FeatureConfig(
        root_weight=args.root_weight,
        depth_decay=args.depth_decay,
        preprocessing_mode=args.preprocessing_mode,
        claim_selection="all",
        eps=1e-12,
    )

    records = load_records(records_path)
    patent_ids, patent_to_index, depth_lookup = build_record_maps(records)

    shard_paths = sorted(shard_dir.glob("shard_*.pt"))

    if len(shard_paths) != args.expected_shards:
        raise FeaturePreparationError(
            f"Expected {args.expected_shards} shards, "
            f"found {len(shard_paths)} in {shard_dir}."
        )

    selected_shards = (
        shard_paths[: args.max_shards] if args.max_shards is not None else shard_paths
    )

    print("=" * 88)
    print("CLAIMSEM FEATURE PREPARATION")
    print("=" * 88)
    print("Split              :", args.split)
    print("Records            :", records_path)
    print("Record count       :", len(records))
    print("Depth entries      :", len(depth_lookup))
    print("Shard directory    :", shard_dir)
    print("Available shards   :", len(shard_paths))
    print("Selected shards    :", len(selected_shards))
    print("Output directory   :", output_dir)
    print("Preprocessing mode :", args.preprocessing_mode)
    print("Root weight        :", args.root_weight)
    print("Depth decay        :", args.depth_decay)

    partial_results: list[dict[str, Any]] = []
    start_time = time.time()

    for shard_index, shard_path in enumerate(
        selected_shards,
        start=1,
    ):
        partial_path = partial_dir / f"{shard_path.stem}_features.npz"
        status = "processed"

        if partial_path.exists() and not args.overwrite:
            try:
                result = load_partial(
                    partial_path,
                    config=config,
                    source_path=shard_path,
                    hidden_dim=args.hidden_dim,
                )
                status = "cached"
            except Exception:
                partial_path.unlink(missing_ok=True)
                result = process_shard(
                    shard_path,
                    depth_lookup=depth_lookup,
                    config=config,
                    hidden_dim=args.hidden_dim,
                    local_cache_dir=args.local_cache_dir,
                )
                save_partial(
                    partial_path,
                    result,
                    config=config,
                )
        else:
            result = process_shard(
                shard_path,
                depth_lookup=depth_lookup,
                config=config,
                hidden_dim=args.hidden_dim,
                local_cache_dir=args.local_cache_dir,
            )
            save_partial(
                partial_path,
                result,
                config=config,
            )

        partial_results.append(result)

        print(
            f"[{shard_index:03d}/{len(selected_shards):03d}] "
            f"{shard_path.name}: {status}; "
            f"patents={len(result['patent_ids'])}, "
            f"claims={result['claim_count']}, "
            f"truncated={result['truncated_count']}"
        )

    if len(selected_shards) != len(shard_paths):
        print()
        print("Partial smoke run completed.")
        print("Final merge skipped because --max-shards was used.")
        return 0

    features, statistics = merge_partials(
        partial_results,
        patent_to_index=patent_to_index,
        n_patents=len(records),
        hidden_dim=args.hidden_dim,
    )

    norms = np.linalg.norm(features, axis=1)

    final_path = output_dir / (
        f"{args.split}_{args.preprocessing_mode}_root12_d010_features.npz"
    )

    metadata = {
        "split": args.split,
        "records_path": str(records_path),
        "shard_dir": str(shard_dir),
        "n_shards": len(shard_paths),
        "n_patents": len(records),
        "hidden_dim": args.hidden_dim,
        "config": config.to_dict(),
        "statistics": statistics,
        "norms": {
            "min": float(norms.min()),
            "mean": float(norms.mean()),
            "max": float(norms.max()),
        },
        "runtime_seconds": time.time() - start_time,
    }

    temporary_final = final_path.with_name(f"{final_path.name}.tmp.npz")
    np.savez_compressed(
        temporary_final,
        features=features,
        patent_ids=patent_ids,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    temporary_final.replace(final_path)

    reference_report = None

    if args.reference_cache is not None:
        reference_report = compare_reference(
            features,
            args.reference_cache,
            reference_key=args.reference_key,
        )

    report = {
        **metadata,
        "final_path": str(final_path),
        "reference_comparison": reference_report,
    }

    report_path = output_dir / "preparation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("FINAL RESULT")
    print("=" * 88)
    print("Feature shape :", features.shape)
    print("Feature dtype :", features.dtype)
    print(
        "Norm min/mean/max:",
        f"{norms.min():.9f}",
        f"{norms.mean():.9f}",
        f"{norms.max():.9f}",
    )
    print("Claims        :", statistics["claim_count"])
    print("Truncated     :", statistics["truncated_count"])
    print("Missing depth :", statistics["missing_depth_count"])
    print("Final cache   :", final_path)
    print("Report        :", report_path)

    if reference_report is not None:
        print(
            "Reference max difference:",
            reference_report["max_absolute_difference"],
        )
        print(
            "Reference mean cosine:",
            reference_report["mean_cosine_similarity"],
        )
        print(
            "Reference match:",
            reference_report["validation_passed"],
        )

        if args.require_reference_match and not reference_report["validation_passed"]:
            raise FeaturePreparationError(
                "Generated features do not match the reference cache."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
