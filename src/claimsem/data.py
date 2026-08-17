"""Patent record loading and normalization for ClaimSem."""

from __future__ import annotations

import json
import pickle
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from claimsem.dependency import (
    DependencyError,
    compute_claim_depths,
    normalize_identifier,
    normalize_parent_ids,
    validate_dependency_graph,
)


class DataError(ValueError):
    """Raised when patent records are malformed or inconsistent."""


@dataclass(frozen=True)
class DatasetSummary:
    """Summary statistics for a normalized patent collection."""

    n_patents: int
    n_claims: int
    n_edges: int
    n_roots: int
    max_depth: int
    cpc_sections: int
    cpc_classes: int
    cpc_subclasses: int

    def to_dict(self) -> dict[str, int]:
        """Convert summary to a JSON-serializable dictionary."""

        return asdict(self)


def load_raw_records(
    path: str | Path,
    *,
    data_format: str | None = None,
) -> list[Mapping[str, Any]]:
    """Load patent records from JSON or pickle."""

    input_path = Path(path).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Patent record file not found: {input_path}"
        )

    if not input_path.is_file():
        raise DataError(
            f"Patent record path is not a file: {input_path}"
        )

    normalized_format = (
        data_format.strip().lower()
        if data_format is not None
        else input_path.suffix.lower().lstrip(".")
    )

    if normalized_format == "json":
        with input_path.open("r", encoding="utf-8") as file:
            records = json.load(file)
    elif normalized_format in {"pkl", "pickle"}:
        with input_path.open("rb") as file:
            records = pickle.load(file)
    else:
        raise DataError(
            "Unsupported record format. Use JSON or pickle."
        )

    if isinstance(records, Mapping):
        for candidate_key in ("records", "data", "patents"):
            candidate = records.get(candidate_key)

            if isinstance(candidate, Sequence) and not isinstance(
                candidate,
                (str, bytes, bytearray),
            ):
                records = candidate
                break

    if not isinstance(records, Sequence) or isinstance(
        records,
        (str, bytes, bytearray),
    ):
        raise DataError(
            "The record file must contain a list of patent records."
        )

    normalized_records = list(records)

    for index, record in enumerate(normalized_records):
        if not isinstance(record, Mapping):
            raise DataError(
                f"Patent record at index {index} must be an object."
            )

    return normalized_records


def _normalize_cpc(
    value: Any,
) -> dict[str, str | None]:
    """Normalize section, class, and subclass labels."""

    labels: dict[str, str | None] = {
        "section": None,
        "class": None,
        "subclass": None,
    }

    if value is None:
        return labels

    if isinstance(value, Mapping):
        for level in labels:
            raw_label = value.get(level)

            if raw_label is not None:
                label = str(raw_label).strip()
                labels[level] = label or None

        return labels

    if isinstance(value, str):
        label = value.strip().upper()

        if not label:
            return labels

        labels["section"] = label[:1] or None
        labels["class"] = label[:3] or None
        labels["subclass"] = label[:4] or None
        return labels

    if isinstance(value, Sequence):
        first_label = next(
            (
                item
                for item in value
                if item is not None and str(item).strip()
            ),
            None,
        )

        return _normalize_cpc(first_label)

    raise DataError(
        "CPC information must be an object, string, sequence, or null."
    )


def _normalize_claim(
    claim: Mapping[str, Any],
    *,
    claim_id_field: str,
    claim_text_field: str,
    parent_ids_field: str,
) -> dict[str, Any]:
    """Normalize one claim record."""

    if claim_id_field not in claim:
        raise DataError(
            f"A claim is missing '{claim_id_field}'."
        )

    claim_id = normalize_identifier(
        claim[claim_id_field],
        field_name=claim_id_field,
    )

    raw_text = claim.get(claim_text_field, "")

    if raw_text is None:
        raw_text = ""

    text = str(raw_text).strip()

    if not text:
        raise DataError(
            f"Claim '{claim_id}' has empty text."
        )

    parent_ids = normalize_parent_ids(
        claim.get(parent_ids_field, [])
    )

    return {
        "claim_id": claim_id,
        "text": text,
        "parent_ids": parent_ids,
    }


def normalize_record(
    record: Mapping[str, Any],
    *,
    patent_id_field: str = "patent_id",
    claims_field: str = "claims",
    claim_id_field: str = "claim_id",
    claim_text_field: str = "text",
    parent_ids_field: str = "parent_ids",
    cpc_field: str = "cpc",
    remove_invalid_references: bool = False,
    require_acyclic_graph: bool = True,
) -> dict[str, Any]:
    """Normalize one patent record and calculate claim depths."""

    if patent_id_field not in record:
        raise DataError(
            f"A patent record is missing '{patent_id_field}'."
        )

    patent_id = normalize_identifier(
        record[patent_id_field],
        field_name=patent_id_field,
    )

    raw_claims = record.get(claims_field)

    if not isinstance(raw_claims, Sequence) or isinstance(
        raw_claims,
        (str, bytes, bytearray),
    ):
        raise DataError(
            f"Patent '{patent_id}' must contain a claim list."
        )

    claims = [
        _normalize_claim(
            claim,
            claim_id_field=claim_id_field,
            claim_text_field=claim_text_field,
            parent_ids_field=parent_ids_field,
        )
        for claim in raw_claims
    ]

    validation = validate_dependency_graph(
        claims,
        patent_id=patent_id,
        remove_invalid_references=remove_invalid_references,
        require_acyclic_graph=require_acyclic_graph,
    )

    if not validation.is_valid:
        raise DataError(
            f"Invalid dependency graph for patent '{patent_id}': "
            f"{validation.to_dict()}"
        )

    if remove_invalid_references:
        known_claims = {
            claim["claim_id"] for claim in claims
        }

        for claim in claims:
            claim["parent_ids"] = [
                parent_id
                for parent_id in claim["parent_ids"]
                if parent_id in known_claims
            ]

    try:
        depths = compute_claim_depths(
            claims,
            remove_invalid_references=remove_invalid_references,
        )
    except DependencyError as error:
        raise DataError(
            f"Could not calculate depths for patent "
            f"'{patent_id}': {error}"
        ) from error

    for claim in claims:
        claim["depth"] = depths[claim["claim_id"]]
        claim["is_root"] = claim["depth"] == 0

    cpc = _normalize_cpc(record.get(cpc_field))

    return {
        "patent_id": patent_id,
        "claims": claims,
        "cpc": cpc,
        "dependency": validation.to_dict(),
    }


def normalize_records(
    records: Sequence[Mapping[str, Any]],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Normalize a patent collection and check patent identifiers."""

    normalized: list[dict[str, Any]] = []
    seen_patent_ids: set[str] = set()

    for index, record in enumerate(records):
        try:
            normalized_record = normalize_record(
                record,
                **kwargs,
            )
        except (DataError, DependencyError) as error:
            raise DataError(
                f"Failed to normalize patent record at index "
                f"{index}: {error}"
            ) from error

        patent_id = normalized_record["patent_id"]

        if patent_id in seen_patent_ids:
            raise DataError(
                f"Duplicate patent identifier: {patent_id}"
            )

        seen_patent_ids.add(patent_id)
        normalized.append(normalized_record)

    if not normalized:
        raise DataError(
            "The patent collection cannot be empty."
        )

    return normalized


def load_records(
    path: str | Path,
    *,
    data_format: str | None = None,
    **normalization_kwargs: Any,
) -> list[dict[str, Any]]:
    """Load and normalize patent records."""

    raw_records = load_raw_records(
        path,
        data_format=data_format,
    )

    return normalize_records(
        raw_records,
        **normalization_kwargs,
    )


def summarize_records(
    records: Sequence[Mapping[str, Any]],
) -> DatasetSummary:
    """Calculate dataset-level summary statistics."""

    n_claims = 0
    n_edges = 0
    n_roots = 0
    max_depth = 0

    labels = {
        "section": set(),
        "class": set(),
        "subclass": set(),
    }

    for record in records:
        claims = record.get("claims", [])
        n_claims += len(claims)

        for claim in claims:
            parent_ids = claim.get("parent_ids", [])
            depth = int(claim.get("depth", 0))

            n_edges += len(parent_ids)
            n_roots += int(depth == 0)
            max_depth = max(max_depth, depth)

        cpc = record.get("cpc", {})

        if isinstance(cpc, Mapping):
            for level in labels:
                label = cpc.get(level)

                if label:
                    labels[level].add(str(label))

    return DatasetSummary(
        n_patents=len(records),
        n_claims=n_claims,
        n_edges=n_edges,
        n_roots=n_roots,
        max_depth=max_depth,
        cpc_sections=len(labels["section"]),
        cpc_classes=len(labels["class"]),
        cpc_subclasses=len(labels["subclass"]),
    )


def extract_cpc_labels(
    records: Sequence[Mapping[str, Any]],
    level: str,
    *,
    require_complete: bool = True,
) -> list[str | None]:
    """Extract one CPC label level in patent order."""

    normalized_level = level.strip().lower()
    valid_levels = {"section", "class", "subclass"}

    if normalized_level not in valid_levels:
        raise DataError(
            f"Unsupported CPC level: {level}"
        )

    output: list[str | None] = []

    for record in records:
        cpc = record.get("cpc", {})

        label = (
            cpc.get(normalized_level)
            if isinstance(cpc, Mapping)
            else None
        )

        normalized_label = (
            str(label).strip()
            if label is not None
            else None
        )

        if normalized_label == "":
            normalized_label = None

        output.append(normalized_label)

    if require_complete:
        missing = [
            index
            for index, label in enumerate(output)
            if label is None
        ]

        if missing:
            preview = ", ".join(
                str(index) for index in missing[:10]
            )
            raise DataError(
                f"Missing CPC '{normalized_level}' labels "
                f"at record indices: {preview}"
            )

    return output


def claim_count_distribution(
    records: Sequence[Mapping[str, Any]],
) -> dict[int, int]:
    """Return patent counts grouped by number of claims."""

    distribution = Counter(
        len(record.get("claims", []))
        for record in records
    )

    return dict(sorted(distribution.items()))


def copy_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return a deep copy of normalized records."""

    return deepcopy(list(records))
