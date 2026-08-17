"""Claim-dependency validation and depth calculation."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence


class DependencyError(ValueError):
    """Raised when a claim-dependency graph is invalid."""


@dataclass(frozen=True)
class DependencyValidationResult:
    """Validation output for one patent claim graph."""

    patent_id: str
    n_claims: int
    n_edges: int
    n_roots: int
    max_depth: int
    roots: tuple[str, ...]
    invalid_references: tuple[tuple[str, str], ...]
    self_references: tuple[str, ...]
    has_cycle: bool
    is_valid: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a JSON-serializable dictionary."""

        result = asdict(self)
        result["roots"] = list(self.roots)
        result["invalid_references"] = [
            list(item) for item in self.invalid_references
        ]
        result["self_references"] = list(self.self_references)
        return result


def normalize_identifier(value: Any, *, field_name: str) -> str:
    """Convert a patent or claim identifier to a non-empty string."""

    if value is None:
        raise DependencyError(f"'{field_name}' cannot be null.")

    if isinstance(value, bool):
        raise DependencyError(
            f"'{field_name}' cannot be Boolean."
        )

    normalized = str(value).strip()

    if not normalized:
        raise DependencyError(
            f"'{field_name}' cannot be empty."
        )

    return normalized


def normalize_parent_ids(value: Any) -> list[str]:
    """Normalize a claim's parent identifier collection."""

    if value is None:
        return []

    if isinstance(value, (str, int)):
        values: Iterable[Any] = [value]
    elif isinstance(value, Sequence):
        values = value
    else:
        raise DependencyError(
            "'parent_ids' must be a sequence, scalar identifier, or null."
        )

    normalized: list[str] = []
    seen: set[str] = set()

    for item in values:
        parent_id = normalize_identifier(
            item,
            field_name="parent_id",
        )

        if parent_id not in seen:
            normalized.append(parent_id)
            seen.add(parent_id)

    return normalized


def _normalize_claims(
    claims: Sequence[Mapping[str, Any]],
    *,
    claim_id_field: str,
    parent_ids_field: str,
) -> tuple[list[str], dict[str, list[str]]]:
    """Normalize claim identifiers and parent references."""

    claim_ids: list[str] = []
    parent_map: dict[str, list[str]] = {}

    for index, claim in enumerate(claims):
        if not isinstance(claim, Mapping):
            raise DependencyError(
                f"Claim at index {index} must be an object."
            )

        if claim_id_field not in claim:
            raise DependencyError(
                f"Claim at index {index} is missing "
                f"'{claim_id_field}'."
            )

        claim_id = normalize_identifier(
            claim[claim_id_field],
            field_name=claim_id_field,
        )

        if claim_id in parent_map:
            raise DependencyError(
                f"Duplicate claim identifier: {claim_id}"
            )

        parent_ids = normalize_parent_ids(
            claim.get(parent_ids_field, [])
        )

        claim_ids.append(claim_id)
        parent_map[claim_id] = parent_ids

    if not claim_ids:
        raise DependencyError(
            "A patent must contain at least one claim."
        )

    return claim_ids, parent_map


def _find_invalid_references(
    claim_ids: Sequence[str],
    parent_map: Mapping[str, Sequence[str]],
) -> list[tuple[str, str]]:
    """Find references to claims not present in the patent."""

    known_claims = set(claim_ids)
    invalid: list[tuple[str, str]] = []

    for child_id in claim_ids:
        for parent_id in parent_map[child_id]:
            if parent_id not in known_claims:
                invalid.append((child_id, parent_id))

    return invalid


def _find_self_references(
    claim_ids: Sequence[str],
    parent_map: Mapping[str, Sequence[str]],
) -> list[str]:
    """Find claims that reference themselves."""

    return [
        claim_id
        for claim_id in claim_ids
        if claim_id in parent_map[claim_id]
    ]


def _build_adjacency(
    claim_ids: Sequence[str],
    parent_map: Mapping[str, Sequence[str]],
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Build parent-to-child adjacency and indegree."""

    claim_set = set(claim_ids)
    children: dict[str, list[str]] = {
        claim_id: [] for claim_id in claim_ids
    }
    indegree: dict[str, int] = {
        claim_id: 0 for claim_id in claim_ids
    }

    for child_id in claim_ids:
        for parent_id in parent_map[child_id]:
            if (
                parent_id in claim_set
                and parent_id != child_id
            ):
                children[parent_id].append(child_id)
                indegree[child_id] += 1

    return children, indegree


def _topological_order_and_depths(
    claim_ids: Sequence[str],
    parent_map: Mapping[str, Sequence[str]],
) -> tuple[list[str], dict[str, int]]:
    """Calculate topological order and longest-path depths."""

    children, indegree = _build_adjacency(
        claim_ids,
        parent_map,
    )

    queue = deque(
        claim_id
        for claim_id in claim_ids
        if indegree[claim_id] == 0
    )

    depths = {
        claim_id: 0
        for claim_id in queue
    }
    order: list[str] = []

    while queue:
        parent_id = queue.popleft()
        order.append(parent_id)
        parent_depth = depths[parent_id]

        for child_id in children[parent_id]:
            candidate_depth = parent_depth + 1
            current_depth = depths.get(child_id, 0)

            if candidate_depth > current_depth:
                depths[child_id] = candidate_depth

            indegree[child_id] -= 1

            if indegree[child_id] == 0:
                queue.append(child_id)

    if len(order) != len(claim_ids):
        raise DependencyError(
            "The claim-dependency graph contains a cycle."
        )

    return order, depths


def compute_claim_depths(
    claims: Sequence[Mapping[str, Any]],
    *,
    claim_id_field: str = "claim_id",
    parent_ids_field: str = "parent_ids",
    remove_invalid_references: bool = False,
) -> dict[str, int]:
    """Compute longest-path dependency depth for every claim."""

    claim_ids, parent_map = _normalize_claims(
        claims,
        claim_id_field=claim_id_field,
        parent_ids_field=parent_ids_field,
    )

    invalid_references = _find_invalid_references(
        claim_ids,
        parent_map,
    )
    self_references = _find_self_references(
        claim_ids,
        parent_map,
    )

    if self_references:
        joined = ", ".join(self_references)
        raise DependencyError(
            f"Self-referencing claims detected: {joined}"
        )

    if invalid_references and not remove_invalid_references:
        formatted = ", ".join(
            f"{child}->{parent}"
            for child, parent in invalid_references
        )
        raise DependencyError(
            f"Invalid parent references detected: {formatted}"
        )

    if invalid_references:
        known_claims = set(claim_ids)
        parent_map = {
            child_id: [
                parent_id
                for parent_id in parents
                if parent_id in known_claims
            ]
            for child_id, parents in parent_map.items()
        }

    _, depths = _topological_order_and_depths(
        claim_ids,
        parent_map,
    )

    return {
        claim_id: int(depths[claim_id])
        for claim_id in claim_ids
    }


def validate_dependency_graph(
    claims: Sequence[Mapping[str, Any]],
    *,
    patent_id: Any = "unknown",
    claim_id_field: str = "claim_id",
    parent_ids_field: str = "parent_ids",
    remove_invalid_references: bool = False,
    require_acyclic_graph: bool = True,
) -> DependencyValidationResult:
    """Validate one patent claim-dependency graph."""

    normalized_patent_id = normalize_identifier(
        patent_id,
        field_name="patent_id",
    )

    claim_ids, parent_map = _normalize_claims(
        claims,
        claim_id_field=claim_id_field,
        parent_ids_field=parent_ids_field,
    )

    invalid_references = _find_invalid_references(
        claim_ids,
        parent_map,
    )
    self_references = _find_self_references(
        claim_ids,
        parent_map,
    )

    known_claims = set(claim_ids)

    cleaned_parent_map = {
        child_id: [
            parent_id
            for parent_id in parents
            if (
                parent_id in known_claims
                and parent_id != child_id
            )
        ]
        for child_id, parents in parent_map.items()
    }

    has_cycle = False
    depths: dict[str, int] = {}

    try:
        _, depths = _topological_order_and_depths(
            claim_ids,
            cleaned_parent_map,
        )
    except DependencyError:
        has_cycle = True

    roots = tuple(
        claim_id
        for claim_id in claim_ids
        if not cleaned_parent_map[claim_id]
    )

    n_edges = sum(
        len(parents)
        for parents in cleaned_parent_map.values()
    )

    is_valid = (
        not self_references
        and (
            remove_invalid_references
            or not invalid_references
        )
        and (
            not require_acyclic_graph
            or not has_cycle
        )
        and bool(roots)
    )

    max_depth = max(depths.values()) if depths else 0

    return DependencyValidationResult(
        patent_id=normalized_patent_id,
        n_claims=len(claim_ids),
        n_edges=n_edges,
        n_roots=len(roots),
        max_depth=int(max_depth),
        roots=roots,
        invalid_references=tuple(invalid_references),
        self_references=tuple(self_references),
        has_cycle=has_cycle,
        is_valid=is_valid,
    )
