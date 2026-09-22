
"""Newly specified train-split stochastic objectives.

This module is not claimed to reproduce the unavailable original
stochastic training implementation. Deterministic inference architectures
remain in reconstructed_model.py and reconstructed_mlp.py.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict

import torch
import torch.nn.functional as F


@contextmanager
def seeded_torch_rng(seed: int, device: torch.device):
    """Temporarily use a deterministic RNG state without contaminating it."""
    devices = []
    if device.type == "cuda":
        if device.index is None:
            devices = [torch.cuda.current_device()]
        else:
            devices = [device.index]

    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        yield


def stochastic_claim_view(
    claim_embeddings: torch.Tensor,
    *,
    seed: int,
    feature_dropout: float = 0.1,
) -> torch.Tensor:
    """Independent inverted-dropout view of frozen claim embeddings."""
    with seeded_torch_rng(seed, claim_embeddings.device):
        return F.dropout(
            claim_embeddings,
            p=float(feature_dropout),
            training=True,
        )


def gat_projected_view(
    model,
    batch,
    *,
    seed: int,
    feature_dropout: float = 0.1,
    model_dropout: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """One stochastic Structural VICReg-GAT view.

    Random components:
    1. claim-feature dropout;
    2. GAT attention-coefficient dropout from GATConv;
    3. projection-output dropout.

    No explicit edge removal is performed.
    """
    model.train()

    with seeded_torch_rng(seed, batch.x.device):
        augmented_x = F.dropout(
            batch.x,
            p=float(feature_dropout),
            training=True,
        )

        output = model(
            claim_embeddings=augmented_x,
            normalized_depth=batch.structural_features[:, 0],
            root_indicator=batch.structural_features[:, 1],
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch_index=batch.batch,
            number_of_documents=batch.n_documents,
        )

        projected = F.dropout(
            output["projected"],
            p=float(model_dropout),
            training=True,
        )

    return {
        **output,
        "projected_for_loss": projected,
    }


def mlp_projected_view(
    model,
    batch,
    *,
    seed: int,
    feature_dropout: float = 0.1,
    model_dropout: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """One stochastic SSL-MLP attention view."""
    model.train()

    with seeded_torch_rng(seed, batch.x.device):
        augmented_x = F.dropout(
            batch.x,
            p=float(feature_dropout),
            training=True,
        )

        output = model(
            claim_embeddings=augmented_x,
            batch_index=batch.batch,
            number_of_documents=batch.n_documents,
        )

        projected = F.dropout(
            output["projected"],
            p=float(model_dropout),
            training=True,
        )

    return {
        **output,
        "projected_for_loss": projected,
    }


def symmetric_info_nce_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> Dict[str, torch.Tensor]:
    """Symmetric in-batch NT-Xent/InfoNCE loss."""
    if first.shape != second.shape:
        raise ValueError("InfoNCE view shapes must match")
    if first.ndim != 2:
        raise ValueError("Expected [batch, feature] tensors")
    if first.shape[0] < 2:
        raise ValueError("InfoNCE requires at least two documents")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    first_normalized = F.normalize(first, dim=1)
    second_normalized = F.normalize(second, dim=1)

    combined = torch.cat(
        [first_normalized, second_normalized],
        dim=0,
    )
    logits = combined @ combined.T
    logits = logits / float(temperature)

    n = first.shape[0]
    total = 2 * n

    diagonal = torch.eye(
        total,
        dtype=torch.bool,
        device=logits.device,
    )
    logits = logits.masked_fill(diagonal, float("-inf"))

    positive_index = (
        torch.arange(total, device=logits.device) + n
    ) % total

    loss = F.cross_entropy(logits, positive_index)

    positive_similarity = torch.cat(
        [
            (first_normalized * second_normalized).sum(dim=1),
            (second_normalized * first_normalized).sum(dim=1),
        ],
        dim=0,
    ).mean()

    return {
        "loss": loss,
        "positive_cosine_similarity": positive_similarity,
        "temperature": torch.tensor(
            float(temperature),
            device=loss.device,
        ),
    }
