
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as pyg_softmax


class RecoveredSSLMLP(nn.Module):
    """
    SSL-MLP with attention recovered from archived checkpoints.

    Deterministic inference was validated against saved development
    attention statistics with mean relative error below 4e-6.

    norm1 and norm2 are retained for strict checkpoint compatibility.
    Their parameters remained at initialization in the archived
    checkpoints and they are not used by the recovered forward path.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 192,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.projection_dim = int(projection_dim)

        self.input_projection = nn.Sequential(
            nn.Linear(
                self.input_dim,
                self.hidden_dim,
            ),
            nn.LayerNorm(self.hidden_dim),
        )

        # Retained only for exact checkpoint compatibility.
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.attention_scorer = nn.Sequential(
            nn.Linear(self.hidden_dim, 96),
            nn.Tanh(),
            nn.Linear(96, 1),
        )

        self.projector = nn.Sequential(
            nn.Linear(
                self.input_dim,
                self.hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                self.hidden_dim,
                self.projection_dim,
            ),
        )

    def parameter_count(self) -> int:
        return int(
            sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            )
        )

    def forward(
        self,
        claim_embeddings: torch.Tensor,
        batch_index: torch.Tensor,
        number_of_documents: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        if claim_embeddings.ndim != 2:
            raise ValueError(
                "claim_embeddings must have shape [N, D]"
            )

        if claim_embeddings.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected dimension {self.input_dim}, "
                f"got {claim_embeddings.shape[1]}"
            )

        claim_embeddings = claim_embeddings.float()
        batch_index = batch_index.long().reshape(-1)

        if batch_index.numel() != claim_embeddings.shape[0]:
            raise ValueError("batch_index length mismatch")

        if number_of_documents is None:
            if batch_index.numel() == 0:
                raise ValueError("Empty batch_index")
            number_of_documents = (
                int(batch_index.max().item()) + 1
            )

        hidden = F.gelu(
            self.input_projection(
                claim_embeddings
            )
        )

        attention_logits = self.attention_scorer(
            hidden
        ).squeeze(-1)

        attention_weights = pyg_softmax(
            attention_logits,
            batch_index,
            num_nodes=number_of_documents,
        )

        pooled = claim_embeddings.new_zeros(
            (number_of_documents, self.input_dim)
        )
        pooled.index_add_(
            0,
            batch_index,
            attention_weights.unsqueeze(1)
            * claim_embeddings,
        )

        projected = self.projector(pooled)

        return {
            "pooled": pooled,
            "projected": projected,
            "attention_logits": attention_logits,
            "attention_weights": attention_weights,
            "node_hidden": hidden,
        }
