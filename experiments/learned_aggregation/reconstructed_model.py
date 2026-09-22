from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import softmax as pyg_softmax


class RecoveredStructuralVICRegGAT(nn.Module):
    """
    Structural VICReg-GAT recovered from archived checkpoints and
    validated against saved DEV and TEST attention statistics.

    Confirmed inference behavior:
      structural features = [depth / fitted_depth_scale, root indicator]
      input hidden = GELU(Linear + LayerNorm)
      GAT block = ELU(GAT(hidden)) + hidden
      document pooling = attention-weighted sum of original claim embeddings
      attention hidden activation = Tanh

    The exact stochastic training-view construction remains separate
    from this deterministic inference implementation.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 192,
        projection_dim: int = 128,
        gat_heads: int = 4,
        edge_dim: int = 2,
        edge_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if hidden_dim % gat_heads != 0:
            raise ValueError("hidden_dim must be divisible by gat_heads")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.projection_dim = int(projection_dim)
        self.gat_heads = int(gat_heads)
        self.edge_dim = int(edge_dim)
        self.edge_dropout = float(edge_dropout)

        structural_dim = 2
        first_head_dim = self.hidden_dim // self.gat_heads

        self.input_projection = nn.Sequential(
            nn.Linear(
                self.input_dim + structural_dim,
                self.hidden_dim,
            ),
            nn.LayerNorm(self.hidden_dim),
        )

        self.conv1 = GATConv(
            in_channels=self.hidden_dim,
            out_channels=first_head_dim,
            heads=self.gat_heads,
            concat=True,
            edge_dim=self.edge_dim,
            dropout=self.edge_dropout,
            add_self_loops=True,
            bias=True,
        )

        self.conv2 = GATConv(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            heads=1,
            concat=False,
            edge_dim=self.edge_dim,
            dropout=self.edge_dropout,
            add_self_loops=True,
            bias=True,
        )

        # Present in the archived checkpoint but not used by the
        # behaviorally recovered deterministic inference path.
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.attention_scorer = nn.Sequential(
            nn.Linear(
                self.hidden_dim + structural_dim,
                96,
            ),
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
        normalized_depth: torch.Tensor,
        root_indicator: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch_index: torch.Tensor,
        number_of_documents: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Deterministic inference/pooling forward.

        Parameters
        ----------
        claim_embeddings:
            Original frozen claim embeddings, shape [N, 768].
        normalized_depth:
            Depth divided by a scale fitted without test labels,
            shape [N]. The recovered original DEV-only experiment
            used development maximum depth 27.
        root_indicator:
            Binary root indicator, shape [N].
        edge_index:
            Bidirectional graph edges, shape [2, E].
        edge_attr:
            Direction one-hot features, shape [E, 2].
            Column 0 corresponds to parent-to-child and column 1
            to child-to-parent.
        batch_index:
            Document index for every claim node, shape [N].
        number_of_documents:
            Number of documents in the batch.
        """
        if claim_embeddings.ndim != 2:
            raise ValueError("claim_embeddings must have shape [N, D]")

        if claim_embeddings.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected embedding dimension {self.input_dim}, "
                f"got {claim_embeddings.shape[1]}"
            )

        claim_embeddings = claim_embeddings.float()
        normalized_depth = normalized_depth.float().reshape(-1)
        root_indicator = root_indicator.float().reshape(-1)
        batch_index = batch_index.long().reshape(-1)
        edge_index = edge_index.long()
        edge_attr = edge_attr.float()

        n_nodes = claim_embeddings.shape[0]

        if normalized_depth.numel() != n_nodes:
            raise ValueError("normalized_depth length mismatch")

        if root_indicator.numel() != n_nodes:
            raise ValueError("root_indicator length mismatch")

        if batch_index.numel() != n_nodes:
            raise ValueError("batch_index length mismatch")

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")

        if edge_attr.shape != (edge_index.shape[1], self.edge_dim):
            raise ValueError("edge_attr must have shape [E, edge_dim]")

        if number_of_documents is None:
            if batch_index.numel() == 0:
                raise ValueError("Empty batch_index")
            number_of_documents = int(batch_index.max().item()) + 1

        # Confirmed feature order: depth first, root second.
        structural_features = torch.stack(
            [normalized_depth, root_indicator],
            dim=1,
        )

        node_input = torch.cat(
            [claim_embeddings, structural_features],
            dim=1,
        )

        hidden = F.gelu(self.input_projection(node_input))

        hidden = (
            F.elu(
                self.conv1(
                    hidden,
                    edge_index,
                    edge_attr=edge_attr,
                )
            )
            + hidden
        )

        hidden = (
            F.elu(
                self.conv2(
                    hidden,
                    edge_index,
                    edge_attr=edge_attr,
                )
            )
            + hidden
        )

        attention_input = torch.cat(
            [hidden, structural_features],
            dim=1,
        )

        attention_logits = self.attention_scorer(attention_input).squeeze(-1)

        attention_weights = pyg_softmax(
            attention_logits,
            batch_index,
            num_nodes=number_of_documents,
        )

        # Pool the original 768-dimensional frozen claim embeddings.
        pooled = claim_embeddings.new_zeros((number_of_documents, self.input_dim))
        pooled.index_add_(
            0,
            batch_index,
            attention_weights.unsqueeze(1) * claim_embeddings,
        )

        projected = self.projector(pooled)

        return {
            "pooled": pooled,
            "projected": projected,
            "attention_logits": attention_logits,
            "attention_weights": attention_weights,
            "node_hidden": hidden,
        }


def vicreg_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    invariance_weight: float = 25.0,
    variance_weight: float = 25.0,
    covariance_weight: float = 1.0,
    variance_target: float = 1.0,
    epsilon: float = 1.0e-4,
) -> Dict[str, torch.Tensor]:
    """
    Standard VICReg objective.

    This loss form is recorded separately from stochastic view
    generation. View generation must be audited before final training.
    """
    if first.shape != second.shape:
        raise ValueError("VICReg view shapes must match")

    if first.ndim != 2:
        raise ValueError("VICReg inputs must have shape [batch, dimension]")

    invariance = F.mse_loss(first, second)

    first_centered = first - first.mean(dim=0)
    second_centered = second - second.mean(dim=0)

    first_std = torch.sqrt(
        first_centered.var(
            dim=0,
            unbiased=False,
        )
        + epsilon
    )
    second_std = torch.sqrt(
        second_centered.var(
            dim=0,
            unbiased=False,
        )
        + epsilon
    )

    variance = 0.5 * (
        F.relu(variance_target - first_std).mean()
        + F.relu(variance_target - second_std).mean()
    )

    batch_size = first.shape[0]
    feature_dim = first.shape[1]

    denominator = max(batch_size - 1, 1)

    first_covariance = (first_centered.T @ first_centered) / denominator

    second_covariance = (second_centered.T @ second_centered) / denominator

    identity = torch.eye(
        feature_dim,
        dtype=torch.bool,
        device=first.device,
    )

    covariance = (
        first_covariance[~identity].pow(2).sum() / feature_dim
        + second_covariance[~identity].pow(2).sum() / feature_dim
    )

    total = (
        float(invariance_weight) * invariance
        + float(variance_weight) * variance
        + float(covariance_weight) * covariance
    )

    return {
        "loss": total,
        "invariance": invariance,
        "variance": variance,
        "covariance": covariance,
        "mean_std_first": first_std.mean(),
        "mean_std_second": second_std.mean(),
    }
