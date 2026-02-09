"""Community Similarity Attention (CSA) model.

This module defines the neural components for the CSA detector.  The
``CrossAttention`` class performs a single-head attention operation
between a query vector (the post embedding) and a set of key/value
vectors (retrieved community centroids).  The ``CSADetector`` class
combines the attention output with the original post embedding and
feeds the concatenated vector through an MLP for final classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    """Single-head cross attention between a query and key/value set.

    This module projects the query and key/value vectors into a shared
    dimension ``hidden_dim``, computes scaled dot-product attention, and
    returns the weighted sum of value projections.  It also returns the
    attention weights for analysis.

    "Post embedding attends over community centroids to pick which communities are most relevant."
    """

    def __init__(self, dim_q: int, dim_kv: int, hidden_dim: int) -> None:
        super().__init__()
        self.wq = nn.Linear(dim_q, hidden_dim, bias=False)
        self.wk = nn.Linear(dim_kv, hidden_dim, bias=False)
        self.wv = nn.Linear(dim_kv, hidden_dim, bias=False)
        self.scale = hidden_dim ** 0.5

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply cross attention.

        Args:
            q: Query tensor of shape [B, D_q].
            k: Key tensor of shape [B, K, D_kv].
            v: Value tensor of shape [B, K, D_kv].

        Returns:
            context: Attention output of shape [B, hidden_dim].
            weights: Attention weights of shape [B, K].
        """
        # project to hidden space
        Q = self.wq(q).unsqueeze(1)         # [B, 1, H]
        K = self.wk(k)                      # [B, K, H]
        V = self.wv(v)                      # [B, K, H]
        # scaled dot product
        attn = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # [B, 1, K]
        weights = torch.softmax(attn, dim=-1)                # [B, 1, K]
        context = torch.bmm(weights, V).squeeze(1)           # [B, H]
        return context, weights.squeeze(1)


class CSADetector(nn.Module):
    """Community Similarity Attention detector.

    The detector takes as input a batch of post embeddings and a batch
    of sets of retrieved community embeddings.  It runs cross attention
    to summarise the community information relative to each post and
    concatenates the resulting context vector with the original post
    embedding.  The concatenated representation is passed through a
    feed-forward network to obtain logits for binary classification.
    """

    def __init__(self, emb_dim: int, hidden_dim: int = 768, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = CrossAttention(dim_q=emb_dim, dim_kv=emb_dim, hidden_dim=hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, post_emb: torch.Tensor, comm_embs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            post_emb: Tensor of shape [B, D] containing L2-normalised post embeddings.
            comm_embs: Tensor of shape [B, K, D] containing the top-K community centroids.

        Returns:
            logits: Tensor of shape [B, 2] containing raw class scores.
            weights: Tensor of shape [B, K] containing attention weights over the communities.
        """
        context, weights = self.attn(post_emb, comm_embs, comm_embs)
        fused = torch.cat([post_emb, context], dim=-1)  # [B, D + H]
        logits = self.mlp(fused)
        return logits, weights


class BaselineDetector(nn.Module):
    """Baseline classifier that uses only the post embedding (no community retrieval).

    This mirrors the MLP used in `CSADetector` but omits the attention/context
    branch so comparisons are fair (same hidden_dim and output structure).
    """

    def __init__(self, emb_dim: int, hidden_dim: int = 768, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, post_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass for baseline classifier.

        Args:
            post_emb: Tensor of shape [B, D] containing L2-normalised post embeddings.

        Returns:
            logits: Tensor of shape [B, 2]
        """
        logits = self.mlp(post_emb)
        return logits
