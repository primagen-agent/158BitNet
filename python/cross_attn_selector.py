"""Cross-attention value selection: query tokens attend to source tokens.

Replaces mean-pooled span scoring with proper cross-attention:
  score(span) = attn_score(query_repr, span_context_repr)
where attention computes per-token alignment between query and source.

This is the architectural fix for the OOD multi-person limitation: mean pooling
cannot encode subject-ownership; attention can, because "Brenna" in the query
can align with "Brenna" in the source preceding "Riga".
"""
import torch
from torch import nn
import math


class CrossAttentionSelector(nn.Module):
    """Cross-attention between query tokens and source tokens for value selection.

    The query representation (from the query's C backbone features) attends
    to source token representations. The attention-weighted source context
    tells us which part of the source is relevant to the query. The value
    is then extracted from the maximally-attended contiguous region.
    """
    def __init__(self, hidden_dim=2048, attn_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim

        # Project query tokens to attention space
        self.q_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        # Project source tokens to attention space
        self.s_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        # Output: score each source token's relevance to the query
        self.out_proj = nn.Linear(attn_dim, 1)

    def forward(self, q_feats, s_feats, mask=None):
        """Compute per-source-token relevance scores.

        q_feats: [n_q, 2048] query token features
        s_feats: [n_s, 2048] source token features
        mask: [n_s] boolean, True = content token (skip JSON wrapper)

        Returns: scores [n_s] — relevance of each source token to the query.
        """
        n_q, n_s = q_feats.shape[0], s_feats.shape[0]

        # Project to attention space
        q = self.q_proj(q_feats)  # [n_q, attn_dim]
        s = self.s_proj(s_feats)  # [n_s, attn_dim]

        # Multi-head attention: query attends to source
        # attn[i,j] = how much query token i aligns with source token j
        attn = torch.softmax(q @ s.T / math.sqrt(self.attn_dim), dim=-1)  # [n_q, n_s]

        # Aggregate: each source token's score = sum over query tokens of
        # (attention weight × query token importance)
        # Query token importance: the output projection of the query token
        q_importance = self.out_proj(q).squeeze(-1)  # [n_q]
        q_importance = torch.softmax(q_importance, dim=0)  # normalize

        # Source score: weighted combination of attention alignment
        # score[j] = Σ_i attn[i,j] * q_importance[i]
        scores = attn.T @ q_importance  # [n_s]

        if mask is not None:
            scores = scores.masked_fill(~mask, -1e30)

        return scores
