from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AuthorTransformerConfig:
    emb_dim: int = 1024
    q_dim: int = 802
    d_model: int = 384
    n_layers: int = 4
    n_heads: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_history: int = 20
    n_delta_buckets: int = 5

    def to_dict(self) -> dict:
        return asdict(self)


def delta_year_buckets(years: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Bucketize adjacent publication-year deltas: 0, 1, 2, 3, 4+."""
    years = years.long()
    deltas = torch.zeros_like(years)
    if years.shape[1] > 1:
        raw = years[:, 1:] - years[:, :-1]
        deltas[:, 1:] = torch.clamp(raw, min=0, max=4)
    deltas = torch.clamp(deltas, min=0, max=4)
    return deltas.masked_fill(~mask.bool(), 0)


class AuthorTransformer(nn.Module):
    def __init__(self, config: AuthorTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.paper_proj = nn.Linear(config.emb_dim + config.q_dim, config.d_model)
        self.delta_emb = nn.Embedding(config.n_delta_buckets, config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)
        self.author_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.cluster_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.q_dim),
        )
        self.embedding_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.emb_dim),
        )
        nn.init.normal_(self.author_token, std=0.02)

    def forward(
        self,
        history_emb: torch.Tensor,
        history_q: torch.Tensor,
        years: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x = torch.cat([history_emb, history_q], dim=-1)
        x = self.paper_proj(x)
        x = x + self.delta_emb(delta_year_buckets(years, mask))
        x = self.input_norm(x)
        x = x.masked_fill(~mask.bool().unsqueeze(-1), 0.0)

        bsz = x.shape[0]
        author = self.author_token.expand(bsz, -1, -1)
        x = torch.cat([author, x], dim=1)
        author_mask = torch.ones((bsz, 1), dtype=torch.bool, device=mask.device)
        full_mask = torch.cat([author_mask, mask.bool()], dim=1)
        encoded = self.encoder(x, src_key_padding_mask=~full_mask)
        h_author = encoded[:, 0]
        cluster_logits = self.cluster_head(h_author)
        pred_cluster = torch.softmax(cluster_logits, dim=-1)
        pred_emb = F.normalize(self.embedding_head(h_author), p=2, dim=-1)
        return {
            "h_author": h_author,
            "cluster_logits": cluster_logits,
            "pred_cluster": pred_cluster,
            "pred_emb": pred_emb,
        }
