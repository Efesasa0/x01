import math

import torch
import torch.nn as nn


class TimeStepEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, with_mlp_block: bool, mlp_hidden_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.with_mlp_block = with_mlp_block

        if with_mlp_block:
            self.mlp = nn.Sequential(
                nn.Linear(embedding_dim, mlp_hidden_dim),
                nn.SiLU(),
                nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            )
            nn.init.normal_(self.mlp[0].weight, std=0.02)
            nn.init.zeros_(self.mlp[0].bias)
            nn.init.normal_(self.mlp[2].weight, std=0.02)
            nn.init.zeros_(self.mlp[2].bias)

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        assert t.dim() == 1
        emb = self.timestep_embedding(t, self.embedding_dim)
        if self.with_mlp_block:
            emb = self.mlp(emb)
        return emb
