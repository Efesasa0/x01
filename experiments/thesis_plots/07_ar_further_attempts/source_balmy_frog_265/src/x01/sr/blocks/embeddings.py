import math

import torch
import torch.nn as nn
from einops import rearrange

from x01.sr.blocks.utils import variance_scaling_uniform_


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


class PatchEmbedding(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, patch_size: int, use_bias: bool):
        super().__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(
            in_channels=input_dim,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=use_bias,
        )
        variance_scaling_uniform_(self.conv.weight, scale=1 / 3)
        if use_bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, _ = x.shape
        num_patches = height // self.patch_size
        x = self.conv(x)
        x = rearrange(x, "b c h w -> b (h w) c", h=num_patches, w=num_patches)
        return x


class SinePositionEmbedding2D(nn.Module):
    def __init__(self, embedding_dim: int, spatial_size: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.spatial_size = spatial_size

    def get_1d_sine_pos_embed_from_grid(self, embedding_dim: int, grid_1d: torch.Tensor) -> torch.Tensor:
        assert embedding_dim % 2 == 0
        omega = torch.arange(embedding_dim // 2, dtype=torch.float32)
        omega /= embedding_dim / 2.0
        omega = 1.0 / 10000**omega

        grid_1d = grid_1d.reshape(-1)
        out = torch.einsum("m,d->md", grid_1d, omega)

        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)

        emb = torch.cat([emb_sin, emb_cos], dim=1)
        return emb

    def get_2d_sine_pos_embed_from_grid(self, embed_dim: int, grid: torch.Tensor) -> torch.Tensor:
        emb_h = self.get_1d_sine_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = self.get_1d_sine_pos_embed_from_grid(embed_dim // 2, grid[1])
        emb = torch.cat([emb_h, emb_w], dim=1)
        return emb

    def get_2d_sine_pos_embed(self, embed_dim: int, length: int) -> torch.Tensor:
        grid_size = int(length**0.5)

        grid_h = torch.arange(grid_size, dtype=torch.float32)
        grid_w = torch.arange(grid_size, dtype=torch.float32)
        grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
        grid = torch.stack(grid, dim=0)
        grid = grid.reshape(2, 1, grid_size, grid_size)
        pos_embed = self.get_2d_sine_pos_embed_from_grid(embed_dim, grid)
        return pos_embed

    def forward(self) -> torch.Tensor:
        emb = self.get_2d_sine_pos_embed(self.embedding_dim, self.spatial_size)[None, ...]
        return emb
