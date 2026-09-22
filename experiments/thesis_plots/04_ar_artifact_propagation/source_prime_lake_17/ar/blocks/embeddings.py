import torch
import torch.nn as nn

from ar.blocks.utils import variance_scaling_uniform_


class OverlapPatchEmbed(nn.Module):  # Preserving spatial, just initial non-linearity from 3 -> 4 channel dim at start
    def __init__(self, input_channels: int, embedding_dims: int, use_bias: bool):
        super().__init__()
        self.projection = nn.Conv2d(input_channels, embedding_dims, kernel_size=3, stride=1, padding=1, bias=use_bias)
        variance_scaling_uniform_(self.projection.weight)
        if use_bias:
            nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)
