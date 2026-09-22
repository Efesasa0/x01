import torch
import torch.nn as nn


class OverlapPatchEmbed(nn.Module):  # linear 2d convolution from (B, input_channels, H, W) to (B, embedding_dims, H, W)
    def __init__(self, input_channels: int, embedding_dims: int, use_bias: bool):
        super().__init__()
        self.projection = nn.Conv2d(input_channels, embedding_dims, kernel_size=3, stride=1, padding=1, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)
