import torch
import torch.nn as nn
import torch.nn.functional as F

from x01.sr.blocks.utils import variance_scaling_uniform_


class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="circular",
            )
            variance_scaling_uniform_(self.conv.weight, scale=1 / 3)
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=0,
            )
            variance_scaling_uniform_(self.conv.weight, scale=1 / 3)
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            # JAX: jnp.pad(x, ((0,0),(0,1),(0,1),(0,0)), mode="wrap") then valid 3x3 stride-2.
            # Torch F.pad order is (W_left, W_right, H_top, H_bottom).
            x = F.pad(x, (0, 1, 0, 1), mode="circular")
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
        return x
