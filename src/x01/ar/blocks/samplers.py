import torch
import torch.nn as nn
import torch.nn.functional as F


class Downsample(nn.Module):  # from (B, C, H, W) to (B, 2C, H/f, W/f) only if f=2 or 4
    def __init__(self, in_features: int, downsample_factor: int):
        super().__init__()
        if downsample_factor not in [
            2,
            4,
        ]:
            raise ValueError("downsample rates other than 2 and 4 are not supported.")
        self.downsample_factor = downsample_factor

        conv_factor = downsample_factor**2 // 2
        self.input_conv = nn.Conv2d(
            in_features, in_features // conv_factor, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_unshuffle(self.input_conv(x), self.downsample_factor)


class Upsample(nn.Module):  # from (B, C, H, W) to (B, C/2, fH, fW) only if f=2 or 4
    def __init__(self, in_features: int, upsample_factor: int):
        super().__init__()
        if upsample_factor not in [2, 4]:
            raise ValueError("upsample rates other than 2 and 4 are not supported.")
        self.upsample_factor = upsample_factor

        conv_factor = upsample_factor**2 // 2
        self.input_conv = nn.Conv2d(
            in_features, in_features * conv_factor, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_shuffle(self.input_conv(x), self.upsample_factor)
