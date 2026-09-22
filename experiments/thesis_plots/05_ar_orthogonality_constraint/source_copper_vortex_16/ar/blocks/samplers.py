import torch
import torch.nn as nn
import torch.nn.functional as F

from ar.blocks.utils import variance_scaling_uniform_


class Downsample(nn.Module):
    def __init__(self, in_features: int, downsample_factor: int):
        super().__init__()
        if downsample_factor not in [
            2,
            4,
        ]:  # TODO: Why do this? because it is a design choice to end up at 32 latent_dim
            raise ValueError("downsample rates other than 2 and 4 are not supported.")
        self.downsample_factor = downsample_factor

        conv_factor = downsample_factor * 2 if downsample_factor == 4 else downsample_factor
        self.input_conv = nn.Conv2d(
            in_features, in_features // conv_factor, kernel_size=3, stride=1, padding=1, bias=False
        )
        variance_scaling_uniform_(self.input_conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_unshuffle(self.input_conv(x), self.downsample_factor)


class Upsample(nn.Module):
    def __init__(self, in_features: int, upsample_factor: int):
        super().__init__()
        if upsample_factor not in [2, 4]:
            raise ValueError("upsample rates other than 2 and 4 are not supported.")
        self.upsample_factor = upsample_factor

        conv_factor = upsample_factor * 2 if upsample_factor == 4 else upsample_factor
        self.input_conv = nn.Conv2d(
            in_features, in_features * conv_factor, kernel_size=3, stride=1, padding=1, bias=False
        )
        variance_scaling_uniform_(self.input_conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_shuffle(self.input_conv(x), self.upsample_factor)
