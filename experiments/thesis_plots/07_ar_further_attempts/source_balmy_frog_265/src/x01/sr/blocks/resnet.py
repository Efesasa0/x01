import torch
import torch.nn as nn

from x01.sr.blocks.utils import lecun_normal_, variance_scaling_uniform_


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_conv_shortcut: bool,
        dropout: float,
        t_embed_channels: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
        )
        variance_scaling_uniform_(self.conv1.weight, scale=1 / 3)
        nn.init.zeros_(self.conv1.bias)

        self.t_embed_proj = nn.Linear(t_embed_channels, out_channels)
        lecun_normal_(self.t_embed_proj.weight)
        nn.init.zeros_(self.t_embed_proj.bias)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
        )
        variance_scaling_uniform_(self.conv2.weight, scale=1 / 3)
        nn.init.zeros_(self.conv2.bias)

        if in_channels != out_channels:
            if use_conv_shortcut:
                self.shortcut = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="circular",
                )
            else:
                self.shortcut = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            variance_scaling_uniform_(self.shortcut.weight, scale=1 / 3)
            nn.init.zeros_(self.shortcut.bias)
        else:
            self.shortcut = nn.Identity()

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        t = self.act(t_embed)
        t = self.t_embed_proj(t)
        t = t[:, :, None, None]  # (b, out_c) -> (b, out_c, 1, 1) broadcast over NCHW
        h = h + t

        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return self.shortcut(x) + h
