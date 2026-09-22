import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from x01.sr.blocks.utils import variance_scaling_uniform_


class ConvAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        def make_1x1():
            conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
            variance_scaling_uniform_(conv.weight, scale=1 / 3)
            nn.init.zeros_(conv.bias)
            return conv

        self.q = make_1x1()
        self.k = make_1x1()
        self.v = make_1x1()
        self.o = make_1x1()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, c, h, w)
        b, c, height, width = x.shape

        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b c (h w)")
        v = rearrange(v, "b c h w -> b c (h w)")

        w = torch.matmul(k, q) / math.sqrt(q.shape[-1])
        w = F.softmax(w, dim=-1)
        w = w.transpose(1, 2)

        # JAX does `.reshape((b, H, W, c))` on a (b, c, hw) tensor with no transpose;
        # we replicate that exact buffer reinterpretation in torch's NCHW layout.
        h = torch.matmul(w, v)
        h = rearrange(h, "b c (height width) -> b c height width", height=height, width=width)
        h = self.o(h)

        return x + h
