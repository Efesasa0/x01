import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from x01.ar.blocks.utils import variance_scaling_uniform_


# Memory efficient transformer implementation.
# Uses transposed attention, calculates relationships across the feature channels.
# Takes input of size Batch, Height, Width, Channels
class MultiDConvHeadTransposedAttention(nn.Module):
    def __init__(self, dims: int, num_heads: int, use_bias: bool):
        super().__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(
            torch.ones(num_heads, 1, 1)
        )  # learnable parameter, how sharp attention scores to be.

        self.qkv = nn.Conv2d(dims, dims * 3, kernel_size=1, bias=use_bias)  # creates 3xdims for each q k v.
        variance_scaling_uniform_(self.qkv.weight)
        if use_bias:
            nn.init.zeros_(self.qkv.bias)

        self.qkv_downconv = nn.Conv2d(
            dims * 3, dims * 3, kernel_size=3, stride=1, padding=1, groups=dims * 3, bias=use_bias
        )
        variance_scaling_uniform_(self.qkv_downconv.weight)
        if use_bias:
            nn.init.zeros_(self.qkv_downconv.bias)

        self.out_projection = nn.Conv2d(dims, dims, kernel_size=1, bias=use_bias)
        variance_scaling_uniform_(self.out_projection.weight)
        if use_bias:
            nn.init.zeros_(self.out_projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        qkv = self.qkv_downconv(self.qkv(x))  # (b, 3c, h, w)
        q, k, v = qkv.chunk(3, dim=1)  # each (b, c, h, w)

        q = rearrange(q, "b (d heads) h w -> b (h w) d heads", heads=self.num_heads)
        k = rearrange(k, "b (d heads) h w -> b (h w) d heads", heads=self.num_heads)
        v = rearrange(v, "b (d heads) h w -> b heads d (h w)", heads=self.num_heads)

        # standardize along spatial dim, matching nnx.standardize(q, axis=1)
        # correction=0 matches JAX's population std (ddof=0)
        q = (q - q.mean(dim=1, keepdim=True)) / (q.std(dim=1, keepdim=True, correction=0) + 1e-5)
        k = (k - k.mean(dim=1, keepdim=True)) / (k.std(dim=1, keepdim=True, correction=0) + 1e-5)

        q = rearrange(q, "b N d heads -> b heads d N")
        k = rearrange(k, "b N d heads -> b heads N d")

        attn = (q @ k) * self.temperature  # (b, heads, d, d)
        attn = attn.softmax(dim=-1)
        out = attn @ v  # (b, heads, d, N)

        out = rearrange(out, "b heads d (h w) -> b (heads d) h w", h=h, w=w)
        return self.out_projection(out)


MDHA = MultiDConvHeadTransposedAttention


class GatedDconvFeedForwardNetwork(nn.Module):
    def __init__(self, dims: int, ffn_expansion_factor: float | int, use_bias: bool):
        super().__init__()
        self.hidden_dims = int(dims * ffn_expansion_factor)

        self.input_projection = nn.Conv2d(dims, self.hidden_dims * 2, kernel_size=1, bias=use_bias)
        variance_scaling_uniform_(self.input_projection.weight)
        if use_bias:
            nn.init.zeros_(self.input_projection.bias)

        self.down_conv = nn.Conv2d(
            self.hidden_dims * 2,
            self.hidden_dims * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.hidden_dims * 2,
            bias=use_bias,
        )
        variance_scaling_uniform_(self.down_conv.weight)
        if use_bias:
            nn.init.zeros_(self.down_conv.bias)

        self.out_projection = nn.Conv2d(self.hidden_dims, dims, kernel_size=1, bias=use_bias)
        variance_scaling_uniform_(self.out_projection.weight)
        if use_bias:
            nn.init.zeros_(self.out_projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down_conv(self.input_projection(x))
        x1, x2 = x.chunk(2, dim=1)
        return self.out_projection(F.gelu(x1, approximate="tanh") * x2)


GDFFN = GatedDconvFeedForwardNetwork


class TransformerBlock(nn.Module):
    def __init__(
        self, dims: int, num_heads: int, ffn_expansion_factor: float | int, use_bias: bool, use_norm_bias: bool
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dims, eps=1e-6, bias=use_norm_bias)
        self.attn = MDHA(dims=dims, num_heads=num_heads, use_bias=use_bias)
        self.norm2 = nn.LayerNorm(dims, eps=1e-6, bias=use_norm_bias)
        self.ffn = GDFFN(dims=dims, ffn_expansion_factor=ffn_expansion_factor, use_bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(rearrange(self.norm1(rearrange(x, "b c h w -> b h w c")), "b h w c -> b c h w"))
        x = x + self.ffn(rearrange(self.norm2(rearrange(x, "b c h w -> b h w c")), "b h w c -> b c h w"))
        return x
