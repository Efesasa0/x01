import torch
import torch.nn as nn
import torch.nn.functional as F

from x01.sr.blocks.attention import ConvAttention
from x01.sr.blocks.embeddings import TimeStepEmbedding
from x01.sr.blocks.resnet import ResNetBlock
from x01.sr.blocks.sampler import Downsample, Upsample
from x01.sr.blocks.utils import lecun_normal_, variance_scaling_uniform_


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dims: int,
        channel_multipliers: tuple[int, ...],
        num_res_blocks: int,
        attention_resolutions: int | tuple[int, ...],
        image_resolution: int,
        dropout_rate: float,
        resample_with_conv: bool,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dims = latent_dims
        self.channel_multipliers = tuple(channel_multipliers)
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = (
            (attention_resolutions,) if isinstance(attention_resolutions, int) else tuple(attention_resolutions)
        )
        self.image_resolution = image_resolution
        self.dropout_rate = dropout_rate
        self.resample_with_conv = resample_with_conv

        self.time_embedding_dim = latent_dims * 4
        self.num_resolutions = len(self.channel_multipliers)

        # time embedding
        self.time_embed = nn.Sequential(
            TimeStepEmbedding(latent_dims, with_mlp_block=False, mlp_hidden_dim=0),
            _make_linear(latent_dims, self.time_embedding_dim),
            nn.SiLU(),
            _make_linear(self.time_embedding_dim, self.time_embedding_dim),
        )

        # condition channel embedding
        self.channel_embed = nn.Sequential(
            _make_conv(in_channels, latent_dims, kernel_size=1, padding=0, circular=False),
            nn.GELU(approximate="tanh"),
            _make_conv(latent_dims, latent_dims, kernel_size=3, padding=1, circular=True),
        )

        self.conv_in = _make_conv(in_channels, latent_dims, kernel_size=3, padding=1, circular=True)
        self.conv_combine = _make_conv(latent_dims * 2, latent_dims, kernel_size=1, padding=0, circular=False)

        # down
        current_resolution = image_resolution
        in_channel_multipliers = (1,) + self.channel_multipliers[:-1]
        self.down = nn.ModuleList()
        block_in = latent_dims
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = latent_dims * in_channel_multipliers[i_level]
            block_out = latent_dims * self.channel_multipliers[i_level]
            for _ in range(num_res_blocks):
                block.append(
                    ResNetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        use_conv_shortcut=False,
                        dropout=dropout_rate,
                        t_embed_channels=self.time_embedding_dim,
                    )
                )
                block_in = block_out
                if current_resolution in self.attention_resolutions:
                    attn.append(ConvAttention(in_channels=block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(in_channels=block_in, with_conv=resample_with_conv)
                current_resolution = current_resolution // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResNetBlock(
            in_channels=block_in,
            out_channels=block_in,
            use_conv_shortcut=False,
            dropout=dropout_rate,
            t_embed_channels=self.time_embedding_dim,
        )
        self.mid.attn_1 = ConvAttention(in_channels=block_in)
        self.mid.block_2 = ResNetBlock(
            in_channels=block_in,
            out_channels=block_in,
            use_conv_shortcut=False,
            dropout=dropout_rate,
            t_embed_channels=self.time_embedding_dim,
        )

        # up
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = latent_dims * self.channel_multipliers[i_level]
            skip_in = latent_dims * self.channel_multipliers[i_level]
            for i_block in range(num_res_blocks + 1):
                if i_block == num_res_blocks:
                    skip_in = latent_dims * in_channel_multipliers[i_level]
                block.append(
                    ResNetBlock(
                        in_channels=block_in + skip_in,
                        out_channels=block_out,
                        use_conv_shortcut=False,
                        dropout=dropout_rate,
                        t_embed_channels=self.time_embedding_dim,
                    )
                )
                block_in = block_out
                if current_resolution in self.attention_resolutions:
                    attn.append(ConvAttention(in_channels=block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.up_sample = Upsample(in_channels=block_in, with_conv=resample_with_conv)
                current_resolution *= 2
            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = _make_conv(block_in, out_channels, kernel_size=3, padding=1, circular=True)

    def forward(
        self,
        x: torch.Tensor,  # (B, C, H, W)
        t: torch.Tensor,  # (B,)
        condition: torch.Tensor | None = None,  # (B, C, H, W)
    ) -> torch.Tensor:
        assert (
            x.shape[-1] == x.shape[-2] == self.image_resolution
        ), f"expecting spatial dims to be the same, got shape {tuple(x.shape)}"

        t_embed = self.time_embed(t)

        x = self.conv_in(x)
        if condition is not None:
            cond_emb = self.channel_embed(condition)
        else:
            cond_emb = torch.zeros_like(x)
        x = torch.cat((x, cond_emb), dim=1)

        # downsampling
        hs = [self.conv_combine(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], t_embed)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, t_embed)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, t_embed)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), t_embed)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].up_sample(h)

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


def _make_linear(in_features: int, out_features: int) -> nn.Linear:
    lin = nn.Linear(in_features, out_features)
    lecun_normal_(lin.weight)
    nn.init.zeros_(lin.bias)
    return lin


def _make_conv(in_c: int, out_c: int, kernel_size: int, padding: int, circular: bool) -> nn.Conv2d:
    conv = nn.Conv2d(
        in_c,
        out_c,
        kernel_size=kernel_size,
        padding=padding,
        padding_mode="circular" if circular else "zeros",
    )
    variance_scaling_uniform_(conv.weight, scale=1 / 3)
    nn.init.zeros_(conv.bias)
    return conv
