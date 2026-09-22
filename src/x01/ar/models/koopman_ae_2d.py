from typing import Literal

import torch
import torch.nn as nn

from x01.ar.blocks.dynamics import DynamicsBackBlock, DynamicsBlock, LowRankDynamicsBackBlock, LowRankDynamicsBlock
from x01.ar.blocks.embeddings import OverlapPatchEmbed
from x01.ar.blocks.samplers import Downsample, Upsample
from x01.ar.blocks.transformer import TransformerBlock


class KoopmanAE2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dims: int = 4,
        num_blocks: tuple = (2, 2, 2, 2),
        spatial_scaling_schedule: tuple = (2, 4, 4),
        num_heads: tuple = (1, 2, 4, 4),
        ffn_expansion_factor: float = 2.66,
        use_bias: bool = False,
        use_norm_bias: bool = True,
        steps: int = 1,
        grid_info: bool = True,
        dynamics_mode: str = "separate",
        dynamics_rank: int | None = None,
    ):
        super().__init__()

        assert len(num_heads) == len(num_blocks) == len(spatial_scaling_schedule) + 1, (
            "the number of heads and"
            " blocks must match, and the"
            " spatial scaling schedule"
            " must be one element shorter"
            " than the number of blocks."
        )  # At last block we dont do spatial scaling with the Downsample

        self.steps = steps
        self.grid_dim = 2 if grid_info else 0
        self.latent_dim = dims * (2 ** len(spatial_scaling_schedule))

        tb_kwargs = dict(
            ffn_expansion_factor=ffn_expansion_factor,
            use_bias=use_bias,
            use_norm_bias=use_norm_bias,
        )

        # --- Encoder ---
        encoder_layers: list[nn.Module] = [
            OverlapPatchEmbed(
                input_channels=self.grid_dim + in_channels,
                embedding_dims=dims,
                use_bias=use_bias,
            )
        ]
        for i in range(len(num_blocks)):
            ch = dims * (2**i)
            for _ in range(num_blocks[i]):
                encoder_layers.append(TransformerBlock(dims=ch, num_heads=num_heads[i], **tb_kwargs))
            if i < len(spatial_scaling_schedule):
                encoder_layers.append(Downsample(in_features=ch, downsample_factor=spatial_scaling_schedule[i]))
        self.encoder = nn.Sequential(*encoder_layers)

        # --- Decoder ---
        decoder_layers: list[nn.Module] = []
        for i, factor in enumerate(spatial_scaling_schedule[::-1]):
            up_level = len(spatial_scaling_schedule) - i  # channel level going into upsample
            tb_level = up_level - 1  # channel level coming out of upsample
            decoder_layers.append(Upsample(in_features=dims * (2**up_level), upsample_factor=factor))
            for _ in range(num_blocks[tb_level]):
                decoder_layers.append(
                    TransformerBlock(dims=dims * (2**tb_level), num_heads=num_heads[tb_level], **tb_kwargs)
                )
        output_conv = nn.Conv2d(dims, out_channels, kernel_size=3, padding=1, bias=use_bias)
        decoder_layers.append(output_conv)
        self.decoder = nn.Sequential(*decoder_layers)

        # --- Dynamics ---
        latent_size = self.latent_dim * 2 * 2  # 32
        if dynamics_rank is None:
            self.dynamics = DynamicsBlock(latent_size)
            self.back_dynamics = DynamicsBackBlock(self.dynamics, mode=dynamics_mode)
        else:
            self.dynamics = LowRankDynamicsBlock(latent_size, rank=dynamics_rank)
            self.back_dynamics = LowRankDynamicsBackBlock(self.dynamics, mode=dynamics_mode)

    def _get_grid(self, H: int, W: int, batch_size: int, device: torch.device) -> torch.Tensor:
        # gridx varies along H (rows), gridy varies along W (columns) — fixes the Flax bug
        # where both axes were incorrectly varied along rows.
        gridx = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1).expand(batch_size, 1, H, W)
        gridy = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W).expand(batch_size, 1, H, W)
        return torch.cat([gridx, gridy], dim=1)  # (N, 2, H, W)

    def forward(
        self, x: torch.Tensor, mode: Literal["forward", "backward"], return_identity: bool = True
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        batch, _C, H, W = x.shape  # (2, 1, 64, 64)
        grid = self._get_grid(H, W, batch, x.device)
        z = self.encoder(torch.cat([x, grid], dim=1))  # (N, latent_dim, 2, 2) i.e., (N, 32, 2, 2)
        qt = z.reshape(batch, -1)  # (N, latent_dim * 2 * 2) i.e., (N, 128)

        if mode == "forward":
            out: list[torch.Tensor] = []
            for _ in range(self.steps):
                qt = self.dynamics(qt)  # Calls forward in time operator qt -> qt+1
                out.append(
                    self.decoder(qt.reshape(batch, self.latent_dim, 2, 2))
                )  # from (N, 128) to (N, 32, 2, 2) to feed to the decoder getting next xt+1
            return out, ([self.decoder(z)] if return_identity else [])

        if mode == "backward":
            out_back: list[torch.Tensor] = []
            for _ in range(self.steps):
                qt = self.back_dynamics(qt)  # Calls backward in time operator: qt -> qt-1
                out_back.append(self.decoder(qt.reshape(batch, self.latent_dim, 2, 2)))
            return out_back, ([self.decoder(z)] if return_identity else [])

        raise ValueError(f"mode must be 'forward' or 'backward', got {mode!r}")

    def encode(self, x: torch.Tensor, mode: Literal["forward", "backward", "identity"] = "identity") -> torch.Tensor:
        batch, _C, H, W = x.shape
        grid = self._get_grid(H, W, batch, x.device)
        z = self.encoder(torch.cat([x, grid], dim=1)).reshape(batch, -1)
        if mode == "forward":
            return self.dynamics(z)
        if mode == "backward":
            return self.back_dynamics(z)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z.reshape(z.shape[0], self.latent_dim, 2, 2))

    def rollout(self, x0: torch.Tensor, T: int, inference_mode: str = "sequential") -> torch.Tensor:
        # sequential: encode -> K -> decode every step (re-encodes the decoded pixel each iteration).
        # koopman: encode once, iterate K in latent, decode each step (no re-encoding).
        device = next(self.parameters()).device
        cur = x0.to(device)
        frames = []
        self.eval()
        with torch.no_grad():
            if inference_mode == "sequential":
                for _ in range(T):
                    pred, _ = self(cur, mode="forward", return_identity=False)
                    cur = pred[0]
                    frames.append(cur.detach().cpu())
            elif inference_mode == "koopman":
                z = self.encode(cur)
                for _ in range(T):
                    z = self.dynamics(z)
                    frames.append(self.decode(z).detach().cpu())
            else:
                raise ValueError(f"unknown inference_mode: {inference_mode!r} (allowed: 'sequential', 'koopman')")
        return torch.cat(frames, dim=0)

    def rollout_from_z(self, z0: torch.Tensor, T: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Strict Koopman continuation: iterate K in latent for T steps, decode each.
        # Returns (frames [T,C,H,W] cpu, z_final on device).
        frames = []
        self.eval()
        with torch.no_grad():
            z = z0
            for _ in range(T):
                z = self.dynamics(z)
                frames.append(self.decode(z).detach().cpu())
        return torch.cat(frames, dim=0), z
