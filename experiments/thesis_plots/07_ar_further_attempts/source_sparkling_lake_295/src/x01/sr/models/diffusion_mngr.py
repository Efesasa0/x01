from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from tqdm.auto import tqdm


class DiffusionManager(nn.Module):
    """DDIM scheduler + sampler. Schedule lives on buffers so it moves with .to(device).

    `alphas_bar` is length T+1 with `alphas_bar[t+1] = prod_{k<=t}(1 - beta_k)` and
    `alphas_bar[0] = 1`, matching the JAX `_compute_alpha` trick of prepending a zero
    beta before cumprod so `bar_alpha_{-1} = 1` is well-defined at the final DDIM step.
    """

    betas: Tensor
    alphas_bar: Tensor
    mask: Tensor

    def __init__(
        self,
        beta_start: float,
        beta_end: float,
        num_diffusion_time_steps: int,
        condition_dropout_rate: float = 0.0,
        x_scale: float = 1.0,
        x_offset: float = 0.0,
        mask_path: str | None = None,
    ):
        super().__init__()
        self.num_diffusion_time_steps = num_diffusion_time_steps
        self.condition_dropout_rate = condition_dropout_rate
        self.x_scale = x_scale
        self.x_offset = x_offset

        betas = torch.linspace(beta_start, beta_end, num_diffusion_time_steps, dtype=torch.float32)
        alphas_bar = torch.cat([torch.ones(1), (1.0 - betas).cumprod(dim=0)])
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_bar", alphas_bar)

        if mask_path is not None:
            m = np.load(mask_path).astype(np.float32)
            mask = torch.from_numpy(m).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        else:
            mask = torch.ones(1, dtype=torch.float32)
        self.register_buffer("mask", mask)

    def compute_alpha(self, t: Tensor) -> Tensor:
        """ᾱ_t broadcast-ready for NCHW: shape (B, 1, 1, 1). Accepts t in [-1, T-1]."""
        return self.alphas_bar[t.long() + 1].view(-1, 1, 1, 1)

    @torch.no_grad()
    def infer_fn(
        self,
        x_init: Tensor,
        model: nn.Module,
        inference_steps: int,
        condition: Tensor | None = None,
        pbar: bool = False,
    ) -> Tensor:
        """SR3-style conditional DDIM sampling.

        `x_init` is the starting state — pure Gaussian noise for SR3 (the noising step below
        with `a_init = bar_alpha_{T-1} ~ 0` collapses to noise anyway). Conditioning flows
        through `condition`. DDIM places `inference_steps` evenly-spaced steps across the
        full [0, T) schedule.
        """
        T = self.num_diffusion_time_steps
        device = x_init.device
        n = x_init.shape[0]

        eps = torch.randn_like(x_init)
        a_init = self.alphas_bar[T]
        x = x_init * a_init.sqrt() + eps * (1.0 - a_init).sqrt()
        x = x * self.mask

        skip = T // inference_steps
        seq = torch.arange(0, T, skip, device=device, dtype=torch.long)
        seq_next = torch.cat([torch.full((1,), -1, device=device, dtype=torch.long), seq[:-1]])

        iterator: Iterable = zip(reversed(seq.tolist()), reversed(seq_next.tolist()))
        if pbar:
            iterator = tqdm(iterator, total=int(seq.numel()))

        x0_t = x
        for i, j in iterator:
            t = torch.full((n,), i, dtype=torch.long, device=device)
            t_next = torch.full((n,), j, dtype=torch.long, device=device)

            at = self.compute_alpha(t)
            at_next = self.compute_alpha(t_next)

            et = model(x, t, condition) * self.mask
            x0_t = (x - et * (1.0 - at).sqrt()) / at.sqrt()
            x = (at_next.sqrt() * x0_t + (1.0 - at_next).sqrt() * et) * self.mask

        return x0_t
