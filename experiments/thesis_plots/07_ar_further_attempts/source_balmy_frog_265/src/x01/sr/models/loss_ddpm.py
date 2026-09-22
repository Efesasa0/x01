from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from x01.sr.models.diffusion_mngr import DiffusionManager


def loss_ddpm(
    manager: DiffusionManager,
    model: nn.Module,
    x: Tensor,
    condition: Tensor | None = None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """DDPM training loss: MSE between true and predicted noise (epsilon-prediction).

    Note: this is the DDPM training objective. DDIM is the sampler used at inference time.

    Samples t ~ U[0, T), epsilon ~ N(0, I), forms x_t = sqrt(alpha_bar_t) * x + sqrt(1 - alpha_bar_t) * eps,
    and asks the model to predict eps. Applies classifier-free-guidance dropout to `condition`
    at rate `manager.condition_dropout_rate` (no-op if rate is 0 or condition is None).
    """
    b = x.shape[0]
    device = x.device

    t = torch.randint(0, manager.num_diffusion_time_steps, (b,), device=device, dtype=torch.long)
    a = manager.compute_alpha(t)
    eps = torch.randn_like(x)
    x_t = a.sqrt() * x + (1.0 - a).sqrt() * eps

    cond = condition
    if cond is not None and manager.condition_dropout_rate > 0.0:
        drop = torch.rand(b, device=device) < manager.condition_dropout_rate
        if drop.any():
            cond = cond.clone()
            cond[drop] = 0.0

    eps_pred = model(x_t, t, cond) * manager.mask
    eps_target = eps * manager.mask

    loss = (eps_pred - eps_target).square().mean()
    return loss, {"loss": loss}
