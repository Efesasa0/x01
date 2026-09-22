from typing import Dict, Tuple

import torch

from ar.models.koopman_ae_2d import KoopmanAE2D

LAMBDA_ORTH = 1e-2


def loss_koopman(
    model: KoopmanAE2D,
    x: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """4-term Koopman loss: forward + backward + reconstruction + consistency + orthogonality."""
    forward_pred, forward_id = model(x, mode="forward")
    backward_pred, backward_id = model(y, mode="backward")

    reconstruction_loss = (x - forward_id[0]).square().mean() + (y - backward_id[0]).square().mean()
    forward_loss = (y - forward_pred[0]).square().mean()
    backward_loss = (x - backward_pred[0]).square().mean()

    K = model.dynamics.dynamics
    K_inv = model.back_dynamics.back_dynamics
    consist_loss = (torch.trace(K @ K_inv) / (torch.trace(K) * torch.trace(K_inv)) - 1).abs()
    I = torch.eye(K.shape[0], device=K.device)
    orth_loss = (K.T @ K - I).pow(2).sum()

    loss = forward_loss + reconstruction_loss + consist_loss + backward_loss + LAMBDA_ORTH * orth_loss

    return loss, {
        "loss": loss,
        "forward_loss": forward_loss,
        "backward_loss": backward_loss,
        "reconstruction_loss": reconstruction_loss,
        "consist_loss": consist_loss,
        "orth_loss": orth_loss,
    }
