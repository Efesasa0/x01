from typing import Dict, Tuple

import torch

from x01.ar.models.koopman_ae_2d import KoopmanAE2D


def loss_koopman(
    model: KoopmanAE2D,
    x: torch.Tensor,
    y: torch.Tensor,
    lambda_recon: float = 1.0,
    lambda_ortho: float = 0.1,
    lambda_bwd: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Koopman loss: recon + pred (fwd+bwd) + Frobenius orthogonality penalty (Eq. 8-9)."""
    forward_pred, forward_id = model(x, mode="forward")
    backward_pred, backward_id = model(y, mode="backward")

    reconstruction_loss = (x - forward_id[0]).square().mean() + (y - backward_id[0]).square().mean()  # MSE Loss
    forward_loss = (y - forward_pred[0]).square().mean()  # Recon Loss
    backward_loss = (x - backward_pred[0]).square().mean()  # Recon Loss

    inner = model.module if hasattr(model, "module") else model
    K = inner.dynamics.dynamics
    I = torch.eye(K.shape[0], device=K.device)
    ortho_loss = (K.T @ K - I).pow(2).sum()

    loss = forward_loss + lambda_recon * reconstruction_loss + lambda_bwd * backward_loss + lambda_ortho * ortho_loss

    return loss, {
        "loss": loss,
        "forward_loss": forward_loss,
        "backward_loss": backward_loss,
        "reconstruction_loss": reconstruction_loss,
        "ortho_loss": ortho_loss,
    }
