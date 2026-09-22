import torch

from x01.ar.models.hs_loss import HsLoss
from x01.ar.models.koopman_ae_2d import KoopmanAE2D
from x01.ar.models.lp_loss import LpLoss
from x01.ar.models.mse_loss import MseLoss


def loss_koopman(
    model: KoopmanAE2D,
    x: torch.Tensor,
    y: torch.Tensor,
    lambda_fwd: float = 1.0,
    lambda_recon: float = 1.0,
    lambda_consist: float = 0.1,
    lambda_bwd: float = 0.1,
    lambda_latent_fwd: float = 0.0,
    lambda_latent_bwd: float = 0.0,
    lambda_ortho_a: float = 0.0,
    lambda_ortho_b: float = 0.0,
    lambda_consist_progressive: float = 0.0,
    loss_mode: str = "lp",
    lp_size_average: bool = False,
    hs_size_average: bool = False,
    mse_size_average: bool = True,
    latent_rollout_steps: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = torch.zeros((), device=x.device, dtype=x.dtype)
    if loss_mode == "lp":
        _lp = LpLoss(size_average=lp_size_average)
    elif loss_mode == "hs":
        _lp = HsLoss(size_average=hs_size_average)
    elif loss_mode == "mse":
        _lp = MseLoss(size_average=mse_size_average)
    else:
        raise ValueError(f"unknown loss_mode {loss_mode!r}, expected 'lp', 'hs', or 'mse'")

    # y is (B, n_rollout, C, H, W): future frames [x_{t+1}, ..., x_{t+n_rollout}].
    n_rollout = y.shape[1]
    assert n_rollout >= 1, f"y must have >= 1 rollout target, got shape {tuple(y.shape)}"

    forward_pred, forward_id = model(x, mode="forward", return_identity=(lambda_recon != 0.0))
    assert len(forward_pred) == n_rollout, f"model.steps ({len(forward_pred)}) must match y n_rollout ({n_rollout})"
    forward_loss = zero
    for k, pred_k in enumerate(forward_pred):
        forward_loss = forward_loss + _lp(pred_k, y[:, k])
    forward_loss = forward_loss / n_rollout

    reconstruction_loss = _lp(forward_id[0], x) if lambda_recon != 0.0 else zero

    if lambda_bwd != 0.0:
        # Seed backward from x_{t+n_rollout} = y[:, -1]. Predicted frames should trace back to x.
        seed_back = y[:, -1]
        backward_pred, _ = model(seed_back, mode="backward", return_identity=False)
        assert len(backward_pred) == n_rollout
        # target for backward_pred[k] is x_{t+n_rollout-1-k}
        back_targets = [y[:, n_rollout - 2 - k] if (n_rollout - 2 - k) >= 0 else x for k in range(n_rollout)]
        backward_loss = zero
        for k, pred_k in enumerate(backward_pred):
            backward_loss = backward_loss + _lp(pred_k, back_targets[k])
        backward_loss = backward_loss / n_rollout
    else:
        backward_loss = zero

    inner = model.module if hasattr(model, "module") else model
    K = inner.dynamics.matrix
    B = inner.back_dynamics.matrix

    if (lambda_latent_fwd != 0.0 or lambda_latent_bwd != 0.0) and latent_rollout_steps > 0:
        L = int(latent_rollout_steps)
        assert n_rollout >= L, f"y has {n_rollout} future frames but latent_rollout_steps={L} requires >= {L}"
        # Encode all L target future frames in one batch: [B, L, C, H, W] -> [B*L, C, H, W] -> [B*L, latent]
        B_size = y.shape[0]
        y_flat = y[:, :L].reshape(B_size * L, *y.shape[2:])
        z_true_flat = inner.encode(y_flat)
        z_true_seq = z_true_flat.reshape(B_size, L, -1)  # [B, L, latent_size]
    else:
        z_true_seq = None

    if lambda_latent_fwd != 0.0 and latent_rollout_steps > 0:
        L = int(latent_rollout_steps)
        z = inner.encode(x)
        latent_fwd_loss = zero
        for k in range(L):
            z = inner.dynamics(z)
            latent_fwd_loss = latent_fwd_loss + ((z - z_true_seq[:, k]) ** 2).mean()
        latent_fwd_loss = latent_fwd_loss / L
    else:
        latent_fwd_loss = zero

    if lambda_latent_bwd != 0.0 and latent_rollout_steps > 0:
        L = int(latent_rollout_steps)
        # seed backward from encode(x_{t+L}) = z_true_seq[:, L-1]. Predict z at (L-1-k-1) for step k=0..L-1.
        z = z_true_seq[:, L - 1]
        z_x = inner.encode(x)  # target for the final step
        latent_bwd_loss = zero
        for k in range(L):
            z = inner.back_dynamics(z)
            idx = L - 2 - k  # index in z_true_seq of the target for this step
            target = z_true_seq[:, idx] if idx >= 0 else z_x
            latent_bwd_loss = latent_bwd_loss + ((z - target) ** 2).mean()
        latent_bwd_loss = latent_bwd_loss / L
    else:
        latent_bwd_loss = zero

    if lambda_consist != 0.0:
        consistency_loss = (torch.trace(K @ B) / (torch.trace(K) * torch.trace(B)) - 1).abs()
    else:
        consistency_loss = zero

    if lambda_ortho_a != 0.0:
        eye_a = torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
        ortho_a_loss = ((K.T @ K - eye_a) ** 2).sum()
    else:
        ortho_a_loss = zero

    if lambda_ortho_b != 0.0:
        eye_b = torch.eye(B.shape[0], device=B.device, dtype=B.dtype)
        ortho_b_loss = ((B.T @ B - eye_b) ** 2).sum()
    else:
        ortho_b_loss = zero

    if lambda_consist_progressive != 0.0:
        AB = K @ B
        BA = B @ K
        d = K.shape[0]
        consist_progressive_loss = zero
        for k in range(1, d + 1):
            eye_k = torch.eye(k, device=K.device, dtype=K.dtype)
            err_ab = AB[:k, :k] - eye_k
            err_ba = BA[:k, :k] - eye_k
            consist_progressive_loss = consist_progressive_loss + ((err_ab**2).sum() + (err_ba**2).sum()) / (2 * k)
    else:
        consist_progressive_loss = zero

    loss = (
        lambda_fwd * forward_loss
        + lambda_recon * reconstruction_loss
        + lambda_bwd * backward_loss
        + lambda_latent_fwd * latent_fwd_loss
        + lambda_latent_bwd * latent_bwd_loss
        + lambda_consist * consistency_loss
        + lambda_ortho_a * ortho_a_loss
        + lambda_ortho_b * ortho_b_loss
        + lambda_consist_progressive * consist_progressive_loss
    )

    return loss, {
        "loss": loss,
        "forward_loss": forward_loss,
        "backward_loss": backward_loss,
        "reconstruction_loss": reconstruction_loss,
        "latent_fwd_loss": latent_fwd_loss,
        "latent_bwd_loss": latent_bwd_loss,
        "consistency_loss": consistency_loss,
        "ortho_a_loss": ortho_a_loss,
        "ortho_b_loss": ortho_b_loss,
        "consist_progressive_loss": consist_progressive_loss,
    }
