import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from x01.ar.models.koopman_ae_2d import KoopmanAE2D


def log_koopman_spectrum(ar_model: KoopmanAE2D) -> None:
    """Print min/mean/max of |eigenvalues| and singular values of the Koopman operator matrix."""
    with torch.no_grad():
        A = ar_model.dynamics.matrix.detach().cpu()
        eig_abs = torch.linalg.eigvals(A).abs()
        svals = torch.linalg.svdvals(A)
    print(
        f"koopman |eigval|: min={float(eig_abs.min()):.4f} "
        f"mean={float(eig_abs.mean()):.4f} max={float(eig_abs.max()):.4f}",
        flush=True,
    )
    print(
        f"koopman svd     : min={float(svals.min()):.4f} mean={float(svals.mean()):.4f} max={float(svals.max()):.4f}",
        flush=True,
    )


def log_latent_norm_drift(
    ar_model: KoopmanAE2D,
    z0: torch.Tensor,
    T: int,
    tag: str,
    save_dir: str | None,
    gt_lr_frames: torch.Tensor | None = None,
    ar_standardize: bool = False,
    ar_std: float = 1.0,
    device: torch.device | None = None,
) -> None:
    """Pure-Koopman probe: iterate K in latent T times from z0, plot ||z_t||_2.

    If gt_lr_frames [T+1,1,H,W] is given, overlay ||E(x_t)||_2.
    """
    ar_model.eval()
    model_norms = []
    with torch.no_grad():
        z = z0
        model_norms.append(float(torch.linalg.vector_norm(z).item()))
        for _ in range(T):
            z = ar_model.dynamics(z)
            model_norms.append(float(torch.linalg.vector_norm(z).item()))
    model_arr = np.asarray(model_norms)

    true_arr = None
    if gt_lr_frames is not None:
        assert gt_lr_frames.shape[0] == T + 1, f"gt_lr_frames has {gt_lr_frames.shape[0]} frames, need {T + 1}"
        assert device is not None, "device required when gt_lr_frames is provided"
        x = gt_lr_frames.to(device)
        if ar_standardize:
            x = x / ar_std
        with torch.no_grad():
            z_all = ar_model.encode(x)
            true_arr = torch.linalg.vector_norm(z_all, dim=1).cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(np.arange(T + 1), model_arr, color="tomato", linewidth=1.5, label="model  ||A^t z0||")
    if true_arr is not None:
        ax.plot(np.arange(T + 1), true_arr, color="steelblue", linewidth=1.5, label="true  ||E(x_t)||")
        ax.legend(fontsize=9)
    ax.set_xlabel("latent step t", fontsize=10)
    ax.set_ylabel("||z||_2", fontsize=10)
    ax.set_title(f"{tag} latent norm (koopman)", fontsize=10)
    fig.tight_layout()
    if wandb.run is not None:
        wandb.log({f"{tag}/latent_norm_drift": wandb.Image(fig)})
    if save_dir is not None:
        fig.savefig(f"{save_dir}/{tag}_latent_norm_drift.png", dpi=120)
    plt.close(fig)


def log_latent_trajectory_error(
    ar_model: KoopmanAE2D,
    z0: torch.Tensor,
    gt_lr_frames: torch.Tensor,
    ar_standardize: bool,
    ar_std: float,
    device: torch.device,
    tag: str,
    save_dir: str | None,
) -> None:
    """||A^t z0 - E(x_t)||_2 for t=0..T. gt_lr_frames [T+1, 1, H_lr, W_lr] raw scale is the true LR trajectory."""
    T_plus_1 = gt_lr_frames.shape[0]
    T = T_plus_1 - 1
    x = gt_lr_frames.to(device)
    if ar_standardize:
        x = x / ar_std
    ar_model.eval()
    distances = []
    with torch.no_grad():
        z_true_all = ar_model.encode(x)  # [T+1, latent_size]
        z_model = z0
        for t in range(T + 1):
            distances.append(float(torch.linalg.vector_norm(z_model - z_true_all[t : t + 1]).item()))
            if t < T:
                z_model = ar_model.dynamics(z_model)
    distances_arr = np.asarray(distances)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(np.arange(T + 1), distances_arr, color="purple", linewidth=1.5)
    ax.set_xlabel("step t within part", fontsize=10)
    ax.set_ylabel("||A^t z0 - E(x_t)||_2", fontsize=10)
    ax.set_title(f"{tag} latent trajectory error (koopman)", fontsize=10)
    fig.tight_layout()
    if wandb.run is not None:
        wandb.log({f"{tag}/latent_trajectory_error": wandb.Image(fig)})
    if save_dir is not None:
        fig.savefig(f"{save_dir}/{tag}_latent_trajectory_error.png", dpi=120)
    plt.close(fig)


def log_encoder_manifold_residual(
    ar_model: KoopmanAE2D,
    z0: torch.Tensor,
    T: int,
    tag: str,
    save_dir: str | None,
) -> None:
    """||E(D(z_t)) - z_t||_2 along the pure-Koopman chain z_0..z_T.

    Measures how far each z_t drifts off the encoder manifold.
    """
    ar_model.eval()
    residuals = []
    with torch.no_grad():
        z = z0
        for t in range(T + 1):
            x_hat = ar_model.decode(z)
            z_hat = ar_model.encode(x_hat)
            residuals.append(float(torch.linalg.vector_norm(z_hat - z).item()))
            if t < T:
                z = ar_model.dynamics(z)
    residuals_arr = np.asarray(residuals)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(np.arange(T + 1), residuals_arr, color="steelblue", linewidth=1.5)
    ax.set_xlabel("latent step t", fontsize=10)
    ax.set_ylabel("||E(D(z_t)) - z_t||_2", fontsize=10)
    ax.set_title(f"{tag} encoder-manifold residual (koopman)", fontsize=10)
    fig.tight_layout()
    if wandb.run is not None:
        wandb.log({f"{tag}/encoder_manifold_residual": wandb.Image(fig)})
    if save_dir is not None:
        fig.savefig(f"{save_dir}/{tag}_encoder_manifold_residual.png", dpi=120)
    plt.close(fig)
