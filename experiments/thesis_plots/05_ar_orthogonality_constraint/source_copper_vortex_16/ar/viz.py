import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import wandb
from ar.models.koopman_ae_2d import KoopmanAE2D

_CMAP_FIELD = "RdBu_r"
_CMAP_ERROR = "inferno"


# Single step image plot
def log_images(
    model: KoopmanAE2D,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    device: torch.device,
    epoch: int,
    std: float,
    tag: str = "val",
) -> None:
    """Single-step prediction: x_t (input)| x_{t+1} (pred)| x_{t+1} (truth)| |error|."""
    x = x_probe.to(device)
    y = y_probe.to(device)
    with torch.no_grad():
        pred, _ = model(x, mode="forward")
    pred = pred[0]

    x_np = (x[0, 0] * std).cpu().numpy()
    pred_np = (pred[0, 0] * std).cpu().numpy()
    y_np = (y[0, 0] * std).cpu().numpy()
    err_np = np.abs(pred_np - y_np)

    vmin = min(x_np.min(), y_np.min())
    vmax = max(x_np.max(), y_np.max())

    fig, axes = plt.subplots(1, 4, figsize=(14, 3), gridspec_kw={"wspace": 0.05})
    for i, (ax, img, title, kwargs) in enumerate(
        zip(
            axes,
            [x_np, pred_np, y_np, err_np],
            ["x_t (input)", "x_{t+1} (pred)", "x_{t+1} (truth)", "|error|"],
            [
                dict(cmap=_CMAP_FIELD, vmin=vmin, vmax=vmax),
                dict(cmap=_CMAP_FIELD, vmin=vmin, vmax=vmax),
                dict(cmap=_CMAP_FIELD, vmin=vmin, vmax=vmax),
                dict(cmap=_CMAP_ERROR, vmin=0, vmax=err_np.max()),
            ],
        )
    ):
        im = ax.imshow(img, **kwargs)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        if i == 3:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("left", size="5%", pad=0.08)
    fig.colorbar(axes[0].images[0], cax=cax, label="vorticity")
    cax.yaxis.set_ticks_position("left")
    cax.yaxis.set_label_position("left")

    fig.suptitle(f"{tag} — epoch {epoch}", fontsize=10)
    wandb.log({f"{tag}/images": wandb.Image(fig), "epoch": epoch})
    plt.close(fig)


# Rollout image plot
def log_rollout(
    model: KoopmanAE2D,
    gt_frames: torch.Tensor,
    device: torch.device,
    epoch: int,
    std: float,
    tag: str = "val",
    n_cols: int = 10,
) -> None:
    """Multi-step AR rollout: 3 rows (GT / pred / |error|) x n_cols timesteps.
    Rollout starts from gt_frames[0], stays in latent space for all T steps.
    """
    T = gt_frames.shape[0]
    timesteps = np.linspace(0, T - 1, n_cols, dtype=int)

    x0 = gt_frames[0:1].to(device)  # (1, 1, H, W)

    with torch.no_grad():
        z = model.encode(x0)  # (1, latent_dim*2*2)
        pred_all = []
        for _ in range(T):
            pred_all.append(model.decode(z).cpu())  # (1, 1, H, W)
            z = model.dynamics(z)  # step latent forward
    pred_all = torch.cat(pred_all, dim=0)  # (T, 1, H, W)

    gt_np = (gt_frames[timesteps, 0] * std).numpy()  # (n_cols, H, W)
    pred_np = (pred_all[timesteps, 0] * std).numpy()
    err_np = np.abs(pred_np - gt_np)

    vmin = min(gt_np.min(), pred_np.min())
    vmax = max(gt_np.max(), pred_np.max())
    err_max = float(err_np.max())

    fig = plt.figure(figsize=(18, 1.8 * 3))
    gs = GridSpec(
        3,
        n_cols + 1,
        figure=fig,
        wspace=0.05,
        hspace=0.1,
        width_ratios=[1] * n_cols + [0.04],
    )
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(3)])
    cax_field = fig.add_subplot(gs[:2, -1])
    cax_err = fig.add_subplot(gs[2, -1])

    im_field = None
    im_err = None

    for col, t in enumerate(timesteps):
        for row, (img, cmap, vlo, vhi) in enumerate(
            zip(
                [gt_np[col], pred_np[col], err_np[col]],
                [_CMAP_FIELD, _CMAP_FIELD, _CMAP_ERROR],
                [vmin, vmin, 0],
                [vmax, vmax, err_max],
            )
        ):
            ax = axes[row, col]
            im = ax.imshow(img, cmap=cmap, vmin=vlo, vmax=vhi)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"t={t}", fontsize=8)
            if row == 1 and col == n_cols - 1:
                im_field = im
            if row == 2 and col == n_cols - 1:
                im_err = im

    for row, label in enumerate(["GT", "pred", "|error|"]):
        axes[row, 0].text(
            -0.05,
            0.5,
            label,
            transform=axes[row, 0].transAxes,
            va="center",
            ha="right",
            fontsize=9,
            rotation=90,
        )

    fig.colorbar(im_field, cax=cax_field, label="vorticity")
    fig.colorbar(im_err, cax=cax_err, label="|error|")
    fig.suptitle(f"{tag} rollout — epoch {epoch}", fontsize=10)

    wandb.log({f"{tag}/rollout": wandb.Image(fig), "epoch": epoch})
    plt.close(fig)


def _temporal_autocorr(frames: np.ndarray) -> np.ndarray:
    """R_uu(tau) for a single trajectory."""
    T = frames.shape[0]
    flat = frames.reshape(T, -1)  # (T, N)
    R = np.empty(T)
    for tau in range(T):
        n = T - tau
        R[tau] = (flat[:n] * flat[tau:]).sum(axis=1).mean()
    R /= R[0]
    return R


def log_autocorr(
    model: KoopmanAE2D,
    gt_trajectories: list[torch.Tensor],  # each (T, 1, H, W) normalized
    device: torch.device,
    epoch: int,
    tag: str = "val",
    save_path: str | None = None,
) -> None:
    """Temporal autocorrelation R_uu(t*) — GT vs AR rollout."""
    gt_R_list, pred_R_list = [], []

    for gt_frames in gt_trajectories:
        T = gt_frames.shape[0]
        x0 = gt_frames[0:1].to(device)

        with torch.no_grad():
            z = model.encode(x0)
            pred_frames = []
            for _ in range(T):
                pred_frames.append(model.decode(z).cpu())
                z = model.dynamics(z)
        pred_frames = torch.cat(pred_frames, dim=0)  # (T, 1, H, W)

        gt_R_list.append(_temporal_autocorr(gt_frames[:, 0].numpy()))
        pred_R_list.append(_temporal_autocorr(pred_frames[:, 0].numpy()))

    gt_R = np.stack(gt_R_list)  # (n_traj, T)
    pred_R = np.stack(pred_R_list)

    T = gt_R.shape[1]
    t_star = np.linspace(0, 1, T)
    gt_mean, gt_std = gt_R.mean(0), gt_R.std(0)
    pred_mean, pred_std = pred_R.mean(0), pred_R.std(0)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.plot(t_star, gt_mean, color="steelblue", linewidth=1.5, label="GT")
    ax.fill_between(t_star, gt_mean - gt_std, gt_mean + gt_std, color="steelblue", alpha=0.2)
    ax.plot(t_star, pred_mean, color="tomato", linewidth=1.5, label="model")
    ax.fill_between(t_star, pred_mean - pred_std, pred_mean + pred_std, color="tomato", alpha=0.2)
    ax.set_xlabel("t*", fontsize=10)
    ax.set_ylabel("R_uu(t*)", fontsize=10)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)
    ax.set_title(f"{tag} autocorrelation — epoch {epoch}", fontsize=10)
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved → {save_path}")
    else:
        wandb.log({f"{tag}/autocorr": wandb.Image(fig), "epoch": epoch})
    plt.close(fig)


#  Energy spectrum


def _energy_spectrum(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radial kinetic energy spectrum averaged over T frames.

    frames: (T, H, W)
    Returns (k_bins, E_k). Each k bin is the mean power across all pixels
    at that radial wavenumber, averaged over all T frames.
    """
    T, H, W = frames.shape
    kx = np.fft.fftfreq(W) * W
    ky = np.fft.fftfreq(H) * H
    KX, KY = np.meshgrid(kx, ky)
    K_rad = np.round(np.sqrt(KX**2 + KY**2)).astype(int)  # (H, W)
    power = np.abs(np.fft.fft2(frames)) ** 2 / (H * W) ** 2  # (T, H, W)
    k_max = min(H, W) // 2
    k_bins = np.arange(1, k_max + 1)
    E = np.array([power[:, K_rad == k].mean() for k in k_bins])
    return k_bins, E


def log_energy_spectrum(
    model: KoopmanAE2D,
    gt_trajectories: list[torch.Tensor],
    device: torch.device,
    epoch: int,
    tag: str = "val",
    save_path: str | None = None,
) -> None:
    """Time-averaged radial energy spectrum <E(k)> — GT vs AR rollout.

    Same interface as log_autocorr. Rollout stays in latent space.
    Plots on log-log axes; bands show ±std across trajectories.
    """
    gt_E_list, pred_E_list = [], []
    k_bins = None

    for gt_frames in gt_trajectories:
        T = gt_frames.shape[0]
        x0 = gt_frames[0:1].to(device)

        with torch.no_grad():
            z = model.encode(x0)
            pred_frames = []
            for _ in range(T):
                pred_frames.append(model.decode(z).cpu())
                z = model.dynamics(z)
        pred_frames = torch.cat(pred_frames, dim=0)  # (T, 1, H, W)

        k_bins, E_gt = _energy_spectrum(gt_frames[:, 0].numpy())
        _, E_pred = _energy_spectrum(pred_frames[:, 0].numpy())
        gt_E_list.append(E_gt)
        pred_E_list.append(E_pred)

    gt_E = np.stack(gt_E_list)  # (n_traj, k_max)
    pred_E = np.stack(pred_E_list)
    gt_mean, gt_std = gt_E.mean(0), gt_E.std(0)
    pred_mean, pred_std = pred_E.mean(0), pred_E.std(0)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.loglog(k_bins, gt_mean, color="steelblue", linewidth=1.5, label="GT")
    ax.fill_between(k_bins, gt_mean - gt_std, gt_mean + gt_std, color="steelblue", alpha=0.2)
    ax.loglog(k_bins, pred_mean, color="tomato", linewidth=1.5, label="model")
    ax.fill_between(k_bins, pred_mean - pred_std, pred_mean + pred_std, color="tomato", alpha=0.2)
    ax.set_xlabel("k", fontsize=10)
    ax.set_ylabel("E(k)", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_title(f"{tag} energy spectrum — epoch {epoch}", fontsize=10)
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved → {save_path}")
    else:
        wandb.log({f"{tag}/energy_spectrum": wandb.Image(fig), "epoch": epoch})
    plt.close(fig)


# Panel (e)
def log_qq(
    model: KoopmanAE2D,
    gt_trajectories: list[torch.Tensor],
    device: torch.device,
    epoch: int,
    tag: str = "val",
    n_quantiles: int = 1000,
    save_path: str | None = None,
) -> None:
    """Q-Q plot of vorticity distributions — GT vs AR rollout.

    Collects all vorticity values across all frames and all trajectories,
    computes n_quantiles evenly-spaced quantiles for GT and model, and plots
    them against each other. Points on the diagonal = perfect match.
    """
    gt_vals, pred_vals = [], []

    for gt_frames in gt_trajectories:
        T = gt_frames.shape[0]
        x0 = gt_frames[0:1].to(device)

        with torch.no_grad():
            z = model.encode(x0)
            pred_frames = []
            for _ in range(T):
                pred_frames.append(model.decode(z).cpu())
                z = model.dynamics(z)
        pred_frames = torch.cat(pred_frames, dim=0)  # (T, 1, H, W)

        gt_vals.append(gt_frames[:, 0].numpy().ravel())
        pred_vals.append(pred_frames[:, 0].numpy().ravel())

    gt_all = np.concatenate(gt_vals)
    pred_all = np.concatenate(pred_vals)

    q = np.linspace(0, 100, n_quantiles)
    gt_q = np.percentile(gt_all, q)
    pred_q = np.percentile(pred_all, q)

    fig, ax = plt.subplots(figsize=(4, 4))
    lim = max(np.abs(gt_q).max(), np.abs(pred_q).max())
    ax.plot([-lim, lim], [-lim, lim], color="k", linewidth=0.8, linestyle="--", label="perfect match")
    ax.scatter(pred_q, gt_q, s=4, color="tomato", alpha=0.6, linewidths=0, label="model")
    ax.set_xlabel("Q_pred", fontsize=10)
    ax.set_ylabel("Q_GT", fontsize=10)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.legend(fontsize=9)
    ax.set_title(f"{tag} Q-Q vorticity — epoch {epoch}", fontsize=10)
    ax.set_aspect("equal")
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved → {save_path}")
    else:
        wandb.log({f"{tag}/qq": wandb.Image(fig), "epoch": epoch})
    plt.close(fig)
