from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

import wandb

_CMAP = "RdBu_r"


def log_images(
    predict_fn: Callable,
    dataset,
    device: torch.device,
    epoch: int,
    std: float,
    tag: str = "val",
    n_probes: int = 5,
    seed: int = 42,
) -> None:
    """1-step prediction at n_probes fixed-random positions. Rows=probes, cols=[x_t, pred, truth, |error|]."""
    rng = np.random.RandomState(seed)
    indices = sorted(rng.choice(len(dataset), n_probes, replace=False).tolist())

    rows = []
    for idx in indices:
        x, y = dataset[idx]
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = predict_fn(x)
        x_np = (x[0, 0] * std).cpu().numpy()
        pred_np = (pred[0, 0] * std).cpu().numpy()
        y_np = (y[0, 0] * std).cpu().numpy()
        rows.append((idx, x_np, pred_np, y_np, np.abs(pred_np - y_np)))

    fig, axes = plt.subplots(n_probes, 4, figsize=(14, 3 * n_probes), gridspec_kw={"wspace": 0.05, "hspace": 0.3})
    col_titles = ["x_t (input)", "x_{t+1} (pred)", "x_{t+1} (truth)", "|error|"]

    for row_i, (idx, x_np, pred_np, y_np, err_np) in enumerate(rows):
        vmin = min(x_np.min(), y_np.min())
        vmax = max(x_np.max(), y_np.max())
        err_lim = err_np.max()
        imgs = [x_np, pred_np, y_np, err_np]
        vmins = [vmin, vmin, vmin, -err_lim]
        vmaxs = [vmax, vmax, vmax, err_lim]

        for col_i, (img, vlo, vhi) in enumerate(zip(imgs, vmins, vmaxs)):
            ax = axes[row_i, col_i]
            ax.imshow(img, cmap=_CMAP, vmin=vlo, vmax=vhi)
            ax.axis("off")
            if row_i == 0:
                ax.set_title(col_titles[col_i], fontsize=9)
        axes[row_i, 0].set_ylabel(f"t={idx}", fontsize=8, rotation=0, labelpad=30, va="center")

    fig.suptitle(f"{tag} 1-step predictions — epoch {epoch}", fontsize=10)
    wandb.log({f"{tag}/images": wandb.Image(fig), "epoch": epoch})
    plt.close(fig)


def log_rollout(
    rollout_fn: Callable,
    gt_frames: torch.Tensor,
    device: torch.device,
    epoch: int,
    std: float,
    tag: str = "val",
    n_cols: int = 10,
) -> None:
    """Multi-step AR rollout: 3 rows (GT / pred / |error|) x n_cols timesteps."""
    T = gt_frames.shape[0]
    timesteps = np.linspace(0, T - 1, n_cols, dtype=int)
    x0 = gt_frames[0:1]  # (1, 1, H, W) on CPU

    with torch.no_grad():
        pred_all = rollout_fn(x0, T)  # (T, 1, H, W) on CPU

    gt_np = (gt_frames[timesteps, 0] * std).numpy()
    pred_np = (pred_all[timesteps, 0] * std).numpy()
    err_np = np.abs(pred_np - gt_np)

    vmin = min(gt_np.min(), pred_np.min())
    vmax = max(gt_np.max(), pred_np.max())
    err_lim = float(err_np.max())

    fig = plt.figure(figsize=(18, 1.8 * 3))
    gs = GridSpec(3, n_cols + 1, figure=fig, wspace=0.05, hspace=0.1, width_ratios=[1] * n_cols + [0.04])
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(3)])
    cax = fig.add_subplot(gs[:, -1])

    im_last = None
    for col, t in enumerate(timesteps):
        for row, (img, vlo, vhi) in enumerate(
            zip(
                [gt_np[col], pred_np[col], err_np[col]],
                [vmin, vmin, -err_lim],
                [vmax, vmax, err_lim],
            )
        ):
            ax = axes[row, col]
            im = ax.imshow(img, cmap=_CMAP, vmin=vlo, vmax=vhi)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"t={t}", fontsize=8)
            if col == n_cols - 1:
                im_last = im

    for row, label in enumerate(["GT", "pred", "|error|"]):
        axes[row, 0].text(
            -0.05, 0.5, label, transform=axes[row, 0].transAxes, va="center", ha="right", fontsize=9, rotation=90
        )

    fig.colorbar(im_last, cax=cax)
    fig.suptitle(f"{tag} rollout — epoch {epoch}", fontsize=10)
    wandb.log({f"{tag}/rollout": wandb.Image(fig), "epoch": epoch})
    plt.close(fig)


def log_short_rollout(
    rollout_fn: Callable,
    gt_frames: torch.Tensor,
    device: torch.device,
    epoch: int,
    std: float,
    tag: str = "val",
    with_recon: bool = True,
) -> None:
    """2-row grid (pred / GT) x 5 timesteps.

    with_recon=True  → columns t=0,1,2,3,4  (t=0 is decode∘encode reconstruction)
    with_recon=False → columns t=1,2,3,4,5  (pure Koopman steps, no reconstruction)
    """
    T = gt_frames.shape[0]
    x0 = gt_frames[0:1]

    with torch.no_grad():
        pred_all = rollout_fn(x0, 6)  # need up to t=5

    if with_recon:
        pred_frames = pred_all[0:5]
        gt_slice = gt_frames[0:5]
        t_labels = list(range(5))
        wandb_key = f"{tag}/short_rollout_with_recon"
        title_suffix = "t=0..4 (t=0 is recon)"
    else:
        pred_frames = pred_all[1:6]
        gt_slice = gt_frames[1:6]
        t_labels = list(range(1, 6))
        wandb_key = f"{tag}/short_rollout_no_recon"
        title_suffix = "t=1..5 (pure Koopman)"

    pred_np = (pred_frames[:, 0] * std).numpy()
    gt_np = (gt_slice[:, 0] * std).numpy()

    vmin = min(gt_np.min(), pred_np.min())
    vmax = max(gt_np.max(), pred_np.max())

    n_cols = 5
    fig, axes = plt.subplots(2, n_cols, figsize=(2.5 * n_cols, 5), gridspec_kw={"wspace": 0.05, "hspace": 0.15})
    for col in range(n_cols):
        for row, (img, label) in enumerate([(pred_np[col], "pred"), (gt_np[col], "GT")]):
            ax = axes[row, col]
            ax.imshow(img, cmap=_CMAP, vmin=vmin, vmax=vmax)
            ax.axis("off")
            if col == 0:
                ax.text(-0.05, 0.5, label, transform=ax.transAxes, va="center", ha="right", fontsize=9, rotation=90)
            if row == 0:
                ax.set_title(f"t={t_labels[col]}", fontsize=8)

    fig.suptitle(f"{tag} short rollout {title_suffix} — epoch {epoch}", fontsize=10)
    wandb.log({wandb_key: wandb.Image(fig), "epoch": epoch})
    plt.close(fig)


def _temporal_autocorr(frames: np.ndarray) -> np.ndarray:
    T = frames.shape[0]
    flat = frames.reshape(T, -1)
    R = np.empty(T)
    for tau in range(T):
        n = T - tau
        R[tau] = (flat[:n] * flat[tau:]).sum(axis=1).mean()
    R /= R[0]
    return R


def log_autocorr(
    rollout_fn: Callable,
    gt_trajectories: list[torch.Tensor],
    device: torch.device,
    epoch: int,
    tag: str = "val",
    save_path: str | None = None,
) -> None:
    """Temporal autocorrelation R_uu(t*) — GT vs AR rollout."""
    gt_R_list, pred_R_list = [], []

    for gt_frames in gt_trajectories:
        T = gt_frames.shape[0]
        with torch.no_grad():
            pred_frames = rollout_fn(gt_frames[0:1], T)

        gt_R_list.append(_temporal_autocorr(gt_frames[:, 0].numpy()))
        pred_R_list.append(_temporal_autocorr(pred_frames[:, 0].numpy()))

    gt_R = np.stack(gt_R_list)
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
    else:
        wandb.log({f"{tag}/autocorr": wandb.Image(fig), "epoch": epoch})
    plt.close(fig)


def _energy_spectrum(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T, H, W = frames.shape
    kx = np.fft.fftfreq(W) * W
    ky = np.fft.fftfreq(H) * H
    KX, KY = np.meshgrid(kx, ky)
    K_rad = np.round(np.sqrt(KX**2 + KY**2)).astype(int)
    power = np.abs(np.fft.fft2(frames)) ** 2 / (H * W) ** 2
    k_max = min(H, W) // 2
    k_bins = np.arange(1, k_max + 1)
    E = np.array([power[:, K_rad == k].mean() for k in k_bins])
    return k_bins, E


def log_energy_spectrum(
    rollout_fn: Callable,
    gt_trajectories: list[torch.Tensor],
    device: torch.device,
    epoch: int,
    tag: str = "val",
    save_path: str | None = None,
) -> None:
    gt_E_list, pred_E_list = [], []
    k_bins = None

    for gt_frames in gt_trajectories:
        T = gt_frames.shape[0]
        with torch.no_grad():
            pred_frames = rollout_fn(gt_frames[0:1], T)

        k_bins, E_gt = _energy_spectrum(gt_frames[:, 0].numpy())
        _, E_pred = _energy_spectrum(pred_frames[:, 0].numpy())
        gt_E_list.append(E_gt)
        pred_E_list.append(E_pred)

    gt_E = np.stack(gt_E_list)
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
    else:
        wandb.log({f"{tag}/energy_spectrum": wandb.Image(fig), "epoch": epoch})
    plt.close(fig)


def log_qq(
    rollout_fn: Callable,
    gt_trajectories: list[torch.Tensor],
    device: torch.device,
    epoch: int,
    tag: str = "val",
    n_quantiles: int = 1000,
    save_path: str | None = None,
) -> None:
    gt_vals, pred_vals = [], []

    for gt_frames in gt_trajectories:
        T = gt_frames.shape[0]
        with torch.no_grad():
            pred_frames = rollout_fn(gt_frames[0:1], T)

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
    else:
        wandb.log({f"{tag}/qq": wandb.Image(fig), "epoch": epoch})
    plt.close(fig)
