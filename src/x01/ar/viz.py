import os
import tempfile
from collections.abc import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

_CMAP = "RdBu_r"


def _log_gif(frames_pil: list, key: str, epoch: int, save_dir: str | None, fps: int = 10) -> None:
    path = (
        f"{save_dir}/{key.replace('/', '_')}_epoch{epoch:04d}.gif"
        if save_dir is not None
        else tempfile.NamedTemporaryFile(suffix=".gif", delete=False).name
    )
    frames_pil[0].save(path, save_all=True, append_images=frames_pil[1:], loop=0, duration=1000 // fps)
    if wandb.run is not None:
        wandb.log({key: wandb.Video(path), "epoch": epoch})
    if save_dir is None:
        os.unlink(path)


def log_rollout_gif(
    rollout_fn: Callable,
    gt_frames: torch.Tensor,
    epoch: int,
    std: float,
    tag: str = "val",
    fps: int = 10,
    save_dir: str | None = None,
    gt_display_size: int = 64,
    lr_pred_frames: torch.Tensor | None = None,
) -> None:
    T = gt_frames.shape[0]
    n_pred = T - 1
    x0 = gt_frames[0:1]
    with torch.no_grad():
        pred_all = rollout_fn(x0, n_pred)

    pred_np = (pred_all[:, 0] * std).numpy()
    gt_lr = F.interpolate(gt_frames[1:, :], size=gt_display_size, mode="area")
    gt_np = (gt_lr[:, 0] * std).numpy()

    p_vmin, p_vmax = float(pred_np.min()), float(pred_np.max())
    g_vmin, g_vmax = float(gt_np.min()), float(gt_np.max())

    show_lr = lr_pred_frames is not None
    if show_lr:
        lr_np = (lr_pred_frames[:n_pred, 0] * std).numpy()
        l_vmin, l_vmax = float(lr_np.min()), float(lr_np.max())
        n_cols = 3
        figsize = (15, 4)
    else:
        n_cols = 2
        figsize = (10, 4)

    pil_frames = []
    for t in range(n_pred):
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, n_cols, figure=fig, wspace=0.15)
        col = 0

        if show_lr:
            ax_l = fig.add_subplot(gs[0, col])
            im_l = ax_l.imshow(lr_np[t], cmap=_CMAP, vmin=l_vmin, vmax=l_vmax)
            ax_l.set_aspect("equal")
            ax_l.axis("off")
            ax_l.set_title(f"{tag} AR (LR) - t={t + 1}/{n_pred}", fontsize=10)
            divl = make_axes_locatable(ax_l)
            caxl = divl.append_axes("right", size="5%", pad=0.08)
            fig.colorbar(im_l, cax=caxl)
            col += 1

        ax_p = fig.add_subplot(gs[0, col])
        im_p = ax_p.imshow(pred_np[t], cmap=_CMAP, vmin=p_vmin, vmax=p_vmax)
        ax_p.set_aspect("equal")
        ax_p.axis("off")
        pred_title = (
            f"{tag} AR+SR - epoch {epoch} - t={t + 1}/{n_pred}"
            if show_lr
            else f"{tag} pred - epoch {epoch} - t={t + 1}/{n_pred}"
        )
        ax_p.set_title(pred_title, fontsize=10)
        divp = make_axes_locatable(ax_p)
        caxp = divp.append_axes("right", size="5%", pad=0.08)
        fig.colorbar(im_p, cax=caxp)
        col += 1

        ax_g = fig.add_subplot(gs[0, col])
        im_g = ax_g.imshow(gt_np[t], cmap=_CMAP, vmin=g_vmin, vmax=g_vmax)
        ax_g.set_aspect("equal")
        ax_g.axis("off")
        ax_g.set_title(f"{tag} GT - t={t + 1}/{n_pred}", fontsize=10)
        divg = make_axes_locatable(ax_g)
        caxg = divg.append_axes("right", size="5%", pad=0.08)
        fig.colorbar(im_g, cax=caxg)

        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
        pil_frames.append(Image.fromarray(buf))
        plt.close(fig)

    _log_gif(pil_frames, f"{tag}/rollout_gif", epoch, save_dir, fps=fps)


def log_eig_spectrum(
    eig_history: dict[str, list[float]],
    epoch: int,
    save_dir: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    xs = list(range(1, epoch + 1))
    ax.plot(xs, eig_history["A_mean"], color="red", linestyle="-", label="A mean")
    ax.plot(xs, eig_history["A_max"], color="red", linestyle="--", label="A max")
    ax.plot(xs, eig_history["A_min"], color="red", linestyle=":", label="A min")
    ax.plot(xs, eig_history["B_mean"], color="blue", linestyle="-", label="B mean")
    ax.plot(xs, eig_history["B_max"], color="blue", linestyle="--", label="B max")
    ax.plot(xs, eig_history["B_min"], color="blue", linestyle=":", label="B min")
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("epoch", fontsize=10)
    ax.set_ylabel("|eigenvalue|", fontsize=10)
    ax.set_title("Eigenvalue magnitudes: A vs B", fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    _log_fig(fig, "epoch/eig_spectrum", epoch, save_dir)


def _log_fig(fig, key: str, epoch: int, save_dir: str | None) -> None:
    if wandb.run is not None:
        wandb.log({key: wandb.Image(fig), "epoch": epoch})
    if save_dir is not None:
        fig.savefig(f"{save_dir}/{key.replace('/', '_')}_epoch{epoch:04d}.png", dpi=120)
    plt.close(fig)


def log_rollout(
    rollout_fn: Callable,
    gt_frames: torch.Tensor,
    epoch: int,
    std: float,
    tag: str = "val",
    n_cols: int = 10,
    save_dir: str | None = None,
) -> None:
    T = gt_frames.shape[0]
    n_pred = T - 1
    timesteps = np.linspace(1, n_pred, n_cols, dtype=int)  # display times t=1..n_pred
    x0 = gt_frames[0:1]  # (1, 1, H, W) on CPU

    with torch.no_grad():
        pred_all = rollout_fn(x0, n_pred)  # (n_pred, 1, H, W), pred_all[i] = predicted t=i+1

    gt_np = (gt_frames[timesteps, 0] * std).numpy()
    pred_np = (pred_all[timesteps - 1, 0] * std).numpy()
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
                strict=True,
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
    _log_fig(fig, f"{tag}/rollout", epoch, save_dir)


def log_short_rollout(
    rollout_fn: Callable,
    gt_frames: torch.Tensor,
    epoch: int,
    std: float,
    tag: str = "val",
    save_dir: str | None = None,
) -> None:
    x0 = gt_frames[0:1]

    with torch.no_grad():
        pred_all = rollout_fn(x0, 5)  # 5 predictions, pred_all[i] = predicted t=i+1

    pred_frames = pred_all
    gt_slice = gt_frames[1:6]
    t_labels = list(range(1, 6))
    wandb_key = f"{tag}/short_rollout"
    title_suffix = "t=1..5"

    pred_np = (pred_frames[:, 0] * std).numpy()
    gt_np = (gt_slice[:, 0] * std).numpy()
    err_np = np.abs(pred_np - gt_np)

    vmin = min(gt_np.min(), pred_np.min())
    vmax = max(gt_np.max(), pred_np.max())
    err_lim = float(err_np.max())

    n_cols = 5
    fig = plt.figure(figsize=(2.5 * n_cols + 0.6, 7.5))
    gs = GridSpec(3, n_cols + 1, figure=fig, wspace=0.05, hspace=0.1, width_ratios=[1] * n_cols + [0.04])
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(3)])
    cax = fig.add_subplot(gs[:, -1])

    im_last = None
    for col in range(n_cols):
        for row, (img, vlo, vhi) in enumerate(
            zip(
                [gt_np[col], pred_np[col], err_np[col]],
                [vmin, vmin, -err_lim],
                [vmax, vmax, err_lim],
                strict=True,
            )
        ):
            ax = axes[row, col]
            im = ax.imshow(img, cmap=_CMAP, vmin=vlo, vmax=vhi)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"t={t_labels[col]}", fontsize=8)
            if col == n_cols - 1 and row == 1:
                im_last = im

    for row, label in enumerate(["GT", "pred", "|error|"]):
        axes[row, 0].text(
            -0.05, 0.5, label, transform=axes[row, 0].transAxes, va="center", ha="right", fontsize=9, rotation=90
        )

    fig.colorbar(im_last, cax=cax)
    fig.suptitle(f"{tag} short rollout {title_suffix} — epoch {epoch}", fontsize=10)
    _log_fig(fig, wandb_key, epoch, save_dir)


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
    epoch: int,
    tag: str = "val",
    save_dir: str | None = None,
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
    ax.set_ylim(-0.5, 1)
    ax.legend(fontsize=9)
    ax.set_title(f"{tag} autocorrelation — epoch {epoch}", fontsize=10)
    fig.tight_layout()
    _log_fig(fig, f"{tag}/autocorr", epoch, save_dir)


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
    epoch: int,
    tag: str = "val",
    save_dir: str | None = None,
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
    ax.set_ylim(1e-5, 1)
    ax.legend(fontsize=9)
    ax.set_title(f"{tag} energy spectrum — epoch {epoch}", fontsize=10)
    fig.tight_layout()
    _log_fig(fig, f"{tag}/energy_spectrum", epoch, save_dir)


def log_qq(
    rollout_fn: Callable,
    gt_trajectories: list[torch.Tensor],
    epoch: int,
    tag: str = "val",
    n_quantiles: int = 1000,
    save_dir: str | None = None,
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
    lim = 35
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
    _log_fig(fig, f"{tag}/qq", epoch, save_dir)


def _mse_per_frame(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return ((pred - gt) ** 2).mean(axis=(0, 2, 3))


def log_mse_vs_frame(
    rollout_fn: Callable,
    gt_trajectories: list[torch.Tensor],
    epoch: int,
    tag: str = "val",
    save_dir: str | None = None,
) -> None:
    pred_list, gt_list = [], []
    for gt_frames in gt_trajectories:
        T = gt_frames.shape[0] - 1
        with torch.no_grad():
            pred = rollout_fn(gt_frames[0:1], T)
        pred_list.append(pred[:, 0].numpy())
        gt_list.append(gt_frames[1 : 1 + T, 0].numpy())
    pred_np = np.stack(pred_list)
    gt_np = np.stack(gt_list)
    mse = _mse_per_frame(gt_np, pred_np)
    t = np.arange(mse.shape[0])

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t, mse, color="tomato", lw=1.5)
    ax.set_xlabel("frame", fontsize=10)
    ax.set_ylabel("MSE (avg over trajs, spatial)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title(f"{tag} rollout MSE vs frame — epoch {epoch}", fontsize=10)
    fig.tight_layout()
    _log_fig(fig, f"{tag}/mse_vs_frame", epoch, save_dir)


def benchmark_fps(rollout_fn: Callable, x0: torch.Tensor, n_warmup: int = 5, n_timed: int = 50) -> None:
    import time

    _ = rollout_fn(x0, n_warmup)
    t0 = time.perf_counter()
    _ = rollout_fn(x0, n_timed)
    dt = time.perf_counter() - t0
    fps = n_timed / dt
    print(f"[benchmark_fps] {n_timed} frames in {dt:.3f}s -> {fps:.2f} fps ({1e3 * dt / n_timed:.2f} ms/frame)")
