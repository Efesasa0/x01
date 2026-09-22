from collections.abc import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from matplotlib.gridspec import GridSpec

_CMAP = "RdBu_r"


def _log_fig(fig, key: str, epoch: int, save_dir: str | None) -> None:
    if wandb.run is not None:
        wandb.log({key: wandb.Image(fig), "epoch": epoch})
    if save_dir is not None:
        fig.savefig(f"{save_dir}/{key.replace('/', '_')}_epoch{epoch:04d}.png", dpi=120)
    plt.close(fig)


def log_samples(
    sample_fn: Callable,
    dataset,
    device: torch.device,
    epoch: int,
    std: float,
    tag: str = "val",
    n_probes: int = 5,
    seed: int = 42,
    save_dir: str | None = None,
) -> None:
    rng = np.random.RandomState(seed)
    indices = sorted(rng.choice(len(dataset), n_probes, replace=False).tolist())

    refs = torch.stack([dataset[i] for i in indices]).to(device)
    with torch.no_grad():
        cond, samples = sample_fn(refs)
    refs_np = (refs * std).cpu().numpy()
    cond_np = (cond * std).numpy()
    samples_np = (samples * std).numpy()
    err_np = np.abs(samples_np - refs_np)

    fig = plt.figure(figsize=(15, 3 * n_probes))
    gs = GridSpec(n_probes, 6, figure=fig, wspace=0.05, hspace=0.3, width_ratios=[1, 1, 1, 0.04, 1, 0.04])
    col_titles = ["reference (HR)", "condition (LR upsampled)", "sample (DDIM)", "|diff|"]
    img_cols = [0, 1, 2, 4]
    cbar_cols = [3, 5]

    for row_i in range(n_probes):
        r, c, s, e = refs_np[row_i, 0], cond_np[row_i, 0], samples_np[row_i, 0], err_np[row_i, 0]
        vmin = float(min(r.min(), c.min(), s.min()))
        vmax = float(max(r.max(), c.max(), s.max()))
        err_lim = float(e.max())
        imgs = [r, c, s, e]
        vmins = [vmin, vmin, vmin, -err_lim]
        vmaxs = [vmax, vmax, vmax, err_lim]
        ims = []
        for col_i, (img, vlo, vhi) in enumerate(zip(imgs, vmins, vmaxs, strict=True)):
            ax = fig.add_subplot(gs[row_i, img_cols[col_i]])
            ims.append(ax.imshow(img, cmap=_CMAP, vmin=vlo, vmax=vhi))
            ax.axis("off")
            if row_i == 0:
                ax.set_title(col_titles[col_i], fontsize=9)
            if col_i == 0:
                ax.set_ylabel(f"idx={indices[row_i]}", fontsize=8, rotation=0, labelpad=30, va="center")
        fig.colorbar(ims[2], cax=fig.add_subplot(gs[row_i, cbar_cols[0]]))
        fig.colorbar(ims[3], cax=fig.add_subplot(gs[row_i, cbar_cols[1]]))

    fig.suptitle(f"{tag} DDIM samples — epoch {epoch}", fontsize=10)
    _log_fig(fig, f"{tag}/samples", epoch, save_dir)


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
    sample_fn: Callable,
    gt_frames: torch.Tensor,
    device: torch.device,
    epoch: int,
    tag: str = "val",
    n_samples: int = 64,
    save_dir: str | None = None,
) -> None:
    refs = gt_frames[:n_samples].to(device)
    with torch.no_grad():
        cond, samples = sample_fn(refs)

    k_bins, E_gt = _energy_spectrum(refs[:, 0].cpu().numpy())
    _, E_cond = _energy_spectrum(cond[:, 0].numpy())
    _, E_pred = _energy_spectrum(samples[:, 0].numpy())

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.loglog(k_bins, E_gt, color="steelblue", linewidth=1.5, label="GT")
    ax.loglog(k_bins, E_cond, color="goldenrod", linewidth=1.5, linestyle="--", label="condition")
    ax.loglog(k_bins, E_pred, color="tomato", linewidth=1.5, label="samples")
    ax.set_xlabel("k", fontsize=10)
    ax.set_ylabel("E(k)", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_title(f"{tag} energy spectrum — epoch {epoch}", fontsize=10)
    fig.tight_layout()
    _log_fig(fig, f"{tag}/energy_spectrum", epoch, save_dir)
