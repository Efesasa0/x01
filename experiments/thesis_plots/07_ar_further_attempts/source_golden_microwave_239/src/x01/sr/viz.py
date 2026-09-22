from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import wandb

_CMAP = "RdBu_r"


def log_samples(
    sample_fn: Callable,
    dataset,
    device: torch.device,
    epoch: int,
    std: float,
    tag: str = "val",
    n_probes: int = 5,
    seed: int = 42,
) -> None:
    """Conditional DDIM samples vs. dataset references.

    `sample_fn(hr_refs)` is expected to return `(condition, samples)`: the LR-blurry guide
    fed to the model and the DDIM-denoised HR output. Rows=probes, cols=[ref, cond, sample, |diff|].
    """
    rng = np.random.RandomState(seed)
    indices = sorted(rng.choice(len(dataset), n_probes, replace=False).tolist())

    refs = torch.stack([dataset[i] for i in indices]).to(device)
    with torch.no_grad():
        cond, samples = sample_fn(refs)
    refs_np = (refs * std).cpu().numpy()
    cond_np = (cond * std).numpy()
    samples_np = (samples * std).numpy()
    err_np = np.abs(samples_np - refs_np)

    fig, axes = plt.subplots(n_probes, 4, figsize=(14, 3 * n_probes), gridspec_kw={"wspace": 0.05, "hspace": 0.3})
    col_titles = ["reference (HR)", "condition (LR upsampled)", "sample (DDIM)", "|diff|"]

    for row_i in range(n_probes):
        r, c, s, e = refs_np[row_i, 0], cond_np[row_i, 0], samples_np[row_i, 0], err_np[row_i, 0]
        vmin = float(min(r.min(), c.min(), s.min()))
        vmax = float(max(r.max(), c.max(), s.max()))
        err_lim = float(e.max())
        imgs = [r, c, s, e]
        vmins = [vmin, vmin, vmin, -err_lim]
        vmaxs = [vmax, vmax, vmax, err_lim]
        for col_i, (img, vlo, vhi) in enumerate(zip(imgs, vmins, vmaxs)):
            ax = axes[row_i, col_i]
            ax.imshow(img, cmap=_CMAP, vmin=vlo, vmax=vhi)
            ax.axis("off")
            if row_i == 0:
                ax.set_title(col_titles[col_i], fontsize=9)
        axes[row_i, 0].set_ylabel(f"idx={indices[row_i]}", fontsize=8, rotation=0, labelpad=30, va="center")

    fig.suptitle(f"{tag} DDIM samples — epoch {epoch}", fontsize=10)
    wandb.log({f"{tag}/samples": wandb.Image(fig), "epoch": epoch})
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
    sample_fn: Callable,
    gt_frames: torch.Tensor,
    device: torch.device,
    epoch: int,
    tag: str = "val",
    n_samples: int = 64,
    save_path: str | None = None,
) -> None:
    """Energy spectrum overlay: GT, condition, and conditional DDIM samples on one loglog axis."""
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

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        wandb.log({f"{tag}/energy_spectrum": wandb.Image(fig), "epoch": epoch})
    plt.close(fig)
