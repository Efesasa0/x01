"""Visualisation utilities for KM flow simulation — images and videos logged to wandb."""

import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import wandb

_CMAP = "RdBu_r"


def log_vorticity_image(omega: np.ndarray, traj_idx: int, frame_idx: int) -> None:
    """Log a single vorticity frame as a wandb image.

    omega: (N, N) float32 — one trajectory, one frame
    """
    omega = np.nan_to_num(omega, nan=0.0, posinf=0.0, neginf=0.0)
    vmax = float(np.percentile(np.abs(omega), 99)) + 1e-8
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(omega, cmap=_CMAP, vmin=-vmax, vmax=vmax, origin="lower")
    fig.colorbar(im, ax=ax, shrink=0.8, label="ω")
    ax.set_title(f"traj={traj_idx}  frame={frame_idx}", fontsize=8)
    ax.axis("off")
    wandb.log({"sim/vorticity": wandb.Image(fig), "sim/traj": traj_idx})
    plt.close(fig)


def log_vorticity_video(frames: np.ndarray, traj_idx: int, fps: int = 10, subsample: int = 1) -> None:
    """Log a subsampled vorticity sequence as a wandb video.

    frames:    (T, N, N) float32 — full recorded sequence for one trajectory
    subsample: take every Nth frame; default 1 (all frames)
    """
    frames = np.nan_to_num(frames[::subsample], nan=0.0, posinf=0.0, neginf=0.0)  # (T', N, N)
    if len(frames) < 2:
        return  # not enough frames for a meaningful video

    vmax = float(np.percentile(np.abs(frames), 99)) + 1e-8
    norm = frames.clip(-vmax, vmax) / vmax  # [-1, 1]

    # apply RdBu_r colormap → RGB so the video matches the static images
    cmap = plt.get_cmap(_CMAP)
    frames_rgb = (cmap((norm + 1.0) / 2.0)[..., :3] * 255).astype(np.uint8)  # (T', H, W, 3)

    pil_frames = [Image.fromarray(f) for f in frames_rgb]

    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    tmp.close()
    pil_frames[0].save(
        tmp.name,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=1000 // fps,  # ms per frame
    )
    wandb.log({"sim/video": wandb.Video(tmp.name, format="gif"), "sim/traj": traj_idx})
    os.unlink(tmp.name)
