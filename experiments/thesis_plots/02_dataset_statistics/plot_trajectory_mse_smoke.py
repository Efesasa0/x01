import argparse
import os
from pathlib import Path

import numpy as np

TMP_ROOT = Path(os.environ.get("TMPDIR", "/tmp"))
MPLCONFIGDIR = TMP_ROOT / "x01_mplconfig"
XDG_CACHE_HOME = TMP_ROOT / "x01_cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[4]
OLD_DATA = ROOT / "kmflow_highres.npy"
NEW_DATA = ROOT / "kmflow_re1000_r256_b400_f320_clean.npy"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Latin Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
    }
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-data", type=Path, default=OLD_DATA)
    parser.add_argument("--new-data", type=Path, default=NEW_DATA)
    parser.add_argument("--reference-traj", type=int, default=0)
    parser.add_argument("--start-traj", type=int, default=1)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--frame-step", type=int, default=2)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--old-alpha", type=float, default=0.35)
    parser.add_argument("--new-alpha", type=float, default=0.55)
    parser.add_argument("--linewidth", type=float, default=9.0)
    parser.add_argument("--legend-handlelength", type=float, default=3.6)
    parser.add_argument("--fig-width", type=float, default=6.2)
    parser.add_argument("--fig-height", type=float, default=3.6)
    parser.add_argument("--left", type=float, default=0.14)
    parser.add_argument("--right", type=float, default=0.96)
    parser.add_argument("--top", type=float, default=0.9)
    parser.add_argument("--bottom", type=float, default=0.16)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("trajectory_mse_smoke.png"))
    return parser.parse_args()


def read_trajectory(data, trajectory_index, frame_indices):
    trajectory = np.asarray(data[trajectory_index, frame_indices], dtype=np.float32)
    if trajectory.ndim == 4 and trajectory.shape[1] == 1:
        trajectory = trajectory[:, 0]
    if trajectory.ndim == 4 and trajectory.shape[-1] == 1:
        trajectory = trajectory[..., 0]
    return trajectory


def compute_mse_curves(path, reference_index, start_index, count, frame_step, eps):
    data = np.load(path, mmap_mode="r")
    frame_indices = np.arange(0, data.shape[1], frame_step)
    reference = read_trajectory(data, reference_index, frame_indices)
    curves = []

    for trajectory_index in range(start_index, min(start_index + count, data.shape[0])):
        trajectory = read_trajectory(data, trajectory_index, frame_indices)
        diff = trajectory - reference
        mse = np.mean(diff * diff, axis=tuple(range(1, diff.ndim)))
        curves.append(mse + eps)

    return frame_indices, np.asarray(curves)


def save_outputs(fig, png_path):
    paths = (png_path, png_path.with_suffix(".pdf"))
    for path in paths:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return paths


def main():
    args = parse_args()
    old_frames, old_curves = compute_mse_curves(
        args.old_data,
        args.reference_traj,
        args.start_traj,
        args.count,
        args.frame_step,
        args.eps,
    )
    new_frames, new_curves = compute_mse_curves(
        args.new_data,
        args.reference_traj,
        args.start_traj,
        args.count,
        args.frame_step,
        args.eps,
    )

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    for curve in old_curves:
        ax.plot(old_frames, curve, color="black", linestyle="-", alpha=args.old_alpha, linewidth=args.linewidth)
    for curve in new_curves:
        ax.plot(new_frames, curve, color="black", linestyle="--", alpha=args.new_alpha, linewidth=args.linewidth)

    handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.2, label="Initial dataset"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.2, label="Generated dataset"),
    ]
    ax.legend(handles=handles, frameon=False, handlelength=args.legend_handlelength)
    ax.set_yscale("log")
    ax.set_xlabel(r"$k$")
    ax.set_ylabel("MSE to reference trajectory")
    ax.set_title("Trajectory-to-trajectory spread")
    ax.grid(True, which="both", color="0.85", linewidth=0.5)
    fig.subplots_adjust(left=args.left, right=args.right, bottom=args.bottom, top=args.top)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    paths = save_outputs(fig, args.out)
    plt.close(fig)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
