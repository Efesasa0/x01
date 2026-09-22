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

ROOT = Path(__file__).resolve().parents[4]
OLD_DATA = ROOT / "kmflow_highres.npy"
NEW_DATA = ROOT / "kmflow_re1000_r256_b400_f320_clean.npy"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Latin Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-data", type=Path, default=OLD_DATA)
    parser.add_argument("--new-data", type=Path, default=NEW_DATA)
    parser.add_argument("--start-traj", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--frame-step", type=int, default=8)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--old-alpha", type=float, default=0.18)
    parser.add_argument("--new-alpha", type=float, default=0.10)
    parser.add_argument("--linewidth", type=float, default=1.5)
    parser.add_argument("--legend-handlelength", type=float, default=3.6)
    parser.add_argument("--fig-width", type=float, default=5.8)
    parser.add_argument("--fig-height", type=float, default=3.6)
    parser.add_argument("--left", type=float, default=0.14)
    parser.add_argument("--right", type=float, default=0.96)
    parser.add_argument("--top", type=float, default=0.88)
    parser.add_argument("--bottom", type=float, default=0.16)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("energy_spectrum_smoke.png"))
    return parser.parse_args()


def energy_spectrum(frames):
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


def read_trajectory(data, trajectory_index, frame_step):
    trajectory = np.asarray(data[trajectory_index, ::frame_step], dtype=np.float32)
    trajectory = np.squeeze(trajectory)
    if trajectory.ndim != 3:
        raise ValueError(f"expected trajectory with shape (T, H, W), got {trajectory.shape}")
    return trajectory


def compute_curves(path, start_index, count, frame_step):
    data = np.load(path, mmap_mode="r")
    stop_index = min(start_index + count, data.shape[0])
    curves = []
    k_bins = None

    for trajectory_index in range(start_index, stop_index):
        trajectory = read_trajectory(data, trajectory_index, frame_step)
        k_bins, spectrum = energy_spectrum(trajectory)
        curves.append(spectrum)

    return k_bins, np.asarray(curves)


def plot_band(ax, k_bins, mean, std, eps, alpha, hatch):
    band = ax.fill_between(
        k_bins,
        np.maximum(mean - std, eps),
        mean + std,
        facecolor="white",
        edgecolor="0.45",
        alpha=alpha,
        linewidth=0.0,
    )
    band.set_hatch(hatch)


def save_outputs(fig, png_path):
    paths = (png_path, png_path.with_suffix(".pdf"))
    for path in paths:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return paths


def main():
    args = parse_args()
    old_k, old_curves = compute_curves(args.old_data, args.start_traj, args.count, args.frame_step)
    new_k, new_curves = compute_curves(args.new_data, args.start_traj, args.count, args.frame_step)

    old_mean = old_curves.mean(axis=0)
    old_std = old_curves.std(axis=0)
    new_mean = new_curves.mean(axis=0)
    new_std = new_curves.std(axis=0)

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    ax.fill_between(
        old_k,
        np.maximum(old_mean - old_std, args.eps),
        old_mean + old_std,
        color="0.75",
        alpha=args.old_alpha,
        linewidth=0,
    )
    plot_band(ax, new_k, new_mean, new_std, args.eps, args.new_alpha, "///")
    ax.loglog(old_k, old_mean, color="black", linestyle="-", linewidth=args.linewidth, label="Initial dataset")
    ax.loglog(new_k, new_mean, color="black", linestyle=":", linewidth=args.linewidth, label="Generated dataset")
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$E(\kappa)$")
    ax.set_title("Energy spectrum")
    ax.legend(frameon=False, handlelength=args.legend_handlelength)
    ax.grid(True, which="both", color="0.88", linewidth=0.5)
    fig.subplots_adjust(left=args.left, right=args.right, bottom=args.bottom, top=args.top)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    paths = save_outputs(fig, args.out)
    plt.close(fig)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
