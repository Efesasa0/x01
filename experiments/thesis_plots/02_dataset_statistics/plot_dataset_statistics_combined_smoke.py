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
from plot_energy_spectrum_smoke import compute_curves as compute_energy_curves
from plot_nan_frame_hist_smoke import SUMMARY, load_histogram
from plot_trajectory_mse_smoke import NEW_DATA, OLD_DATA, compute_mse_curves

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Latin Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
    }
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide-width", type=float, default=8.5)
    parser.add_argument("--wide-height", type=float, default=2.2)
    parser.add_argument("--small-width", type=float, default=3.75)
    parser.add_argument("--small-height", type=float, default=2.2)
    parser.add_argument("--top-middle-margin", type=float, default=1.00)
    parser.add_argument("--middle-bottom-margin", type=float, default=1.00)
    parser.add_argument("--bottom-column-margin", type=float, default=1.00)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("dataset_statistics_combined_smoke.png"))
    return parser.parse_args()


def add_axes(fig, fig_width, fig_height, x, y, width, height):
    return fig.add_axes([x / fig_width, y / fig_height, width / fig_width, height / fig_height])


def draw_mse(ax):
    old_frames, old_curves = compute_mse_curves(OLD_DATA, 0, 1, 10, 2, 1e-12)
    new_frames, new_curves = compute_mse_curves(NEW_DATA, 0, 1, 10, 2, 1e-12)

    for curve in old_curves:
        ax.plot(old_frames, curve, color="black", linestyle="-", alpha=0.35, linewidth=0.9)
    for curve in new_curves:
        ax.plot(new_frames, curve, color="black", linestyle=":", alpha=0.55, linewidth=0.9)

    handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.2, label="Initial dataset"),
        Line2D([0], [0], color="black", linestyle=":", linewidth=1.2, label="Generated dataset"),
    ]
    ax.legend(handles=handles, frameon=False, handlelength=3.6, fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel(r"$k$")
    ax.set_ylabel("MSE to reference trajectory")
    ax.set_title("(a) Trajectory-to-trajectory spread")
    ax.grid(True, which="both", color="0.85", linewidth=0.5)


def draw_nan_hist(ax):
    bins, values, dirty_count = load_histogram(SUMMARY)
    valid_count = 400 - dirty_count

    ax.bar(
        bins[:-1],
        values,
        width=np.diff(bins),
        align="edge",
        facecolor="white",
        edgecolor="black",
        linewidth=0.8,
        hatch="///",
    )
    ax.text(
        0.98,
        0.93,
        f"{valid_count} retained\n{dirty_count} removed",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    ax.set_xlabel(r"First non-finite $k$")
    ax.set_ylabel("Trajectory count")
    ax.set_title("(b) Rejected trajectories")
    ax.set_ylim(0, 16)
    ax.set_yticks([0, 5, 10, 15])
    ax.grid(True, axis="y", color="0.85", linewidth=0.5)


def read_full_trajectory(data, trajectory_index):
    trajectory = np.asarray(data[trajectory_index], dtype=np.float32)
    trajectory = np.squeeze(trajectory)
    if trajectory.ndim != 3:
        raise ValueError(f"expected trajectory with shape (T, H, W), got {trajectory.shape}")
    return trajectory


def temporal_autocorr_full(frames, chunk_size=4096):
    T = frames.shape[0]
    flat = frames.reshape(T, -1)
    corr_sum = np.zeros(T, dtype=np.float64)

    for start in range(0, flat.shape[1], chunk_size):
        chunk = flat[:, start : start + chunk_size]
        spectrum = np.fft.rfft(chunk, n=2 * T, axis=0)
        corr = np.fft.irfft(spectrum * np.conj(spectrum), n=2 * T, axis=0)[:T]
        corr_sum += corr.sum(axis=1)

    R = corr_sum / np.arange(T, 0, -1)
    R /= R[0]
    return R


def compute_full_autocorr_curves(path, start_index, count):
    data = np.load(path, mmap_mode="r")
    stop_index = min(start_index + count, data.shape[0])
    curves = []

    for trajectory_index in range(start_index, stop_index):
        trajectory = read_full_trajectory(data, trajectory_index)
        curves.append(temporal_autocorr_full(trajectory))

    curves = np.asarray(curves)
    t_star = np.linspace(0, 1, curves.shape[1])
    return t_star, curves


def count_trajectories(path):
    data = np.load(path, mmap_mode="r")
    return data.shape[0]


def draw_autocorr(ax):
    old_t, old_curves = compute_full_autocorr_curves(OLD_DATA, 0, count_trajectories(OLD_DATA))
    new_t, new_curves = compute_full_autocorr_curves(NEW_DATA, 0, count_trajectories(NEW_DATA))
    old_mean, old_std = old_curves.mean(axis=0), old_curves.std(axis=0)
    new_mean, new_std = new_curves.mean(axis=0), new_curves.std(axis=0)

    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.fill_between(old_t, old_mean - old_std, old_mean + old_std, color="0.75", alpha=0.18, linewidth=0)
    new_band = ax.fill_between(
        new_t,
        new_mean - new_std,
        new_mean + new_std,
        facecolor="white",
        edgecolor="0.45",
        alpha=0.10,
        linewidth=0.0,
    )
    new_band.set_hatch("///")
    ax.plot(old_t, old_mean, color="black", linestyle="-", linewidth=1.5, label="Initial dataset")
    ax.plot(new_t, new_mean, color="black", linestyle=":", linewidth=1.5, label="Generated dataset")
    ax.set_xlabel(r"$k^\ast$")
    ax.set_ylabel(r"$R(k^\ast)$")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 1.05)
    ax.set_title("(c) Temporal autocorrelation")
    ax.legend(frameon=False, handlelength=3.0, fontsize=8)
    ax.grid(True, color="0.88", linewidth=0.5)


def draw_energy(ax):
    old_k, old_curves = compute_energy_curves(OLD_DATA, 0, count_trajectories(OLD_DATA), 1)
    new_k, new_curves = compute_energy_curves(NEW_DATA, 0, count_trajectories(NEW_DATA), 1)
    old_mean, old_std = old_curves.mean(axis=0), old_curves.std(axis=0)
    new_mean, new_std = new_curves.mean(axis=0), new_curves.std(axis=0)
    eps = 1e-12

    ax.fill_between(
        old_k,
        np.maximum(old_mean - old_std, eps),
        old_mean + old_std,
        color="0.75",
        alpha=0.18,
        linewidth=0,
    )
    new_band = ax.fill_between(
        new_k,
        np.maximum(new_mean - new_std, eps),
        new_mean + new_std,
        facecolor="white",
        edgecolor="0.45",
        alpha=0.10,
        linewidth=0.0,
    )
    new_band.set_hatch("///")
    ax.loglog(old_k, old_mean, color="black", linestyle="-", linewidth=1.5, label="Initial dataset")
    ax.loglog(new_k, new_mean, color="black", linestyle=":", linewidth=1.5, label="Generated dataset")
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$E(\kappa)$")
    ax.set_title(r"(d) Energy spectrum")
    ax.legend(frameon=False, handlelength=3.0, fontsize=8)
    ax.grid(True, which="both", color="0.88", linewidth=0.5)


def save_outputs(fig, png_path):
    paths = (png_path, png_path.with_suffix(".pdf"))
    for path in paths:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return paths


def main():
    args = parse_args()
    left_margin = 0.75
    right_margin = 0.25
    top_margin = 0.35
    bottom_margin = 0.55

    bottom_width = 2 * args.small_width + args.bottom_column_margin
    body_width = max(args.wide_width, bottom_width)
    body_height = 2 * args.wide_height + args.small_height + args.top_middle_margin + args.middle_bottom_margin
    fig_width = left_margin + body_width + right_margin
    fig_height = bottom_margin + body_height + top_margin

    y_small = bottom_margin
    y_nan = y_small + args.small_height + args.middle_bottom_margin
    y_mse = y_nan + args.wide_height + args.top_middle_margin
    x_wide = left_margin + (body_width - args.wide_width) / 2
    x_small = left_margin + (body_width - bottom_width) / 2

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax_mse = add_axes(fig, fig_width, fig_height, x_wide, y_mse, args.wide_width, args.wide_height)
    ax_nan = add_axes(fig, fig_width, fig_height, x_wide, y_nan, args.wide_width, args.wide_height)
    ax_auto = add_axes(fig, fig_width, fig_height, x_small, y_small, args.small_width, args.small_height)
    ax_energy = add_axes(
        fig,
        fig_width,
        fig_height,
        x_small + args.small_width + args.bottom_column_margin,
        y_small,
        args.small_width,
        args.small_height,
    )

    draw_mse(ax_mse)
    draw_nan_hist(ax_nan)
    draw_autocorr(ax_auto)
    draw_energy(ax_energy)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    paths = save_outputs(fig, args.out)
    plt.close(fig)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
