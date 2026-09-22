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

HERE = Path(__file__).resolve().parent
SMOKE_DATA = HERE / "blowup_rollout_smoke.npz"
FULL_DATA = HERE / "blowup_rollout_data.npz"
SMALL_WIDTH = 3.75
SMALL_HEIGHT = 2.2
PANEL_MARGIN = 1.0

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Latin Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
    }
)


def parse_args():
    default_data = FULL_DATA if FULL_DATA.exists() else SMOKE_DATA
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--frame-column-margin", "--frame-margin", dest="frame_column_margin", type=float, default=0.11)
    parser.add_argument("--rollout-row-margin", "--rollout-margin", dest="rollout_row_margin", type=float, default=0.675)
    parser.add_argument("--colorbar-width", type=float, default=0.016)
    parser.add_argument("--colorbar-height", type=float, default=1.0)
    parser.add_argument("--colorbar-margin", type=float, default=0.05)
    parser.add_argument("--title-closeness", type=float, default=0.97)
    parser.add_argument("--frame-title-pad", type=float, default=3.0)
    return parser.parse_args()


def output_paths(data_path):
    suffix = "_smoke" if "smoke" in data_path.stem else ""
    return (
        HERE / f"blowup_diagnostics{suffix}.png",
        HERE / f"blowup_rollouts{suffix}.png",
    )


def figure_output_paths(png_path):
    return (png_path, png_path.with_suffix(".pdf"))


def save_figure(fig, png_path):
    outputs = figure_output_paths(png_path)
    for path in outputs:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return outputs


def load_data(path):
    return np.load(path, allow_pickle=False)


def add_axes(fig, fig_width, fig_height, x, y, width, height):
    return fig.add_axes([x / fig_width, y / fig_height, width / fig_width, height / fig_height])


def draw_spectrum(ax, data):
    ax.plot(data["eig_epochs"], data["eig_forward_min"], color="black", linestyle=":", linewidth=1.4, label="min")
    ax.plot(data["eig_epochs"], data["eig_forward_mean"], color="black", linestyle="-", linewidth=1.4, label="mean")
    ax.plot(data["eig_epochs"], data["eig_forward_max"], color="black", linestyle="--", linewidth=1.4, label="max")
    ax.axhline(1.0, color="0.45", linestyle="-.", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$|\lambda(K)|$")
    ax.set_title("(a) Forward spectrum")
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.legend(frameon=False, handlelength=2.8, fontsize=8)
    ax.grid(True, color="0.88", linewidth=0.5)


def draw_losses(ax, data):
    ax.plot(data["loss_epochs"], data["train_loss"], color="black", linestyle="-", linewidth=1.4, label="train")
    ax.plot(data["loss_epochs"], data["val_loss"], color="black", linestyle="--", linewidth=1.4, label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("(b) Training curves")
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.legend(frameon=False, handlelength=2.8, fontsize=8)
    ax.grid(True, color="0.88", linewidth=0.5)


def draw_max_vorticity(ax, data):
    max_vorticity = np.maximum(data["max_abs_vorticity"], 1e-12)
    ax.plot(data["epochs"], max_vorticity, color="black", linestyle="-", marker="o", markersize=3.5, linewidth=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"max $|\omega|$")
    ax.set_title("(c) Rollout magnitude")
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, which="both", color="0.88", linewidth=0.5)


def plot_diagnostics(data, out_path):
    panel_width = SMALL_WIDTH
    panel_height = SMALL_HEIGHT
    panel_margin = PANEL_MARGIN
    left_margin = 0.75
    right_margin = 0.96
    top_margin = 0.35
    bottom_margin = 0.55

    body_width = 3 * panel_width + 2 * panel_margin
    fig_width = left_margin + body_width + right_margin
    fig_height = bottom_margin + panel_height + top_margin
    y = bottom_margin
    x0 = left_margin

    fig = plt.figure(figsize=(fig_width, fig_height))
    axes = [
        add_axes(fig, fig_width, fig_height, x0 + i * (panel_width + panel_margin), y, panel_width, panel_height)
        for i in range(3)
    ]

    draw_spectrum(axes[0], data)
    draw_losses(axes[1], data)
    draw_max_vorticity(axes[2], data)

    outputs = save_figure(fig, out_path)
    plt.close(fig)
    return outputs


def selected_frames(n_frames):
    frame_idx = np.asarray([0, 25, 50, 100, 200, 319], dtype=int)
    if np.any(frame_idx >= n_frames):
        raise ValueError(f"requested frames {frame_idx.tolist()} but data only has {n_frames} frames")
    return frame_idx


def display_rollouts(data):
    epochs = data["epochs"]
    rows = [data["gt"]]
    labels = ["GT"]

    for epoch in data["display_epochs"]:
        idx = np.flatnonzero(epochs == epoch)
        if len(idx) != 1:
            raise ValueError(f"display epoch {epoch} not found in saved epochs {epochs}")
        rows.append(data["pred"][idx[0]])
        labels.append(f"Epoch {int(epoch)}")

    return rows, labels


def layout_margins(args):
    frame_column_margin = getattr(args, "frame_column_margin", getattr(args, "frame_margin", 0.06))
    rollout_row_margin = getattr(args, "rollout_row_margin", getattr(args, "rollout_margin", 0.06))
    return frame_column_margin, rollout_row_margin


def rollout_layout(n_rows, n_cols, frame_column_margin, rollout_row_margin):
    image_size = 2.0
    left_margin = 0.9
    right_margin = 1.0
    bottom_margin = 0.35
    top_margin = 0.8

    body_width = n_cols * image_size + (n_cols - 1) * frame_column_margin
    body_height = n_rows * image_size + (n_rows - 1) * rollout_row_margin
    fig_width = left_margin + body_width + right_margin
    fig_height = bottom_margin + body_height + top_margin
    return fig_width, fig_height, left_margin, bottom_margin, image_size


def build_rollout_figure(data, args):
    rows, labels = display_rollouts(data)
    frame_idx = selected_frames(data["gt"].shape[0])
    frame_column_margin, rollout_row_margin = layout_margins(args)
    n_rows = len(rows)
    n_cols = len(frame_idx)
    fig_width, fig_height, left_margin, bottom_margin, image_size = rollout_layout(
        n_rows,
        n_cols,
        frame_column_margin,
        rollout_row_margin,
    )

    fig = plt.figure(figsize=(fig_width, fig_height))
    axes = np.empty((n_rows, n_cols), dtype=object)
    row_ims = []

    for row, frames in enumerate(rows):
        selected = frames[frame_idx].reshape(-1)
        row_vmax = float(np.percentile(np.abs(selected), 99.0))
        row_vmax = max(row_vmax, 1e-12)
        row_vmin = -row_vmax
        im_row = None
        for col, timestep in enumerate(frame_idx):
            x = left_margin + col * (image_size + frame_column_margin)
            y = bottom_margin + (n_rows - row - 1) * (image_size + rollout_row_margin)
            ax = add_axes(fig, fig_width, fig_height, x, y, image_size, image_size)
            axes[row, col] = ax
            im_row = ax.imshow(frames[timestep], cmap="RdBu_r", vmin=row_vmin, vmax=row_vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"k={int(timestep)}", fontsize=8, pad=args.frame_title_pad)
            if col == 0:
                ax.set_ylabel(labels[row], fontsize=11)
        row_ims.append(im_row)

    fig.suptitle("Numerical-instability rollout snapshots", fontsize=13, y=args.title_closeness)
    for row in range(n_rows):
        row_axes = [axes[row, col] for col in range(n_cols)]
        boxes = [ax.get_position() for ax in row_axes]
        x1 = max(box.x1 for box in boxes)
        y0 = min(box.y0 for box in boxes)
        y1 = max(box.y1 for box in boxes)
        colorbar_height = (y1 - y0) * args.colorbar_height
        colorbar_y = y0 + ((y1 - y0) - colorbar_height) / 2
        colorbar_ax = fig.add_axes([
            x1 + args.colorbar_margin,
            colorbar_y,
            args.colorbar_width,
            colorbar_height,
        ])
        fig.colorbar(row_ims[row], cax=colorbar_ax)
    return fig


def plot_rollouts(data, out_path, args):
    fig = build_rollout_figure(data, args)
    outputs = save_figure(fig, out_path)
    plt.close(fig)
    return outputs


def main():
    args = parse_args()
    data = load_data(args.data)
    diagnostics_out, rollouts_out = output_paths(args.data)
    diagnostics_paths = plot_diagnostics(data, diagnostics_out)
    rollout_paths = plot_rollouts(data, rollouts_out, args)
    for path in (*diagnostics_paths, *rollout_paths):
        print(path)


if __name__ == "__main__":
    main()
