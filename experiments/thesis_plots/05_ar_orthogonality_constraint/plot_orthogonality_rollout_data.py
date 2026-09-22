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
SMOKE_DATA = HERE / "orthogonality_rollout_smoke.npz"
FULL_DATA = HERE / "orthogonality_rollout_data.npz"

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
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--frame-column-margin", "--frame-margin", dest="frame_column_margin", type=float, default=0.11)
    parser.add_argument("--rollout-row-margin", "--rollout-margin", dest="rollout_row_margin", type=float, default=0.675)
    parser.add_argument("--rollout-diagnostics-margin", type=float, default=1.0)
    parser.add_argument("--diagnostic-column-margin", type=float, default=1.0)
    parser.add_argument("--diagnostic-height", type=float, default=2.2)
    parser.add_argument("--colorbar-width", type=float, default=0.016)
    parser.add_argument("--colorbar-height", type=float, default=1.0)
    parser.add_argument("--colorbar-margin", type=float, default=0.05)
    parser.add_argument("--title-closeness", type=float, default=0.97)
    parser.add_argument("--frame-title-pad", type=float, default=3.0)
    return parser.parse_args()


def output_path(data_path):
    suffix = "_smoke" if "smoke" in data_path.stem else ""
    return HERE / f"orthogonality_summary{suffix}.png"


def load_data(path):
    return np.load(path, allow_pickle=False)


def add_axes(fig, fig_width, fig_height, x, y, width, height):
    return fig.add_axes([x / fig_width, y / fig_height, width / fig_width, height / fig_height])


def figure_output_paths(png_path):
    return (png_path, png_path.with_suffix(".pdf"))


def save_figure(fig, png_path):
    outputs = figure_output_paths(png_path)
    for path in outputs:
        fig.savefig(path, dpi=200, bbox_inches="tight")
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


def draw_rollout_mse(ax, data):
    timesteps = data["timesteps"]
    mse_epochs = data["mse_epochs"]
    rollout_mse = np.maximum(data["rollout_mse"], 1e-12)
    styles = ("-", "--", ":", "-.")

    for idx, epoch in enumerate(mse_epochs):
        shade = 0.72 - 0.55 * idx / max(len(mse_epochs) - 1, 1)
        linewidth = 1.5 if epoch in (0, 50, 100) else 0.9
        label = f"{int(epoch)}" if epoch in (0, 50, 100) else None
        ax.plot(
            timesteps,
            rollout_mse[idx],
            color=str(shade),
            linestyle=styles[idx % len(styles)],
            linewidth=linewidth,
            label=label,
        )

    ax.set_xlabel(r"$k$")
    ax.set_ylabel("Mean MSE")
    ax.set_title("(b) Mean rollout MSE")
    ax.legend(title="Epoch", frameon=False, handlelength=2.4, fontsize=7, title_fontsize=8)
    ax.grid(True, color="0.88", linewidth=0.5)


def draw_spectrum(ax, data):
    ax.plot(data["eig_epochs"], data["eig_forward_min"], color="black", linestyle=":", linewidth=1.4, label="min")
    ax.plot(data["eig_epochs"], data["eig_forward_mean"], color="black", linestyle="-", linewidth=1.4, label="mean")
    ax.plot(data["eig_epochs"], data["eig_forward_max"], color="black", linestyle="--", linewidth=1.4, label="max")
    ax.axhline(1.0, color="0.45", linestyle="-.", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$|\lambda(K)|$")
    ax.set_title("(c) Forward spectrum")
    ax.legend(frameon=False, handlelength=2.8, fontsize=8)
    ax.grid(True, color="0.88", linewidth=0.5)


def draw_losses(ax, data):
    ax.plot(data["loss_epochs"], data["train_loss"], color="black", linestyle="-", linewidth=1.4, label="train")
    ax.plot(data["loss_epochs"], data["val_loss"], color="black", linestyle="--", linewidth=1.4, label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("(d) Training curves")
    ax.legend(frameon=False, handlelength=2.8, fontsize=8)
    ax.grid(True, color="0.88", linewidth=0.5)


def layout(args, n_rows, n_cols):
    image_size = 2.0
    left_margin = 0.9
    right_margin = 1.0
    bottom_margin = 0.6
    top_margin = 0.8

    rollout_width = n_cols * image_size + (n_cols - 1) * args.frame_column_margin
    rollout_height = n_rows * image_size + (n_rows - 1) * args.rollout_row_margin
    diagnostics_width = (rollout_width - 2 * args.diagnostic_column_margin) / 3
    body_height = args.diagnostic_height + args.rollout_diagnostics_margin + rollout_height
    fig_width = left_margin + rollout_width + right_margin
    fig_height = bottom_margin + body_height + top_margin
    rollout_bottom = bottom_margin + args.diagnostic_height + args.rollout_diagnostics_margin

    return {
        "image_size": image_size,
        "left_margin": left_margin,
        "bottom_margin": bottom_margin,
        "rollout_width": rollout_width,
        "rollout_bottom": rollout_bottom,
        "diagnostics_width": diagnostics_width,
        "fig_width": fig_width,
        "fig_height": fig_height,
    }


def build_figure(data, args):
    rows, labels = display_rollouts(data)
    frame_idx = selected_frames(data["gt"].shape[0])
    n_rows = len(rows)
    n_cols = len(frame_idx)
    sizes = layout(args, n_rows, n_cols)

    fig = plt.figure(figsize=(sizes["fig_width"], sizes["fig_height"]))
    rollout_axes = np.empty((n_rows, n_cols), dtype=object)
    row_ims = []

    for row, frames in enumerate(rows):
        selected = frames[frame_idx].reshape(-1)
        row_vmax = float(np.percentile(np.abs(selected), 99.0))
        row_vmax = max(row_vmax, 1e-12)
        row_vmin = -row_vmax
        im_row = None
        for col, timestep in enumerate(frame_idx):
            x = sizes["left_margin"] + col * (sizes["image_size"] + args.frame_column_margin)
            y = sizes["rollout_bottom"] + (n_rows - row - 1) * (sizes["image_size"] + args.rollout_row_margin)
            ax = add_axes(fig, sizes["fig_width"], sizes["fig_height"], x, y, sizes["image_size"], sizes["image_size"])
            rollout_axes[row, col] = ax
            im_row = ax.imshow(frames[timestep], cmap="RdBu_r", vmin=row_vmin, vmax=row_vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"k={int(timestep)}", fontsize=8, pad=args.frame_title_pad)
            if col == 0:
                ax.set_ylabel(labels[row], fontsize=11)
        row_ims.append(im_row)

    mse_ax = add_axes(
        fig,
        sizes["fig_width"],
        sizes["fig_height"],
        sizes["left_margin"],
        sizes["bottom_margin"],
        sizes["diagnostics_width"],
        args.diagnostic_height,
    )
    spectrum_ax = add_axes(
        fig,
        sizes["fig_width"],
        sizes["fig_height"],
        sizes["left_margin"] + sizes["diagnostics_width"] + args.diagnostic_column_margin,
        sizes["bottom_margin"],
        sizes["diagnostics_width"],
        args.diagnostic_height,
    )
    loss_ax = add_axes(
        fig,
        sizes["fig_width"],
        sizes["fig_height"],
        sizes["left_margin"] + 2 * (sizes["diagnostics_width"] + args.diagnostic_column_margin),
        sizes["bottom_margin"],
        sizes["diagnostics_width"],
        args.diagnostic_height,
    )
    draw_rollout_mse(mse_ax, data)
    draw_spectrum(spectrum_ax, data)
    draw_losses(loss_ax, data)

    fig.suptitle("Orthogonality-constrained rollout diagnostics", fontsize=13, y=args.title_closeness)
    for row in range(n_rows):
        row_axes = [rollout_axes[row, col] for col in range(n_cols)]
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


def plot_summary(data, out_path, args):
    fig = build_figure(data, args)
    outputs = save_figure(fig, out_path)
    plt.close(fig)
    return outputs


def main():
    args = parse_args()
    data = load_data(args.data)
    out_path = args.output if args.output is not None else output_path(args.data)
    paths = plot_summary(data, out_path, args)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
