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
CMAP = "RdBu_r"

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
    parser.add_argument("--old-traj", type=int, default=0)
    parser.add_argument("--new-traj", type=int, default=0)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 25, 50, 100, 200, 319])
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--image-size", "--frame-size", dest="image_size", type=float, default=2.0)
    parser.add_argument("--panel-margin", type=float, default=None)
    parser.add_argument(
        "--image-column-margin",
        "--frame-column-margin",
        "--frame-margin",
        dest="image_column_margin",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--image-row-margin",
        "--rollout-row-margin",
        "--rollout-margin",
        dest="image_row_margin",
        type=float,
        default=None,
    )
    parser.add_argument("--wspace", type=float, default=None)
    parser.add_argument("--hspace", type=float, default=None)
    parser.add_argument("--colorbar-width", type=float, default=0.016)
    parser.add_argument("--colorbar-height", "--legend-length", dest="colorbar_height", type=float, default=1.0)
    parser.add_argument("--colorbar-margin", "--legend-margin", dest="colorbar_margin", type=float, default=0.05)
    parser.add_argument("--top-margin", "--top", dest="top_margin", type=float, default=0.8)
    parser.add_argument("--bottom-margin", "--bottom", dest="bottom_margin", type=float, default=0.35)
    parser.add_argument("--left-margin", "--left", dest="left_margin", type=float, default=0.9)
    parser.add_argument("--right-margin", "--right", dest="right_margin", type=float, default=1.0)
    parser.add_argument("--title-y", "--title-closeness", dest="title_y", type=float, default=0.97)
    parser.add_argument("--frame-title-pad", type=float, default=3.0)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--no-title", action="store_true")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("dataset_samples.png"))
    return parser.parse_args()


def load_row(path, traj_idx, frames):
    data = np.load(path, mmap_mode="r")
    return np.asarray(data[traj_idx, frames], dtype=np.float32)


def save_outputs(fig, png_path, dpi):
    paths = (png_path, png_path.with_suffix(".pdf"))
    for path in paths:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return paths


def add_axes(fig, fig_width, fig_height, x, y, width, height):
    return fig.add_axes([x / fig_width, y / fig_height, width / fig_width, height / fig_height])


def layout_margins(args):
    column_margin = args.image_column_margin
    if column_margin is None:
        column_margin = args.wspace if args.wspace is not None else args.panel_margin
    if column_margin is None:
        column_margin = 0.11

    row_margin = args.image_row_margin
    if row_margin is None:
        row_margin = args.hspace if args.hspace is not None else args.panel_margin
    if row_margin is None:
        row_margin = 0.675

    return float(column_margin), float(row_margin)


def add_row_colorbar(fig, axes_row, im, args):
    boxes = [ax.get_position() for ax in axes_row]
    x1 = max(box.x1 for box in boxes)
    y0 = min(box.y0 for box in boxes)
    y1 = max(box.y1 for box in boxes)
    bar_height = (y1 - y0) * args.colorbar_height
    bar_y = y0 + ((y1 - y0) - bar_height) / 2
    cax = fig.add_axes([x1 + args.colorbar_margin, bar_y, args.colorbar_width, bar_height])
    fig.colorbar(im, cax=cax)


def build_figure(old_frames, new_frames, frame_labels, vmin, vmax, args):
    rows = [
        ("Initial dataset", old_frames),
        ("Generated dataset", new_frames),
    ]
    n_rows = len(rows)
    n_cols = len(frame_labels)
    column_margin, row_margin = layout_margins(args)

    image_width = n_cols * args.image_size + (n_cols - 1) * column_margin
    image_height = n_rows * args.image_size + (n_rows - 1) * row_margin
    fig_width = args.left_margin + image_width + args.right_margin
    fig_height = args.bottom_margin + image_height + args.top_margin

    fig = plt.figure(figsize=(fig_width, fig_height))
    axes = np.empty((n_rows, n_cols), dtype=object)
    row_ims = []

    for row_idx, (label, frames) in enumerate(rows):
        y = args.bottom_margin + (n_rows - row_idx - 1) * (args.image_size + row_margin)
        im_row = None
        for col_idx, frame in enumerate(frames):
            x = args.left_margin + col_idx * (args.image_size + column_margin)
            ax = add_axes(fig, fig_width, fig_height, x, y, args.image_size, args.image_size)
            axes[row_idx, col_idx] = ax
            im_row = ax.imshow(frame, cmap=CMAP, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(f"k={int(frame_labels[col_idx])}", fontsize=8, pad=args.frame_title_pad)
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=11)
        row_ims.append(im_row)

    for row_idx, im in enumerate(row_ims):
        if im is not None:
            add_row_colorbar(fig, axes[row_idx], im, args)

    if not args.no_title:
        fig.suptitle("Dataset sample trajectory vorticity fields", fontsize=13, y=args.title_y)
    return fig


def main():
    args = parse_args()
    old_frames = load_row(args.old_data, args.old_traj, args.frames)
    new_frames = load_row(args.new_data, args.new_traj, args.frames)

    selected = np.concatenate([old_frames.reshape(-1), new_frames.reshape(-1)])
    vmax = np.percentile(np.abs(selected), args.percentile)
    vmin = -vmax

    fig = build_figure(old_frames, new_frames, args.frames, vmin, vmax, args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    paths = save_outputs(fig, args.out, args.dpi)
    plt.close(fig)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
