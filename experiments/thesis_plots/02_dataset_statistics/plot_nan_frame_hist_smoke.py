import argparse
import json
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

SUMMARY = Path(__file__).with_name("wandb-summary.json")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Latin Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
    }
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--total-trajectories", type=int, default=400)
    parser.add_argument("--bar-alpha", type=float, default=1.0)
    parser.add_argument("--hatch", type=str, default="///")
    parser.add_argument("--fig-width", type=float, default=9.0)
    parser.add_argument("--fig-height", type=float, default=3.6)
    parser.add_argument("--left", type=float, default=0.14)
    parser.add_argument("--right", type=float, default=0.96)
    parser.add_argument("--top", type=float, default=0.88)
    parser.add_argument("--bottom", type=float, default=0.16)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("nan_frame_hist_smoke.png"))
    return parser.parse_args()


def load_histogram(path):
    with path.open("r") as f:
        summary = json.load(f)

    histogram = summary["nan/first_nan_frame_hist"]
    dirty_count = int(summary["nan/total_dirty_trajs"])
    bins = np.asarray(histogram["bins"], dtype=np.float64)
    values = np.asarray(histogram["values"], dtype=np.float64)
    return bins, values, dirty_count


def save_outputs(fig, png_path):
    paths = (png_path, png_path.with_suffix(".pdf"))
    for path in paths:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return paths


def main():
    args = parse_args()
    bins, values, dirty_count = load_histogram(args.summary)
    valid_count = args.total_trajectories - dirty_count

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    widths = np.diff(bins)
    ax.bar(
        bins[:-1],
        values,
        width=widths,
        align="edge",
        facecolor="white",
        edgecolor="black",
        linewidth=0.8,
        hatch=args.hatch,
        alpha=args.bar_alpha,
    )

    ax.text(
        0.98,
        0.95,
        f"{valid_count} retained\n{dirty_count} removed",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )
    ax.set_xlabel(r"First non-finite $k$")
    ax.set_ylabel("Trajectory count")
    ax.set_title("Rejected trajectories")
    ax.grid(True, axis="y", color="0.85", linewidth=0.5)
    fig.subplots_adjust(left=args.left, right=args.right, bottom=args.bottom, top=args.top)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    paths = save_outputs(fig, args.out)
    plt.close(fig)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
