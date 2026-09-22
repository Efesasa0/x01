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

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "lsq_linearizability_mse.png"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Latin Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
    }
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--width", type=float, default=6.8)
    parser.add_argument("--height", type=float, default=3.2)
    parser.add_argument("--legend-length", type=float, default=2.8)
    parser.add_argument("--legend-font-size", type=float, default=8.0)
    parser.add_argument("--title-pad", type=float, default=8.0)
    parser.add_argument("--ymin", type=float, default=1e-3)
    parser.add_argument("--ymax", type=float, default=2.0)
    parser.add_argument("--linear-y", action="store_true")
    return parser.parse_args()


def summary_paths():
    return [
        HERE / "lsq_summary_N01.json",
        HERE / "lsq_summary_N02.json",
        HERE / "lsq_summary_N08.json",
        HERE / "lsq_summary_N40.json",
    ]


def load_lsq_curve(path):
    data = json.loads(path.read_text())
    lsq = data["methods"]["lsq"]
    per_traj = lsq["rollout_mse_per_t_per_traj"]
    curves = [np.asarray(curve, dtype=np.float64) for _, curve in sorted(per_traj.items(), key=lambda item: int(item[0]))]
    min_len = min(len(curve) for curve in curves)
    stacked = np.stack([curve[:min_len] for curve in curves], axis=0)
    mean = stacked.mean(axis=0)

    return {
        "N": int(data["pick_n_trajs"]),
        "timesteps": np.arange(min_len),
        "mean_mse": mean,
        "final_mean": float(lsq["rollout_mse_final_mean"]),
        "latent_rmse_rel": float(lsq["latent_1step_rmse_rel_joint"]),
        "n_pairs": int(data["n_pairs"]),
    }


def draw_plot(ax, records, args):
    linestyles = {
        1: "-",
        2: "--",
        8: ":",
        40: "-.",
    }
    linewidths = {
        1: 1.5,
        2: 1.5,
        8: 1.7,
        40: 1.7,
    }

    for idx, record in enumerate(records):
        shade = 0.15 + 0.18 * idx
        ax.plot(
            record["timesteps"],
            np.maximum(record["mean_mse"], 1e-12),
            color=str(shade),
            linestyle=linestyles[record["N"]],
            linewidth=linewidths[record["N"]],
            label=rf"$N={record['N']}$",
        )

    if not args.linear_y:
        ax.set_yscale("log")
        ax.set_ylim(args.ymin, args.ymax)
    ax.set_xlabel(r"$k$")
    ylabel = "Rollout MSE"
    if not args.linear_y:
        ylabel += " (log scale)"
    ax.set_ylabel(ylabel)
    ax.set_title("Least-squares latent linearizability probe", pad=args.title_pad)
    ax.legend(frameon=False, handlelength=args.legend_length, fontsize=args.legend_font_size)
    ax.grid(True, which="both", color="0.88", linewidth=0.5)


def build_figure(args):
    records = [load_lsq_curve(path) for path in summary_paths()]
    records.sort(key=lambda record: record["N"])
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    draw_plot(ax, records, args)
    fig.tight_layout()
    return fig


def figure_output_paths(png_path):
    return (png_path, png_path.with_suffix(".pdf"))


def save_figure(fig, png_path):
    outputs = figure_output_paths(png_path)
    for path in outputs:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return outputs


def main():
    args = parse_args()
    fig = build_figure(args)
    paths = save_figure(fig, args.output)
    plt.close(fig)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
