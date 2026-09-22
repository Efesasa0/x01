import argparse
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SMOKE_DATA = HERE / "lsq_latent_geometry_smoke.npz"
FULL_DATA = HERE / "lsq_latent_geometry_data.npz"
DATA = FULL_DATA if FULL_DATA.exists() else SMOKE_DATA
ROLLOUT_CMAP = "RdBu_r"


def output_paths(data_path):
    suffix = "_smoke" if "smoke" in Path(data_path).stem else ""
    return {
        "pca": HERE / f"lsq_latent_pca_projection{suffix}.png",
        "displacement": HERE / f"lsq_latent_displacement_distribution{suffix}.png",
        "covariance": HERE / f"lsq_latent_covariance_spectrum{suffix}.png",
    }

TMP_ROOT = Path(os.environ.get("TMPDIR", "/tmp"))
MPLCONFIGDIR = TMP_ROOT / "x01_mplconfig"
XDG_CACHE_HOME = TMP_ROOT / "x01_cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Latin Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
    }
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--pca-width", type=float, default=6.8)
    parser.add_argument("--pca-height", type=float, default=5.4)
    parser.add_argument("--pca-point-size", type=float, default=3.0)
    parser.add_argument("--pca-alpha", type=float, default=0.55)
    parser.add_argument("--pca-wspace", type=float, default=0.30)
    parser.add_argument("--pca-hspace", type=float, default=0.42)
    parser.add_argument("--pca-colorbar-width", type=float, default=0.028)
    parser.add_argument("--pca-colorbar-gap-scale", type=float, default=0.45)
    parser.add_argument("--distribution-width", type=float, default=6.8)
    parser.add_argument("--distribution-height", type=float, default=3.1)
    parser.add_argument("--covariance-width", type=float, default=6.8)
    parser.add_argument("--covariance-height", type=float, default=2.85)
    parser.add_argument("--covariance-wspace", type=float, default=0.30)
    parser.add_argument("--legend-length", type=float, default=2.4)
    parser.add_argument("--legend-font-size", type=float, default=8.0)
    parser.add_argument("--title-pad", type=float, default=8.0)
    parser.add_argument("--max-rank", type=int, default=80)
    return parser.parse_args()


def case_labels():
    return ["N01", "N02", "N08", "N40"]


def case_N(data, label):
    return int(np.asarray(data[f"{label}_N"]).item())


def line_style(index):
    styles = ["-", "--", ":", "-."]
    shades = ["0.15", "0.30", "0.45", "0.60"]
    return styles[index], shades[index]


def load_data(path):
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    return data, metadata


def set_panel_ticks(ax):
    ax.tick_params(axis="both", labelsize=8, length=2.5, width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def draw_pca_projection(data, args):
    fig, axes = plt.subplots(2, 2, figsize=(args.pca_width, args.pca_height))
    axes = axes.ravel()
    scatter = None

    for idx, label in enumerate(case_labels()):
        ax = axes[idx]
        coords = data[f"{label}_pca_coords"]
        time = data[f"{label}_pca_time"].astype(np.float64)
        t_max = max(float(time.max()), 1.0)
        color_value = time / t_max
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=color_value,
            cmap=ROLLOUT_CMAP,
            s=args.pca_point_size,
            alpha=args.pca_alpha,
            linewidths=0,
        )
        ax.axhline(0.0, color="0.84", linewidth=0.5, zorder=0)
        ax.axvline(0.0, color="0.84", linewidth=0.5, zorder=0)
        ax.set_title(rf"$N={case_N(data, label)}$", pad=args.title_pad, fontsize=10)
        ax.set_xlabel("PC1", fontsize=9)
        ax.set_ylabel("PC2", fontsize=9)
        set_panel_ticks(ax)

    fig.subplots_adjust(
        left=0.08,
        right=0.86,
        bottom=0.08,
        top=0.92,
        wspace=args.pca_wspace,
        hspace=args.pca_hspace,
    )
    if scatter is not None:
        top_box = axes[1].get_position()
        bot_box = axes[3].get_position()
        row_gap = top_box.y0 - bot_box.y1
        cbar_left = top_box.x1 + row_gap * args.pca_colorbar_gap_scale
        cbar_bottom = bot_box.y0
        cbar_height = top_box.y1 - bot_box.y0
        cax = fig.add_axes([cbar_left, cbar_bottom, args.pca_colorbar_width, cbar_height])
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label(r"normalized $k$", fontsize=9)
        cbar.ax.tick_params(labelsize=8, length=2.5, width=0.6)
    return fig


def percentile_table(values):
    return [
        np.percentile(values, 5),
        np.percentile(values, 25),
        np.percentile(values, 50),
        np.percentile(values, 75),
        np.percentile(values, 95),
    ]


def draw_distribution_panel(ax, data, key, ylabel, title):
    labels = case_labels()
    Ns = [case_N(data, label) for label in labels]
    values = [data[f"{label}_{key}"].astype(np.float64) for label in labels]

    box = ax.boxplot(
        values,
        positions=np.arange(len(labels)) + 1,
        widths=0.52,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "0.0", "linewidth": 1.0},
        boxprops={"facecolor": "0.92", "edgecolor": "0.0", "linewidth": 0.8},
        whiskerprops={"color": "0.0", "linewidth": 0.8},
        capprops={"color": "0.0", "linewidth": 0.8},
    )
    for patch, shade in zip(box["boxes"], ["0.95", "0.88", "0.80", "0.72"], strict=True):
        patch.set_facecolor(shade)

    medians = [percentile_table(v)[2] for v in values]
    ax.plot(np.arange(len(labels)) + 1, medians, color="0.0", linewidth=0.9)
    ax.set_xticks(np.arange(len(labels)) + 1)
    ax.set_xticklabels([rf"$N={N}$" for N in Ns])
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, pad=6, fontsize=10)
    ax.grid(axis="y", color="0.88", linewidth=0.5)
    set_panel_ticks(ax)


def draw_displacement_distribution(data, args):
    fig, axes = plt.subplots(1, 2, figsize=(args.distribution_width, args.distribution_height))
    draw_distribution_panel(
        axes[0],
        data,
        "displacement_norms",
        r"$\|z_{t+1}-z_t\|_2$",
        "absolute latent displacement",
    )
    draw_distribution_panel(
        axes[1],
        data,
        "relative_displacement_norms",
        r"$\|z_{t+1}-z_t\|_2 / \|z_t\|_2$",
        "relative latent displacement",
    )
    fig.tight_layout()
    return fig


def draw_covariance_spectrum(data, args):
    fig, axes = plt.subplots(1, 2, figsize=(args.covariance_width, args.covariance_height))
    rank_limit = max(args.max_rank, 1)

    for idx, label in enumerate(case_labels()):
        explained = np.maximum(data[f"{label}_cov_explained"].astype(np.float64), 1e-12)
        rank = np.arange(1, min(rank_limit, explained.size) + 1)
        linestyle, shade = line_style(idx)
        axes[0].plot(
            rank,
            explained[: rank.size],
            color=shade,
            linestyle=linestyle,
            linewidth=1.5,
            label=rf"$N={case_N(data, label)}$",
        )
        axes[1].plot(
            rank,
            np.cumsum(explained[: rank.size]),
            color=shade,
            linestyle=linestyle,
            linewidth=1.5,
            label=rf"$N={case_N(data, label)}$",
        )

    axes[0].set_yscale("log")
    axes[0].set_xlabel("latent covariance rank", fontsize=9)
    axes[0].set_ylabel("explained variance", fontsize=9)
    axes[0].set_title("covariance spectrum", pad=6, fontsize=10)

    axes[1].axhline(0.9, color="0.72", linestyle="--", linewidth=0.8)
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_xlabel("latent covariance rank", fontsize=9)
    axes[1].set_ylabel("cumulative explained variance", fontsize=9)
    axes[1].set_title("cumulative spectrum", pad=6, fontsize=10)

    for ax in axes:
        ax.grid(True, which="both", color="0.88", linewidth=0.5)
        ax.legend(frameon=False, handlelength=args.legend_length, fontsize=args.legend_font_size)
        set_panel_ticks(ax)

    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.18, top=0.84, wspace=args.covariance_wspace)
    pca_axis_height = (0.92 - 0.08) / (2.0 + args.pca_hspace)
    pca_row_gap = args.pca_hspace * pca_axis_height
    gutter_left = 0.86 + pca_row_gap * args.pca_colorbar_gap_scale
    fig.add_artist(
        Rectangle(
            (gutter_left, 0.08),
            0.08,
            0.84,
            transform=fig.transFigure,
            facecolor="none",
            edgecolor="none",
            linewidth=0.0,
        )
    )
    return fig


def figure_output_paths(png_path):
    return (png_path, png_path.with_suffix(".pdf"))


def save_figure(fig, png_path):
    outputs = figure_output_paths(png_path)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return outputs


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    paths = save_figure(fig, path)
    plt.close(fig)
    for saved_path in paths:
        print(saved_path)


def build_figures(args):
    data, _metadata = load_data(args.data)
    return {
        "pca": draw_pca_projection(data, args),
        "displacement": draw_displacement_distribution(data, args),
        "covariance": draw_covariance_spectrum(data, args),
    }


def main():
    args = parse_args()
    figures = build_figures(args)
    outs = output_paths(args.data)
    save(figures["pca"], args.output_dir / outs["pca"].name)
    save(figures["displacement"], args.output_dir / outs["displacement"].name)
    save(figures["covariance"], args.output_dir / outs["covariance"].name)


if __name__ == "__main__":
    main()
