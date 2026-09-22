from __future__ import annotations

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
SMOKE_DATA = HERE / "ar_sequential_inference_smoke.npz"
FULL_DATA = HERE / "ar_sequential_inference_data.npz"
SMOKE_OUTPUT = HERE / "ar_sequential_inference_smoke.png"
FULL_OUTPUT = HERE / "ar_sequential_inference.png"

CMAP = "RdBu_r"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Latin Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    }
)


def default_data_path() -> Path:
    return FULL_DATA if FULL_DATA.exists() else SMOKE_DATA


def default_output_path(data_path: Path) -> Path:
    return SMOKE_OUTPUT if "smoke" in data_path.stem else FULL_OUTPUT


def figure_output_paths(png_path: Path) -> tuple[Path, Path]:
    return png_path, png_path.with_suffix(".pdf")


def save_figure(fig, png_path: Path, dpi: int = 240) -> tuple[Path, Path]:
    outputs = figure_output_paths(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    for path in outputs:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return outputs


def parse_args() -> argparse.Namespace:
    data_path = default_data_path()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=data_path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--image-size", "--frame-size", dest="image_size", type=float, default=2.0)
    parser.add_argument(
        "--image-column-margin",
        "--frame-column-margin",
        "--frame-margin",
        dest="image_column_margin",
        type=float,
        default=0.11,
    )
    parser.add_argument(
        "--image-row-margin",
        "--rollout-row-margin",
        "--rollout-margin",
        dest="image_row_margin",
        type=float,
        default=0.675,
    )
    parser.add_argument("--panel-height", "--diagnostic-height", dest="panel_height", type=float, default=2.2)
    parser.add_argument("--panel-gap", "--diagnostic-column-margin", dest="panel_gap", type=float, default=1.0)
    parser.add_argument("--section-gap", "--rollout-diagnostics-margin", dest="section_gap", type=float, default=1.0)
    parser.add_argument("--left-margin", type=float, default=0.9)
    parser.add_argument("--right-margin", type=float, default=1.0)
    parser.add_argument("--bottom-margin", type=float, default=0.6)
    parser.add_argument("--top-margin", type=float, default=0.8)
    parser.add_argument("--colorbar-width", type=float, default=0.016)
    parser.add_argument("--colorbar-height", type=float, default=1.0)
    parser.add_argument("--colorbar-margin", type=float, default=0.05)
    parser.add_argument("--title-y", "--title-closeness", dest="title_y", type=float, default=0.97)
    parser.add_argument("--frame-title-pad", type=float, default=3.0)
    parser.add_argument("--line-width", type=float, default=1.15)
    parser.add_argument("--font-size", type=float, default=7.0)
    parser.add_argument("--label-font-size", type=float, default=8.0)
    parser.add_argument("--row-label-font-size", type=float, default=11.0)
    parser.add_argument("--title-font-size", type=float, default=8.6)
    parser.add_argument("--legend-font-size", type=float, default=6.8)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--no-title", action="store_true")
    return parser.parse_args()


def load_data(path: Path):
    return np.load(path, allow_pickle=False)


def add_axes(fig, fig_width, fig_height, x, y, width, height):
    return fig.add_axes([x / fig_width, y / fig_height, width / fig_width, height / fig_height])


def robust_vlim(*arrays: np.ndarray) -> tuple[float, float]:
    joined = np.concatenate([np.asarray(arr).reshape(-1) for arr in arrays])
    finite = joined[np.isfinite(joined)]
    vmax = float(np.percentile(np.abs(finite), 99.5)) if finite.size else 1.0
    vmax = max(vmax, 1e-6)
    return -vmax, vmax


def positive_band(mean, std, floor=1e-16):
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    return np.maximum(mean - std, floor), np.maximum(mean + std, floor)


def style_axis(ax, args):
    ax.tick_params(axis="both", labelsize=args.font_size, length=2.5, width=0.6)
    ax.grid(True, color="0.88", linewidth=0.45)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)


def draw_rollout(fig, data, args, fig_width, fig_height, y0):
    gt = data["rollout_gt"]
    pred = data["rollout_pred"]
    times = data["rollout_t"]
    n_cols = gt.shape[0]
    rows = [
        ("GT", gt),
        ("AR", pred),
    ]

    axes = np.empty((len(rows), n_cols), dtype=object)
    row_ims = []
    for row_idx, (label, values) in enumerate(rows):
        vmin, vmax = robust_vlim(values)
        row_y = y0 + (len(rows) - row_idx - 1) * (args.image_size + args.image_row_margin)
        im_row = None
        for col_idx in range(n_cols):
            x = args.left_margin + col_idx * (args.image_size + args.image_column_margin)
            ax = add_axes(fig, fig_width, fig_height, x, row_y, args.image_size, args.image_size)
            axes[row_idx, col_idx] = ax
            im_row = ax.imshow(values[col_idx], cmap=CMAP, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(f"k={int(times[col_idx])}", fontsize=args.font_size, pad=args.frame_title_pad)
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=args.row_label_font_size)
        row_ims.append(im_row)

    return axes, row_ims


def add_row_colorbar(fig, axes_row, im, args):
    boxes = [ax.get_position() for ax in axes_row]
    x1 = max(box.x1 for box in boxes)
    y0 = min(box.y0 for box in boxes)
    y1 = max(box.y1 for box in boxes)
    bar_height = (y1 - y0) * args.colorbar_height
    bar_y = y0 + ((y1 - y0) - bar_height) / 2
    cax = fig.add_axes([x1 + args.colorbar_margin, bar_y, args.colorbar_width, bar_height])
    fig.colorbar(im, cax=cax)


def draw_autocorr(ax, data, args):
    t = data["t_star"]
    gt = data["autocorr_gt_mean"]
    pred = data["autocorr_pred_mean"]
    ax.axhline(0.0, color="0.55", linestyle=":", linewidth=0.65)
    ax.plot(t, gt, color="black", linestyle="-", linewidth=args.line_width, label="GT")
    ax.plot(t, pred, color="black", linestyle="--", linewidth=args.line_width, label="AR")
    ax.set_xlim(float(t.min()), float(t.max()))
    ax.set_xlabel(r"$k^\ast$", fontsize=args.label_font_size)
    ax.set_ylabel(r"$R(k^\ast)$", fontsize=args.label_font_size)
    ax.set_title("(a) Autocorr.", fontsize=args.title_font_size, pad=4)
    ax.legend(frameon=False, fontsize=args.legend_font_size, handlelength=2.4)
    style_axis(ax, args)


def draw_energy(ax, data, args):
    k = data["k_bins"]
    gt = data["energy_gt_mean"]
    pred = data["energy_pred_mean"]
    ax.loglog(k, np.maximum(gt, 1e-16), color="black", linestyle="-", linewidth=args.line_width, label="GT")
    ax.loglog(k, np.maximum(pred, 1e-16), color="black", linestyle="--", linewidth=args.line_width, label="AR")
    ax.set_xlabel(r"$\kappa$", fontsize=args.label_font_size)
    ax.set_ylabel(r"$E(\kappa)$", fontsize=args.label_font_size)
    ax.set_title(r"(b) $E(\kappa)$", fontsize=args.title_font_size, pad=4)
    ax.legend(frameon=False, fontsize=args.legend_font_size, handlelength=2.4)
    style_axis(ax, args)


def draw_qq(ax, data, args):
    gt = data["qq_gt"]
    pred = data["qq_pred"]
    lim = float(max(np.nanmax(np.abs(gt)), np.nanmax(np.abs(pred)), 1e-6))
    ax.plot([-lim, lim], [-lim, lim], color="0.35", linestyle=":", linewidth=0.75)
    ax.scatter(gt, pred, s=3.0, color="black", alpha=0.55, linewidths=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$Q_{\mathrm{GT}}$", fontsize=args.label_font_size)
    ax.set_ylabel(r"$Q_{\mathrm{AR}}$", fontsize=args.label_font_size)
    ax.set_title("(c) Q–Q", fontsize=args.title_font_size, pad=4)
    style_axis(ax, args)


def build_figure(data, args):
    n_cols = int(data["rollout_gt"].shape[0])
    n_rows = 2
    image_width = n_cols * args.image_size + (n_cols - 1) * args.image_column_margin
    image_height = n_rows * args.image_size + (n_rows - 1) * args.image_row_margin
    panel_width = (image_width - 2 * args.panel_gap) / 3
    fig_width = args.left_margin + image_width + args.right_margin
    fig_height = args.bottom_margin + args.panel_height + args.section_gap + image_height + args.top_margin
    fig = plt.figure(figsize=(fig_width, fig_height))

    panel_y = args.bottom_margin
    drawers = (draw_autocorr, draw_energy, draw_qq)
    for col_idx, drawer in enumerate(drawers):
        x = args.left_margin + col_idx * (panel_width + args.panel_gap)
        ax = add_axes(fig, fig_width, fig_height, x, panel_y, panel_width, args.panel_height)
        drawer(ax, data, args)

    image_y = args.bottom_margin + args.panel_height + args.section_gap
    axes, row_ims = draw_rollout(fig, data, args, fig_width, fig_height, image_y)
    for row_idx, im in enumerate(row_ims):
        if im is None:
            continue
        add_row_colorbar(fig, axes[row_idx], im, args)

    if not args.no_title:
        fig.suptitle("Sequential AR inference", fontsize=args.title_font_size + 1.8, y=args.title_y)
    return fig


def plot_summary(data_path: Path, output_path: Path, args) -> tuple[Path, Path]:
    data = load_data(data_path)
    fig = build_figure(data, args)
    outputs = save_figure(fig, output_path, dpi=args.dpi)
    plt.close(fig)
    for path in outputs:
        print(path)
    return outputs


def main() -> None:
    args = parse_args()
    out = args.output or default_output_path(args.data)
    plot_summary(args.data, out, args)


if __name__ == "__main__":
    main()
