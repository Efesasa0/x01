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
SMOKE_DATA = HERE / "ar_further_attempts_smoke.npz"
FULL_DATA = HERE / "ar_further_attempts_data.npz"
SMOKE_OUTPUT = HERE / "ar_further_attempts_summary_smoke.png"
FULL_OUTPUT = HERE / "ar_further_attempts_summary.png"

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


def save_figure(fig, png_path: Path, dpi: int = 220) -> tuple[Path, Path]:
    outputs = figure_output_paths(png_path)
    for path in outputs:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return outputs


def parse_args() -> argparse.Namespace:
    data_path = default_data_path()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=data_path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--panel-width", type=float, default=2.15)
    parser.add_argument("--panel-height", type=float, default=1.62)
    parser.add_argument("--column-margin", type=float, default=0.62)
    parser.add_argument("--row-margin", type=float, default=0.62)
    parser.add_argument("--left-margin", type=float, default=1.78)
    parser.add_argument("--right-margin", type=float, default=0.28)
    parser.add_argument("--bottom-margin", type=float, default=0.58)
    parser.add_argument("--top-margin", type=float, default=0.74)
    parser.add_argument("--title-y", type=float, default=0.985)
    parser.add_argument("--row-label-x", type=float, default=-0.34)
    parser.add_argument("--line-width", type=float, default=1.05)
    parser.add_argument("--font-size", type=float, default=7.2)
    parser.add_argument("--title-font-size", type=float, default=8.2)
    parser.add_argument("--label-font-size", type=float, default=8.0)
    parser.add_argument("--legend-font-size", type=float, default=6.6)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--no-title", action="store_true")
    return parser.parse_args()


def load_data(path: Path):
    return np.load(path, allow_pickle=False)


def add_axes(fig, fig_width, fig_height, x, y, width, height):
    return fig.add_axes([x / fig_width, y / fig_height, width / fig_width, height / fig_height])


def finite_xy(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def positive_band(mean, std, floor=1e-16):
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    low = np.maximum(mean - std, floor)
    high = np.maximum(mean + std, floor)
    return low, high


def style_axis(ax, args):
    ax.tick_params(axis="both", labelsize=args.font_size, length=2.5, width=0.6)
    ax.grid(True, color="0.88", linewidth=0.45)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)


def maybe_top_title(ax, title, row_idx, args):
    if row_idx == 0:
        ax.set_title(title, fontsize=args.title_font_size, pad=5)


def draw_autocorr(ax, data, row_idx, args):
    t = data["t_star"]
    gt = data["autocorr_gt_mean"][row_idx]
    pred = data["autocorr_pred_mean"][row_idx]

    ax.axhline(0.0, color="0.55", linestyle=":", linewidth=0.65)
    ax.plot(t, gt, color="black", linestyle="-", linewidth=args.line_width, label="GT")
    ax.plot(t, pred, color="black", linestyle="--", linewidth=args.line_width, label="model")
    ax.set_xlim(float(t.min()), float(t.max()))
    ax.set_xlabel(r"$k^\ast$", fontsize=args.label_font_size)
    ax.set_ylabel(r"$R(k^\ast)$", fontsize=args.label_font_size)
    maybe_top_title(ax, "Autocorr.", row_idx, args)
    if row_idx == 0:
        ax.legend(frameon=False, fontsize=args.legend_font_size, loc="upper right", handlelength=2.4)
    style_axis(ax, args)


def draw_energy(ax, data, row_idx, args):
    k = data["k_bins"]
    gt = data["energy_gt_mean"][row_idx]
    pred = data["energy_pred_mean"][row_idx]

    ax.loglog(k, np.maximum(gt, 1e-16), color="black", linestyle="-", linewidth=args.line_width, label="GT")
    ax.loglog(k, np.maximum(pred, 1e-16), color="black", linestyle="--", linewidth=args.line_width, label="model")
    ax.set_xlabel(r"$\kappa$", fontsize=args.label_font_size)
    ax.set_ylabel(r"$E(\kappa)$", fontsize=args.label_font_size)
    maybe_top_title(ax, r"$E(\kappa)$", row_idx, args)
    style_axis(ax, args)


def draw_qq(ax, data, row_idx, args):
    gt = data["qq_gt"][row_idx]
    pred = data["qq_pred"][row_idx]
    lim = float(max(np.nanmax(np.abs(gt)), np.nanmax(np.abs(pred)), 1e-6))
    ax.plot([-lim, lim], [-lim, lim], color="0.35", linestyle=":", linewidth=0.75)
    ax.scatter(pred, gt, s=3.0, color="black", alpha=0.55, linewidths=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$Q_{\mathrm{model}}$", fontsize=args.label_font_size)
    ax.set_ylabel(r"$Q_{\mathrm{GT}}$", fontsize=args.label_font_size)
    maybe_top_title(ax, "Q–Q", row_idx, args)
    style_axis(ax, args)


def draw_spectrum(ax, data, row_idx, args):
    x_min, y_min = finite_xy(data["eig_epochs"][row_idx], data["eig_forward_min"][row_idx])
    x_mean, y_mean = finite_xy(data["eig_epochs"][row_idx], data["eig_forward_mean"][row_idx])
    x_max, y_max = finite_xy(data["eig_epochs"][row_idx], data["eig_forward_max"][row_idx])
    ax.plot(x_min, y_min, color="black", linestyle=":", linewidth=args.line_width, label="min")
    ax.plot(x_mean, y_mean, color="black", linestyle="-", linewidth=args.line_width, label="mean")
    ax.plot(x_max, y_max, color="black", linestyle="--", linewidth=args.line_width, label="max")
    ax.axhline(1.0, color="0.55", linestyle="-.", linewidth=0.7)
    ax.set_xlabel("Epoch", fontsize=args.label_font_size)
    ax.set_ylabel(r"$|\lambda(K)|$", fontsize=args.label_font_size)
    maybe_top_title(ax, "Forward spectrum", row_idx, args)
    if row_idx == 0:
        ax.legend(frameon=False, fontsize=args.legend_font_size, loc="best", handlelength=2.0)
    style_axis(ax, args)


def draw_losses(ax, data, row_idx, args):
    x_train, train = finite_xy(data["loss_epochs"][row_idx], data["train_loss"][row_idx])
    x_val, val = finite_xy(data["loss_epochs"][row_idx], data["val_loss"][row_idx])
    ax.plot(x_train, train, color="black", linestyle="-", linewidth=args.line_width, label="train")
    ax.plot(x_val, val, color="black", linestyle="--", linewidth=args.line_width, label="validation")
    ax.set_xlabel("Epoch", fontsize=args.label_font_size)
    ax.set_ylabel("Loss", fontsize=args.label_font_size)
    maybe_top_title(ax, "Training curves", row_idx, args)
    if row_idx == 0:
        ax.legend(frameon=False, fontsize=args.legend_font_size, loc="best", handlelength=2.0)
    style_axis(ax, args)


PANEL_DRAWERS = (
    draw_autocorr,
    draw_energy,
    draw_qq,
    draw_spectrum,
    draw_losses,
)


def layout(data, args):
    n_rows = len(data["experiment_ids"])
    n_cols = len(PANEL_DRAWERS)
    body_width = n_cols * args.panel_width + (n_cols - 1) * args.column_margin
    body_height = n_rows * args.panel_height + (n_rows - 1) * args.row_margin
    fig_width = args.left_margin + body_width + args.right_margin
    fig_height = args.bottom_margin + body_height + args.top_margin
    return n_rows, n_cols, fig_width, fig_height


def build_figure(data, args):
    n_rows, n_cols, fig_width, fig_height = layout(data, args)
    fig = plt.figure(figsize=(fig_width, fig_height))
    axes = np.empty((n_rows, n_cols), dtype=object)
    labels = [str(label) for label in data["experiment_labels"]]

    for row_idx in range(n_rows):
        for col_idx, drawer in enumerate(PANEL_DRAWERS):
            x = args.left_margin + col_idx * (args.panel_width + args.column_margin)
            y = args.bottom_margin + (n_rows - row_idx - 1) * (args.panel_height + args.row_margin)
            ax = add_axes(fig, fig_width, fig_height, x, y, args.panel_width, args.panel_height)
            axes[row_idx, col_idx] = ax
            drawer(ax, data, row_idx, args)
            if col_idx == 0:
                ax.text(
                    args.row_label_x,
                    0.5,
                    labels[row_idx],
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=args.label_font_size + 0.8,
                )

    if not args.no_title:
        fig.suptitle(
            "Strict single-operator AR ablations",
            fontsize=args.title_font_size + 2.0,
            y=args.title_y,
        )
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
