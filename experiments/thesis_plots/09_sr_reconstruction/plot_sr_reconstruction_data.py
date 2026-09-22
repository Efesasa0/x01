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
SMOKE_DATA = HERE / "sr_reconstruction_smoke.npz"
FULL_DATA = HERE / "sr_reconstruction_data.npz"
SMOKE_SAMPLES_OUTPUT = HERE / "sr_reconstruction_samples_smoke.png"
FULL_SAMPLES_OUTPUT = HERE / "sr_reconstruction_samples.png"
SMOKE_ENERGY_OUTPUT = HERE / "sr_reconstruction_energy_smoke.png"
FULL_ENERGY_OUTPUT = HERE / "sr_reconstruction_energy.png"

CMAP = "RdBu_r"
DIFF_CMAP = "Reds"

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


def default_output_paths(data_path: Path) -> tuple[Path, Path]:
    if "smoke" in data_path.stem:
        return SMOKE_SAMPLES_OUTPUT, SMOKE_ENERGY_OUTPUT
    return FULL_SAMPLES_OUTPUT, FULL_ENERGY_OUTPUT


def parse_args() -> argparse.Namespace:
    data_path = default_data_path()
    samples_output, energy_output = default_output_paths(data_path)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=data_path)
    parser.add_argument("--samples-output", type=Path, default=samples_output)
    parser.add_argument("--energy-output", type=Path, default=energy_output)
    parser.add_argument("--sample-image-size", type=float, default=1.5)
    parser.add_argument("--sample-column-margin", type=float, default=0.08)
    parser.add_argument("--sample-row-margin", type=float, default=0.50)
    parser.add_argument("--sample-left-margin", type=float, default=0.75)
    parser.add_argument("--sample-right-margin", type=float, default=0.4)
    parser.add_argument("--sample-bottom-margin", type=float, default=0.35)
    parser.add_argument("--sample-top-margin", type=float, default=0.55)
    parser.add_argument("--sample-colorbar-width", type=float, default=0.18)
    parser.add_argument("--sample-state-cbar-left", type=float, default=0.50)
    parser.add_argument("--sample-state-cbar-right", type=float, default=0.64)
    parser.add_argument("--sample-err-cbar-left", type=float, default=0.50)
    parser.add_argument("--sample-title-font-size", type=float, default=8.0)
    parser.add_argument("--sample-row-label-font-size", type=float, default=7.5)
    parser.add_argument("--sample-row-label-pad", type=float, default=12.0)
    parser.add_argument("--sample-title-pad", type=float, default=3.0)
    parser.add_argument("--energy-width", type=float, default=7.57)
    parser.add_argument("--energy-height", type=float, default=3.2)
    parser.add_argument("--energy-tick-font-size", type=float, default=7.0)
    parser.add_argument("--energy-label-font-size", type=float, default=8.0)
    parser.add_argument("--energy-legend-font-size", type=float, default=6.8)
    parser.add_argument("--energy-legend-label-spacing", type=float, default=0.5)
    parser.add_argument("--energy-title-pad", type=float, default=8.0)
    parser.add_argument("--energy-plot-floor", type=float, default=1e-16)
    parser.add_argument("--line-width", type=float, default=1.15)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--physical-energy", action="store_true")
    return parser.parse_args()


def load_data(path: Path):
    return np.load(path, allow_pickle=False)


def figure_output_paths(png_path: Path) -> tuple[Path, Path]:
    return png_path, png_path.with_suffix(".pdf")


def save_figure(fig, png_path: Path, dpi: int = 240) -> tuple[Path, Path]:
    outputs = figure_output_paths(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    for path in outputs:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return outputs


def add_axes(fig, fig_width, fig_height, x, y, width, height):
    return fig.add_axes([x / fig_width, y / fig_height, width / fig_width, height / fig_height])


def add_row_colorbar(fig, axes_row, im, fig_width, margin, width, tick_font_size):
    boxes = [ax.get_position() for ax in axes_row]
    x1 = max(box.x1 for box in boxes)
    y0 = min(box.y0 for box in boxes)
    y1 = max(box.y1 for box in boxes)
    cax = fig.add_axes([x1 + margin / fig_width, y0, width / fig_width, y1 - y0])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=tick_font_size, length=2.5, width=0.6)
    return cbar


def finite_positive_xy(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    return x[mask], y[mask]


def positive_max(*arrays):
    values = np.concatenate([np.asarray(array).reshape(-1) for array in arrays])
    values = values[np.isfinite(values) & (values > 0)]
    return float(values.max()) if values.size else 1.0


def style_energy_axis(ax, args):
    ax.tick_params(axis="both", labelsize=args.energy_tick_font_size, length=2.5, width=0.6)
    ax.grid(True, which="both", color="0.88", linewidth=0.45)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)


def build_samples_figure(data, args):
    refs = data["sample_reference"]
    cond = data["sample_condition"]
    pred = data["sample_prediction"]
    err = data["sample_abs_error"]

    n_rows = refs.shape[0]
    image_size = args.sample_image_size
    state_width = 3 * image_size + 2 * args.sample_column_margin

    body_width = (
        state_width
        + args.sample_state_cbar_left
        + args.sample_colorbar_width
        + args.sample_state_cbar_right
        + image_size
        + args.sample_err_cbar_left
        + args.sample_colorbar_width
    )
    body_height = n_rows * image_size + (n_rows - 1) * args.sample_row_margin
    fig_width = args.sample_left_margin + body_width + args.sample_right_margin
    fig_height = args.sample_bottom_margin + body_height + args.sample_top_margin

    fig = plt.figure(figsize=(fig_width, fig_height))
    col_titles = ["reference (HR)", "condition (LR upsampled)", "sample (DDIM)", "|diff|"]

    err_x = (
        args.sample_left_margin
        + state_width
        + args.sample_state_cbar_left
        + args.sample_colorbar_width
        + args.sample_state_cbar_right
    )

    for row_idx in range(n_rows):
        row_y = args.sample_bottom_margin + (n_rows - row_idx - 1) * (image_size + args.sample_row_margin)
        r = refs[row_idx]
        c = cond[row_idx]
        s = pred[row_idx]
        e = err[row_idx]
        vmin = float(min(r.min(), c.min(), s.min()))
        vmax = float(max(r.max(), c.max(), s.max()))
        err_lim = max(float(e.max()), 1e-6)

        state_columns = [
            (col_titles[0], r, vmin, vmax),
            (col_titles[1], c, vmin, vmax),
            (col_titles[2], s, vmin, vmax),
        ]
        state_ims = []
        state_axes = []
        for col_idx, (title, values, vlo, vhi) in enumerate(state_columns):
            x = args.sample_left_margin + col_idx * (image_size + args.sample_column_margin)
            ax = add_axes(fig, fig_width, fig_height, x, row_y, image_size, image_size)
            state_axes.append(ax)
            state_ims.append(ax.imshow(values, cmap=CMAP, vmin=vlo, vmax=vhi))
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(title, fontsize=args.sample_title_font_size, pad=args.sample_title_pad)
            if col_idx == 0:
                ax.set_ylabel(
                    f"sample {row_idx + 1}",
                    fontsize=args.sample_row_label_font_size,
                    rotation=90,
                    labelpad=args.sample_row_label_pad,
                    va="center",
                )

        add_row_colorbar(
            fig,
            state_axes,
            state_ims[-1],
            fig_width,
            args.sample_state_cbar_left,
            args.sample_colorbar_width,
            args.sample_row_label_font_size,
        )

        err_ax = add_axes(fig, fig_width, fig_height, err_x, row_y, image_size, image_size)
        err_im = err_ax.imshow(e, cmap=DIFF_CMAP, vmin=0.0, vmax=err_lim)
        err_ax.set_xticks([])
        err_ax.set_yticks([])
        if row_idx == 0:
            err_ax.set_title(col_titles[3], fontsize=args.sample_title_font_size, pad=args.sample_title_pad)

        add_row_colorbar(
            fig,
            [err_ax],
            err_im,
            fig_width,
            args.sample_err_cbar_left,
            args.sample_colorbar_width,
            args.sample_row_label_font_size,
        )

    return fig


def build_energy_figure(data, args):
    kappa = data["energy_k_bins"]
    suffix = "_physical" if args.physical_energy and "energy_reference_physical" in data.files else ""
    ref = data[f"energy_reference{suffix}"]
    cond = data[f"energy_condition{suffix}"]
    pred = data[f"energy_prediction{suffix}"]
    plot_floor = max(float(args.energy_plot_floor), 0.0)

    fig, ax = plt.subplots(figsize=(args.energy_width, args.energy_height))
    for values, linestyle, label in (
        (ref, "-", "GT"),
        (cond, "--", "condition"),
        (pred, ":", "samples"),
    ):
        x, y = finite_positive_xy(kappa, values)
        if plot_floor > 0.0:
            y = np.maximum(y, plot_floor)
        ax.loglog(x, y, color="black", linestyle=linestyle, linewidth=args.line_width, label=label)
    ax.set_xlabel("κ", fontsize=args.energy_label_font_size)
    ax.set_ylabel("E(κ)", fontsize=args.energy_label_font_size)
    ax.set_title("SR reconstruction energy spectrum", pad=args.energy_title_pad)
    ax.legend(
        fontsize=args.energy_legend_font_size,
        frameon=False,
        handlelength=2.4,
        labelspacing=args.energy_legend_label_spacing,
    )
    style_energy_axis(ax, args)
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.20, top=0.94)
    return fig


def plot_outputs(data_path: Path, samples_output: Path, energy_output: Path, args) -> tuple[Path, ...]:
    data = load_data(data_path)

    fig = build_samples_figure(data, args)
    samples_paths = save_figure(fig, samples_output, dpi=args.dpi)
    plt.close(fig)

    fig = build_energy_figure(data, args)
    energy_paths = save_figure(fig, energy_output, dpi=args.dpi)
    plt.close(fig)

    outputs = (*samples_paths, *energy_paths)
    for path in outputs:
        print(path)
    return outputs


def main() -> None:
    args = parse_args()
    plot_outputs(args.data, args.samples_output, args.energy_output, args)


if __name__ == "__main__":
    main()
