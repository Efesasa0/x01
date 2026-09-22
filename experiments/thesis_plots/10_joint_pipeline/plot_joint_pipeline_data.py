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
SMOKE_DATA = HERE / "joint_pipeline_smoke.npz"
FULL_DATA = HERE / "joint_pipeline_data.npz"
STAGE_C_FULL_DATA = HERE / "joint_pipeline_stage_c_full.npz"
SMOKE_DIAGNOSTICS_OUTPUT = HERE / "joint_pipeline_diagnostics_smoke.png"
FULL_DIAGNOSTICS_OUTPUT = HERE / "joint_pipeline_diagnostics.png"
SMOKE_LONG_OUTPUT = HERE / "joint_pipeline_long_horizon_smoke.png"
FULL_LONG_OUTPUT = HERE / "joint_pipeline_long_horizon.png"
DEFAULT_STAGE_A_TIMES = (0, 25, 50, 100, 200, 319)

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


def default_output_paths(data_path: Path) -> tuple[Path, Path]:
    if "smoke" in data_path.stem:
        return SMOKE_DIAGNOSTICS_OUTPUT, SMOKE_LONG_OUTPUT
    return FULL_DIAGNOSTICS_OUTPUT, FULL_LONG_OUTPUT


def parse_int_sequence(value: str) -> tuple[int, ...]:
    try:
        times = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not times:
        raise argparse.ArgumentTypeError("at least one value is required")
    return times


def parse_args() -> argparse.Namespace:
    data_path = default_data_path()
    diagnostics_output, long_output = default_output_paths(data_path)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=data_path)
    parser.add_argument("--stage-c-full-data", type=Path, default=None)
    parser.add_argument("--diagnostics-output", type=Path, default=diagnostics_output)
    parser.add_argument("--long-output", type=Path, default=long_output)
    parser.add_argument("--stage-a-times", type=parse_int_sequence, default=DEFAULT_STAGE_A_TIMES)
    parser.add_argument("--stage-a-columns", type=int, default=None)
    parser.add_argument("--stage-a-image-size", type=float, default=2.0)
    parser.add_argument("--stage-a-column-margin", type=float, default=0.11)
    parser.add_argument("--stage-a-row-margin", type=float, default=0.675)
    parser.add_argument("--panel-height", type=float, default=2.2)
    parser.add_argument("--panel-gap", type=float, default=1.0)
    parser.add_argument("--section-gap", type=float, default=1.0)
    parser.add_argument("--left-margin", type=float, default=0.9)
    parser.add_argument("--right-margin", type=float, default=1.0)
    parser.add_argument("--bottom-margin", type=float, default=0.6)
    parser.add_argument("--top-margin", type=float, default=0.8)
    parser.add_argument("--colorbar-width", type=float, default=0.016)
    parser.add_argument("--colorbar-height", type=float, default=1.0)
    parser.add_argument("--colorbar-margin", type=float, default=0.05)
    parser.add_argument("--line-width", type=float, default=1.4)
    parser.add_argument("--font-size", type=float, default=8.0)
    parser.add_argument("--label-font-size", type=float, default=9.0)
    parser.add_argument("--row-label-font-size", type=float, default=11.0)
    parser.add_argument("--title-font-size", type=float, default=9.6)
    parser.add_argument("--legend-font-size", type=float, default=8.0)
    parser.add_argument("--long-image-size", type=float, default=2.0)
    parser.add_argument("--long-column-margin", type=float, default=0.11)
    parser.add_argument("--long-left-margin", type=float, default=0.9)
    parser.add_argument("--long-right-margin", type=float, default=1.0)
    parser.add_argument("--long-bottom-margin", type=float, default=0.35)
    parser.add_argument("--long-top-margin", type=float, default=0.55)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--no-title", action="store_true")
    return parser.parse_args()


def load_data(path: Path):
    return np.load(path, allow_pickle=False)


def load_stage_c_full_data(data, args):
    candidates = []
    if args.stage_c_full_data is not None:
        candidates.append(args.stage_c_full_data)
    if "stage_c_full_output" in data.files:
        value = str(np.asarray(data["stage_c_full_output"]).item())
        if value:
            candidates.append(Path(value))
    candidates.append(STAGE_C_FULL_DATA)

    for path in candidates:
        if path.exists():
            return np.load(path, allow_pickle=False)
    return None


def add_axes(fig, fig_width, fig_height, x, y, width, height):
    return fig.add_axes([x / fig_width, y / fig_height, width / fig_width, height / fig_height])


def figure_output_paths(png_path: Path) -> tuple[Path, Path]:
    return png_path, png_path.with_suffix(".pdf")


def save_figure(fig, png_path: Path, dpi: int = 240) -> tuple[Path, Path]:
    outputs = figure_output_paths(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    for path in outputs:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return outputs


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


def stage_a_keep_indices(data, args) -> np.ndarray:
    full_times = np.asarray(data["stage_a_t"], dtype=np.int32)
    requested = getattr(args, "stage_a_times", DEFAULT_STAGE_A_TIMES)
    if isinstance(requested, str):
        requested = parse_int_sequence(requested)
    if requested:
        keep = []
        for timestep in requested:
            matches = np.flatnonzero(full_times == int(timestep))
            if len(matches) != 1:
                raise ValueError(f"stage A timestep {timestep} not found in saved times {full_times.tolist()}")
            keep.append(int(matches[0]))
        return np.asarray(keep, dtype=int)

    n_cols = min(int(args.stage_a_columns), len(full_times))
    return np.linspace(0, len(full_times) - 1, n_cols, dtype=int)


def stage_c_keep_indices(stage_c_full, times: np.ndarray) -> np.ndarray:
    full_times = np.asarray(stage_c_full["stage_c_t"], dtype=np.int32)
    keep = []
    for timestep in times:
        matches = np.flatnonzero(full_times == int(timestep))
        if len(matches) != 1:
            raise ValueError(f"stage C timestep {timestep} not found in saved times")
        keep.append(int(matches[0]))
    return np.asarray(keep, dtype=int)


def rollout_rows(data, stage_c_full, args):
    full_gt = data["stage_a_lr_gt"]
    full_pred = data["stage_a_lr_pred"]
    full_times = data["stage_a_t"]
    keep = stage_a_keep_indices(data, args)
    gt = full_gt[keep]
    pred = full_pred[keep]
    times = full_times[keep]
    rows = [
        ("LR reference", gt),
        ("LR AR rollout", pred),
    ]
    if stage_c_full is not None and "stage_c_hr_pred" in stage_c_full.files:
        hr_keep = stage_c_keep_indices(stage_c_full, times)
        rows.append(("HR AR–SR rollout", stage_c_full["stage_c_hr_pred"][hr_keep]))
    return times, rows


def draw_stage_a(fig, data, stage_c_full, args, fig_width, fig_height, y0):
    times, rows = rollout_rows(data, stage_c_full, args)
    n_cols = len(times)
    axes = np.empty((len(rows), n_cols), dtype=object)
    row_ims = []
    for row_idx, (label, values) in enumerate(rows):
        row_y = y0 + (len(rows) - row_idx - 1) * (args.stage_a_image_size + args.stage_a_row_margin)
        vmin, vmax = robust_vlim(values)
        row_im = None
        for col_idx in range(n_cols):
            x = args.left_margin + col_idx * (args.stage_a_image_size + args.stage_a_column_margin)
            ax = add_axes(fig, fig_width, fig_height, x, row_y, args.stage_a_image_size, args.stage_a_image_size)
            axes[row_idx, col_idx] = ax
            row_im = ax.imshow(values[col_idx], cmap=CMAP, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(f"k={int(times[col_idx])}", fontsize=args.font_size, pad=3.0)
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=args.row_label_font_size)
        row_ims.append(row_im)
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
    k_star = data["stage_b_t_star"]
    gt = data["stage_b_autocorr_gt_mean"]
    pred = data["stage_b_autocorr_pred_mean"]
    ax.axhline(0.0, color="0.55", linestyle=":", linewidth=0.65)
    ax.plot(k_star, gt, color="black", linestyle="-", linewidth=args.line_width, label="GT")
    ax.plot(k_star, pred, color="black", linestyle="--", linewidth=args.line_width, label="joint")
    ax.set_xlim(float(k_star.min()), float(k_star.max()))
    ax.set_xlabel(r"$k^\ast$", fontsize=args.label_font_size)
    ax.set_ylabel(r"$R(k^\ast)$", fontsize=args.label_font_size)
    ax.set_title("(a) HR autocorr.", fontsize=args.title_font_size, pad=4)
    ax.legend(frameon=False, fontsize=args.legend_font_size, handlelength=2.4)
    style_axis(ax, args)


def draw_energy(ax, data, args):
    kappa = data["stage_b_k_bins"]
    gt = data["stage_b_energy_gt_mean"]
    pred = data["stage_b_energy_pred_mean"]
    ax.loglog(kappa, np.maximum(gt, 1e-16), color="black", linestyle="-", linewidth=args.line_width, label="GT")
    if "stage_b_energy_condition_mean" in data.files:
        cond = data["stage_b_energy_condition_mean"]
        ax.loglog(
            kappa,
            np.maximum(cond, 1e-16),
            color="black",
            linestyle=":",
            linewidth=args.line_width,
            label="LR upsample",
        )
    ax.loglog(kappa, np.maximum(pred, 1e-16), color="black", linestyle="--", linewidth=args.line_width, label="joint")
    ax.set_xlabel("κ", fontsize=args.label_font_size)
    ax.set_ylabel("E(κ)", fontsize=args.label_font_size)
    ax.set_title("(b) HR E(κ)", fontsize=args.title_font_size, pad=4)
    ax.legend(frameon=False, fontsize=args.legend_font_size, handlelength=2.4)
    style_axis(ax, args)


def draw_qq(ax, data, args):
    gt = data["stage_b_qq_gt"]
    pred = data["stage_b_qq_pred"]
    lim = float(max(np.nanmax(np.abs(gt)), np.nanmax(np.abs(pred)), 1e-6))
    ax.plot([-lim, lim], [-lim, lim], color="0.35", linestyle=":", linewidth=0.75)
    ax.scatter(gt, pred, s=3.0, color="black", alpha=0.55, linewidths=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$Q_{\mathrm{GT}}$", fontsize=args.label_font_size)
    ax.set_ylabel(r"$Q_{\mathrm{joint}}$", fontsize=args.label_font_size)
    ax.set_title("(c) HR Q–Q", fontsize=args.title_font_size, pad=4)
    style_axis(ax, args)


def build_diagnostics_figure(data, stage_c_full, args):
    n_cols = len(stage_a_keep_indices(data, args))
    n_rows = len(rollout_rows(data, stage_c_full, args)[1])
    image_width = n_cols * args.stage_a_image_size + (n_cols - 1) * args.stage_a_column_margin
    image_height = n_rows * args.stage_a_image_size + (n_rows - 1) * args.stage_a_row_margin
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
    axes, row_ims = draw_stage_a(fig, data, stage_c_full, args, fig_width, fig_height, image_y)
    for row_idx, row_im in enumerate(row_ims):
        if row_im is not None:
            add_row_colorbar(fig, axes[row_idx], row_im, args)
    if not args.no_title:
        fig.suptitle(
            "Joint AR–SR rollout diagnostics",
            fontsize=args.title_font_size + 1.4,
            y=0.985,
        )
    return fig


def build_long_horizon_figure(data, args):
    frames = data["stage_c_snapshot_hr_pred"]
    times = data["stage_c_snapshot_t"]
    n_cols = frames.shape[0]
    body_width = n_cols * args.long_image_size + (n_cols - 1) * args.long_column_margin
    fig_width = args.long_left_margin + body_width + args.long_right_margin
    fig_height = args.long_bottom_margin + args.long_image_size + args.long_top_margin
    fig = plt.figure(figsize=(fig_width, fig_height))
    vmin, vmax = robust_vlim(frames)
    axes = np.empty(n_cols, dtype=object)
    state_im = None
    for col_idx in range(n_cols):
        x = args.long_left_margin + col_idx * (args.long_image_size + args.long_column_margin)
        ax = add_axes(fig, fig_width, fig_height, x, args.long_bottom_margin, args.long_image_size, args.long_image_size)
        axes[col_idx] = ax
        state_im = ax.imshow(frames[col_idx], cmap=CMAP, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"k={int(times[col_idx])}", fontsize=args.font_size, pad=3.0)
    if state_im is not None:
        add_row_colorbar(fig, axes, state_im, args)
    if not args.no_title:
        fig.suptitle("Long-horizon joint rollout", fontsize=args.title_font_size + 1.2, y=0.99)
    return fig


def plot_outputs(data_path: Path, diagnostics_output: Path, long_output: Path, args) -> tuple[Path, ...]:
    data = load_data(data_path)
    stage_c_full = load_stage_c_full_data(data, args)

    fig = build_diagnostics_figure(data, stage_c_full, args)
    diagnostics_paths = save_figure(fig, diagnostics_output, dpi=args.dpi)
    plt.close(fig)

    fig = build_long_horizon_figure(data, args)
    long_paths = save_figure(fig, long_output, dpi=args.dpi)
    plt.close(fig)

    outputs = (*diagnostics_paths, *long_paths)
    for path in outputs:
        print(path)
    return outputs


def main() -> None:
    args = parse_args()
    plot_outputs(args.data, args.diagnostics_output, args.long_output, args)


if __name__ == "__main__":
    main()
