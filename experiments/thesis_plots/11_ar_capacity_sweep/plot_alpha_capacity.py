from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np

TMP_ROOT = Path(os.environ.get("TMPDIR", "/tmp"))
MPLCONFIGDIR = TMP_ROOT / "x01_mplconfig"
XDG_CACHE_HOME = TMP_ROOT / "x01_cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
COMBO_FILE = HERE / "alpha_capacity_combos.npz"
SMOKE_DATA = HERE / "alpha_capacity_smoke.npz"
RESULTS_DIR = HERE / "results" / "alpha"
SMOKE_OUTPUT_ROOT = HERE / "figures" / "alpha_smoke"
FULL_OUTPUT_ROOT = HERE / "figures" / "alpha"

CMAP = "RdBu_r"
ERR_CMAP = "magma"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Latin Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--combo-file", type=Path, default=COMBO_FILE)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--combo-id", action="append", default=[])
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--gif-duration-ms", type=int, default=40)
    parser.add_argument("--no-gifs", action="store_true")
    parser.add_argument("--partial-heatmap", action="store_true")
    parser.add_argument("--heatmap-only", action="store_true")
    parser.add_argument("--loss-terms-only", action="store_true")
    parser.add_argument("--gifs-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--heatmap-colorbar-width", type=float, default=0.016)
    parser.add_argument("--heatmap-colorbar-margin", type=float, default=0.05)
    parser.add_argument("--font-size", type=float, default=7.0)
    parser.add_argument("--label-font-size", type=float, default=8.0)
    parser.add_argument("--title-font-size", type=float, default=8.6)
    parser.add_argument("--summary-image-size", type=float, default=1.46)
    parser.add_argument("--frame-column-margin", "--frame-margin", dest="frame_column_margin", type=float, default=0.11)
    parser.add_argument("--rollout-row-margin", "--rollout-margin", dest="rollout_row_margin", type=float, default=0.42)
    parser.add_argument("--summary-left-margin", type=float, default=0.9)
    parser.add_argument("--summary-right-margin", type=float, default=1.0)
    parser.add_argument("--summary-bottom-margin", type=float, default=0.58)
    parser.add_argument("--summary-top-margin", type=float, default=0.78)
    parser.add_argument("--summary-section-gap", type=float, default=0.62)
    parser.add_argument("--summary-panel-height", type=float, default=1.48)
    parser.add_argument("--summary-panel-gap", type=float, default=0.42)
    parser.add_argument("--colorbar-width", type=float, default=0.016)
    parser.add_argument("--colorbar-height", type=float, default=1.0)
    parser.add_argument("--colorbar-margin", type=float, default=0.05)
    parser.add_argument("--title-closeness", type=float, default=0.987)
    parser.add_argument("--frame-title-pad", type=float, default=3.0)
    return parser.parse_args()


def default_data_path() -> Path | None:
    if SMOKE_DATA.exists():
        return SMOKE_DATA
    return None


def default_output_root(data_path: Path | None) -> Path:
    if data_path is not None and "smoke" in data_path.stem:
        return SMOKE_OUTPUT_ROOT
    return FULL_OUTPUT_ROOT


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def collect_result_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("*/alpha_capacity_*.npz"))


def concat_files(paths: list[Path]) -> dict[str, np.ndarray]:
    loaded = [load_npz(path) for path in paths]
    keys = loaded[0].keys()
    per_combo = {
        "combo_index",
        "chunk_id",
        "combo_id",
        "encoder_profile",
        "encoder_level",
        "dims",
        "num_blocks",
        "num_heads",
        "total_params",
        "encoder_decoder_params",
        "koopman_params",
        "autocorr_score_val",
        "autocorr_gt_mean",
        "autocorr_gt_std",
        "autocorr_pred_mean",
        "autocorr_pred_std",
        "energy_gt_mean",
        "energy_gt_std",
        "energy_pred_mean",
        "energy_pred_std",
        "qq_gt",
        "qq_pred",
        "rollout_gt",
        "rollout_pred",
        "rollout_abs_error",
        "rollout_gif_gt",
        "rollout_gif_pred",
        "summary_gt_rollout",
        "summary_pred_rollout",
        "train_raw_loss",
        "val_weighted_loss",
        "val_forward_loss",
        "val_backward_loss",
        "val_reconstruction_loss",
        "val_latent_fwd_loss",
        "val_latent_bwd_loss",
        "val_consistency_loss",
        "val_ortho_a_loss",
        "val_ortho_b_loss",
        "val_consist_progressive_loss",
        "eig_A_min",
        "eig_A_mean",
        "eig_A_max",
        "eig_B_min",
        "eig_B_mean",
        "eig_B_max",
        "train_std",
        "checkpoint_path",
        "train_log_path",
    }
    out: dict[str, np.ndarray] = {}
    for key in keys:
        values = [item[key] for item in loaded]
        out[key] = np.concatenate(values, axis=0) if key in per_combo else values[0]
    order = np.argsort(out["combo_index"])
    for key in per_combo:
        if key in out:
            out[key] = out[key][order]
    return out


def load_data(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], Path | None]:
    if args.data is not None and args.data.exists():
        return load_npz(args.data), args.data
    paths = collect_result_files(args.results_dir)
    if paths:
        return concat_files(paths), None
    data_path = default_data_path()
    if data_path is not None and data_path.exists():
        return load_npz(data_path), data_path
    raise FileNotFoundError("no data file given and no per-combo results found")


def save_figure(fig, path: Path, dpi: int) -> tuple[Path, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    png = path.with_suffix(".png")
    pdf = path.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(png)
    print(pdf)
    return png, pdf


def add_axes(fig, fig_width: float, fig_height: float, x: float, y: float, width: float, height: float):
    return fig.add_axes([x / fig_width, y / fig_height, width / fig_width, height / fig_height])


def robust_vlim(*arrays: np.ndarray) -> tuple[float, float]:
    joined = np.concatenate([np.asarray(arr).reshape(-1) for arr in arrays])
    finite = joined[np.isfinite(joined)]
    vmax = float(np.percentile(np.abs(finite), 99.5)) if finite.size else 1.0
    vmax = max(vmax, 1e-6)
    return -vmax, vmax


def positive_band(mean: np.ndarray, std: np.ndarray, floor: float = 1e-16) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    return np.maximum(mean - std, floor), np.maximum(mean + std, floor)


def finite_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def style_axis(ax, font_size: float = 7.0):
    ax.tick_params(axis="both", labelsize=font_size, length=2.5, width=0.6)
    ax.grid(True, color="0.88", linewidth=0.45)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)


def complete_combo_set(data: dict[str, np.ndarray], combo_file: Path) -> bool:
    if not combo_file.exists():
        return False
    combos = load_npz(combo_file)
    return set(map(str, combos["combo_id"])) == set(map(str, data["combo_id"]))


def encoder_config_label(data: dict[str, np.ndarray], profile: str) -> str:
    idx = int(np.nonzero(data["encoder_profile"] == profile)[0][0])
    blocks = ",".join(str(int(x)) for x in data["num_blocks"][idx])
    heads = ",".join(str(int(x)) for x in data["num_heads"][idx])
    return f"blocks [{blocks}]\nheads [{heads}]"


def draw_heatmap(data: dict[str, np.ndarray], output_root: Path, combo_file: Path, args: argparse.Namespace) -> Path | None:
    if not args.partial_heatmap and not complete_combo_set(data, combo_file):
        print("heatmap skipped: not all combo ids are present")
        return None
    profiles = sorted(np.unique(data["encoder_profile"]), key=lambda name: int(name.split("_")[0][1:]))
    dims = sorted(np.unique(data["dims"]).astype(int).tolist())
    matrix = np.full((len(profiles), len(dims)), np.nan, dtype=np.float32)
    labels = np.full((len(profiles), len(dims)), "", dtype=object)
    for i, profile in enumerate(profiles):
        for j, dim in enumerate(dims):
            mask = (data["encoder_profile"] == profile) & (data["dims"] == dim)
            if mask.any():
                idx = int(np.nonzero(mask)[0][0])
                matrix[i, j] = float(data["autocorr_score_val"][idx])
                labels[i, j] = f"{matrix[i, j]:.2f}\n{data['total_params'][idx] / 1e6:.1f}M"

    fig, ax = plt.subplots(figsize=(6.45, 3.05))
    fig.subplots_adjust(left=0.38, right=0.84, bottom=0.19, top=0.82)
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Greys")
    ax.set_xticks(np.arange(len(dims)), labels=[str(d) for d in dims], fontsize=args.label_font_size)
    ax.set_yticks(
        np.arange(len(profiles)),
        labels=[encoder_config_label(data, profile) for profile in profiles],
        fontsize=args.font_size,
    )
    ax.set_xlabel("base width", fontsize=args.label_font_size)
    ax.set_ylabel("encoder-decoder profile", fontsize=args.label_font_size)
    ax.set_title(
        "Stage 1 capacity sweep",
        fontsize=args.title_font_size + 1.0,
        pad=6,
    )
    for i in range(len(profiles)):
        for j in range(len(dims)):
            if np.isfinite(matrix[i, j]):
                color = "white" if matrix[i, j] >= 0.58 else "black"
                ax.text(j, i, labels[i, j], ha="center", va="center", fontsize=args.font_size, color=color)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
    ax.tick_params(axis="both", length=2.5, width=0.6)
    box = ax.get_position()
    cax = fig.add_axes([box.x1 + args.heatmap_colorbar_margin, box.y0, args.heatmap_colorbar_width, box.height])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"$S_R \in [0,1]$", fontsize=args.label_font_size)
    cbar.ax.tick_params(labelsize=args.font_size, length=2.5, width=0.6)
    cbar.outline.set_linewidth(0.65)
    path = output_root / "alpha_capacity_heatmap"
    save_figure(fig, path, args.dpi)
    return path.with_suffix(".png")


def image_to_rgb(frame: np.ndarray, vmin: float, vmax: float, cmap_name: str = CMAP) -> Image.Image:
    cmap = plt.get_cmap(cmap_name)
    norm = np.clip((frame - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)
    rgba = (cmap(norm) * 255).astype(np.uint8)
    return Image.fromarray(rgba[:, :, :3])


def make_rollout_gif(data: dict[str, np.ndarray], idx: int, out_path: Path, duration_ms: int) -> Path:
    if "rollout_gif_gt" in data and "rollout_gif_pred" in data:
        gt = data["rollout_gif_gt"][idx]
        pred = data["rollout_gif_pred"][idx]
        times = data["rollout_gif_t"] if "rollout_gif_t" in data else np.arange(1, gt.shape[0] + 1)
    else:
        gt = data["rollout_gt"][idx]
        pred = data["rollout_pred"][idx]
        times = data["rollout_t"]
    vmin, vmax = robust_vlim(gt, pred)
    n_frames = len(times)
    scale = 4
    gap = 12
    header = 28
    label_band = 18
    frame_h, frame_w = gt.shape[-2], gt.shape[-1]
    panel_w = frame_w * scale
    panel_h = frame_h * scale
    canvas_w = panel_w * 2 + gap
    canvas_h = header + label_band + panel_h
    frames = []
    for j, t in enumerate(times):
        gt_img = image_to_rgb(gt[j], vmin, vmax).resize((panel_w, panel_h), Image.Resampling.NEAREST)
        pred_img = image_to_rgb(pred[j], vmin, vmax).resize((panel_w, panel_h), Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 5), f"validation rollout k={int(t)} ({j + 1}/{n_frames})", fill=(0, 0, 0))
        draw.text((4, header), "prediction", fill=(0, 0, 0))
        draw.text((panel_w + gap + 4, header), "reference", fill=(0, 0, 0))
        canvas.paste(pred_img, (0, header + label_band))
        canvas.paste(gt_img, (panel_w + gap, header + label_band))
        frames.append(canvas)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    print(out_path)
    return out_path


def summary_rollout_rows(data: dict[str, np.ndarray], idx: int) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    if "summary_gt_rollout" in data and "summary_pred_rollout" in data:
        times = data["summary_rollout_t"]
        epochs = data["summary_rollout_epochs"]
        rows = [("GT", data["summary_gt_rollout"][idx])]
        rows.extend((f"Epoch {int(epoch)}", data["summary_pred_rollout"][idx, row]) for row, epoch in enumerate(epochs))
        return times, rows
    return data["rollout_t"], [("GT", data["rollout_gt"][idx]), ("model", data["rollout_pred"][idx])]


def summary_layout(n_rows: int, n_cols: int, n_panels: int, args: argparse.Namespace) -> dict[str, float]:
    image_size = args.summary_image_size
    body_width = n_cols * image_size + (n_cols - 1) * args.frame_column_margin
    rollout_height = n_rows * image_size + (n_rows - 1) * args.rollout_row_margin
    panel_gap = args.summary_panel_gap
    panel_width = (body_width - (n_panels - 1) * panel_gap) / n_panels
    if panel_width <= 0:
        raise ValueError("summary panel gap is too large for the available width")

    fig_width = args.summary_left_margin + body_width + args.summary_right_margin
    fig_height = (
        args.summary_bottom_margin
        + args.summary_panel_height
        + args.summary_section_gap
        + rollout_height
        + args.summary_top_margin
    )
    return {
        "fig_width": fig_width,
        "fig_height": fig_height,
        "body_width": body_width,
        "rollout_bottom": args.summary_bottom_margin + args.summary_panel_height + args.summary_section_gap,
        "panel_width": panel_width,
    }


def draw_rollout_grid(
    fig,
    data: dict[str, np.ndarray],
    idx: int,
    args: argparse.Namespace,
    layout: dict[str, float],
) -> None:
    times, rows = summary_rollout_rows(data, idx)
    n_rows = len(rows)
    n_cols = len(times)
    fig_width = layout["fig_width"]
    fig_height = layout["fig_height"]
    axes = np.empty((n_rows, n_cols), dtype=object)
    row_ims = []

    for row, (label, values) in enumerate(rows):
        selected = np.asarray(values).reshape(-1)
        finite = selected[np.isfinite(selected)]
        row_vmax = float(np.percentile(np.abs(finite), 99.0)) if finite.size else 1.0
        row_vmax = max(row_vmax, 1e-12)
        row_vmin = -row_vmax
        im_row = None
        for col, timestep in enumerate(times):
            x = args.summary_left_margin + col * (args.summary_image_size + args.frame_column_margin)
            y = layout["rollout_bottom"] + (n_rows - row - 1) * (
                args.summary_image_size + args.rollout_row_margin
            )
            ax = add_axes(fig, fig_width, fig_height, x, y, args.summary_image_size, args.summary_image_size)
            axes[row, col] = ax
            im_row = ax.imshow(values[col], cmap=CMAP, vmin=row_vmin, vmax=row_vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"k={int(timestep)}", fontsize=args.font_size, pad=args.frame_title_pad)
            if col == 0:
                ax.set_ylabel(label, fontsize=args.label_font_size + 3.0)
            for spine in ax.spines.values():
                spine.set_linewidth(0.65)
        row_ims.append(im_row)

    for row in range(n_rows):
        row_axes = [axes[row, col] for col in range(n_cols)]
        boxes = [ax.get_position() for ax in row_axes]
        x1 = max(box.x1 for box in boxes)
        y0 = min(box.y0 for box in boxes)
        y1 = max(box.y1 for box in boxes)
        colorbar_height = (y1 - y0) * args.colorbar_height
        colorbar_y = y0 + ((y1 - y0) - colorbar_height) / 2
        colorbar_ax = fig.add_axes(
            [
                x1 + args.colorbar_margin,
                colorbar_y,
                args.colorbar_width,
                colorbar_height,
            ]
        )
        cbar = fig.colorbar(row_ims[row], cax=colorbar_ax)
        cbar.ax.tick_params(labelsize=args.font_size, length=2.5, width=0.6)
        cbar.outline.set_linewidth(0.65)


def draw_autocorr(ax, data: dict[str, np.ndarray], idx: int, args: argparse.Namespace):
    t = data["t_star"]
    gt = data["autocorr_gt_mean"][idx]
    pred = data["autocorr_pred_mean"][idx]
    ax.axhline(0.0, color="0.55", linestyle=":", linewidth=0.65)
    ax.plot(t, gt, color="black", linestyle="-", linewidth=1.05, label="GT")
    ax.plot(t, pred, color="black", linestyle="--", linewidth=1.05, label="model")
    ax.set_xlim(float(t.min()), float(t.max()))
    ax.set_title("Autocorr.", fontsize=args.title_font_size, pad=5)
    ax.set_xlabel(r"$k^\ast$", fontsize=args.label_font_size)
    ax.set_ylabel(r"$R(k^\ast)$", fontsize=args.label_font_size)
    ax.legend(frameon=False, fontsize=6.6, loc="upper right", handlelength=2.4)
    style_axis(ax, args.font_size)


def draw_energy(ax, data: dict[str, np.ndarray], idx: int, args: argparse.Namespace):
    k = data["k_bins"]
    gt = data["energy_gt_mean"][idx]
    pred = data["energy_pred_mean"][idx]
    ax.loglog(k, np.maximum(gt, 1e-16), color="black", linestyle="-", linewidth=1.05, label="GT")
    ax.loglog(k, np.maximum(pred, 1e-16), color="black", linestyle="--", linewidth=1.05, label="model")
    ax.set_title(r"$E(\kappa)$", fontsize=args.title_font_size, pad=5)
    ax.set_xlabel(r"$\kappa$", fontsize=args.label_font_size)
    ax.set_ylabel(r"$E(\kappa)$", fontsize=args.label_font_size)
    style_axis(ax, args.font_size)


def draw_qq(ax, data: dict[str, np.ndarray], idx: int, args: argparse.Namespace):
    gt = data["qq_gt"][idx]
    pred = data["qq_pred"][idx]
    lim = float(max(np.nanmax(np.abs(gt)), np.nanmax(np.abs(pred)), 1e-6))
    ax.plot([-lim, lim], [-lim, lim], color="0.35", linestyle=":", linewidth=0.75)
    ax.scatter(pred, gt, s=3.0, color="black", alpha=0.55, linewidths=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Q–Q", fontsize=args.title_font_size, pad=5)
    ax.set_xlabel(r"$Q_{\mathrm{model}}$", fontsize=args.label_font_size)
    ax.set_ylabel(r"$Q_{\mathrm{GT}}$", fontsize=args.label_font_size)
    style_axis(ax, args.font_size)


def draw_train_val(ax, data: dict[str, np.ndarray], idx: int, args: argparse.Namespace):
    x_train, train = finite_xy(data["loss_epochs"], data["train_raw_loss"][idx])
    x_val, val = finite_xy(data["loss_epochs"], data["val_weighted_loss"][idx])
    ax.plot(x_train, train, color="black", linestyle="-", linewidth=1.05, label="train")
    ax.plot(x_val, val, color="black", linestyle="--", linewidth=1.05, label="validation")
    ax.set_title("Training curves", fontsize=args.title_font_size, pad=5)
    ax.set_xlabel("Epoch", fontsize=args.label_font_size)
    ax.set_ylabel("Loss", fontsize=args.label_font_size)
    ax.legend(frameon=False, fontsize=6.6, loc="best", handlelength=2.0)
    style_axis(ax, args.font_size)


def draw_spectra(ax, data: dict[str, np.ndarray], idx: int, args: argparse.Namespace):
    e = data["loss_epochs"]
    for key, style, label in (("eig_A_min", ":", "min"), ("eig_A_mean", "-", "mean"), ("eig_A_max", "--", "max")):
        x, y = finite_xy(e, data[key][idx])
        ax.plot(x, y, color="black", linestyle=style, linewidth=1.05, label=label)
    ax.axhline(1.0, color="0.55", linestyle="-.", linewidth=0.7)
    ax.set_title("Forward spectrum", fontsize=args.title_font_size, pad=5)
    ax.set_xlabel("Epoch", fontsize=args.label_font_size)
    ax.set_ylabel(r"$|\lambda(K)|$", fontsize=args.label_font_size)
    ax.legend(frameon=False, fontsize=6.6, loc="best", handlelength=2.0)
    style_axis(ax, args.font_size)


def make_summary(data: dict[str, np.ndarray], idx: int, out_path: Path, dpi: int, args: argparse.Namespace) -> Path:
    combo_id = str(data["combo_id"][idx])
    score = float(data["autocorr_score_val"][idx])
    _times, rows = summary_rollout_rows(data, idx)
    bottom = [
        draw_autocorr,
        draw_energy,
        draw_qq,
        draw_train_val,
    ]
    layout = summary_layout(len(rows), len(_times), len(bottom), args)
    fig = plt.figure(figsize=(layout["fig_width"], layout["fig_height"]))
    draw_rollout_grid(fig, data, idx, args, layout)
    for j, drawer in enumerate(bottom):
        x = args.summary_left_margin + j * (layout["panel_width"] + args.summary_panel_gap)
        ax = add_axes(
            fig,
            layout["fig_width"],
            layout["fig_height"],
            x,
            args.summary_bottom_margin,
            layout["panel_width"],
            args.summary_panel_height,
        )
        drawer(ax, data, idx, args)
    fig.suptitle(
        f"{combo_id}: dims={int(data['dims'][idx])}, {str(data['encoder_profile'][idx]).replace('_', ' ')}, "
        f"$S_R$={score:.2f}",
        fontsize=args.title_font_size + 2.0,
        y=args.title_closeness,
    )
    save_figure(fig, out_path, dpi)
    return out_path.with_suffix(".png")


def make_loss_terms(data: dict[str, np.ndarray], idx: int, out_path: Path, dpi: int) -> Path:
    combo_id = str(data["combo_id"][idx])
    e = data["loss_epochs"]
    terms = [
        ("fwd", "val_forward_loss"),
        ("bwd", "val_backward_loss"),
        ("recon", "val_reconstruction_loss"),
        ("latent fwd", "val_latent_fwd_loss"),
        ("ortho A", "val_ortho_a_loss"),
        ("weighted", "val_weighted_loss"),
    ]
    fig, axes = plt.subplots(1, len(terms), figsize=(10.2, 1.95), sharex=True)
    for term_idx, (label, key) in enumerate(terms):
        ax = axes[term_idx]
        if key in data:
            y = np.asarray(data[key][idx], dtype=np.float64)
            finite = np.isfinite(y)
            active = finite.any() and np.nanmax(np.abs(y[finite])) >= 1e-12
        else:
            y = np.zeros_like(e, dtype=np.float64)
            active = False

        if active:
            ax.plot(e, np.maximum(y, 1e-12), color="black", linewidth=1.05)
            ax.set_yscale("log")
        else:
            ax.text(0.5, 0.5, "inactive", transform=ax.transAxes, ha="center", va="center", fontsize=7.0)
            ax.set_ylim(0.0, 1.0)

        ax.set_title(label, fontsize=8.0, pad=4)
        ax.set_xlabel("epoch", fontsize=7.4)
        if term_idx == 0:
            ax.set_ylabel("validation loss", fontsize=7.4)
        style_axis(ax)

    fig.suptitle(f"{combo_id}: validation loss terms", fontsize=9.0, y=1.02)
    fig.tight_layout(pad=0.35, w_pad=0.55)
    save_figure(fig, out_path, dpi)
    return out_path.with_suffix(".png")


def plot_all(data: dict[str, np.ndarray], args: argparse.Namespace, output_root: Path) -> dict[str, list[Path]]:
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, list[Path]] = {"heatmap": [], "summary": [], "loss_terms": [], "gif": []}
    if not args.loss_terms_only and not args.gifs_only and not args.summary_only:
        heatmap = draw_heatmap(data, output_root, args.combo_file, args)
        if heatmap is not None:
            outputs["heatmap"].append(heatmap)
    if args.heatmap_only:
        return outputs

    selected = np.arange(len(data["combo_id"]))
    if args.combo_id:
        selected = selected[np.isin(data["combo_id"], np.asarray(args.combo_id))]
    for idx in selected:
        combo_id = str(data["combo_id"][idx])
        combo_dir = output_root / combo_id
        if not args.loss_terms_only and not args.summary_only and not args.no_gifs:
            outputs["gif"].append(make_rollout_gif(data, int(idx), combo_dir / "rollout.gif", args.gif_duration_ms))
        if not args.loss_terms_only and not args.gifs_only:
            outputs["summary"].append(make_summary(data, int(idx), combo_dir / "summary", args.dpi, args))
        if not args.gifs_only and not args.summary_only:
            outputs["loss_terms"].append(make_loss_terms(data, int(idx), combo_dir / "loss_terms", args.dpi))
    return outputs


def main() -> None:
    args = parse_args()
    data, data_path = load_data(args)
    output_root = args.output_root or default_output_root(data_path)
    outputs = plot_all(data, args, output_root)
    print(f"wrote figures under {output_root}")
    print(f"heatmaps={len(outputs['heatmap'])} summaries={len(outputs['summary'])} gifs={len(outputs['gif'])}")


if __name__ == "__main__":
    main()
