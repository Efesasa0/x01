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

import plot_alpha_capacity as alpha_plot

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results" / "beta"
OUTPUT_ROOT = HERE / "figures" / "beta"
DEFAULT_BETA_COMBOS = ("alpha_e2_d064", "alpha_e1_d064", "alpha_e1_d048")

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
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--combo-id", action="append", default=[])
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--loss-terms-only", action="store_true")
    parser.add_argument("--heatmap-only", action="store_true")
    parser.add_argument("--font-size", type=float, default=7.0)
    parser.add_argument("--label-font-size", type=float, default=8.0)
    parser.add_argument("--title-font-size", type=float, default=8.6)
    parser.add_argument("--heatmap-colorbar-width", type=float, default=0.016)
    parser.add_argument("--heatmap-colorbar-margin", type=float, default=0.05)
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


def load_npz(path: Path) -> dict[str, np.ndarray]:
    return alpha_plot.load_npz(path)


def collect_result_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("*/beta_capacity_*.npz"))


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
        "epoch_diagnostic_epochs",
        "epoch_autocorr_score_val",
        "epoch_rollout_std_ratio_val",
        "epoch_rollout_abs99_ratio_val",
        "epoch_checkpoint_path",
        "best_autocorr_epoch",
        "best_autocorr_score_val",
        "final_epoch_autocorr_score_val",
        "final_to_best_autocorr_gap",
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


def load_primary_data(args: argparse.Namespace) -> dict[str, np.ndarray]:
    if args.data is not None:
        return load_npz(args.data)
    paths = collect_result_files(args.results_dir)
    if not paths:
        raise FileNotFoundError("no beta data file given and no per-combo beta results found")
    return concat_files(paths)


def load_full_beta_comparison(args: argparse.Namespace) -> dict[str, np.ndarray] | None:
    paths = {path.parent.name: path for path in collect_result_files(args.results_dir)}
    if not all(combo_id in paths for combo_id in DEFAULT_BETA_COMBOS):
        print("beta heatmap skipped: waiting for all three beta candidate npz files")
        return None
    return concat_files([paths[combo_id] for combo_id in DEFAULT_BETA_COMBOS])


def beta_label(combo_id: str) -> str:
    return combo_id.replace("alpha_", "").replace("_", " ")


def draw_epoch_heatmap(data: dict[str, np.ndarray], output_root: Path, args: argparse.Namespace) -> Path | None:
    required = set(DEFAULT_BETA_COMBOS)
    available = set(map(str, data["combo_id"]))
    if not required.issubset(available):
        print("beta heatmap skipped: incomplete beta candidates")
        return None
    if "epoch_autocorr_score_val" not in data or "epoch_diagnostic_epochs" not in data:
        print("beta heatmap skipped: epoch diagnostics missing")
        return None

    rows = []
    for combo_id in DEFAULT_BETA_COMBOS:
        idx = int(np.nonzero(data["combo_id"] == combo_id)[0][0])
        rows.append(idx)

    epochs = np.asarray(data["epoch_diagnostic_epochs"][rows[0]], dtype=np.int32)
    matrix = np.asarray(data["epoch_autocorr_score_val"][rows], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(7.0, 2.45))
    fig.subplots_adjust(left=0.18, right=0.86, bottom=0.24, top=0.80)
    im = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="Greys")
    ax.set_yticks(np.arange(len(rows)), labels=[beta_label(str(data["combo_id"][idx])) for idx in rows])
    tick_step = 2 if len(epochs) <= 16 else 5
    xticks = np.arange(0, len(epochs), tick_step)
    if len(epochs) - 1 not in xticks:
        xticks = np.append(xticks, len(epochs) - 1)
    ax.set_xticks(xticks, labels=[str(int(epochs[i])) for i in xticks])
    ax.set_xlabel("epoch", fontsize=args.label_font_size)
    ax.set_ylabel("candidate", fontsize=args.label_font_size)
    ax.set_title(
        "Stage 2 capacity continuation",
        fontsize=args.title_font_size + 1.0,
        pad=6,
    )
    ax.tick_params(axis="both", labelsize=args.font_size, length=2.5, width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)

    for row_idx, idx in enumerate(rows):
        best_col = int(np.nanargmax(matrix[row_idx]))
        ax.text(
            best_col,
            row_idx,
            f"{matrix[row_idx, best_col]:.2f}",
            ha="center",
            va="center",
            fontsize=args.font_size,
            color="white" if matrix[row_idx, best_col] >= 0.58 else "black",
        )

    box = ax.get_position()
    cax = fig.add_axes([box.x1 + args.heatmap_colorbar_margin, box.y0, args.heatmap_colorbar_width, box.height])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"$S_R \in [0,1]$", fontsize=args.label_font_size)
    cbar.ax.tick_params(labelsize=args.font_size, length=2.5, width=0.6)
    cbar.outline.set_linewidth(0.65)

    path = output_root / "beta_capacity_epoch_heatmap"
    alpha_plot.save_figure(fig, path, args.dpi)
    return path.with_suffix(".png")


def plot_all(data: dict[str, np.ndarray], comparison_data: dict[str, np.ndarray] | None, args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not args.summary_only and not args.loss_terms_only:
        if comparison_data is not None:
            draw_epoch_heatmap(comparison_data, args.output_root, args)
        else:
            print("beta heatmap skipped")
    if args.heatmap_only:
        return

    selected = np.arange(len(data["combo_id"]))
    if args.combo_id:
        selected = selected[np.isin(data["combo_id"], np.asarray(args.combo_id))]

    for idx in selected:
        combo_id = str(data["combo_id"][idx])
        combo_dir = args.output_root / combo_id
        if not args.loss_terms_only:
            alpha_plot.make_summary(data, int(idx), combo_dir / "summary", args.dpi, args)
        if not args.summary_only:
            alpha_plot.make_loss_terms(data, int(idx), combo_dir / "loss_terms", args.dpi)


def main() -> None:
    args = parse_args()
    data = load_primary_data(args)
    if set(DEFAULT_BETA_COMBOS).issubset(set(map(str, data["combo_id"]))):
        comparison_data = data
    else:
        comparison_data = load_full_beta_comparison(args)
    plot_all(data, comparison_data, args)
    print(f"wrote beta figures under {args.output_root}")


if __name__ == "__main__":
    main()
