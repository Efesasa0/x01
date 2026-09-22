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
GAMMA_COMBO_FILE = HERE / "gamma_loss_combos.npz"
RESULTS_DIR = HERE / "results" / "gamma"
OUTPUT_ROOT = HERE / "figures" / "gamma"

LOSS_KEYS = ("lambda_fwd", "lambda_bwd", "lambda_recon", "lambda_latent_fwd", "lambda_ortho_a")
LOSS_LABELS = ("fwd", "bwd", "recon", "latent", "ortho")

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
    parser.add_argument("--gamma-combo-file", type=Path, default=GAMMA_COMBO_FILE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--gamma-combo-id", action="append", default=[])
    parser.add_argument("--top-k-summaries", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--partial-ranking", action="store_true")
    parser.add_argument("--ranking-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--loss-terms-only", action="store_true")
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
    return sorted(results_dir.glob("*/gamma_loss_sweep_*.npz"))


def per_combo_keys() -> set[str]:
    return {
        "combo_index",
        "chunk_id",
        "combo_id",
        "capacity_combo_id",
        "gamma_combo_index",
        "gamma_chunk_id",
        "gamma_combo_id",
        "is_default",
        "encoder_profile",
        "encoder_level",
        "dims",
        "num_blocks",
        "num_heads",
        "total_params",
        "encoder_decoder_params",
        "koopman_params",
        *LOSS_KEYS,
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
        "epoch_checkpoint_path",
        "best_autocorr_epoch",
        "best_autocorr_score_val",
        "final_epoch_autocorr_score_val",
        "final_to_best_autocorr_gap",
    }


def concat_files(paths: list[Path]) -> dict[str, np.ndarray]:
    loaded = [load_npz(path) for path in paths]
    keys = loaded[0].keys()
    combo_keys = per_combo_keys()
    out: dict[str, np.ndarray] = {}
    for key in keys:
        values = [item[key] for item in loaded]
        out[key] = np.concatenate(values, axis=0) if key in combo_keys else values[0]
    if "gamma_combo_index" in out:
        order = np.argsort(out["gamma_combo_index"])
    else:
        order = np.argsort(out["combo_index"])
    for key in combo_keys:
        if key in out:
            out[key] = out[key][order]
    return out


def load_primary_data(args: argparse.Namespace) -> dict[str, np.ndarray]:
    if args.data is not None:
        return load_npz(args.data)
    paths = collect_result_files(args.results_dir)
    if not paths:
        raise FileNotFoundError("no gamma data file given and no per-combo gamma results found")
    return concat_files(paths)


def expected_gamma_ids(combo_file: Path) -> set[str]:
    data = load_npz(combo_file)
    return set(map(str, data["gamma_combo_id"]))


def load_full_gamma_comparison(args: argparse.Namespace, fallback: dict[str, np.ndarray]) -> dict[str, np.ndarray] | None:
    expected = expected_gamma_ids(args.gamma_combo_file)
    if expected.issubset(set(map(str, fallback["gamma_combo_id"]))):
        return fallback
    paths = {path.parent.name: path for path in collect_result_files(args.results_dir)}
    if not expected.issubset(set(paths)):
        missing = sorted(expected - set(paths))
        print(f"gamma comparison uses completed coefficient npz files; {len(missing)} planned runs are missing")
        return fallback
    return concat_files([paths[combo_id] for combo_id in sorted(expected)])


def score_arrays(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "best_autocorr_score_val" in data:
        best = np.asarray(data["best_autocorr_score_val"], dtype=np.float32)
    else:
        best = np.asarray(data["autocorr_score_val"], dtype=np.float32)
    if "final_epoch_autocorr_score_val" in data:
        final = np.asarray(data["final_epoch_autocorr_score_val"], dtype=np.float32)
    else:
        final = np.asarray(data["autocorr_score_val"], dtype=np.float32)
    if "best_autocorr_epoch" in data:
        best_epoch = np.asarray(data["best_autocorr_epoch"], dtype=np.int32)
    else:
        best_epoch = np.full(len(best), -1, dtype=np.int32)
    return best, final, best_epoch


def gamma_short_label(value: str) -> str:
    if value == "gamma_default":
        return "default"
    match = value.split("_")
    return "_".join(match[:2]) if len(match) >= 2 else value


def coeff_text(value: float) -> str:
    if np.isclose(value, 0.0):
        return "0"
    if np.isclose(value, 1.0):
        return "1"
    exponent = int(round(np.log10(value)))
    return rf"$10^{{{exponent}}}$"


def coeff_code(value: float) -> float:
    if np.isclose(value, 0.0):
        return 0.0
    return float(np.log10(value) + 4.0)


def sorted_rows(data: dict[str, np.ndarray]) -> np.ndarray:
    best, _final, _best_epoch = score_arrays(data)
    return np.argsort(best)[::-1]


def draw_ranking(data: dict[str, np.ndarray], output_root: Path, args: argparse.Namespace) -> Path:
    order = sorted_rows(data)
    best, final, best_epoch = score_arrays(data)
    n = len(order)
    fig_height = max(3.0, 0.24 * n + 1.25)
    fig, (ax_bar, ax_coeff) = plt.subplots(
        1,
        2,
        figsize=(8.3, fig_height),
        gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.08},
    )
    y = np.arange(n)
    labels = [gamma_short_label(str(data["gamma_combo_id"][idx])) for idx in order]
    ax_bar.barh(y, best[order], color="0.25", height=0.72, label="best")
    ax_bar.scatter(final[order], y, marker="|", s=54, color="black", label="final")
    for row, idx in enumerate(order):
        suffix = f"e{int(best_epoch[idx])}" if int(best_epoch[idx]) >= 0 else ""
        ax_bar.text(min(float(best[idx]) + 0.015, 0.98), row, suffix, fontsize=args.font_size, va="center")
    ax_bar.set_yticks(y, labels=labels)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0.0, 1.0)
    ax_bar.set_xlabel(r"validation score $S_R$", fontsize=args.label_font_size)
    ax_bar.set_title("best and final", fontsize=args.title_font_size, pad=5)
    ax_bar.legend(frameon=False, fontsize=args.font_size, loc="lower right", handlelength=1.4)
    alpha_plot.style_axis(ax_bar, args.font_size)

    matrix = np.asarray([[coeff_code(float(data[key][idx])) for key in LOSS_KEYS] for idx in order], dtype=np.float32)
    im = ax_coeff.imshow(matrix, aspect="auto", cmap="Greys", vmin=0.0, vmax=4.0)
    ax_coeff.set_xticks(np.arange(len(LOSS_KEYS)), labels=LOSS_LABELS, rotation=35, ha="right")
    ax_coeff.set_yticks(y, labels=[])
    ax_coeff.set_title("loss coefficients", fontsize=args.title_font_size, pad=5)
    ax_coeff.tick_params(axis="both", labelsize=args.font_size, length=2.5, width=0.6)
    for row, idx in enumerate(order):
        for col, key in enumerate(LOSS_KEYS):
            value = float(data[key][idx])
            ax_coeff.text(
                col,
                row,
                coeff_text(value),
                ha="center",
                va="center",
                fontsize=args.font_size,
                color="white" if coeff_code(value) >= 3.0 else "black",
            )
    for spine in ax_coeff.spines.values():
        spine.set_linewidth(0.65)

    box = ax_coeff.get_position()
    cax = fig.add_axes([box.x1 + args.heatmap_colorbar_margin, box.y0, args.heatmap_colorbar_width, box.height])
    cbar = fig.colorbar(im, cax=cax, ticks=[0, 1, 2, 3, 4])
    cbar.ax.set_yticklabels(["0", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", "1"])
    cbar.ax.tick_params(labelsize=args.font_size, length=2.5, width=0.6)
    cbar.outline.set_linewidth(0.65)
    fig.suptitle("Stage 3 loss-coefficient sweep", fontsize=10.0, y=0.99)
    path = output_root / "gamma_loss_sweep_ranking"
    alpha_plot.save_figure(fig, path, args.dpi)
    return path.with_suffix(".png")


def epoch_array(data: dict[str, np.ndarray], idx: int) -> tuple[np.ndarray, np.ndarray]:
    epochs = np.asarray(data["epoch_diagnostic_epochs"][idx], dtype=np.int32)
    scores = np.asarray(data["epoch_autocorr_score_val"][idx], dtype=np.float32)
    return epochs, scores


def draw_epoch_heatmap(data: dict[str, np.ndarray], output_root: Path, args: argparse.Namespace) -> Path | None:
    if "epoch_autocorr_score_val" not in data or "epoch_diagnostic_epochs" not in data:
        print("gamma epoch heatmap skipped: epoch diagnostics missing")
        return None
    order = sorted_rows(data)
    epochs, _scores = epoch_array(data, int(order[0]))
    matrix = np.stack([epoch_array(data, int(idx))[1] for idx in order]).astype(np.float32)
    n = len(order)
    fig_height = max(3.0, 0.24 * n + 1.25)
    fig, ax = plt.subplots(figsize=(7.2, fig_height))
    fig.subplots_adjust(left=0.18, right=0.86, bottom=0.12, top=0.90)
    im = ax.imshow(matrix, aspect="auto", cmap="Greys", vmin=0.0, vmax=1.0)
    ax.set_yticks(np.arange(n), labels=[gamma_short_label(str(data["gamma_combo_id"][idx])) for idx in order])
    ax.set_xticks(np.arange(len(epochs)), labels=[str(int(epoch)) for epoch in epochs])
    ax.set_xlabel("epoch", fontsize=args.label_font_size)
    ax.set_ylabel("coefficient setting", fontsize=args.label_font_size)
    ax.set_title("Stage 3 loss sweep", fontsize=args.title_font_size + 1.0, pad=6)
    ax.tick_params(axis="both", labelsize=args.font_size, length=2.5, width=0.6)
    for row, idx in enumerate(order):
        col = int(np.nanargmax(matrix[row]))
        ax.text(
            col,
            row,
            f"{matrix[row, col]:.2f}",
            ha="center",
            va="center",
            fontsize=args.font_size,
            color="white" if matrix[row, col] >= 0.58 else "black",
        )
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
    box = ax.get_position()
    cax = fig.add_axes([box.x1 + args.heatmap_colorbar_margin, box.y0, args.heatmap_colorbar_width, box.height])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"$S_R \in [0,1]$", fontsize=args.label_font_size)
    cbar.ax.tick_params(labelsize=args.font_size, length=2.5, width=0.6)
    cbar.outline.set_linewidth(0.65)
    path = output_root / "gamma_loss_sweep_epoch_heatmap"
    alpha_plot.save_figure(fig, path, args.dpi)
    return path.with_suffix(".png")


def selected_summary_indices(data: dict[str, np.ndarray], args: argparse.Namespace) -> np.ndarray:
    selected = np.arange(len(data["combo_id"]))
    if args.gamma_combo_id:
        selected = selected[np.isin(data["gamma_combo_id"], np.asarray(args.gamma_combo_id))]
        return selected
    if args.top_k_summaries <= 0:
        return selected
    return sorted_rows(data)[: min(args.top_k_summaries, len(selected))]


def plot_all(data: dict[str, np.ndarray], comparison_data: dict[str, np.ndarray] | None, args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not args.summary_only and not args.loss_terms_only:
        if comparison_data is not None:
            draw_ranking(comparison_data, args.output_root, args)
            draw_epoch_heatmap(comparison_data, args.output_root, args)
        else:
            print("gamma ranking and epoch heatmap skipped")
    if args.ranking_only:
        return

    for idx in selected_summary_indices(data, args):
        gamma_id = str(data["gamma_combo_id"][idx]) if "gamma_combo_id" in data else str(data["combo_id"][idx])
        combo_dir = args.output_root / gamma_id
        if not args.loss_terms_only:
            alpha_plot.make_summary(data, int(idx), combo_dir / "summary", args.dpi, args)
        if not args.summary_only:
            alpha_plot.make_loss_terms(data, int(idx), combo_dir / "loss_terms", args.dpi)


def main() -> None:
    args = parse_args()
    data = load_primary_data(args)
    comparison_data = load_full_gamma_comparison(args, data)
    plot_all(data, comparison_data, args)
    print(f"wrote gamma figures under {args.output_root}")


if __name__ == "__main__":
    main()
