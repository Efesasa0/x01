from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import produce_alpha_capacity as alpha

HERE = Path(__file__).resolve().parent
COMBO_PATH = HERE / "alpha_capacity_combos.npz"
SMOKE_OUTPUT = HERE / "beta_capacity_smoke.npz"
RESULTS_DIR = HERE / "results" / "beta"
CHECKPOINT_DIR = HERE / "checkpoints" / "beta"
PLOT_DIR = HERE / "plots" / "beta"
LOG_DIR = HERE / "logs" / "beta"
FIGURE_DIR = HERE / "figures" / "beta"
PLOTTER = HERE / "plot_beta_capacity.py"

DEFAULT_BETA_COMBOS = ("alpha_e2_d064", "alpha_e1_d064", "alpha_e1_d048")
DEFAULT_SUMMARY_EPOCHS = (10, 20, 30)

PER_COMBO_KEYS = {
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


def configure_alpha_paths() -> None:
    alpha.SMOKE_OUTPUT = SMOKE_OUTPUT
    alpha.RESULTS_DIR = RESULTS_DIR
    alpha.CHECKPOINT_DIR = CHECKPOINT_DIR
    alpha.PLOT_DIR = PLOT_DIR
    alpha.LOG_DIR = LOG_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="write synthetic beta data without training")
    parser.add_argument("--combo-file", type=Path, default=COMBO_PATH)
    parser.add_argument("--chunk-id", type=int, default=None)
    parser.add_argument("--combo-id", action="append", default=[])
    parser.add_argument("--combo-index", type=int, action="append", default=[])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=alpha.DATA_PATH)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train", type=int, default=252)
    parser.add_argument("--n-val", type=int, default=31)
    parser.add_argument("--n-test", type=int, default=31)
    parser.add_argument("--t-in", type=int, default=0)
    parser.add_argument("--t", type=int, default=320)
    parser.add_argument("--diagnostic-trajectories", type=int, default=31)
    parser.add_argument("--epoch-diagnostic-trajectories", type=int, default=31)
    parser.add_argument("--epoch-diagnostic-every", type=int, default=1)
    parser.add_argument("--skip-epoch-diagnostics", action="store_true")
    parser.add_argument("--rollout-frames", type=int, default=320)
    parser.add_argument("--qq-quantiles", type=int, default=1000)
    parser.add_argument("--ckpt-every", type=int, default=1)
    parser.add_argument("--plot-every", type=int, default=10)
    parser.add_argument(
        "--summary-rollout-times",
        type=int,
        nargs="+",
        default=list(alpha.DEFAULT_SUMMARY_ROLLOUT_TIMES),
    )
    parser.add_argument("--summary-epochs", type=int, nargs="+", default=list(DEFAULT_SUMMARY_EPOCHS))
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-precomputed-std", action="store_true")
    parser.add_argument("--train-std", type=float, default=4.5722)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--viz-during-train", action="store_true")
    parser.add_argument("--skip-plotter", action="store_true")
    return parser.parse_args()


def selected_combo_indices(combos: dict[str, np.ndarray], args: argparse.Namespace) -> np.ndarray:
    has_filter = args.chunk_id is not None or bool(args.combo_id) or bool(args.combo_index)
    if has_filter:
        return alpha.select_combo_indices(combos, args)

    mask = np.isin(combos["combo_id"], np.asarray(DEFAULT_BETA_COMBOS))
    indices = np.nonzero(mask)[0]
    if indices.size != len(DEFAULT_BETA_COMBOS):
        missing = sorted(set(DEFAULT_BETA_COMBOS) - set(map(str, combos["combo_id"])))
        raise ValueError(f"missing default beta combos: {missing}")
    return indices


def result_path_for_combo(combo_id: str) -> Path:
    return RESULTS_DIR / combo_id / f"beta_capacity_{combo_id}.npz"


def output_path_for_selection(args: argparse.Namespace, written: list[Path]) -> Path | None:
    if len(written) == 1:
        return written[0]
    if args.chunk_id is not None:
        return RESULTS_DIR / f"beta_capacity_chunk{args.chunk_id:02d}.npz"
    return RESULTS_DIR / "beta_capacity_selected.npz"


def run_plotter(args: argparse.Namespace, data_path: Path | None) -> None:
    if args.skip_plotter or data_path is None:
        return
    cmd = [
        sys.executable,
        str(PLOTTER),
        "--data",
        str(data_path.resolve()),
        "--output-root",
        str(FIGURE_DIR.resolve()),
    ]
    print("plotting:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=HERE, check=True)


def update_metadata(arrays: dict[str, Any], row: dict[str, Any], args: argparse.Namespace) -> None:
    raw = arrays.get("metadata_json", np.asarray("{}"))
    try:
        metadata = json.loads(str(np.asarray(raw).item()))
    except Exception:
        metadata = {}
    metadata.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stage": "Beta",
            "beta_default_combos": list(DEFAULT_BETA_COMBOS),
            "combo": alpha.serializable_row(row),
            "summary_epochs": list(map(int, args.summary_epochs)),
            "checkpoint_every": int(args.ckpt_every),
            "epoch_diagnostics": not bool(args.skip_epoch_diagnostics),
        }
    )
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))


def validation_frame_cache(
    args: argparse.Namespace,
    train_std: float,
    n_diag: int,
) -> tuple[list[torch.Tensor], list[np.ndarray], np.ndarray, float, float]:
    full = np.load(args.data, mmap_mode="r")
    val_start = args.n_train
    frames = min(args.rollout_frames, args.t)
    gt_tensors: list[torch.Tensor] = []
    gt_rollouts: list[np.ndarray] = []
    gt_corrs = []
    gt_values = []

    for offset in range(n_diag):
        arr = np.asarray(full[val_start + offset, args.t_in : args.t_in + frames, ::4, ::4], dtype=np.float32)
        arr = arr / train_std
        tensor = torch.from_numpy(arr).unsqueeze(1)
        gt_rollout = (arr * train_std).astype(np.float32)
        gt_future = gt_rollout[1:]
        gt_tensors.append(tensor)
        gt_rollouts.append(gt_rollout)
        gt_corrs.append(alpha.temporal_autocorr(gt_future))
        gt_values.append(gt_future.reshape(-1))

    gt_mean = np.stack(gt_corrs).mean(axis=0).astype(np.float32)
    gt_flat = np.concatenate(gt_values)
    gt_std = float(np.std(gt_flat) + 1e-8)
    gt_abs99 = float(np.percentile(np.abs(gt_flat), 99.0) + 1e-8)
    return gt_tensors, gt_rollouts, gt_mean, gt_std, gt_abs99


def diagnostic_epoch_sequence(args: argparse.Namespace) -> np.ndarray:
    if args.epoch_diagnostic_every < 1:
        raise ValueError("--epoch-diagnostic-every must be >= 1")
    epochs = list(range(1, args.epochs + 1, args.epoch_diagnostic_every))
    if args.epochs not in epochs:
        epochs.append(args.epochs)
    for epoch in args.summary_epochs:
        if 1 <= int(epoch) <= args.epochs and int(epoch) not in epochs:
            epochs.append(int(epoch))
    return np.asarray(sorted(set(epochs)), dtype=np.int32)


def make_epoch_diagnostics(args: argparse.Namespace, ckpt_dir: Path, train_std: float) -> dict[str, Any]:
    if args.skip_epoch_diagnostics:
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = diagnostic_epoch_sequence(args)
    n_diag = min(args.epoch_diagnostic_trajectories, args.n_val)
    gt_tensors, _gt_rollouts, gt_mean, gt_std, gt_abs99 = validation_frame_cache(args, train_std, n_diag)

    scores = []
    std_ratios = []
    abs99_ratios = []
    ckpt_paths = []
    for epoch in epochs:
        ckpt_path = alpha.checkpoint_for_epoch(ckpt_dir, int(epoch))
        model, _ckpt = alpha.load_model(ckpt_path, device)
        pred_corrs = []
        pred_values = []
        with torch.no_grad():
            for gt_tensor_cpu in gt_tensors:
                gt_tensor = gt_tensor_cpu.to(device)
                _gt_rollout, pred_rollout = alpha.model_rollout_full(model, gt_tensor, train_std)
                pred_future = pred_rollout[1:]
                pred_corrs.append(alpha.temporal_autocorr(pred_future))
                pred_values.append(pred_future.reshape(-1))
        pred_mean = np.stack(pred_corrs).mean(axis=0).astype(np.float32)
        pred_flat = np.concatenate(pred_values)
        scores.append(alpha.autocorr_score(gt_mean, pred_mean))
        std_ratios.append(np.asarray(float(np.std(pred_flat) / gt_std), dtype=np.float32))
        abs99_ratios.append(np.asarray(float(np.percentile(np.abs(pred_flat), 99.0) / gt_abs99), dtype=np.float32))
        ckpt_paths.append(str(ckpt_path))
        print(
            f"epoch diagnostic {ckpt_dir.name} epoch={int(epoch)} "
            f"autocorr={float(scores[-1]):.4f} std_ratio={float(std_ratios[-1]):.3f} "
            f"abs99_ratio={float(abs99_ratios[-1]):.3f}",
            flush=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    score_arr = np.asarray(scores, dtype=np.float32)
    best_idx = int(np.nanargmax(score_arr))
    return {
        "epoch_diagnostic_epochs": epochs.astype(np.int32),
        "epoch_autocorr_score_val": score_arr,
        "epoch_rollout_std_ratio_val": np.asarray(std_ratios, dtype=np.float32),
        "epoch_rollout_abs99_ratio_val": np.asarray(abs99_ratios, dtype=np.float32),
        "epoch_checkpoint_path": np.asarray(ckpt_paths),
        "best_autocorr_epoch": np.asarray(int(epochs[best_idx]), dtype=np.int32),
        "best_autocorr_score_val": np.asarray(float(score_arr[best_idx]), dtype=np.float32),
        "final_epoch_autocorr_score_val": np.asarray(float(score_arr[-1]), dtype=np.float32),
        "final_to_best_autocorr_gap": np.asarray(float(score_arr[best_idx] - score_arr[-1]), dtype=np.float32),
    }


def add_per_combo_arrays(arrays: dict[str, Any], extra: dict[str, Any]) -> None:
    for key, value in extra.items():
        arr = np.asarray(value)
        arrays[key] = arr.reshape(1) if arr.ndim == 0 else arr[np.newaxis, ...]


def usable_checkpoint(args: argparse.Namespace, combo_id: str) -> tuple[Path, Path] | None:
    try:
        ckpt_path = alpha.latest_checkpoint(CHECKPOINT_DIR / combo_id)
    except FileNotFoundError:
        return None
    epoch = alpha.checkpoint_epoch(ckpt_path) or -1
    needed = max(args.epochs, max(args.summary_epochs))
    if epoch < needed:
        print(f"checkpoint too early for beta reuse: epoch {epoch} < {needed}", flush=True)
        return None
    return ckpt_path, alpha.latest_log(LOG_DIR / combo_id)


def make_real_combo_data(args: argparse.Namespace, row: dict[str, Any], ckpt_path: Path, log_path: Path) -> dict[str, Any]:
    arrays = alpha.make_real_combo_data(args, row, ckpt_path, log_path)
    update_metadata(arrays, row, args)
    train_std = float(np.asarray(arrays["train_std"]).reshape(-1)[0])
    add_per_combo_arrays(arrays, make_epoch_diagnostics(args, ckpt_path.parent, train_std))
    return arrays


def selected_combo_dict(combos: dict[str, np.ndarray], selected: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for key, value in combos.items():
        if key == "metadata_json":
            out[key] = value
        else:
            out[key] = value[selected]
    return out


def make_smoke_data(combos: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, Any]:
    selected = selected_combo_indices(combos, args)
    arrays = alpha.make_smoke_data(selected_combo_dict(combos, selected), args)
    n = len(arrays["combo_id"])
    epochs = diagnostic_epoch_sequence(args)
    rng = np.random.default_rng(209)
    scores = np.clip(
        0.35
        + 0.45 * np.linspace(0.25, 1.0, len(epochs))[None, :]
        + rng.normal(0.0, 0.04, size=(n, len(epochs))),
        0.0,
        1.0,
    ).astype(np.float32)
    arrays["epoch_diagnostic_epochs"] = np.tile(epochs, (n, 1)).astype(np.int32)
    arrays["epoch_autocorr_score_val"] = scores
    arrays["epoch_rollout_std_ratio_val"] = np.clip(1.0 + rng.normal(0.0, 0.2, size=scores.shape), 0.2, 3.0).astype(
        np.float32
    )
    arrays["epoch_rollout_abs99_ratio_val"] = np.clip(
        1.0 + rng.normal(0.0, 0.25, size=scores.shape), 0.2, 4.0
    ).astype(np.float32)
    arrays["epoch_checkpoint_path"] = np.asarray([["" for _ in epochs] for _ in range(n)])
    best_idx = np.nanargmax(scores, axis=1)
    arrays["best_autocorr_epoch"] = epochs[best_idx].astype(np.int32)
    arrays["best_autocorr_score_val"] = scores[np.arange(n), best_idx].astype(np.float32)
    arrays["final_epoch_autocorr_score_val"] = scores[:, -1].astype(np.float32)
    arrays["final_to_best_autocorr_gap"] = (arrays["best_autocorr_score_val"] - scores[:, -1]).astype(np.float32)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "smoke": True,
        "stage": "Beta",
        "diagnostic_split": "validation",
        "summary_epochs": list(map(int, args.summary_epochs)),
        "note": "Synthetic beta data mirrors the producer schema for plotter development.",
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    return arrays


def concat_results(paths: list[Path]) -> dict[str, Any]:
    loaded = [np.load(path, allow_pickle=False) for path in paths]
    try:
        out: dict[str, Any] = {}
        for key in loaded[0].files:
            values = [data[key] for data in loaded]
            out[key] = np.concatenate(values, axis=0) if key in PER_COMBO_KEYS else values[0]
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "smoke": False,
            "stage": "Beta",
            "merged_files": [str(path) for path in paths],
            "diagnostic_split": "validation",
        }
        out["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
        return out
    finally:
        for data in loaded:
            data.close()


def main() -> None:
    configure_alpha_paths()
    args = parse_args()
    combos = alpha.load_combos(args.combo_file)

    if args.smoke:
        smoke_output = args.output or SMOKE_OUTPUT
        alpha.save_npz(smoke_output, **make_smoke_data(combos, args))
        run_plotter(args, smoke_output)
        return

    selected = selected_combo_indices(combos, args)
    written: list[Path] = []
    for idx in selected:
        row = alpha.combo_row(combos, int(idx))
        combo_id = row["combo_id"]
        out_path = result_path_for_combo(combo_id)
        if out_path.exists() and args.skip_existing and not args.overwrite:
            print(f"skip existing: {out_path}", flush=True)
            written.append(out_path)
            continue

        t0 = time.perf_counter()
        reused = usable_checkpoint(args, combo_id) if args.reuse_checkpoint else None
        if reused is not None:
            ckpt_path, log_path = reused
            print(f"reuse checkpoint: {ckpt_path}", flush=True)
        else:
            ckpt_path, log_path = alpha.run_training(args, row)

        arrays = make_real_combo_data(args, row, ckpt_path, log_path)
        alpha.save_npz(out_path, **arrays)
        written.append(out_path)
        print(f"{combo_id} beta done in {(time.perf_counter() - t0) / 3600:.2f}h", flush=True)

    plot_data = output_path_for_selection(args, written)
    if plot_data is not None and len(written) > 1:
        alpha.save_npz(plot_data, **concat_results(written))
    run_plotter(args, plot_data)


if __name__ == "__main__":
    main()
