from __future__ import annotations

import argparse
import json
import os
import re
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
CAPACITY_COMBO_FILE = HERE / "alpha_capacity_combos.npz"
GAMMA_COMBO_FILE = HERE / "gamma_loss_combos.npz"
SMOKE_OUTPUT = HERE / "gamma_loss_sweep_smoke.npz"
RESULTS_DIR = HERE / "results" / "gamma"
CHECKPOINT_DIR = HERE / "checkpoints" / "gamma"
PLOT_DIR = HERE / "plots" / "gamma"
LOG_DIR = HERE / "logs" / "gamma"
FIGURE_DIR = HERE / "figures" / "gamma"
PLOTTER = HERE / "plot_gamma_loss_sweep.py"

DEFAULT_CAPACITY_COMBO_ID = "alpha_e2_d064"
DEFAULT_SUMMARY_EPOCHS = (2, 4, 8)

LOSS_KEYS = (
    "lambda_fwd",
    "lambda_bwd",
    "lambda_recon",
    "lambda_latent_fwd",
    "lambda_ortho_a",
)

PER_COMBO_KEYS = {
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


def configure_alpha_paths() -> None:
    alpha.SMOKE_OUTPUT = SMOKE_OUTPUT
    alpha.RESULTS_DIR = RESULTS_DIR
    alpha.CHECKPOINT_DIR = CHECKPOINT_DIR
    alpha.PLOT_DIR = PLOT_DIR
    alpha.LOG_DIR = LOG_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="write synthetic gamma data without training")
    parser.add_argument("--capacity-combo-file", type=Path, default=CAPACITY_COMBO_FILE)
    parser.add_argument("--gamma-combo-file", type=Path, default=GAMMA_COMBO_FILE)
    parser.add_argument("--capacity-combo-id", type=str, default=DEFAULT_CAPACITY_COMBO_ID)
    parser.add_argument("--chunk-id", type=int, default=None)
    parser.add_argument("--gamma-combo-id", action="append", default=[])
    parser.add_argument("--gamma-combo-index", type=int, action="append", default=[])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=alpha.DATA_PATH)
    parser.add_argument("--epochs", type=int, default=10)
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
    parser.add_argument("--reuse-default-alpha", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--viz-during-train", action="store_true")
    parser.add_argument("--skip-plotter", action="store_true")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def gamma_row(combos: dict[str, np.ndarray], idx: int) -> dict[str, Any]:
    row = {}
    for key, values in combos.items():
        if key == "metadata_json":
            continue
        value = values[idx]
        if isinstance(value, np.generic):
            value = value.item()
        row[key] = value
    row["gamma_combo_id"] = str(row["gamma_combo_id"])
    row["is_default"] = bool(row["is_default"])
    for key in LOSS_KEYS:
        row[key] = float(row[key])
    return row


def select_gamma_indices(combos: dict[str, np.ndarray], args: argparse.Namespace) -> np.ndarray:
    mask = np.ones(len(combos["gamma_combo_id"]), dtype=bool)
    filters = []
    if args.chunk_id is not None:
        filters.append(combos["gamma_chunk_id"] == args.chunk_id)
    if args.gamma_combo_id:
        filters.append(np.isin(combos["gamma_combo_id"], np.asarray(args.gamma_combo_id)))
    if args.gamma_combo_index:
        filters.append(np.isin(combos["gamma_combo_index"], np.asarray(args.gamma_combo_index)))
    if filters:
        mask = np.logical_or.reduce(filters)
    indices = np.nonzero(mask)[0]
    if indices.size == 0:
        raise ValueError("no gamma combinations selected")
    return indices


def capacity_row(capacity_combos: dict[str, np.ndarray], combo_id: str) -> dict[str, Any]:
    indices = np.nonzero(capacity_combos["combo_id"] == combo_id)[0]
    if indices.size != 1:
        raise ValueError(f"capacity combo id not found uniquely: {combo_id}")
    return alpha.combo_row(capacity_combos, int(indices[0]))


def replace_override(overrides: list[str], prefix: str, value: str) -> list[str]:
    replaced = False
    out = []
    for item in overrides:
        if item.startswith(prefix):
            out.append(f"{prefix}{value}")
            replaced = True
        else:
            out.append(item)
    if not replaced:
        out.append(f"{prefix}{value}")
    return out


def train_overrides(
    args: argparse.Namespace,
    capacity: dict[str, Any],
    gamma: dict[str, Any],
    ckpt_dir: Path,
    plot_dir: Path,
    run_tag: str,
) -> list[str]:
    overrides = alpha.train_overrides(args, capacity, ckpt_dir, plot_dir, run_tag)
    replacements = {
        "ar.lambda_fwd=": gamma["lambda_fwd"],
        "ar.lambda_bwd=": gamma["lambda_bwd"],
        "ar.lambda_recon=": gamma["lambda_recon"],
        "ar.lambda_latent_fwd=": gamma["lambda_latent_fwd"],
        "ar.lambda_ortho_a=": gamma["lambda_ortho_a"],
        "ar.loss_mode=": "lp",
        "ar.lp_size_average=": "true",
    }
    for prefix, value in replacements.items():
        overrides = replace_override(overrides, prefix, str(value).lower() if isinstance(value, bool) else str(value))
    return overrides


def latest_checkpoint(path: Path) -> Path:
    return alpha.latest_checkpoint(path)


def latest_log(path: Path) -> Path:
    return alpha.latest_log(path)


def run_training(args: argparse.Namespace, capacity: dict[str, Any], gamma: dict[str, Any]) -> tuple[Path, Path]:
    gamma_id = gamma["gamma_combo_id"]
    ckpt_dir = CHECKPOINT_DIR / gamma_id
    plot_dir = PLOT_DIR / gamma_id
    log_dir = LOG_DIR / gamma_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{gamma_id}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.out"
    cmd = [
        sys.executable,
        str(alpha.X01_DIR / "train_ar.py"),
        *train_overrides(args, capacity, gamma, ckpt_dir, plot_dir, gamma_id),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(alpha.X01_DIR / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["WANDB_MODE"] = "disabled"
    print("running:", " ".join(cmd), flush=True)
    print("log:", log_path, flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=HERE,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"train_ar.py failed for {gamma_id}; see {log_path}")
    return latest_checkpoint(ckpt_dir), log_path


def usable_checkpoint(args: argparse.Namespace, gamma_id: str) -> tuple[Path, Path] | None:
    try:
        ckpt_path = latest_checkpoint(CHECKPOINT_DIR / gamma_id)
    except FileNotFoundError:
        return None
    epoch = alpha.checkpoint_epoch(ckpt_path) or -1
    needed = max(args.epochs, max(args.summary_epochs))
    if epoch < needed:
        print(f"checkpoint too early for gamma reuse: epoch {epoch} < {needed}", flush=True)
        return None
    return ckpt_path, latest_log(LOG_DIR / gamma_id)


def usable_default_alpha_checkpoint(args: argparse.Namespace) -> tuple[Path, Path] | None:
    if not args.reuse_default_alpha:
        return None
    try:
        ckpt_path = latest_checkpoint(HERE / "checkpoints" / "alpha" / args.capacity_combo_id)
    except FileNotFoundError:
        return None
    epoch = alpha.checkpoint_epoch(ckpt_path) or -1
    needed = max(args.epochs, max(args.summary_epochs))
    if epoch < needed:
        print(f"default alpha checkpoint too early: epoch {epoch} < {needed}", flush=True)
        return None
    return ckpt_path, latest_log(HERE / "logs" / "alpha" / args.capacity_combo_id)


def validation_frame_cache(
    args: argparse.Namespace,
    train_std: float,
    n_diag: int,
) -> tuple[list[torch.Tensor], np.ndarray]:
    full = np.load(args.data, mmap_mode="r")
    val_start = args.n_train
    frames = min(args.rollout_frames, args.t)
    gt_tensors: list[torch.Tensor] = []
    gt_corrs = []
    for offset in range(n_diag):
        arr = np.asarray(full[val_start + offset, args.t_in : args.t_in + frames, ::4, ::4], dtype=np.float32)
        arr = arr / train_std
        tensor = torch.from_numpy(arr).unsqueeze(1)
        gt_tensors.append(tensor)
        gt_corrs.append(alpha.temporal_autocorr((arr * train_std).astype(np.float32)[1:]))
    return gt_tensors, np.stack(gt_corrs).mean(axis=0).astype(np.float32)


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
    gt_tensors, gt_mean = validation_frame_cache(args, train_std, n_diag)

    scores = []
    ckpt_paths = []
    for epoch in epochs:
        ckpt_path = alpha.checkpoint_for_epoch(ckpt_dir, int(epoch))
        model, _ckpt = alpha.load_model(ckpt_path, device)
        pred_corrs = []
        with torch.no_grad():
            for gt_tensor_cpu in gt_tensors:
                gt_tensor = gt_tensor_cpu.to(device)
                _gt_rollout, pred_rollout = alpha.model_rollout_full(model, gt_tensor, train_std)
                pred_corrs.append(alpha.temporal_autocorr(pred_rollout[1:]))
        pred_mean = np.stack(pred_corrs).mean(axis=0).astype(np.float32)
        score = alpha.autocorr_score(gt_mean, pred_mean)
        scores.append(score)
        ckpt_paths.append(str(ckpt_path))
        print(f"epoch diagnostic {ckpt_dir.name} epoch={int(epoch)} autocorr={float(score):.4f}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    score_arr = np.asarray(scores, dtype=np.float32)
    best_idx = int(np.nanargmax(score_arr))
    return {
        "epoch_diagnostic_epochs": epochs.astype(np.int32),
        "epoch_autocorr_score_val": score_arr,
        "epoch_checkpoint_path": np.asarray(ckpt_paths),
        "best_autocorr_epoch": np.asarray(int(epochs[best_idx]), dtype=np.int32),
        "best_autocorr_score_val": np.asarray(float(score_arr[best_idx]), dtype=np.float32),
        "final_epoch_autocorr_score_val": np.asarray(float(score_arr[-1]), dtype=np.float32),
        "final_to_best_autocorr_gap": np.asarray(float(score_arr[best_idx] - score_arr[-1]), dtype=np.float32),
    }


def add_gamma_axes(arrays: dict[str, Any], capacity: dict[str, Any], gamma: dict[str, Any]) -> None:
    gamma_id = gamma["gamma_combo_id"]
    arrays["combo_index"] = np.asarray([int(gamma["gamma_combo_index"])], dtype=np.int32)
    arrays["chunk_id"] = np.asarray([int(gamma["gamma_chunk_id"])], dtype=np.int32)
    arrays["combo_id"] = np.asarray([gamma_id])
    arrays["capacity_combo_id"] = np.asarray([capacity["combo_id"]])
    arrays["gamma_combo_index"] = np.asarray([int(gamma["gamma_combo_index"])], dtype=np.int32)
    arrays["gamma_chunk_id"] = np.asarray([int(gamma["gamma_chunk_id"])], dtype=np.int32)
    arrays["gamma_combo_id"] = np.asarray([gamma_id])
    arrays["is_default"] = np.asarray([bool(gamma["is_default"])])
    for key in LOSS_KEYS:
        arrays[key] = np.asarray([float(gamma[key])], dtype=np.float32)


def add_per_combo_arrays(arrays: dict[str, Any], extra: dict[str, Any]) -> None:
    for key, value in extra.items():
        arr = np.asarray(value)
        arrays[key] = arr.reshape(1) if arr.ndim == 0 else arr[np.newaxis, ...]


def update_metadata(arrays: dict[str, Any], capacity: dict[str, Any], gamma: dict[str, Any], args: argparse.Namespace) -> None:
    raw = arrays.get("metadata_json", np.asarray("{}"))
    try:
        metadata = json.loads(str(np.asarray(raw).item()))
    except Exception:
        metadata = {}
    metadata.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stage": "Gamma",
            "capacity_combo_id": capacity["combo_id"],
            "gamma_combo": {key: gamma[key] for key in ("gamma_combo_id", *LOSS_KEYS)},
            "diagnostic_split": "validation",
            "selection_metric": "best validation autocorrelation over epochs",
            "loss_mode": "lp",
            "lp_size_average": True,
            "summary_epochs": list(map(int, args.summary_epochs)),
            "epoch_diagnostics": not bool(args.skip_epoch_diagnostics),
        }
    )
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))


def make_real_combo_data(
    args: argparse.Namespace,
    capacity: dict[str, Any],
    gamma: dict[str, Any],
    ckpt_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    arrays = alpha.make_real_combo_data(args, capacity, ckpt_path, log_path)
    add_gamma_axes(arrays, capacity, gamma)
    update_metadata(arrays, capacity, gamma, args)
    train_std = float(np.asarray(arrays["train_std"]).reshape(-1)[0])
    add_per_combo_arrays(arrays, make_epoch_diagnostics(args, ckpt_path.parent, train_std))
    return arrays


def selected_gamma_dict(combos: dict[str, np.ndarray], selected: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for key, value in combos.items():
        if key == "metadata_json":
            out[key] = value
        else:
            out[key] = value[selected]
    return out


def make_smoke_data(capacity: dict[str, Any], gamma_combos: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, Any]:
    selected = select_gamma_indices(gamma_combos, args)
    selected_gamma = selected_gamma_dict(gamma_combos, selected)
    base = {
        "combo_index": selected_gamma["gamma_combo_index"],
        "chunk_id": selected_gamma["gamma_chunk_id"],
        "combo_id": selected_gamma["gamma_combo_id"],
        "encoder_profile": np.asarray([capacity["encoder_profile"]] * len(selected)),
        "encoder_level": np.asarray([capacity["encoder_level"]] * len(selected), dtype=np.int32),
        "dims": np.asarray([capacity["dims"]] * len(selected), dtype=np.int32),
        "num_blocks": np.tile(np.asarray(capacity["num_blocks"], dtype=np.int32), (len(selected), 1)),
        "num_heads": np.tile(np.asarray(capacity["num_heads"], dtype=np.int32), (len(selected), 1)),
        "total_params": np.asarray([capacity["total_params"]] * len(selected), dtype=np.int64),
        "encoder_decoder_params": np.asarray([capacity["encoder_decoder_params"]] * len(selected), dtype=np.int64),
        "koopman_params": np.asarray([capacity["koopman_params"]] * len(selected), dtype=np.int64),
        "metadata_json": selected_gamma.get("metadata_json", np.asarray("{}")),
    }
    arrays = alpha.make_smoke_data(base, args)
    n = len(selected)
    arrays["capacity_combo_id"] = np.asarray([capacity["combo_id"]] * n)
    arrays["gamma_combo_index"] = selected_gamma["gamma_combo_index"]
    arrays["gamma_chunk_id"] = selected_gamma["gamma_chunk_id"]
    arrays["gamma_combo_id"] = selected_gamma["gamma_combo_id"]
    arrays["is_default"] = selected_gamma["is_default"]
    for key in LOSS_KEYS:
        arrays[key] = selected_gamma[key]

    epochs = diagnostic_epoch_sequence(args)
    rng = np.random.default_rng(411)
    score_base = np.linspace(0.35, 0.78, len(epochs), dtype=np.float32)
    noise = rng.normal(0.0, 0.04, size=(n, len(epochs))).astype(np.float32)
    scores = np.clip(score_base[None, :] + noise, 0.0, 1.0)
    arrays["epoch_diagnostic_epochs"] = np.tile(epochs, (n, 1)).astype(np.int32)
    arrays["epoch_autocorr_score_val"] = scores.astype(np.float32)
    arrays["epoch_checkpoint_path"] = np.asarray([["" for _ in epochs] for _ in range(n)])
    best_idx = np.nanargmax(scores, axis=1)
    arrays["best_autocorr_epoch"] = epochs[best_idx].astype(np.int32)
    arrays["best_autocorr_score_val"] = scores[np.arange(n), best_idx].astype(np.float32)
    arrays["final_epoch_autocorr_score_val"] = scores[:, -1].astype(np.float32)
    arrays["final_to_best_autocorr_gap"] = (arrays["best_autocorr_score_val"] - scores[:, -1]).astype(np.float32)
    arrays["metadata_json"] = np.asarray(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "smoke": True,
                "stage": "Gamma",
                "capacity_combo_id": capacity["combo_id"],
                "diagnostic_split": "validation",
            },
            sort_keys=True,
        )
    )
    return arrays


def concat_results(paths: list[Path]) -> dict[str, Any]:
    loaded = [np.load(path, allow_pickle=False) for path in paths]
    try:
        out: dict[str, Any] = {}
        for key in loaded[0].files:
            values = [data[key] for data in loaded]
            out[key] = np.concatenate(values, axis=0) if key in PER_COMBO_KEYS else values[0]
        order = np.argsort(out["gamma_combo_index"])
        for key in PER_COMBO_KEYS:
            if key in out:
                out[key] = out[key][order]
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "smoke": False,
            "stage": "Gamma",
            "merged_files": [str(path) for path in paths],
            "diagnostic_split": "validation",
        }
        out["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
        return out
    finally:
        for data in loaded:
            data.close()


def result_path_for_gamma(gamma_id: str) -> Path:
    return RESULTS_DIR / gamma_id / f"gamma_loss_sweep_{gamma_id}.npz"


def run_plotter(args: argparse.Namespace, data_path: Path | None) -> None:
    if args.skip_plotter or data_path is None:
        return
    cmd = [
        sys.executable,
        str(PLOTTER),
        "--gamma-combo-file",
        str(args.gamma_combo_file.resolve()),
        "--data",
        str(data_path.resolve()),
        "--output-root",
        str(FIGURE_DIR.resolve()),
    ]
    print("plotting:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=HERE, check=True)


def output_path_for_selection(args: argparse.Namespace, written: list[Path]) -> Path | None:
    if len(written) == 1:
        return written[0]
    if args.chunk_id is not None:
        return RESULTS_DIR / f"gamma_loss_sweep_chunk{args.chunk_id:02d}.npz"
    return RESULTS_DIR / "gamma_loss_sweep_selected.npz"


def main() -> None:
    configure_alpha_paths()
    args = parse_args()
    capacity_combos = alpha.load_combos(args.capacity_combo_file)
    gamma_combos = load_npz(args.gamma_combo_file)
    capacity = capacity_row(capacity_combos, args.capacity_combo_id)

    if args.smoke:
        smoke_output = args.output or SMOKE_OUTPUT
        alpha.save_npz(smoke_output, **make_smoke_data(capacity, gamma_combos, args))
        run_plotter(args, smoke_output)
        return

    selected = select_gamma_indices(gamma_combos, args)
    written: list[Path] = []
    for idx in selected:
        gamma = gamma_row(gamma_combos, int(idx))
        gamma_id = gamma["gamma_combo_id"]
        out_path = result_path_for_gamma(gamma_id)
        if out_path.exists() and args.skip_existing and not args.overwrite:
            print(f"skip existing: {out_path}", flush=True)
            written.append(out_path)
            continue

        t0 = time.perf_counter()
        reused = usable_checkpoint(args, gamma_id) if args.reuse_checkpoint else None
        if reused is None and gamma["is_default"]:
            reused = usable_default_alpha_checkpoint(args)
        if reused is not None:
            ckpt_path, log_path = reused
            print(f"reuse checkpoint: {ckpt_path}", flush=True)
        else:
            ckpt_path, log_path = run_training(args, capacity, gamma)

        arrays = make_real_combo_data(args, capacity, gamma, ckpt_path, log_path)
        alpha.save_npz(out_path, **arrays)
        written.append(out_path)
        print(f"{gamma_id} gamma done in {(time.perf_counter() - t0) / 3600:.2f}h", flush=True)

    plot_data = output_path_for_selection(args, written)
    if plot_data is not None and len(written) > 1:
        alpha.save_npz(plot_data, **concat_results(written))
    run_plotter(args, plot_data)


if __name__ == "__main__":
    main()
