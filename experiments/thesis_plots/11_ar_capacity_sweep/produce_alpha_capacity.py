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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
X01_DIR = ROOT / "x01"
DATA_PATH = ROOT.parent / "kmflow_re1000_r256_b400_f320_clean.npy"
COMBO_PATH = HERE / "alpha_capacity_combos.npz"
SMOKE_OUTPUT = HERE / "alpha_capacity_smoke.npz"
RESULTS_DIR = HERE / "results" / "alpha"
CHECKPOINT_DIR = HERE / "checkpoints" / "alpha"
PLOT_DIR = HERE / "plots" / "alpha"
LOG_DIR = HERE / "logs" / "alpha"
PLOTTER = HERE / "plot_alpha_capacity.py"

DEFAULT_ROLLOUT_TIMES = (1, 2, 3, 4, 5, 10)
DEFAULT_SUMMARY_ROLLOUT_TIMES = (0, 25, 50, 100, 200, 319)
DEFAULT_SUMMARY_EPOCHS = (2, 4, 8)
LOSS_RECIPE = {
    "lambda_fwd": 1.0,
    "lambda_recon": 1.0,
    "lambda_bwd": 1.0,
    "lambda_latent_fwd": 1.0,
    "lambda_ortho_a": 0.1,
    "lambda_consist": 0.0,
    "dynamics_mode": "single",
    "pixel_rollout_steps": 1,
    "latent_rollout_steps": 1,
    "loss_mode": "lp",
    "lp_size_average": True,
}

TMP_ROOT = Path(os.environ.get("TMPDIR", "/tmp"))
MPLCONFIGDIR = TMP_ROOT / "x01_mplconfig"
XDG_CACHE_HOME = TMP_ROOT / "x01_cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))
os.environ.setdefault("WANDB_MODE", "disabled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="write synthetic all-combo data without training")
    parser.add_argument("--combo-file", type=Path, default=COMBO_PATH)
    parser.add_argument("--chunk-id", type=int, default=None)
    parser.add_argument("--combo-id", action="append", default=[])
    parser.add_argument("--combo-index", type=int, action="append", default=[])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
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
    parser.add_argument("--rollout-frames", type=int, default=320)
    parser.add_argument("--qq-quantiles", type=int, default=1000)
    parser.add_argument("--ckpt-every", type=int, default=1)
    parser.add_argument("--plot-every", type=int, default=10)
    parser.add_argument("--summary-rollout-times", type=int, nargs="+", default=list(DEFAULT_SUMMARY_ROLLOUT_TIMES))
    parser.add_argument("--summary-epochs", type=int, nargs="+", default=list(DEFAULT_SUMMARY_EPOCHS))
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-precomputed-std", action="store_true")
    parser.add_argument("--train-std", type=float, default=4.5722)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reuse-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="if a combo checkpoint already exists, post-process it instead of retraining",
    )
    parser.add_argument("--viz-during-train", action="store_true")
    parser.add_argument("--skip-plotter", action="store_true")
    return parser.parse_args()


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(path, flush=True)


def run_plotter(args: argparse.Namespace, data_path: Path | None = None, output_root: Path | None = None) -> None:
    if args.skip_plotter:
        return
    cmd = [sys.executable, str(PLOTTER), "--combo-file", str(args.combo_file.resolve())]
    if data_path is not None:
        cmd.extend(["--data", str(data_path.resolve())])
    if output_root is not None:
        cmd.extend(["--output-root", str(output_root.resolve())])
    for combo_id in args.combo_id:
        cmd.extend(["--combo-id", combo_id])
    print("plotting:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=HERE, check=True)


def load_combos(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"combo file does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def select_combo_indices(combos: dict[str, np.ndarray], args: argparse.Namespace) -> np.ndarray:
    mask = np.ones(len(combos["combo_id"]), dtype=bool)
    filters = []
    if args.chunk_id is not None:
        filters.append(combos["chunk_id"] == args.chunk_id)
    if args.combo_id:
        filters.append(np.isin(combos["combo_id"], np.asarray(args.combo_id)))
    if args.combo_index:
        filters.append(np.isin(combos["combo_index"], np.asarray(args.combo_index)))
    if filters:
        mask = np.logical_or.reduce(filters)
    indices = np.nonzero(mask)[0]
    if indices.size == 0:
        raise ValueError("no combos selected")
    return indices


def combo_row(combos: dict[str, np.ndarray], idx: int) -> dict[str, Any]:
    row = {}
    for key, values in combos.items():
        if key == "metadata_json":
            continue
        value = values[idx]
        if isinstance(value, np.generic):
            value = value.item()
        row[key] = value
    row["combo_id"] = str(row["combo_id"])
    row["encoder_profile"] = str(row["encoder_profile"])
    row["num_blocks"] = tuple(int(x) for x in np.asarray(row["num_blocks"]).tolist())
    row["num_heads"] = tuple(int(x) for x in np.asarray(row["num_heads"]).tolist())
    return row


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), *args], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def data_stamp(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {"path": str(path), "exists": True, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def train_overrides(args: argparse.Namespace, row: dict[str, Any], ckpt_dir: Path, plot_dir: Path, run_tag: str) -> list[str]:
    blocks = ",".join(str(x) for x in row["num_blocks"])
    heads = ",".join(str(x) for x in row["num_heads"])
    return [
        f"seed={args.seed}",
        f"data.path={args.data.resolve()}",
        f"data.n_train={args.n_train}",
        f"data.n_val={args.n_val}",
        f"data.n_test={args.n_test}",
        f"data.T_in={args.t_in}",
        f"data.T={args.t}",
        f"data.standardize={str(args.standardize).lower()}",
        f"data.use_precomputed_std={str(args.use_precomputed_std).lower()}",
        f"data.train_std={args.train_std}",
        f"training.epochs={args.epochs}",
        f"training.batch_size={args.batch_size}",
        f"training.num_workers={args.num_workers}",
        f"training.weight_decay=0.0",
        f"training.scheduler=none",
        "training.smoke=false",
        "wandb.enabled=false",
        f"viz.enabled={str(args.viz_during_train).lower()}",
        "save.enabled=true",
        f"save.ckpt_every={args.ckpt_every}",
        f"save.plot_every={args.plot_every}",
        f"save.ckpt_dir={ckpt_dir.resolve()}",
        f"save.plot_dir={plot_dir.resolve()}",
        f"save.run_tag={run_tag}",
        "inference_mode=koopman",
        f"ar.dims={int(row['dims'])}",
        f"ar.num_blocks=[{blocks}]",
        f"ar.num_heads=[{heads}]",
        "ar.dynamics_mode=single",
        "ar.dynamics_rank=null",
        "ar.pixel_rollout_steps=1",
        "ar.latent_rollout_steps=1",
        "ar.lambda_fwd=1.0",
        "ar.lambda_recon=1.0",
        "ar.lambda_bwd=1.0",
        "ar.lambda_consist=0.0",
        "ar.lambda_latent_fwd=1.0",
        "ar.lambda_latent_bwd=0.0",
        "ar.lambda_ortho_a=0.1",
        "ar.lambda_ortho_b=0.0",
        "ar.lambda_consist_progressive=0.0",
        "ar.loss_mode=lp",
        "ar.lp_size_average=true",
    ]


def run_training(args: argparse.Namespace, row: dict[str, Any]) -> tuple[Path, Path]:
    combo_id = row["combo_id"]
    ckpt_dir = CHECKPOINT_DIR / combo_id
    plot_dir = PLOT_DIR / combo_id
    log_dir = LOG_DIR / combo_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{combo_id}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.out"
    cmd = [sys.executable, str(X01_DIR / "train_ar.py"), *train_overrides(args, row, ckpt_dir, plot_dir, combo_id)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(X01_DIR / "src") + os.pathsep + env.get("PYTHONPATH", "")
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
        raise RuntimeError(f"train_ar.py failed for {combo_id}; see {log_path}")
    return latest_checkpoint(ckpt_dir), log_path


def latest_checkpoint(path: Path) -> Path:
    candidates = sorted(path.rglob("*_ar_epoch*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no AR checkpoints found in {path}")
    return max(candidates, key=lambda p: (checkpoint_epoch(p) or -1, p.stat().st_mtime_ns))


def latest_log(path: Path) -> Path:
    candidates = sorted(path.glob("*.out"))
    if not candidates:
        return Path("")
    return max(candidates, key=lambda p: p.stat().st_mtime_ns)


def checkpoint_epoch(path: Path) -> int | None:
    match = re.search(r"_epoch(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else None


def checkpoint_for_epoch(path: Path, epoch: int) -> Path:
    for candidate in sorted(path.rglob("*_ar_epoch*.pt")):
        if checkpoint_epoch(candidate) == epoch:
            return candidate
    raise FileNotFoundError(
        f"missing checkpoint for epoch {epoch} in {path}; use --ckpt-every 1 or adjust --summary-epochs"
    )


def parse_history(log_path: Path | None) -> dict[str, np.ndarray]:
    keys = [
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
    ]
    if log_path is None or not log_path.exists():
        return {"loss_epochs": np.asarray([], dtype=np.int32), **{key: np.asarray([], dtype=np.float32) for key in keys}}

    epoch_re = re.compile(r"epoch\s+(\d+)/(\d+)\s+train_raw=([0-9.eE+-]+)\s+\|\s+w_val=([0-9.eE+-]+)")
    raw_val_re = re.compile(
        r"raw_val fwd=([0-9.eE+-]+) recon=([0-9.eE+-]+) consist=([0-9.eE+-]+) "
        r"bwd=([0-9.eE+-]+) latent_fwd=([0-9.eE+-]+) latent_bwd=([0-9.eE+-]+) "
        r"ortho_a=([0-9.eE+-]+) ortho_b=([0-9.eE+-]+) consist_prog=([0-9.eE+-]+)"
    )
    unit_re = re.compile(r"A_unit_dev=([0-9.eE+-]+)\s+B_unit_dev=([0-9.eE+-]+)")
    eig_re = re.compile(r"A_min=([0-9.eE+-]+).*A_mean=([0-9.eE+-]+).*A_max=([0-9.eE+-]+).*B_min=([0-9.eE+-]+).*B_mean=([0-9.eE+-]+).*B_max=([0-9.eE+-]+)")

    epochs = []
    history = {key: [] for key in keys}
    pending_epoch = False
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = epoch_re.search(line)
        if match:
            epochs.append(int(match.group(1)))
            history["train_raw_loss"].append(float(match.group(3)))
            history["val_weighted_loss"].append(float(match.group(4)))
            pending_epoch = True
            continue
        match = raw_val_re.search(line)
        if match and pending_epoch:
            names = [
                "val_forward_loss",
                "val_reconstruction_loss",
                "val_consistency_loss",
                "val_backward_loss",
                "val_latent_fwd_loss",
                "val_latent_bwd_loss",
                "val_ortho_a_loss",
                "val_ortho_b_loss",
                "val_consist_progressive_loss",
            ]
            for name, value in zip(names, match.groups()):
                history[name].append(float(value))
            continue
        if unit_re.search(line):
            pending_epoch = False
            continue
        match = eig_re.search(line)
        if match:
            for name, value in zip(
                ["eig_A_min", "eig_A_mean", "eig_A_max", "eig_B_min", "eig_B_mean", "eig_B_max"],
                match.groups(),
            ):
                history[name].append(float(value))

    n_epochs = len(epochs)
    out = {"loss_epochs": np.asarray(epochs, dtype=np.int32)}
    for key in keys:
        values = history[key]
        if len(values) < n_epochs:
            values = values + [np.nan] * (n_epochs - len(values))
        out[key] = np.asarray(values[:n_epochs], dtype=np.float32)
    return out


def strip_state_dict_prefixes(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def load_model(ckpt_path: Path, device: torch.device):
    sys.path.insert(0, str(X01_DIR / "src"))
    from x01.ar import KoopmanAE2D

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = KoopmanAE2D(
        in_channels=1,
        out_channels=1,
        dims=int(cfg["dims"]),
        num_blocks=tuple(cfg["num_blocks"]),
        num_heads=tuple(cfg["num_heads"]),
        dynamics_mode=str(cfg.get("dynamics_mode", "single")),
        dynamics_rank=cfg.get("dynamics_rank"),
        steps=int(cfg.get("pixel_rollout_steps", 1) or 1),
    ).to(device)
    model.load_state_dict(strip_state_dict_prefixes(ckpt["model"]))
    model.eval()
    return model, ckpt


def load_lr_trajectory(full: np.ndarray, idx: int, t_in: int, frames: int, train_std: float, device: torch.device) -> torch.Tensor:
    arr = np.asarray(full[idx, t_in : t_in + frames, ::4, ::4], dtype=np.float32)
    arr = arr / train_std
    return torch.from_numpy(arr).unsqueeze(1).to(device)


def rollout_future(model, gt_frames: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model.rollout(gt_frames[0:1], gt_frames.shape[0] - 1, inference_mode="koopman")


def model_rollout_full(model, gt_frames: torch.Tensor, train_std: float) -> tuple[np.ndarray, np.ndarray]:
    pred_frames = rollout_future(model, gt_frames)
    gt_rollout = (gt_frames[:, 0].detach().cpu().numpy() * train_std).astype(np.float32)
    pred_future = (pred_frames[:, 0].detach().cpu().numpy() * train_std).astype(np.float32)
    pred_rollout = np.concatenate([gt_rollout[:1], pred_future], axis=0).astype(np.float32)
    return gt_rollout, pred_rollout


def temporal_autocorr(frames: np.ndarray) -> np.ndarray:
    frames = frames.astype(np.float64, copy=False)
    t_len = frames.shape[0]
    flat = frames.reshape(t_len, -1)
    corr = np.empty(t_len, dtype=np.float64)
    for tau in range(t_len):
        n = t_len - tau
        corr[tau] = (flat[:n] * flat[tau:]).sum(axis=1).mean()
    if abs(corr[0]) > 1e-12:
        corr /= corr[0]
    return corr.astype(np.float32)


def autocorr_score(gt_mean: np.ndarray, pred_mean: np.ndarray) -> np.float32:
    denom = float(np.trapezoid(np.abs(gt_mean)) + 1e-8)
    error = float(np.trapezoid(np.abs(pred_mean - gt_mean)))
    return np.asarray(np.clip(1.0 - error / denom, 0.0, 1.0), dtype=np.float32)


def energy_spectrum(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _t, h, w = frames.shape
    kx = np.fft.fftfreq(w) * w
    ky = np.fft.fftfreq(h) * h
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_rad = np.round(np.sqrt(kx_grid**2 + ky_grid**2)).astype(int)
    power = np.abs(np.fft.fft2(frames)) ** 2 / (h * w) ** 2
    k_max = min(h, w) // 2
    k_bins = np.arange(1, k_max + 1)
    spectrum = np.array([power[:, k_rad == k].mean() for k in k_bins], dtype=np.float32)
    return k_bins.astype(np.int32), spectrum


def diagnostic_arrays(gt: np.ndarray, pred: np.ndarray, n_quantiles: int) -> dict[str, np.ndarray]:
    gt_r, pred_r = [], []
    gt_e, pred_e = [], []
    gt_vals, pred_vals = [], []
    k_bins = None
    for gt_frames, pred_frames in zip(gt, pred):
        gt_r.append(temporal_autocorr(gt_frames))
        pred_r.append(temporal_autocorr(pred_frames))
        k_bins, gt_spec = energy_spectrum(gt_frames)
        _, pred_spec = energy_spectrum(pred_frames)
        gt_e.append(gt_spec)
        pred_e.append(pred_spec)
        gt_vals.append(gt_frames.ravel())
        pred_vals.append(pred_frames.ravel())

    gt_r_arr = np.stack(gt_r)
    pred_r_arr = np.stack(pred_r)
    gt_e_arr = np.stack(gt_e)
    pred_e_arr = np.stack(pred_e)
    quantiles = np.linspace(0, 100, n_quantiles, dtype=np.float32)
    gt_q = np.percentile(np.concatenate(gt_vals), quantiles).astype(np.float32)
    pred_q = np.percentile(np.concatenate(pred_vals), quantiles).astype(np.float32)
    gt_mean = gt_r_arr.mean(axis=0).astype(np.float32)
    pred_mean = pred_r_arr.mean(axis=0).astype(np.float32)

    return {
        "t_star": np.linspace(0, 1, gt_r_arr.shape[1], dtype=np.float32),
        "autocorr_score_val": autocorr_score(gt_mean, pred_mean),
        "autocorr_gt_mean": gt_mean,
        "autocorr_gt_std": gt_r_arr.std(axis=0).astype(np.float32),
        "autocorr_pred_mean": pred_mean,
        "autocorr_pred_std": pred_r_arr.std(axis=0).astype(np.float32),
        "k_bins": k_bins.astype(np.int32),
        "energy_gt_mean": gt_e_arr.mean(axis=0).astype(np.float32),
        "energy_gt_std": gt_e_arr.std(axis=0).astype(np.float32),
        "energy_pred_mean": pred_e_arr.mean(axis=0).astype(np.float32),
        "energy_pred_std": pred_e_arr.std(axis=0).astype(np.float32),
        "qq_quantiles": quantiles,
        "qq_gt": gt_q,
        "qq_pred": pred_q,
    }


def rollout_display(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timesteps = np.asarray(DEFAULT_ROLLOUT_TIMES, dtype=np.int32)
    if np.any(timesteps >= min(gt.shape[0], pred.shape[0])):
        raise ValueError("rollout times exceed available frames")
    return timesteps, gt[timesteps].astype(np.float32), pred[timesteps].astype(np.float32)


def summary_rollout_display(
    args: argparse.Namespace,
    ckpt_dir: Path,
    gt_frames: torch.Tensor,
    gt_rollout: np.ndarray,
    train_std: float,
    device: torch.device,
) -> dict[str, np.ndarray]:
    timesteps = np.asarray(args.summary_rollout_times, dtype=np.int32)
    if np.any(timesteps >= gt_rollout.shape[0]):
        raise ValueError("summary rollout times exceed available frames")

    preds = []
    epochs = np.asarray(args.summary_epochs, dtype=np.int32)
    for epoch in epochs:
        model_epoch, _ckpt_epoch = load_model(checkpoint_for_epoch(ckpt_dir, int(epoch)), device)
        _gt_epoch, pred_epoch = model_rollout_full(model_epoch, gt_frames, train_std)
        preds.append(pred_epoch[timesteps].astype(np.float32))
        del model_epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "summary_rollout_t": timesteps,
        "summary_rollout_epochs": epochs,
        "summary_gt_rollout": gt_rollout[timesteps].astype(np.float32),
        "summary_pred_rollout": np.stack(preds).astype(np.float32),
    }


def make_real_combo_data(args: argparse.Namespace, row: dict[str, Any], ckpt_path: Path, log_path: Path) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(ckpt_path, device)
    train_std = float(ckpt["train_std"])
    frames = min(args.rollout_frames, args.t)
    full = np.load(args.data, mmap_mode="r")
    val_start = args.n_train
    n_diag = min(args.diagnostic_trajectories, args.n_val)

    gt_list, pred_list = [], []
    rollout_t = rollout_gt = rollout_pred = None
    first_gt_frames = None
    first_gt_rollout = None
    first_pred_rollout = None
    for offset in range(n_diag):
        gt_frames = load_lr_trajectory(full, val_start + offset, args.t_in, frames, train_std, device)
        gt_rollout, pred_rollout = model_rollout_full(model, gt_frames, train_std)
        gt_future = gt_rollout[1:]
        pred_future = pred_rollout[1:]
        gt_list.append(gt_future)
        pred_list.append(pred_future)
        if offset == 0:
            first_gt_frames = gt_frames
            first_gt_rollout = gt_rollout
            first_pred_rollout = pred_rollout
            rollout_t, rollout_gt, rollout_pred = rollout_display(gt_rollout, pred_rollout)

    diagnostics = diagnostic_arrays(np.stack(gt_list), np.stack(pred_list), args.qq_quantiles)
    history = parse_history(log_path)
    if first_gt_frames is None or first_gt_rollout is None or first_pred_rollout is None:
        raise RuntimeError("no validation trajectory was available for summary rollout")
    summary_rollouts = summary_rollout_display(
        args,
        ckpt_path.parent,
        first_gt_frames,
        first_gt_rollout,
        train_std,
        device,
    )
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "smoke": False,
        "stage": "Alpha",
        "combo": serializable_row(row),
        "loss_recipe": LOSS_RECIPE,
        "diagnostic_split": "validation",
        "diagnostic_trajectories": int(n_diag),
        "checkpoint_path": str(ckpt_path),
        "train_log_path": str(log_path),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "rollout_gif_frames": int(first_gt_rollout.shape[0] - 1),
        "data": data_stamp(args.data),
        "git_head": git_value(["rev-parse", "HEAD"]),
        "git_branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": git_value(["status", "--short"]),
    }
    return with_combo_axes(
        row,
        {
            **diagnostics,
            **history,
            "rollout_t": rollout_t.astype(np.int32),
            "rollout_gt": rollout_gt.astype(np.float32),
            "rollout_pred": rollout_pred.astype(np.float32),
            "rollout_abs_error": np.abs(rollout_pred - rollout_gt).astype(np.float32),
            "rollout_gif_t": np.arange(1, first_gt_rollout.shape[0], dtype=np.int32),
            "rollout_gif_gt": first_gt_rollout[1:].astype(np.float32),
            "rollout_gif_pred": first_pred_rollout[1:].astype(np.float32),
            **summary_rollouts,
            "train_std": np.asarray(train_std, dtype=np.float32),
            "checkpoint_path": np.asarray(str(ckpt_path)),
            "train_log_path": np.asarray(str(log_path)),
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        },
    )


def serializable_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in row.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, np.generic):
            out[key] = value.item()
        elif isinstance(value, tuple):
            out[key] = list(value)
        else:
            out[key] = value
    return out


def with_combo_axes(row: dict[str, Any], arrays: dict[str, Any]) -> dict[str, Any]:
    prefixed = {
        "combo_index": np.asarray([row["combo_index"]], dtype=np.int32),
        "chunk_id": np.asarray([row["chunk_id"]], dtype=np.int32),
        "combo_id": np.asarray([row["combo_id"]]),
        "encoder_profile": np.asarray([row["encoder_profile"]]),
        "encoder_level": np.asarray([row["encoder_level"]], dtype=np.int32),
        "dims": np.asarray([row["dims"]], dtype=np.int32),
        "num_blocks": np.asarray([row["num_blocks"]], dtype=np.int32),
        "num_heads": np.asarray([row["num_heads"]], dtype=np.int32),
        "total_params": np.asarray([row["total_params"]], dtype=np.int64),
        "encoder_decoder_params": np.asarray([row["encoder_decoder_params"]], dtype=np.int64),
        "koopman_params": np.asarray([row["koopman_params"]], dtype=np.int64),
    }
    for key, value in arrays.items():
        arr = np.asarray(value)
        if key in (
            "t_star",
            "k_bins",
            "qq_quantiles",
            "loss_epochs",
            "rollout_t",
            "rollout_gif_t",
            "summary_rollout_t",
            "summary_rollout_epochs",
            "metadata_json",
        ):
            prefixed[key] = arr
        elif arr.ndim == 0 and key not in ("checkpoint_path", "train_log_path"):
            prefixed[key] = arr.reshape(1)
        elif key in ("checkpoint_path", "train_log_path"):
            prefixed[key] = arr.reshape(1)
        else:
            prefixed[key] = arr[np.newaxis, ...]
    return prefixed


def make_smoke_data(combos: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(110)
    n = len(combos["combo_id"])
    n_frames = 319
    h = w = 64
    n_rollout = len(DEFAULT_ROLLOUT_TIMES)
    n_epochs = args.epochs
    q = np.linspace(0, 100, args.qq_quantiles, dtype=np.float32)

    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    base = []
    for t in range(n_frames + 1):
        phase = 2 * np.pi * t / 90.0
        base.append((np.sin(3 * X + phase) * np.cos(2 * Y - 0.4 * phase)).astype(np.float32))
    base = np.asarray(base)

    t_star = np.linspace(0, 1, n_frames, dtype=np.float32)
    k_bins = np.arange(1, 33, dtype=np.int32)
    epochs = np.arange(1, n_epochs + 1, dtype=np.int32)
    rollout_t = np.asarray(DEFAULT_ROLLOUT_TIMES, dtype=np.int32)
    summary_rollout_t = np.asarray(args.summary_rollout_times, dtype=np.int32)
    summary_epochs = np.asarray(args.summary_epochs, dtype=np.int32)
    if np.any(summary_rollout_t > n_frames):
        raise ValueError("summary rollout times exceed smoke trajectory length")

    autocorr_gt = temporal_autocorr(base[1:])
    energy_k, energy_gt = energy_spectrum(base[1:])
    assert np.array_equal(k_bins, energy_k)
    qq_gt = np.percentile(base[1:].reshape(-1), q).astype(np.float32)

    arrays: dict[str, Any] = {
        "combo_index": combos["combo_index"],
        "chunk_id": combos["chunk_id"],
        "combo_id": combos["combo_id"],
        "encoder_profile": combos["encoder_profile"],
        "encoder_level": combos["encoder_level"],
        "dims": combos["dims"],
        "num_blocks": combos["num_blocks"],
        "num_heads": combos["num_heads"],
        "total_params": combos["total_params"],
        "encoder_decoder_params": combos["encoder_decoder_params"],
        "koopman_params": combos["koopman_params"],
        "t_star": t_star,
        "k_bins": k_bins,
        "qq_quantiles": q,
        "loss_epochs": epochs,
        "rollout_t": rollout_t,
        "summary_rollout_t": summary_rollout_t,
        "summary_rollout_epochs": summary_epochs,
    }

    shape_t = (n, n_frames)
    shape_k = (n, len(k_bins))
    shape_q = (n, len(q))
    shape_e = (n, n_epochs)
    arrays["autocorr_gt_mean"] = np.tile(autocorr_gt, (n, 1)).astype(np.float32)
    arrays["autocorr_gt_std"] = np.zeros(shape_t, dtype=np.float32)
    arrays["energy_gt_mean"] = np.tile(energy_gt, (n, 1)).astype(np.float32)
    arrays["energy_gt_std"] = np.zeros(shape_k, dtype=np.float32)
    arrays["qq_gt"] = np.tile(qq_gt, (n, 1)).astype(np.float32)

    pred_corr = []
    scores = []
    energy_pred = []
    qq_pred = []
    rollout_gt = []
    rollout_pred = []
    rollout_gif_gt = []
    rollout_gif_pred = []
    summary_gt = []
    summary_pred = []
    train_loss = np.zeros(shape_e, dtype=np.float32)
    val_loss = np.zeros(shape_e, dtype=np.float32)
    loss_terms = {name: np.zeros(shape_e, dtype=np.float32) for name in [
        "val_forward_loss",
        "val_backward_loss",
        "val_reconstruction_loss",
        "val_latent_fwd_loss",
        "val_latent_bwd_loss",
        "val_consistency_loss",
        "val_ortho_a_loss",
        "val_ortho_b_loss",
        "val_consist_progressive_loss",
    ]}
    eig = {name: np.zeros(shape_e, dtype=np.float32) for name in [
        "eig_A_min",
        "eig_A_mean",
        "eig_A_max",
        "eig_B_min",
        "eig_B_mean",
        "eig_B_max",
    ]}

    dims = combos["dims"].astype(np.float32)
    levels = combos["encoder_level"].astype(np.float32)
    dim_span = max(float(dims.max() - dims.min()), 1.0)
    level_span = max(float(levels.max() - levels.min()), 1.0)
    quality = 0.35 + 0.38 * (dims - dims.min()) / dim_span + 0.12 * (levels - levels.min()) / level_span
    quality = np.clip(quality + rng.normal(0, 0.04, size=n), 0.25, 0.92)
    for i in range(n):
        drift = (1.0 - quality[i]) * np.linspace(0, 0.8, n_frames + 1, dtype=np.float32)[:, None, None]
        noise = rng.normal(0, 0.035 * (1.0 - quality[i]), size=base.shape).astype(np.float32)
        pred = base + drift * np.sin(4 * X - Y)[None, ...] + noise
        pred[0] = base[0]
        corr = temporal_autocorr(pred[1:])
        pred_corr.append(corr)
        scores.append(autocorr_score(autocorr_gt, corr))
        _, e_pred = energy_spectrum(pred[1:])
        energy_pred.append(e_pred)
        qq_pred.append(np.percentile(pred[1:].reshape(-1), q).astype(np.float32))
        rollout_gt.append(base[rollout_t])
        rollout_pred.append(pred[rollout_t])
        rollout_gif_gt.append(base[1:])
        rollout_gif_pred.append(pred[1:])
        summary_gt.append(base[summary_rollout_t])

        epoch_rows = []
        for epoch in summary_epochs:
            progress = float(np.clip(epoch / max(1, n_epochs), 0.0, 1.0))
            epoch_quality = np.clip(0.25 + quality[i] * (0.25 + 0.75 * progress), 0.20, 0.96)
            epoch_drift = (1.0 - epoch_quality) * np.linspace(0, 0.8, n_frames + 1, dtype=np.float32)[:, None, None]
            epoch_noise = rng.normal(0, 0.035 * (1.0 - epoch_quality), size=base.shape).astype(np.float32)
            pred_epoch = base + epoch_drift * np.sin(4 * X - Y)[None, ...] + epoch_noise
            pred_epoch[0] = base[0]
            epoch_rows.append(pred_epoch[summary_rollout_t])
        summary_pred.append(np.stack(epoch_rows))

        decay = np.exp(-np.linspace(0, 2.5 + quality[i], n_epochs))
        train_loss[i] = (1.5 * decay + 0.08 * rng.random(n_epochs)).astype(np.float32)
        val_loss[i] = (1.7 * decay + 0.12 * rng.random(n_epochs) + 0.05 * (1.0 - quality[i])).astype(np.float32)
        loss_terms["val_forward_loss"][i] = 0.45 * val_loss[i]
        loss_terms["val_backward_loss"][i] = 0.22 * val_loss[i]
        loss_terms["val_reconstruction_loss"][i] = 0.25 * val_loss[i]
        loss_terms["val_latent_fwd_loss"][i] = 0.18 * val_loss[i]
        loss_terms["val_ortho_a_loss"][i] = np.maximum(0.01, (1.0 - quality[i]) * np.exp(-np.linspace(0, 1.2, n_epochs)))
        eig["eig_A_min"][i] = 0.82 + 0.11 * quality[i] + 0.02 * np.linspace(0, 1, n_epochs)
        eig["eig_A_mean"][i] = 0.93 + 0.05 * quality[i] + 0.01 * np.linspace(0, 1, n_epochs)
        eig["eig_A_max"][i] = 1.03 + 0.03 * (1.0 - quality[i]) + 0.01 * np.sin(np.linspace(0, 2, n_epochs))
        eig["eig_B_min"][i] = eig["eig_A_min"][i]
        eig["eig_B_mean"][i] = eig["eig_A_mean"][i]
        eig["eig_B_max"][i] = eig["eig_A_max"][i]

    arrays["autocorr_score_val"] = np.asarray(scores, dtype=np.float32)
    arrays["autocorr_pred_mean"] = np.asarray(pred_corr, dtype=np.float32)
    arrays["autocorr_pred_std"] = np.full(shape_t, 0.02, dtype=np.float32)
    arrays["energy_pred_mean"] = np.asarray(energy_pred, dtype=np.float32)
    arrays["energy_pred_std"] = np.full(shape_k, 1e-6, dtype=np.float32)
    arrays["qq_pred"] = np.asarray(qq_pred, dtype=np.float32)
    arrays["rollout_gt"] = np.asarray(rollout_gt, dtype=np.float32)
    arrays["rollout_pred"] = np.asarray(rollout_pred, dtype=np.float32)
    arrays["rollout_abs_error"] = np.abs(arrays["rollout_pred"] - arrays["rollout_gt"]).astype(np.float32)
    arrays["rollout_gif_t"] = np.arange(1, n_frames + 1, dtype=np.int32)
    arrays["rollout_gif_gt"] = np.asarray(rollout_gif_gt, dtype=np.float32)
    arrays["rollout_gif_pred"] = np.asarray(rollout_gif_pred, dtype=np.float32)
    arrays["summary_gt_rollout"] = np.asarray(summary_gt, dtype=np.float32)
    arrays["summary_pred_rollout"] = np.asarray(summary_pred, dtype=np.float32)
    arrays["train_raw_loss"] = train_loss
    arrays["val_weighted_loss"] = val_loss
    arrays.update(loss_terms)
    arrays.update(eig)
    arrays["train_std"] = np.full(n, 4.5722, dtype=np.float32)
    arrays["checkpoint_path"] = np.asarray(["" for _ in range(n)])
    arrays["train_log_path"] = np.asarray(["" for _ in range(n)])
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "smoke": True,
        "stage": "Alpha",
        "diagnostic_split": "validation",
        "loss_recipe": LOSS_RECIPE,
        "note": "Synthetic data mirrors the producer schema for plotter development.",
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    return arrays


def concat_combo_results(paths: list[Path]) -> dict[str, Any]:
    loaded = [np.load(path, allow_pickle=False) for path in paths]
    try:
        keys = loaded[0].files
        out: dict[str, Any] = {}
        per_combo_keys = {
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
        for key in keys:
            values = [data[key] for data in loaded]
            out[key] = np.concatenate(values, axis=0) if key in per_combo_keys else values[0]
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "smoke": False,
            "stage": "Alpha",
            "merged_files": [str(path) for path in paths],
            "diagnostic_split": "validation",
        }
        out["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
        return out
    finally:
        for data in loaded:
            data.close()


def result_path_for_combo(combo_id: str) -> Path:
    return RESULTS_DIR / combo_id / f"alpha_capacity_{combo_id}.npz"


def main() -> None:
    args = parse_args()
    combos = load_combos(args.combo_file)
    if args.smoke:
        smoke_output = args.output or SMOKE_OUTPUT
        save_npz(smoke_output, **make_smoke_data(combos, args))
        run_plotter(args, smoke_output, HERE / "figures" / "alpha_smoke")
        return

    selected = select_combo_indices(combos, args)
    written: list[Path] = []
    for idx in selected:
        row = combo_row(combos, int(idx))
        out_path = result_path_for_combo(row["combo_id"])
        if out_path.exists() and args.skip_existing and not args.overwrite:
            print(f"skip existing: {out_path}", flush=True)
            written.append(out_path)
            continue
        t0 = time.perf_counter()
        combo_id = row["combo_id"]
        if args.reuse_checkpoint:
            try:
                ckpt_path = latest_checkpoint(CHECKPOINT_DIR / combo_id)
                log_path = latest_log(LOG_DIR / combo_id)
                print(f"reuse checkpoint: {ckpt_path}", flush=True)
            except FileNotFoundError:
                ckpt_path, log_path = run_training(args, row)
        else:
            ckpt_path, log_path = run_training(args, row)
        arrays = make_real_combo_data(args, row, ckpt_path, log_path)
        save_npz(out_path, **arrays)
        written.append(out_path)
        print(f"{row['combo_id']} done in {(time.perf_counter() - t0) / 3600:.2f}h", flush=True)

    if args.chunk_id is not None and written:
        chunk_out = RESULTS_DIR / f"alpha_capacity_chunk{args.chunk_id:02d}.npz"
        save_npz(chunk_out, **concat_combo_results(written))
        run_plotter(args, chunk_out)
    elif written:
        run_plotter(args)


if __name__ == "__main__":
    main()
