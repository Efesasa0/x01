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
ROOT = Path(__file__).resolve().parents[3]
X01_DIR = ROOT
DATA_PATH = ROOT.parent / "kmflow_re1000_r256_b400_f320_clean.npy"
FULL_OUTPUT = HERE / "ar_sequential_inference_data.npz"
SMOKE_OUTPUT = HERE / "ar_sequential_inference_smoke.npz"
DEFAULT_ROLLOUT_TIMES = (0, 25, 50, 100, 200, 319)

TMP_ROOT = Path(os.environ.get("TMPDIR", "/tmp"))
MPLCONFIGDIR = TMP_ROOT / "x01_mplconfig"
XDG_CACHE_HOME = TMP_ROOT / "x01_cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))
os.environ.setdefault("WANDB_MODE", "disabled")

LOSS_RECIPE = {
    "lambda_fwd": 1.0,
    "lambda_recon": 1.0,
    "lambda_bwd": 0.1,
    "lambda_consist": 0.1,
    "lambda_latent_fwd": 0.0,
    "lambda_latent_bwd": 0.0,
    "lambda_ortho_a": 0.0,
    "lambda_ortho_b": 0.0,
    "lambda_consist_progressive": 0.0,
    "loss_mode": "lp",
    "lp_size_average": True,
    "pixel_rollout_steps": 1,
    "latent_rollout_steps": 0,
    "dynamics_mode": "separate",
    "inference_mode": "sequential",
}


def parse_rollout_times(value: str) -> tuple[int, ...]:
    try:
        times = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rollout times must be comma-separated integers") from exc
    if not times:
        raise argparse.ArgumentTypeError("at least one rollout time is required")
    return times


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="write deterministic synthetic data without training")
    parser.add_argument("--skip-train", action="store_true", help="post-process an existing checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=None, help="checkpoint to post-process")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--run-tag", default="ar08_sequential_inference")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--ckpt-root", type=Path, default=HERE / "checkpoints")
    parser.add_argument("--plot-root", type=Path, default=HERE / "plots")
    parser.add_argument("--log-root", type=Path, default=HERE / "logs")
    parser.add_argument("--ckpt-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=252)
    parser.add_argument("--n-val", type=int, default=31)
    parser.add_argument("--n-test", type=int, default=31)
    parser.add_argument("--t-in", type=int, default=0)
    parser.add_argument("--t", type=int, default=320)
    parser.add_argument("--diagnostic-trajectories", type=int, default=31)
    parser.add_argument("--rollout-frames", type=int, default=320)
    parser.add_argument("--rollout-times", type=parse_rollout_times, default=DEFAULT_ROLLOUT_TIMES)
    parser.add_argument("--rollout-columns", type=int, default=None)
    parser.add_argument("--qq-quantiles", type=int, default=1000)
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-precomputed-std", action="store_true")
    parser.add_argument("--train-std", type=float, default=4.5722)
    parser.add_argument("--viz-during-train", action="store_true")
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="additional Hydra override passed to train_ar.py; may be repeated",
    )
    return parser.parse_args()


def output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output
    return SMOKE_OUTPUT if args.smoke else FULL_OUTPUT


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(path, flush=True)


def data_stamp(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def git_value(args: list[str]) -> str:
    try:
        return (
            subprocess.check_output(["git", "-C", str(X01_DIR), *args], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def make_smoke_data() -> dict[str, Any]:
    rng = np.random.default_rng(8)
    n_frames = 320
    h = w = 64
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    gt = []
    pred = []
    for t in range(n_frames):
        phase = 2 * np.pi * t / 80.0
        gt_t = np.sin(3 * X + phase) * np.cos(2 * Y - 0.4 * phase)
        drift = 0.08 * (t / n_frames) * np.sin(4 * X - Y)
        pred_t = gt_t + drift + rng.normal(0.0, 0.01, size=gt_t.shape)
        gt.append(gt_t.astype(np.float32))
        pred.append(pred_t.astype(np.float32))
    gt = np.asarray(gt, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    pred[0] = gt[0]

    diagnostics = diagnostic_arrays(gt[None, 1:, ...], pred[None, 1:, ...], 1000)
    rollout_t, rollout_gt, rollout_pred = rollout_display(gt, pred, DEFAULT_ROLLOUT_TIMES)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "smoke": True,
        "loss_recipe": LOSS_RECIPE,
        "diagnostic_alignment": "future_aligned",
        "rollout_times": list(DEFAULT_ROLLOUT_TIMES),
    }
    return {
        **diagnostics,
        "rollout_t": rollout_t,
        "rollout_gt": rollout_gt,
        "rollout_pred": rollout_pred,
        "rollout_abs_error": np.abs(rollout_pred - rollout_gt).astype(np.float32),
        "train_std": np.asarray(4.5722, dtype=np.float32),
        "checkpoint_path": np.asarray("", dtype="U1"),
        "train_log_path": np.asarray("", dtype="U1"),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }


def train_overrides(args: argparse.Namespace) -> list[str]:
    overrides = [
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
        "training.smoke=false",
        "wandb.enabled=false",
        f"viz.enabled={str(args.viz_during_train).lower()}",
        "save.enabled=true",
        f"save.ckpt_every={args.ckpt_every}",
        f"save.plot_every={args.ckpt_every}",
        f"save.ckpt_dir={args.ckpt_root.resolve()}",
        f"save.plot_dir={args.plot_root.resolve()}",
        f"save.run_tag={args.run_tag}",
        "inference_mode=sequential",
        "ar.dynamics_mode=separate",
        "ar.dynamics_rank=null",
        "ar.pixel_rollout_steps=1",
        "ar.latent_rollout_steps=0",
        "ar.lambda_fwd=1.0",
        "ar.lambda_recon=1.0",
        "ar.lambda_bwd=0.1",
        "ar.lambda_consist=0.1",
        "ar.lambda_latent_fwd=0.0",
        "ar.lambda_latent_bwd=0.0",
        "ar.lambda_ortho_a=0.0",
        "ar.lambda_ortho_b=0.0",
        "ar.lambda_consist_progressive=0.0",
        "ar.loss_mode=lp",
        "ar.lp_size_average=true",
    ]
    overrides.extend(args.extra_override)
    return overrides


def run_training(args: argparse.Namespace) -> tuple[Path, Path]:
    args.ckpt_root.mkdir(parents=True, exist_ok=True)
    args.plot_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)

    log_path = args.log_root / f"{args.run_tag}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.out"
    cmd = [sys.executable, str(X01_DIR / "train_ar.py"), *train_overrides(args)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(X01_DIR / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["WANDB_MODE"] = "disabled"
    env.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
    env.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))

    print("running:", " ".join(cmd), flush=True)
    print("log:", log_path, flush=True)
    with log_path.open("w") as log_file:
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
        raise RuntimeError(f"train_ar.py failed with exit code {return_code}; see {log_path}")

    ckpt_dir = parse_saved_dir(log_path, "ckpts")
    if ckpt_dir is None:
        ckpt_dir = newest_checkpoint_dir(args.ckpt_root)
    ckpt_path = latest_checkpoint(ckpt_dir)
    return ckpt_path, log_path


def parse_saved_dir(log_path: Path, label: str) -> Path | None:
    pattern = re.compile(rf"{re.escape(label)}\s*->\s*(.+)$")
    for line in log_path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            return Path(match.group(1).strip())
    return None


def newest_checkpoint_dir(root: Path) -> Path:
    candidates = [path for path in root.glob("*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no checkpoint run directories found in {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def latest_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    checkpoints = sorted(path.glob("*_ar_epoch*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no AR checkpoints found in {path}")
    return checkpoints[-1]


def parse_loss_history(log_path: Path | None) -> dict[str, np.ndarray]:
    if log_path is None or not log_path.exists():
        return {
            "loss_epochs": np.asarray([], dtype=np.int32),
            "train_raw_loss": np.asarray([], dtype=np.float32),
            "val_weighted_loss": np.asarray([], dtype=np.float32),
        }
    epoch_re = re.compile(
        r"epoch\s+(\d+)/(\d+)\s+train_raw=([0-9.eE+-]+)\s+\|\s+w_val=([0-9.eE+-]+)"
    )
    epochs: list[int] = []
    train_raw: list[float] = []
    val_weighted: list[float] = []
    for line in log_path.read_text(errors="replace").splitlines():
        match = epoch_re.search(line)
        if match:
            epochs.append(int(match.group(1)))
            train_raw.append(float(match.group(3)))
            val_weighted.append(float(match.group(4)))
    return {
        "loss_epochs": np.asarray(epochs, dtype=np.int32),
        "train_raw_loss": np.asarray(train_raw, dtype=np.float32),
        "val_weighted_loss": np.asarray(val_weighted, dtype=np.float32),
    }


def strip_state_dict_prefixes(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "_orig_mod."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device):
    sys.path.insert(0, str(X01_DIR / "src"))
    from x01.ar import KoopmanAE2D

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    steps = int(cfg.get("pixel_rollout_steps", 1) or 1)
    model = KoopmanAE2D(
        in_channels=1,
        out_channels=1,
        dims=int(cfg.get("dims", 16)),
        num_blocks=tuple(cfg.get("num_blocks", [4, 6, 6, 8])),
        num_heads=tuple(cfg.get("num_heads", [1, 2, 4, 8])),
        dynamics_mode=str(cfg.get("dynamics_mode", "separate")),
        dynamics_rank=cfg.get("dynamics_rank", None),
        steps=steps,
    ).to(device)
    model.load_state_dict(strip_state_dict_prefixes(ckpt["model"]))
    model.eval()
    return model, ckpt


def load_trajectory(
    full: np.ndarray,
    idx: int,
    t_in: int,
    frames: int,
    train_std: float,
    device: torch.device,
) -> torch.Tensor:
    arr = np.asarray(full[idx, t_in : t_in + frames, ::4, ::4], dtype=np.float32)
    arr = arr / train_std
    return torch.from_numpy(arr).unsqueeze(1).to(device)


def rollout_future(model, gt_frames: torch.Tensor) -> torch.Tensor:
    n_pred = gt_frames.shape[0] - 1
    with torch.no_grad():
        return model.rollout(gt_frames[0:1], n_pred, inference_mode="sequential")


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

    return {
        "t_star": np.linspace(0, 1, gt_r_arr.shape[1], dtype=np.float32),
        "autocorr_gt_mean": gt_r_arr.mean(axis=0).astype(np.float32),
        "autocorr_gt_std": gt_r_arr.std(axis=0).astype(np.float32),
        "autocorr_pred_mean": pred_r_arr.mean(axis=0).astype(np.float32),
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


def rollout_display(
    gt: np.ndarray,
    pred: np.ndarray,
    rollout_times: tuple[int, ...],
    fallback_columns: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_frames = min(gt.shape[0], pred.shape[0])
    if rollout_times:
        timesteps = np.asarray(rollout_times, dtype=np.int32)
    elif fallback_columns is not None:
        timesteps = np.linspace(0, n_frames - 1, fallback_columns, dtype=np.int32)
    else:
        raise ValueError("rollout_times must be set unless fallback_columns is provided")

    if np.any(timesteps < 0) or np.any(timesteps >= n_frames):
        raise ValueError(f"rollout times {timesteps.tolist()} exceed available frame range 0..{n_frames - 1}")
    return timesteps, gt[timesteps].astype(np.float32), pred[timesteps].astype(np.float32)


def make_diagnostic_data(args: argparse.Namespace, ckpt_path: Path, log_path: Path | None) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model_from_checkpoint(ckpt_path, device)
    train_std = float(ckpt["train_std"])
    frames = min(args.rollout_frames, args.t)
    if frames < 2:
        raise ValueError("rollout-frames must be at least 2")

    full = np.load(args.data, mmap_mode="r")
    test_start = args.n_train + args.n_val
    n_diag = min(args.diagnostic_trajectories, args.n_test)
    gt_list, pred_list = [], []
    rollout_gt = rollout_pred = rollout_t = None

    for offset in range(n_diag):
        gt_frames = load_trajectory(full, test_start + offset, args.t_in, frames, train_std, device)
        pred_frames = rollout_future(model, gt_frames)
        gt_rollout = (gt_frames[:, 0].detach().cpu().numpy() * train_std).astype(np.float32)
        gt_future = (gt_frames[1:, 0].detach().cpu().numpy() * train_std).astype(np.float32)
        pred_future = (pred_frames[:, 0].detach().cpu().numpy() * train_std).astype(np.float32)
        pred_rollout = np.concatenate([gt_rollout[:1], pred_future], axis=0).astype(np.float32)
        gt_list.append(gt_future)
        pred_list.append(pred_future)
        if offset == 0:
            rollout_t, rollout_gt, rollout_pred = rollout_display(
                gt_rollout,
                pred_rollout,
                args.rollout_times,
                args.rollout_columns,
            )

    gt_arr = np.stack(gt_list)
    pred_arr = np.stack(pred_list)
    diagnostics = diagnostic_arrays(gt_arr, pred_arr, args.qq_quantiles)
    history = parse_loss_history(log_path)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "smoke": False,
        "source": "current x01 codebase",
        "git_head": git_value(["rev-parse", "HEAD"]),
        "git_branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": git_value(["status", "--short"]),
        "data": data_stamp(args.data),
        "checkpoint_path": str(ckpt_path),
        "train_log_path": str(log_path) if log_path is not None else "",
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint_config": ckpt.get("config", {}),
        "loss_recipe": LOSS_RECIPE,
        "standardize": bool(args.standardize),
        "train_std": train_std,
        "diagnostic_split": "test",
        "diagnostic_trajectories": int(n_diag),
        "diagnostic_alignment": "future_aligned",
        "diagnostics_are_unstandardized": True,
        "rollout_times": list(args.rollout_times),
        "historical_notes": {
            "023": "non-standardized sequential reference, train std 1.0000, reached epoch 100",
            "027": "standardized main reference, train std 4.5722, cancelled around epoch 66",
        },
    }

    return {
        **diagnostics,
        **history,
        "rollout_t": rollout_t.astype(np.int32),
        "rollout_gt": rollout_gt.astype(np.float32),
        "rollout_pred": rollout_pred.astype(np.float32),
        "rollout_abs_error": np.abs(rollout_pred - rollout_gt).astype(np.float32),
        "train_std": np.asarray(train_std, dtype=np.float32),
        "checkpoint_path": np.asarray(str(ckpt_path)),
        "train_log_path": np.asarray(str(log_path) if log_path is not None else ""),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }


def main() -> None:
    args = parse_args()
    out = output_path(args)
    if args.smoke:
        save_npz(out, **make_smoke_data())
        return

    if args.checkpoint is not None:
        ckpt_path = latest_checkpoint(args.checkpoint)
        log_path = None
    elif args.skip_train:
        ckpt_path = latest_checkpoint(newest_checkpoint_dir(args.ckpt_root))
        log_path = None
    else:
        t0 = time.perf_counter()
        ckpt_path, log_path = run_training(args)
        print(f"training and checkpoint discovery finished in {time.perf_counter() - t0:.1f}s", flush=True)

    arrays = make_diagnostic_data(args, ckpt_path, log_path)
    save_npz(out, **arrays)


if __name__ == "__main__":
    main()
