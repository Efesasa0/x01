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
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
THESIS_PLOTS = HERE.parent
ROOT = Path(__file__).resolve().parents[3]
X01_DIR = ROOT
DATA_PATH = ROOT.parent / "kmflow_re1000_r256_b400_f320_clean.npy"
RUN_JOINT = X01_DIR / "run_joint.py"
X01_SRC = X01_DIR / "src"
AR_CKPT_ROOT = THESIS_PLOTS / "08_ar_sequential_inference" / "checkpoints"
SR_CKPT_ROOT = THESIS_PLOTS / "09_sr_reconstruction" / "checkpoints"
FULL_OUTPUT = HERE / "joint_pipeline_data.npz"
SMOKE_OUTPUT = HERE / "joint_pipeline_smoke.npz"
STAGE_C_FULL_OUTPUT = HERE / "joint_pipeline_stage_c_full.npz"
REFERENCE_PLOT_ROOT = HERE / "plots" / "reference_run_joint"
LOG_ROOT = HERE / "logs"

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
    parser.add_argument("--smoke", action="store_true", help="write deterministic synthetic data without checkpoints")
    parser.add_argument(
        "--reference-runner",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="call x01/run_joint.py before local NPZ capture",
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--stage-c-full-output", type=Path, default=STAGE_C_FULL_OUTPUT)
    parser.add_argument("--ar-checkpoint", type=Path, default=None)
    parser.add_argument("--sr-checkpoint", type=Path, default=None)
    parser.add_argument("--ar-root", type=Path, default=AR_CKPT_ROOT)
    parser.add_argument("--sr-root", type=Path, default=SR_CKPT_ROOT)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--test-size", type=int, default=30)
    parser.add_argument("--viz-offset", type=int, default=-1)
    parser.add_argument("--t-in", type=int, default=0)
    parser.add_argument("--data-frames", type=int, default=320)
    parser.add_argument("--stage-a-steps", type=int, default=32)
    parser.add_argument("--stage-c-steps", type=int, default=1024)
    parser.add_argument("--stage-c-snapshots", type=int, nargs="+", default=[0, 200, 400, 600, 800, 1000])
    parser.add_argument("--sr-batch-size", type=int, default=32)
    parser.add_argument("--inference-mode", choices=("sequential", "koopman"), default="sequential")
    parser.add_argument("--run-tag", default="thesis_joint10")
    parser.add_argument("--reference-plot-root", type=Path, default=REFERENCE_PLOT_ROOT)
    parser.add_argument("--log-root", type=Path, default=LOG_ROOT)
    parser.add_argument("--stage-b-seed", type=int, default=123)
    parser.add_argument("--stage-c-seed", type=int, default=456)
    parser.add_argument("--qq-quantiles", type=int, default=1000)
    parser.add_argument("--max-qq-values", type=int, default=2_000_000)
    parser.add_argument("--save-stage-c-full", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output
    return SMOKE_OUTPUT if args.smoke else FULL_OUTPUT


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(path, flush=True)


def checkpoint_epoch(path: Path) -> int:
    match = re.search(r"epoch(\d+)", path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint(path: Path, pattern: str) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"checkpoint root does not exist: {path}")
    candidates = list(path.rglob(pattern))
    if not candidates:
        raise FileNotFoundError(f"no checkpoints matching {pattern!r} under {path}")
    return max(candidates, key=lambda p: (checkpoint_epoch(p), p.stat().st_mtime_ns))


def resolve_ar_checkpoint(args: argparse.Namespace) -> Path:
    return latest_checkpoint(args.ar_checkpoint or args.ar_root, "*_ar_epoch*.pt")


def resolve_sr_checkpoint(args: argparse.Namespace) -> Path:
    return latest_checkpoint(args.sr_checkpoint or args.sr_root, "*_sr_epoch*.pt")


def run_joint_command(args: argparse.Namespace, ar_ckpt: Path, sr_ckpt: Path) -> list[str]:
    return [
        sys.executable,
        str(RUN_JOINT),
        f"data.path={args.data.resolve()}",
        f"data.test_size={args.test_size}",
        f"checkpoints.ar_path={ar_ckpt.resolve()}",
        f"checkpoints.sr_path={sr_ckpt.resolve()}",
        f"inference_mode={args.inference_mode}",
        f"pipeline.part1_frames={args.stage_a_steps}",
        f"pipeline.part2_frames={args.data_frames}",
        f"pipeline.part3_frames={args.stage_c_steps}",
        f"pipeline.sr_batch_size={args.sr_batch_size}",
        "viz.enabled=true",
        "save.enabled=true",
        f"save.plot_dir={args.reference_plot_root.resolve()}",
        f"save.run_tag={args.run_tag}",
    ]


def run_reference_runner(args: argparse.Namespace, ar_ckpt: Path, sr_ckpt: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {"enabled": bool(args.reference_runner), "plot_dir": "", "log_path": ""}
    if not args.reference_runner:
        metadata["status"] = "skipped"
        return metadata
    if not RUN_JOINT.exists():
        raise FileNotFoundError(RUN_JOINT)

    args.log_root.mkdir(parents=True, exist_ok=True)
    log_path = args.log_root / f"reference_run_joint_{datetime.now().strftime('%Y%m%dT%H%M%S')}.out"
    cmd = run_joint_command(args, ar_ckpt, sr_ckpt)
    metadata["command"] = cmd
    metadata["log_path"] = str(log_path)
    print("\n=== Reference runner: x01/run_joint.py ===", flush=True)
    print(" ".join(cmd), flush=True)

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env["PYTHONPATH"] = str(X01_SRC) + os.pathsep + env.get("PYTHONPATH", "")
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

    metadata["return_code"] = return_code
    if return_code != 0:
        metadata["status"] = "failed"
        raise subprocess.CalledProcessError(return_code, cmd)

    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"save\s+: plots -> (.+)", text)
    if match:
        metadata["plot_dir"] = match.group(1).strip()
    metadata["status"] = "completed"
    return metadata


def collect_reference_artifacts(plot_dir: str) -> dict[str, list[str]]:
    if not plot_dir:
        return {}
    root = Path(plot_dir)
    if not root.exists():
        return {}
    grouped: dict[str, list[str]] = {"part1": [], "part2": [], "part3": [], "other": []}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        key = "other"
        for candidate in ("part1", "part2", "part3"):
            if candidate in path.name:
                key = candidate
                break
        grouped[key].append(str(path))
    return grouped


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


def load_ar(ckpt_path: Path, device: torch.device):
    sys.path.insert(0, str(X01_DIR / "src"))
    from x01.ar import KoopmanAE2D

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = KoopmanAE2D(
        in_channels=1,
        out_channels=1,
        dims=int(cfg["dims"]),
        num_blocks=tuple(cfg["num_blocks"]),
        num_heads=tuple(cfg["num_heads"]),
        dynamics_mode=str(cfg.get("dynamics_mode", "separate")),
        dynamics_rank=cfg.get("dynamics_rank"),
        steps=int(cfg.get("pixel_rollout_steps", 1) or 1),
    ).to(device)
    model.load_state_dict(strip_state_dict_prefixes(ckpt["model"]))
    model.eval()
    return model, ckpt, bool(cfg.get("standardize", False)), float(ckpt["train_std"])


def load_sr(ckpt_path: Path, device: torch.device):
    sys.path.insert(0, str(X01_DIR / "src"))
    from x01.sr import DiffusionManager, UNet

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = UNet(
        in_channels=int(cfg["in_channels"]),
        out_channels=int(cfg["out_channels"]),
        latent_dims=int(cfg["latent_dims"]),
        channel_multipliers=tuple(cfg["channel_multipliers"]),
        num_res_blocks=int(cfg["num_res_blocks"]),
        attention_resolutions=tuple(cfg["attention_resolutions"]),
        image_resolution=int(cfg["image_resolution"]),
        dropout_rate=float(cfg["dropout_rate"]),
        resample_with_conv=bool(cfg["resample_with_conv"]),
    ).to(device)
    state_key = "ema" if "ema" in ckpt else "model"
    model.load_state_dict(strip_state_dict_prefixes(ckpt[state_key]))
    model.eval()

    manager = DiffusionManager(
        beta_start=float(cfg["beta_start"]),
        beta_end=float(cfg["beta_end"]),
        num_diffusion_time_steps=int(cfg["num_diffusion_time_steps"]),
        condition_dropout_rate=float(cfg.get("condition_dropout_rate", 0.0)),
    ).to(device)
    if "manager" in ckpt:
        manager.load_state_dict(ckpt["manager"])
    manager.eval()
    return model, manager, ckpt, state_key


def ar_rollout_raw(
    ar_model,
    x0_lr_raw: torch.Tensor,
    steps: int,
    ar_standardize: bool,
    ar_std: float,
    device: torch.device,
) -> torch.Tensor:
    x = x0_lr_raw.to(device)
    if ar_standardize:
        x = x / ar_std
    with torch.no_grad():
        pred = ar_model.rollout(x, steps, inference_mode="sequential")
    if ar_standardize:
        pred = pred * ar_std
    return pred.detach().cpu()


@torch.no_grad()
def sr_refine_raw(
    sr_model,
    manager,
    lr_frames_raw: torch.Tensor,
    upsample_factor: int,
    inference_steps: int,
    batch_size: int,
    sr_standardize: bool,
    sr_std: float,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    cond_chunks = []
    sample_chunks = []
    for start in range(0, lr_frames_raw.shape[0], batch_size):
        lr_chunk = lr_frames_raw[start : start + batch_size].to(device)
        cond_raw = F.interpolate(lr_chunk, scale_factor=upsample_factor, mode="nearest")
        cond = cond_raw / sr_std if sr_standardize else cond_raw
        x_noise = torch.randn_like(cond)
        sample = manager.infer_fn(x_noise, sr_model, inference_steps=inference_steps, condition=cond, pbar=False)
        if sr_standardize:
            sample = sample * sr_std
        cond_chunks.append(cond_raw.detach().cpu())
        sample_chunks.append(sample.detach().cpu())
    return torch.cat(cond_chunks, dim=0), torch.cat(sample_chunks, dim=0)


def temporal_autocorr(frames: np.ndarray) -> np.ndarray:
    frames = frames.astype(np.float64, copy=False)
    n_time = frames.shape[0]
    flat = frames.reshape(n_time, -1)
    corr = np.empty(n_time, dtype=np.float64)
    for tau in range(n_time):
        n = n_time - tau
        corr[tau] = (flat[:n] * flat[tau:]).sum(axis=1).mean()
    if abs(corr[0]) > 1e-12:
        corr /= corr[0]
    return corr.astype(np.float32)


def energy_spectrum(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _n, height, width = frames.shape
    kx = np.fft.fftfreq(width) * width
    ky = np.fft.fftfreq(height) * height
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_rad = np.round(np.sqrt(kx_grid**2 + ky_grid**2)).astype(int)
    power = np.abs(np.fft.fft2(frames)) ** 2 / (height * width) ** 2
    k_max = min(height, width) // 2
    k_bins = np.arange(1, k_max + 1)
    spectrum = np.array([power[:, k_rad == k].mean() for k in k_bins], dtype=np.float32)
    return k_bins.astype(np.int32), spectrum


def sample_for_qq(
    gt: np.ndarray,
    pred: np.ndarray,
    rng: np.random.Generator,
    n_values: int,
) -> tuple[np.ndarray, np.ndarray]:
    gt_flat = gt.reshape(-1)
    pred_flat = pred.reshape(-1)
    n = min(n_values, gt_flat.size, pred_flat.size)
    idx = rng.integers(0, min(gt_flat.size, pred_flat.size), size=n)
    return gt_flat[idx].astype(np.float32), pred_flat[idx].astype(np.float32)


def trajectory_indices(full: np.ndarray, test_size: int) -> list[int]:
    if test_size <= 0 or test_size > full.shape[0]:
        raise ValueError(f"test_size={test_size} is outside [1, {full.shape[0]}]")
    start = full.shape[0] - test_size
    return list(range(start, full.shape[0]))


def choose_viz_index(indices: list[int], viz_offset: int) -> int:
    if viz_offset < 0:
        offset = len(indices) + viz_offset
    else:
        offset = viz_offset
    if offset < 0 or offset >= len(indices):
        raise ValueError(f"viz_offset={viz_offset} is outside the selected {len(indices)} test trajectories")
    return indices[offset]


def load_hr_trajectory(full: np.ndarray, idx: int, t_in: int, n_frames: int) -> np.ndarray:
    if t_in + n_frames > full.shape[1]:
        raise ValueError(f"requested t_in+n_frames={t_in + n_frames}, but data has {full.shape[1]} frames")
    return np.asarray(full[idx, t_in : t_in + n_frames], dtype=np.float32)


def make_lr_seed(hr_traj: np.ndarray, factor: int) -> torch.Tensor:
    lr0 = hr_traj[0, ::factor, ::factor]
    return torch.from_numpy(lr0).unsqueeze(0).unsqueeze(0)


def compute_stage_a(
    full: np.ndarray,
    viz_idx: int,
    args: argparse.Namespace,
    factor: int,
    ar_model,
    ar_standardize: bool,
    ar_std: float,
    device: torch.device,
) -> dict[str, np.ndarray]:
    n_frames = int(args.data_frames)
    if n_frames < 1:
        raise ValueError("data_frames must be at least 1")
    n_steps = n_frames - 1
    hr_traj = load_hr_trajectory(full, viz_idx, args.t_in, n_frames)
    x0_lr = make_lr_seed(hr_traj, factor)
    lr_pred = ar_rollout_raw(ar_model, x0_lr, n_steps, ar_standardize, ar_std, device)
    lr_gt = hr_traj[:, ::factor, ::factor].astype(np.float32)
    lr_pred_np = np.concatenate(
        [lr_gt[:1], lr_pred[:, 0].numpy().astype(np.float32)],
        axis=0,
    )
    return {
        "stage_a_t": np.arange(n_frames, dtype=np.int32),
        "stage_a_lr_gt": lr_gt.astype(np.float32),
        "stage_a_lr_pred": lr_pred_np,
        "stage_a_lr_abs_error": np.abs(lr_pred_np - lr_gt).astype(np.float32),
        "stage_a_mse_per_frame": np.square(lr_pred_np - lr_gt).mean(axis=(1, 2)).astype(np.float32),
    }


def compute_stage_b(
    full: np.ndarray,
    test_idxs: list[int],
    args: argparse.Namespace,
    factor: int,
    ar_model,
    sr_model,
    manager,
    ar_standardize: bool,
    ar_std: float,
    sr_standardize: bool,
    sr_std: float,
    inference_steps: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    n_frames = int(args.data_frames)
    n_steps = n_frames - 1
    if n_frames < 2:
        raise ValueError("data_frames must be at least 2")

    autocorr_gt, autocorr_pred = [], []
    energy_gt, energy_pred, energy_condition = [], [], []
    mse_per_traj = []
    qq_gt_chunks, qq_pred_chunks = [], []
    rng = np.random.default_rng(args.stage_b_seed)
    qq_values_per_traj = max(1, args.max_qq_values // max(1, len(test_idxs)))
    k_bins = None

    for offset, idx in enumerate(test_idxs):
        t0 = time.perf_counter()
        hr_traj = load_hr_trajectory(full, idx, args.t_in, args.data_frames)
        x0_lr = make_lr_seed(hr_traj, factor)
        lr_pred = ar_rollout_raw(ar_model, x0_lr, n_steps, ar_standardize, ar_std, device)
        cond_hr, hr_pred = sr_refine_raw(
            sr_model,
            manager,
            lr_pred,
            upsample_factor=factor,
            inference_steps=inference_steps,
            batch_size=args.sr_batch_size,
            sr_standardize=sr_standardize,
            sr_std=sr_std,
            device=device,
            seed=args.stage_b_seed + offset,
        )

        gt_full = hr_traj[:n_frames].astype(np.float32)
        pred_future = hr_pred[:, 0].numpy().astype(np.float32)
        pred_np = np.concatenate([gt_full[:1], pred_future], axis=0).astype(np.float32)
        cond0 = np.repeat(
            np.repeat(x0_lr[0, 0].numpy(), factor, axis=0),
            factor,
            axis=1,
        ).astype(np.float32)
        cond_np = np.concatenate([cond0[None, ...], cond_hr[:, 0].numpy().astype(np.float32)], axis=0)
        mse_per_traj.append(np.square(pred_np - gt_full).mean(axis=(1, 2)).astype(np.float32))

        autocorr_gt.append(temporal_autocorr(gt_full))
        autocorr_pred.append(temporal_autocorr(pred_np))

        k_bins, gt_e = energy_spectrum(gt_full)
        _, pred_e = energy_spectrum(pred_np)
        _, cond_e = energy_spectrum(cond_np)
        energy_gt.append(gt_e)
        energy_pred.append(pred_e)
        energy_condition.append(cond_e)

        qq_gt, qq_pred = sample_for_qq(gt_full, pred_np, rng, qq_values_per_traj)
        qq_gt_chunks.append(qq_gt)
        qq_pred_chunks.append(qq_pred)
        print(
            f"stage B traj {offset + 1:02d}/{len(test_idxs)} idx={idx} "
            f"mse={mse_per_traj[-1].mean():.6f} t={time.perf_counter() - t0:.1f}s",
            flush=True,
        )

    autocorr_gt_arr = np.stack(autocorr_gt)
    autocorr_pred_arr = np.stack(autocorr_pred)
    energy_gt_arr = np.stack(energy_gt)
    energy_pred_arr = np.stack(energy_pred)
    energy_condition_arr = np.stack(energy_condition)
    mse_arr = np.stack(mse_per_traj)
    quantiles = np.linspace(0, 100, args.qq_quantiles, dtype=np.float32)
    qq_gt_all = np.concatenate(qq_gt_chunks)
    qq_pred_all = np.concatenate(qq_pred_chunks)

    return {
        "stage_b_t": np.arange(n_frames, dtype=np.int32),
        "stage_b_t_star": np.linspace(0, 1, n_frames, dtype=np.float32),
        "stage_b_autocorr_gt_mean": autocorr_gt_arr.mean(axis=0).astype(np.float32),
        "stage_b_autocorr_gt_std": autocorr_gt_arr.std(axis=0).astype(np.float32),
        "stage_b_autocorr_pred_mean": autocorr_pred_arr.mean(axis=0).astype(np.float32),
        "stage_b_autocorr_pred_std": autocorr_pred_arr.std(axis=0).astype(np.float32),
        "stage_b_k_bins": k_bins.astype(np.int32),
        "stage_b_energy_gt_mean": energy_gt_arr.mean(axis=0).astype(np.float32),
        "stage_b_energy_gt_std": energy_gt_arr.std(axis=0).astype(np.float32),
        "stage_b_energy_pred_mean": energy_pred_arr.mean(axis=0).astype(np.float32),
        "stage_b_energy_pred_std": energy_pred_arr.std(axis=0).astype(np.float32),
        "stage_b_energy_condition_mean": energy_condition_arr.mean(axis=0).astype(np.float32),
        "stage_b_energy_condition_std": energy_condition_arr.std(axis=0).astype(np.float32),
        "stage_b_qq_quantiles": quantiles,
        "stage_b_qq_gt": np.percentile(qq_gt_all, quantiles).astype(np.float32),
        "stage_b_qq_pred": np.percentile(qq_pred_all, quantiles).astype(np.float32),
        "stage_b_mse_mean": mse_arr.mean(axis=0).astype(np.float32),
        "stage_b_mse_std": mse_arr.std(axis=0).astype(np.float32),
        "stage_b_qq_sample_count": np.asarray(qq_gt_all.size, dtype=np.int32),
    }


def compute_stage_c(
    full: np.ndarray,
    viz_idx: int,
    args: argparse.Namespace,
    factor: int,
    ar_model,
    sr_model,
    manager,
    ar_standardize: bool,
    ar_std: float,
    sr_standardize: bool,
    sr_std: float,
    inference_steps: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    n_data_frames = int(args.data_frames)
    n_total_frames = int(args.stage_c_steps)
    if n_data_frames < 2:
        raise ValueError("data_frames must be at least 2")
    if n_total_frames <= n_data_frames:
        raise ValueError(f"stage_c_steps ({n_total_frames}) must exceed data_frames ({n_data_frames})")

    hr_data_traj = load_hr_trajectory(full, viz_idx, args.t_in, n_data_frames)
    x0_lr = make_lr_seed(hr_data_traj, factor)
    lr_data_future = ar_rollout_raw(ar_model, x0_lr, n_data_frames - 1, ar_standardize, ar_std, device)
    _cond_data_hr, hr_data_future = sr_refine_raw(
        sr_model,
        manager,
        lr_data_future,
        upsample_factor=factor,
        inference_steps=inference_steps,
        batch_size=args.sr_batch_size,
        sr_standardize=sr_standardize,
        sr_std=sr_std,
        device=device,
        seed=args.stage_c_seed,
    )
    lr_data_np = np.concatenate(
        [x0_lr[:, 0].numpy().astype(np.float32), lr_data_future[:, 0].numpy().astype(np.float32)],
        axis=0,
    )
    hr_data_np = np.concatenate(
        [hr_data_traj[:1].astype(np.float32), hr_data_future[:, 0].numpy().astype(np.float32)],
        axis=0,
    )

    x0_lr_long = torch.from_numpy(lr_data_np[-1:]).unsqueeze(1)
    extension_steps = n_total_frames - n_data_frames
    print(
        f"Stage C seed: test trajectory {viz_idx}, timestep {n_data_frames - 1}; "
        f"extending {extension_steps} frames",
        flush=True,
    )
    lr_extension = ar_rollout_raw(ar_model, x0_lr_long, extension_steps, ar_standardize, ar_std, device)
    _cond_ext_hr, hr_extension = sr_refine_raw(
        sr_model,
        manager,
        lr_extension,
        upsample_factor=factor,
        inference_steps=inference_steps,
        batch_size=args.sr_batch_size,
        sr_standardize=sr_standardize,
        sr_std=sr_std,
        device=device,
        seed=args.stage_c_seed + 1,
    )
    snapshot_t = np.asarray(args.stage_c_snapshots, dtype=np.int32)
    if np.any(snapshot_t < 0) or np.any(snapshot_t >= n_total_frames):
        raise ValueError(f"stage-c snapshots must be within [0, {n_total_frames})")

    hr_np = np.concatenate([hr_data_np, hr_extension[:, 0].numpy().astype(np.float32)], axis=0)
    lr_np = np.concatenate([lr_data_np, lr_extension[:, 0].numpy().astype(np.float32)], axis=0)
    main = {
        "stage_c_t": np.arange(n_total_frames, dtype=np.int32),
        "stage_c_snapshot_t": snapshot_t,
        "stage_c_snapshot_hr_pred": hr_np[snapshot_t].astype(np.float32),
        "stage_c_snapshot_lr_pred": lr_np[snapshot_t].astype(np.float32),
        "stage_c_seed_t": np.asarray(n_data_frames - 1, dtype=np.int32),
        "stage_c_full_output": np.asarray(str(args.stage_c_full_output)),
    }
    full_out = {
        "stage_c_t": np.arange(n_total_frames, dtype=np.int32),
        "stage_c_lr_pred": lr_np,
        "stage_c_hr_pred": hr_np,
    }
    return main, full_out


def git_value(args: list[str]) -> str:
    try:
        return (
            __import__("subprocess")
            .check_output(["git", "-C", str(X01_DIR), *args], stderr=__import__("subprocess").DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


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


def make_smoke_data() -> dict[str, Any]:
    rng = np.random.default_rng(10)
    lr_h = lr_w = 16
    hr_h = hr_w = 64
    data_frames = 320
    stage_c_steps = 1024
    snapshots = np.asarray([0, 200, 400, 600, 800, 1000], dtype=np.int32)

    y_lr = np.linspace(-1.0, 1.0, lr_h, dtype=np.float32)
    x_lr = np.linspace(-1.0, 1.0, lr_w, dtype=np.float32)
    xl, yl = np.meshgrid(x_lr, y_lr)
    y_hr = np.linspace(-1.0, 1.0, hr_h, dtype=np.float32)
    x_hr = np.linspace(-1.0, 1.0, hr_w, dtype=np.float32)
    xh, yh = np.meshgrid(x_hr, y_hr)

    lr_gt = []
    lr_pred = []
    hr_gt = []
    hr_pred = []
    for t in range(stage_c_steps):
        phase = 2 * np.pi * t / 60.0
        lr_frame = np.sin(2 * xl + phase) * np.cos(3 * yl - 0.3 * phase)
        hr_frame = np.sin(2 * xh + phase) * np.cos(3 * yh - 0.3 * phase)
        hr_frame += 0.15 * np.sin(15 * xh - 9 * yh + phase)
        lr_gt.append(lr_frame.astype(np.float32))
        lr_pred.append((lr_frame + 0.05 * (t / stage_c_steps) * np.sin(4 * xl)).astype(np.float32))
        hr_gt.append(hr_frame.astype(np.float32))
        hr_pred.append(
            (hr_frame + 0.04 * (t / stage_c_steps) * np.cos(5 * xh - 2 * yh) + rng.normal(0, 0.015, xh.shape)).astype(
                np.float32
            )
        )
    lr_gt = np.asarray(lr_gt, dtype=np.float32)
    lr_pred = np.asarray(lr_pred, dtype=np.float32)
    hr_gt = np.asarray(hr_gt, dtype=np.float32)
    hr_pred = np.asarray(hr_pred, dtype=np.float32)
    lr_pred[0] = lr_gt[0]
    hr_pred[0] = hr_gt[0]

    k_bins, e_gt = energy_spectrum(hr_gt[:data_frames])
    _, e_pred = energy_spectrum(hr_pred[:data_frames])
    _, e_cond = energy_spectrum(np.repeat(np.repeat(lr_pred[:data_frames], 4, axis=1), 4, axis=2))
    quantiles = np.linspace(0, 100, 1000, dtype=np.float32)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "smoke": True,
        "stage_b_alignment": "t0_to_319",
        "test_indices": list(range(283, 314)),
        "viz_index": 313,
        "stage_c_seed_t": data_frames - 1,
        "stage_c_full_saved": False,
    }

    return {
        "stage_a_t": np.arange(data_frames, dtype=np.int32),
        "stage_a_lr_gt": lr_gt[:data_frames],
        "stage_a_lr_pred": lr_pred[:data_frames],
        "stage_a_lr_abs_error": np.abs(lr_pred[:data_frames] - lr_gt[:data_frames]).astype(np.float32),
        "stage_a_mse_per_frame": np.square(lr_pred[:data_frames] - lr_gt[:data_frames])
        .mean(axis=(1, 2))
        .astype(np.float32),
        "stage_b_t": np.arange(data_frames, dtype=np.int32),
        "stage_b_t_star": np.linspace(0, 1, data_frames, dtype=np.float32),
        "stage_b_autocorr_gt_mean": temporal_autocorr(hr_gt[:data_frames]),
        "stage_b_autocorr_gt_std": np.zeros(data_frames, dtype=np.float32),
        "stage_b_autocorr_pred_mean": temporal_autocorr(hr_pred[:data_frames]),
        "stage_b_autocorr_pred_std": np.zeros(data_frames, dtype=np.float32),
        "stage_b_k_bins": k_bins,
        "stage_b_energy_gt_mean": e_gt,
        "stage_b_energy_gt_std": np.zeros_like(e_gt),
        "stage_b_energy_pred_mean": e_pred,
        "stage_b_energy_pred_std": np.zeros_like(e_pred),
        "stage_b_energy_condition_mean": e_cond,
        "stage_b_energy_condition_std": np.zeros_like(e_cond),
        "stage_b_qq_quantiles": quantiles,
        "stage_b_qq_gt": np.percentile(hr_gt[:data_frames].reshape(-1), quantiles).astype(np.float32),
        "stage_b_qq_pred": np.percentile(hr_pred[:data_frames].reshape(-1), quantiles).astype(np.float32),
        "stage_b_mse_mean": np.square(hr_pred[:data_frames] - hr_gt[:data_frames])
        .mean(axis=(1, 2))
        .astype(np.float32),
        "stage_b_mse_std": np.zeros(data_frames, dtype=np.float32),
        "stage_b_qq_sample_count": np.asarray(hr_gt[:data_frames].size, dtype=np.int32),
        "stage_c_t": np.arange(stage_c_steps, dtype=np.int32),
        "stage_c_snapshot_t": snapshots,
        "stage_c_snapshot_hr_pred": hr_pred[snapshots],
        "stage_c_snapshot_lr_pred": lr_pred[snapshots],
        "stage_c_seed_t": np.asarray(data_frames - 1, dtype=np.int32),
        "stage_c_full_output": np.asarray(""),
        "ar_checkpoint_path": np.asarray(""),
        "sr_checkpoint_path": np.asarray(""),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }


def make_real_data(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    device = torch.device(args.device)
    ar_ckpt_path = resolve_ar_checkpoint(args)
    sr_ckpt_path = resolve_sr_checkpoint(args)
    print(f"AR checkpoint: {ar_ckpt_path}", flush=True)
    print(f"SR checkpoint: {sr_ckpt_path}", flush=True)
    reference_runner = run_reference_runner(args, ar_ckpt_path, sr_ckpt_path)

    ar_model, ar_ckpt, ar_standardize, ar_std = load_ar(ar_ckpt_path, device)
    sr_model, manager, sr_ckpt, sr_state_key = load_sr(sr_ckpt_path, device)
    sr_cfg = sr_ckpt["config"]
    factor = int(sr_cfg["condition_downsample_factor"])
    inference_steps = int(sr_cfg["inference_steps"])
    sr_standardize = bool(sr_cfg.get("standardize", False))
    sr_std = float(sr_ckpt["train_std"])

    print(f"AR standardize={ar_standardize} train_std={ar_std:.4f}", flush=True)
    print(f"SR standardize={sr_standardize} train_std={sr_std:.4f}", flush=True)
    print(f"factor={factor} inference_steps={inference_steps}", flush=True)

    full = np.load(args.data, mmap_mode="r")
    test_idxs = trajectory_indices(full, args.test_size)
    viz_idx = choose_viz_index(test_idxs, args.viz_offset)
    print(f"test trajectories: {test_idxs[0]}..{test_idxs[-1]} n={len(test_idxs)}", flush=True)
    print(f"visualized trajectory: {viz_idx}", flush=True)

    t0 = time.perf_counter()
    print("\n=== Stage A: LR debug rollout ===", flush=True)
    stage_a = compute_stage_a(full, viz_idx, args, factor, ar_model, ar_standardize, ar_std, device)
    print(f"Stage A done in {time.perf_counter() - t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    print("\n=== Stage B: full-data-length HR diagnostics ===", flush=True)
    stage_b = compute_stage_b(
        full,
        test_idxs,
        args,
        factor,
        ar_model,
        sr_model,
        manager,
        ar_standardize,
        ar_std,
        sr_standardize,
        sr_std,
        inference_steps,
        device,
    )
    print(f"Stage B done in {time.perf_counter() - t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    print("\n=== Stage C: single-trajectory long horizon ===", flush=True)
    stage_c, stage_c_full = compute_stage_c(
        full,
        viz_idx,
        args,
        factor,
        ar_model,
        sr_model,
        manager,
        ar_standardize,
        ar_std,
        sr_standardize,
        sr_std,
        inference_steps,
        device,
    )
    print(f"Stage C done in {time.perf_counter() - t0:.1f}s", flush=True)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "smoke": False,
        "source": "current x01 codebase",
        "source_runner": str(RUN_JOINT),
        "reference_runner": reference_runner,
        "reference_runner_artifacts": collect_reference_artifacts(str(reference_runner.get("plot_dir", ""))),
        "git_head": git_value(["rev-parse", "HEAD"]),
        "git_branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": git_value(["status", "--short"]),
        "data": data_stamp(args.data),
        "ar_checkpoint_path": str(ar_ckpt_path),
        "sr_checkpoint_path": str(sr_ckpt_path),
        "ar_checkpoint_epoch": int(ar_ckpt.get("epoch", -1)),
        "sr_checkpoint_epoch": int(sr_ckpt.get("epoch", -1)),
        "ar_standardize": ar_standardize,
        "ar_train_std": ar_std,
        "sr_standardize": sr_standardize,
        "sr_train_std": sr_std,
        "sr_state_dict_used": sr_state_key,
        "sr_condition_downsample_factor": factor,
        "sr_inference_steps": inference_steps,
        "test_indices": test_idxs,
        "viz_index": viz_idx,
        "stage_a_steps": int(len(stage_a["stage_a_t"])),
        "stage_b_steps": int(len(stage_b["stage_b_t"])),
        "stage_c_steps": int(len(stage_c["stage_c_t"])),
        "stage_b_alignment": "t0_to_319",
        "stage_c_seed_t": int(stage_c["stage_c_seed_t"]),
        "stage_mapping": {
            "stage_a": "full data-length low-resolution AR strip over t=0..data_frames-1 for the selected test trajectory",
            "stage_b": "full data-length high-resolution AR-SR diagnostics over t=0..data_frames-1 for all selected test trajectories",
            "stage_c": "single-trajectory long-horizon snapshots seeded from the t=data_frames-1 output, matching run_joint part3",
        },
        "stage_b_qq_uses_subsample": args.max_qq_values > 0,
        "stage_c_full_saved": bool(args.save_stage_c_full),
        "stage_c_full_output": str(args.stage_c_full_output) if args.save_stage_c_full else "",
        "historical_reference": {
            "run": "2026-07-17T18-18-26__main__26",
            "batch": "x01-writeup/results/newplots-10/batches/026.out",
            "note": "historical result used old non-standardized AR epoch 50 and SR epoch 100",
        },
    }
    main = {
        **stage_a,
        **stage_b,
        **stage_c,
        "ar_checkpoint_path": np.asarray(str(ar_ckpt_path)),
        "sr_checkpoint_path": np.asarray(str(sr_ckpt_path)),
        "ar_train_std": np.asarray(ar_std, dtype=np.float32),
        "sr_train_std": np.asarray(sr_std, dtype=np.float32),
        "reference_runner_plot_dir": np.asarray(str(reference_runner.get("plot_dir", ""))),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    stage_c_full["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    return main, stage_c_full


def main() -> None:
    args = parse_args()
    out = output_path(args)
    if args.smoke:
        save_npz(out, **make_smoke_data())
        return

    arrays, stage_c_full = make_real_data(args)
    save_npz(out, **arrays)
    if args.save_stage_c_full:
        save_npz(args.stage_c_full_output, **stage_c_full)


if __name__ == "__main__":
    main()
