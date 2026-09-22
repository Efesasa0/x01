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
ROOT = Path(__file__).resolve().parents[3]
X01_DIR = ROOT
DATA_PATH = ROOT.parent / "kmflow_re1000_r256_b400_f320_clean.npy"
FULL_OUTPUT = HERE / "sr_reconstruction_data.npz"
SMOKE_OUTPUT = HERE / "sr_reconstruction_smoke.npz"

TMP_ROOT = Path(os.environ.get("TMPDIR", "/tmp"))
MPLCONFIGDIR = TMP_ROOT / "x01_mplconfig"
XDG_CACHE_HOME = TMP_ROOT / "x01_cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))
os.environ.setdefault("WANDB_MODE", "disabled")

SR_RECIPE = {
    "model": "conditional DDIM/SR",
    "loss": "DDPM noise prediction",
    "sampling_model": "EMA UNet",
    "epochs": 100,
    "batch_size": 32,
    "lr": 1.0e-4,
    "scheduler": "none",
    "ema_decay": 0.9999,
    "condition_downsample_factor": 4,
    "inference_steps": 40,
    "n_train": 254,
    "n_val": 30,
    "n_test": 30,
    "standardize": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="write deterministic synthetic data without training")
    parser.add_argument("--skip-train", action="store_true", help="post-process the newest local checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=None, help="checkpoint file or run directory to post-process")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--run-tag", default="sr09_reconstruction")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--ckpt-root", type=Path, default=HERE / "checkpoints")
    parser.add_argument("--plot-root", type=Path, default=HERE / "plots")
    parser.add_argument("--log-root", type=Path, default=HERE / "logs")
    parser.add_argument("--ckpt-every", type=int, default=10)
    parser.add_argument("--plot-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--energy-seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=254)
    parser.add_argument("--n-val", type=int, default=30)
    parser.add_argument("--n-test", type=int, default=30)
    parser.add_argument("--t-in", type=int, default=0)
    parser.add_argument("--t", type=int, default=320)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--n-energy-frames", type=int, default=64)
    parser.add_argument("--sample-batch-size", type=int, default=8)
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--viz-during-train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="additional Hydra override passed to train_sr.py; may be repeated",
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


def synthesize_condition(x_hr: torch.Tensor, factor: int) -> torch.Tensor:
    x_lr = x_hr[:, :, ::factor, ::factor]
    return F.interpolate(x_lr, scale_factor=factor, mode="nearest")


def energy_spectrum(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _t, height, width = frames.shape
    kx = np.fft.fftfreq(width) * width
    ky = np.fft.fftfreq(height) * height
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_rad = np.round(np.sqrt(kx_grid**2 + ky_grid**2)).astype(int)
    power = np.abs(np.fft.fft2(frames)) ** 2 / (height * width) ** 2
    k_max = min(height, width) // 2
    k_bins = np.arange(1, k_max + 1)
    spectrum = np.array([power[:, k_rad == k].mean() for k in k_bins], dtype=np.float32)
    return k_bins.astype(np.int32), spectrum


def make_smoke_data() -> dict[str, Any]:
    rng = np.random.default_rng(9)
    n_samples = 5
    height = width = 256
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x, y)
    refs = []
    for idx in range(n_samples):
        phase = 2 * np.pi * idx / n_samples
        base = np.sin(3 * x_grid + phase) * np.cos(2 * y_grid - 0.3 * phase)
        detail = 0.18 * np.sin(19 * x_grid - 7 * y_grid + phase)
        refs.append((4.0 * (base + detail)).astype(np.float32))
    refs = np.asarray(refs, dtype=np.float32)

    refs_t = torch.from_numpy(refs).unsqueeze(1)
    cond = synthesize_condition(refs_t, 4).squeeze(1).numpy().astype(np.float32)
    samples = (0.82 * refs + 0.18 * cond + rng.normal(0.0, 0.08, size=refs.shape)).astype(np.float32)

    energy_frames = []
    for frame in range(64):
        phase = 2 * np.pi * frame / 64.0
        base = np.sin(2 * x_grid + phase) * np.cos(4 * y_grid - 0.5 * phase)
        detail = 0.15 * np.cos(15 * x_grid + 11 * y_grid + phase)
        energy_frames.append((base + detail).astype(np.float32))
    energy_ref = np.asarray(energy_frames, dtype=np.float32)
    energy_cond = synthesize_condition(torch.from_numpy(energy_ref).unsqueeze(1), 4).squeeze(1).numpy()
    energy_pred = (0.84 * energy_ref + 0.16 * energy_cond).astype(np.float32)
    k_bins, e_ref = energy_spectrum(energy_ref)
    _, e_cond = energy_spectrum(energy_cond)
    _, e_pred = energy_spectrum(energy_pred)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "smoke": True,
        "recipe": SR_RECIPE,
        "energy_units": "synthetic physical units",
    }
    return {
        "sample_indices": np.arange(n_samples, dtype=np.int32),
        "sample_reference": refs,
        "sample_condition": cond,
        "sample_prediction": samples,
        "sample_abs_error": np.abs(samples - refs).astype(np.float32),
        "energy_probe_indices": np.arange(64, dtype=np.int32),
        "energy_k_bins": k_bins,
        "energy_reference": e_ref,
        "energy_condition": e_cond,
        "energy_prediction": e_pred,
        "train_std": np.asarray(4.5725, dtype=np.float32),
        "loss_epochs": np.arange(1, 101, dtype=np.int32),
        "train_loss": np.geomspace(0.0125, 0.0017, 100).astype(np.float32),
        "val_loss": np.geomspace(0.0050, 0.0015, 100).astype(np.float32),
        "test_loss": np.asarray(0.0018, dtype=np.float32),
        "checkpoint_path": np.asarray(""),
        "train_log_path": np.asarray(""),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }


def train_overrides(args: argparse.Namespace) -> list[str]:
    overrides = [
        f"seed={args.seed}",
        f"data.path={args.data.resolve()}",
        "data.downsample=1",
        f"data.n_train={args.n_train}",
        f"data.n_val={args.n_val}",
        f"data.n_test={args.n_test}",
        f"data.T_in={args.t_in}",
        f"data.T={args.t}",
        f"data.standardize={str(args.standardize).lower()}",
        f"training.epochs={args.epochs}",
        f"training.batch_size={args.batch_size}",
        f"training.num_workers={args.num_workers}",
        "training.lr=1.0e-4",
        "training.grad_clip=1.0",
        "training.scheduler=none",
        "training.ema_decay=0.9999",
        "training.smoke=false",
        "wandb.enabled=false",
        f"viz.enabled={str(args.viz_during_train).lower()}",
        "viz.every=10",
        "save.enabled=true",
        f"save.ckpt_every={args.ckpt_every}",
        f"save.plot_every={args.plot_every}",
        f"save.ckpt_dir={args.ckpt_root.resolve()}",
        f"save.plot_dir={args.plot_root.resolve()}",
        f"save.run_tag={args.run_tag}",
        "sr.latent_dims=64",
        "sr.channel_multipliers=[1,1,1,2]",
        "sr.num_res_blocks=1",
        "sr.attention_resolutions=[16]",
        "sr.image_resolution=256",
        "sr.condition_downsample_factor=4",
        "sr.beta_start=1.0e-4",
        "sr.beta_end=2.0e-2",
        "sr.num_diffusion_time_steps=1000",
        "sr.condition_dropout_rate=0.0",
        "sr.inference_steps=40",
    ]
    overrides.extend(args.extra_override)
    return overrides


def run_training(args: argparse.Namespace) -> tuple[Path, Path]:
    args.ckpt_root.mkdir(parents=True, exist_ok=True)
    args.plot_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)

    log_path = args.log_root / f"{args.run_tag}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.out"
    cmd = [sys.executable, str(X01_DIR / "train_sr.py"), *train_overrides(args)]
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
        raise RuntimeError(f"train_sr.py failed with exit code {return_code}; see {log_path}")

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
    checkpoints = sorted(path.glob("*_sr_epoch*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no SR checkpoints found in {path}")
    return checkpoints[-1]


def parse_loss_history(log_path: Path | None) -> dict[str, np.ndarray]:
    empty = {
        "loss_epochs": np.asarray([], dtype=np.int32),
        "train_loss": np.asarray([], dtype=np.float32),
        "val_loss": np.asarray([], dtype=np.float32),
        "test_loss": np.asarray(np.nan, dtype=np.float32),
    }
    if log_path is None or not log_path.exists():
        return empty

    epoch_re = re.compile(r"epoch\s+(\d+)/(\d+)\s+train=([0-9.eE+-]+)\s+val=([0-9.eE+-]+)")
    test_re = re.compile(r"test\s+loss=([0-9.eE+-]+)")
    epochs: list[int] = []
    train: list[float] = []
    val: list[float] = []
    test_loss = np.nan
    for line in log_path.read_text(errors="replace").splitlines():
        epoch_match = epoch_re.search(line)
        if epoch_match:
            epochs.append(int(epoch_match.group(1)))
            train.append(float(epoch_match.group(3)))
            val.append(float(epoch_match.group(4)))
        test_match = test_re.search(line)
        if test_match:
            test_loss = float(test_match.group(1))
    return {
        "loss_epochs": np.asarray(epochs, dtype=np.int32),
        "train_loss": np.asarray(train, dtype=np.float32),
        "val_loss": np.asarray(val, dtype=np.float32),
        "test_loss": np.asarray(test_loss, dtype=np.float32),
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


def load_sr_from_checkpoint(ckpt_path: Path, device: torch.device):
    sys.path.insert(0, str(X01_DIR / "src"))
    from x01.sr import DiffusionManager, UNet

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    model = UNet(
        in_channels=int(cfg.get("in_channels", 1)),
        out_channels=int(cfg.get("out_channels", 1)),
        latent_dims=int(cfg.get("latent_dims", 64)),
        channel_multipliers=tuple(cfg.get("channel_multipliers", [1, 1, 1, 2])),
        num_res_blocks=int(cfg.get("num_res_blocks", 1)),
        attention_resolutions=tuple(cfg.get("attention_resolutions", [16])),
        image_resolution=int(cfg.get("image_resolution", 256)),
        dropout_rate=float(cfg.get("dropout_rate", 0.0)),
        resample_with_conv=bool(cfg.get("resample_with_conv", True)),
    ).to(device)
    state_key = "ema" if "ema" in ckpt else "model"
    model.load_state_dict(strip_state_dict_prefixes(ckpt[state_key]))
    model.eval()

    manager = DiffusionManager(
        beta_start=float(cfg.get("beta_start", 1.0e-4)),
        beta_end=float(cfg.get("beta_end", 2.0e-2)),
        num_diffusion_time_steps=int(cfg.get("num_diffusion_time_steps", 1000)),
        condition_dropout_rate=float(cfg.get("condition_dropout_rate", 0.0)),
    ).to(device)
    if "manager" in ckpt:
        manager.load_state_dict(ckpt["manager"])
    manager.eval()
    return model, manager, ckpt, state_key


def flat_test_indices(n_total: int, n_select: int, seed: int) -> np.ndarray:
    n_select = min(n_select, n_total)
    rng = np.random.RandomState(seed)
    return np.asarray(sorted(rng.choice(n_total, n_select, replace=False).tolist()), dtype=np.int32)


def load_test_frames(
    data_path: Path,
    indices: np.ndarray,
    n_train: int,
    n_val: int,
    t_in: int,
    frames_per_traj: int,
    train_std: float,
    device: torch.device,
) -> torch.Tensor:
    full = np.load(data_path, mmap_mode="r")
    test_start = n_train + n_val
    frames = []
    for flat_idx in indices:
        traj_offset = int(flat_idx) // frames_per_traj
        frame_offset = int(flat_idx) % frames_per_traj
        arr = np.asarray(full[test_start + traj_offset, t_in + frame_offset, ::1, ::1], dtype=np.float32)
        frames.append(arr / train_std)
    stacked = np.stack(frames).astype(np.float32)
    return torch.from_numpy(stacked).unsqueeze(1).to(device)


@torch.no_grad()
def sample_in_batches(
    refs: torch.Tensor,
    model: torch.nn.Module,
    manager: torch.nn.Module,
    condition_downsample_factor: int,
    inference_steps: int,
    batch_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cond_chunks = []
    sample_chunks = []
    torch.manual_seed(seed)
    if refs.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    for start in range(0, refs.shape[0], batch_size):
        ref_batch = refs[start : start + batch_size]
        cond = synthesize_condition(ref_batch, condition_downsample_factor)
        x_noise = torch.randn_like(cond)
        sample = manager.infer_fn(
            x_noise,
            model,
            inference_steps=inference_steps,
            condition=cond,
            pbar=False,
        )
        cond_chunks.append(cond.detach().cpu())
        sample_chunks.append(sample.detach().cpu())
    return torch.cat(cond_chunks, dim=0), torch.cat(sample_chunks, dim=0)


def make_diagnostic_data(args: argparse.Namespace, ckpt_path: Path, log_path: Path | None) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, manager, ckpt, state_key = load_sr_from_checkpoint(ckpt_path, device)
    train_std = float(ckpt["train_std"])
    cfg = ckpt.get("config", {})
    condition_factor = int(cfg.get("condition_downsample_factor", 4))
    inference_steps = int(cfg.get("inference_steps", 40))
    n_total_test_frames = args.n_test * args.t

    sample_indices = flat_test_indices(n_total_test_frames, args.n_samples, args.sample_seed)
    energy_indices = flat_test_indices(n_total_test_frames, args.n_energy_frames, args.energy_seed)
    sample_refs = load_test_frames(
        args.data, sample_indices, args.n_train, args.n_val, args.t_in, args.t, train_std, device
    )
    energy_refs = load_test_frames(
        args.data, energy_indices, args.n_train, args.n_val, args.t_in, args.t, train_std, device
    )

    sample_cond, sample_pred = sample_in_batches(
        sample_refs,
        model,
        manager,
        condition_factor,
        inference_steps,
        args.sample_batch_size,
        args.sample_seed,
    )
    energy_cond, energy_pred = sample_in_batches(
        energy_refs,
        model,
        manager,
        condition_factor,
        inference_steps,
        args.sample_batch_size,
        args.energy_seed,
    )

    sample_reference = (sample_refs[:, 0].detach().cpu().numpy() * train_std).astype(np.float32)
    sample_condition = (sample_cond[:, 0].numpy() * train_std).astype(np.float32)
    sample_prediction = (sample_pred[:, 0].numpy() * train_std).astype(np.float32)
    sample_abs_error = np.abs(sample_prediction - sample_reference).astype(np.float32)

    refs_norm = energy_refs[:, 0].detach().cpu().numpy().astype(np.float32)
    cond_norm = energy_cond[:, 0].numpy().astype(np.float32)
    pred_norm = energy_pred[:, 0].numpy().astype(np.float32)
    k_bins, energy_reference = energy_spectrum(refs_norm)
    _, energy_condition = energy_spectrum(cond_norm)
    _, energy_prediction = energy_spectrum(pred_norm)

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
        "checkpoint_config": cfg,
        "state_dict_used_for_sampling": state_key,
        "recipe": SR_RECIPE,
        "standardize": bool(args.standardize),
        "train_std": train_std,
        "sample_indices_seed": int(args.sample_seed),
        "energy_indices_seed": int(args.energy_seed),
        "energy_units": "standardized vorticity units, matching x01.sr.viz.log_energy_spectrum",
        "sample_units": "unstandardized vorticity units",
        "historical_reference": {
            "run": "2026-07-15T23-09-17__joint__24",
            "batch": "x01-writeup/results/newplots-10/batches/024.out",
            "train_std": 4.5725,
            "test_loss": 0.0018,
        },
    }

    return {
        "sample_indices": sample_indices,
        "sample_reference": sample_reference,
        "sample_condition": sample_condition,
        "sample_prediction": sample_prediction,
        "sample_abs_error": sample_abs_error,
        "energy_probe_indices": energy_indices,
        "energy_k_bins": k_bins,
        "energy_reference": energy_reference.astype(np.float32),
        "energy_condition": energy_condition.astype(np.float32),
        "energy_prediction": energy_prediction.astype(np.float32),
        "energy_reference_physical": (energy_reference * train_std**2).astype(np.float32),
        "energy_condition_physical": (energy_condition * train_std**2).astype(np.float32),
        "energy_prediction_physical": (energy_prediction * train_std**2).astype(np.float32),
        "train_std": np.asarray(train_std, dtype=np.float32),
        "checkpoint_path": np.asarray(str(ckpt_path)),
        "train_log_path": np.asarray(str(log_path) if log_path is not None else ""),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        **history,
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
