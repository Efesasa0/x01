import argparse
import gc
import importlib
import inspect
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT.parent / "kmflow_re1000_r256_b400_f320_clean.npy"
SMOKE_OUTPUT = HERE / "ar_further_attempts_smoke.npz"
FULL_OUTPUT = HERE / "ar_further_attempts_data.npz"

TMP_ROOT = Path(os.environ.get("TMPDIR", "/tmp"))
MPLCONFIGDIR = TMP_ROOT / "x01_mplconfig"
XDG_CACHE_HOME = TMP_ROOT / "x01_cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))

NUM_BLOCKS = (4, 6, 6, 8)
INIT_SCALE = 1.0
LR = 1e-4
GRAD_CLIP = 1.0
BATCH_SIZE = 32
DIAGNOSTIC_TRAJECTORIES = None
QQ_QUANTILES = 1000


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    label: str
    source_dir: str
    source_branch: str
    source_commit: str
    command: str
    seed: int
    dims: int
    num_heads: tuple[int, int, int, int]
    epochs: int
    lambda_recon: float | None
    lambda_ortho: float
    lambda_bwd: float
    lambda_fwd: float | None
    lambda_latent_fwd: float | None
    expected_params: int | None
    source_markers: tuple[tuple[str, str], ...]


EXPERIMENTS = (
    ExperimentSpec(
        experiment_id="baseline_single_k",
        label="Strict single-$K$ baseline",
        source_dir="source_balmy_frog_265",
        source_branch="balmy-frog-265",
        source_commit="be9772ffbb263b471e943a674868274797d6b999",
        command="-m train_ar seed=42",
        seed=42,
        dims=16,
        num_heads=(1, 2, 4, 8),
        epochs=12,
        lambda_recon=1.0,
        lambda_ortho=0.1,
        lambda_bwd=1.0,
        lambda_fwd=None,
        lambda_latent_fwd=None,
        expected_params=None,
        source_markers=(
            ("src/x01/ar/blocks/dynamics.py", "return x @ self._forward.dynamics.T"),
            ("src/x01/ar/models/loss_koopman.py", "lambda_recon"),
            ("conf/ar/koopman_big.yaml", "dims: 16"),
        ),
    ),
    ExperimentSpec(
        experiment_id="latent_step_consistency",
        label="Latent-step consistency",
        source_dir="source_sparkling_lake_295",
        source_branch="sparkling-lake-295",
        source_commit="c520975d680f79b20e8feeaf1b5aeb34fe7778b2",
        command="-m train_ar wandb.tags=[latentfw] ar.lambda_latent_fwd=1.0",
        seed=42,
        dims=16,
        num_heads=(1, 2, 4, 8),
        epochs=12,
        lambda_recon=1.0,
        lambda_ortho=0.1,
        lambda_bwd=1.0,
        lambda_fwd=1.0,
        lambda_latent_fwd=1.0,
        expected_params=None,
        source_markers=(
            ("src/x01/ar/blocks/dynamics.py", "return x @ self._forward.dynamics.T"),
            ("src/x01/ar/models/loss_koopman.py", "latent_fwd_loss"),
            ("conf/ar/koopman_big.yaml", "lambda_latent_fwd"),
        ),
    ),
    ExperimentSpec(
        experiment_id="increased_capacity",
        label="Increased capacity",
        source_dir="source_golden_microwave_239",
        source_branch="golden-microwave-239",
        source_commit="f987d46c3e971508205b64db713c526061feed45",
        command="-m train_ar ar.dims=32 training.epochs=20",
        seed=42,
        dims=32,
        num_heads=(2, 4, 8, 16),
        epochs=20,
        lambda_recon=None,
        lambda_ortho=0.1,
        lambda_bwd=1.0,
        lambda_fwd=None,
        lambda_latent_fwd=None,
        expected_params=16_727_216,
        source_markers=(
            ("src/x01/ar/blocks/dynamics.py", "return x @ self._forward.dynamics.T"),
            ("src/x01/ar/models/loss_koopman.py", "loss = forward_loss + reconstruction_loss"),
            ("conf/ar/koopman_bigbig.yaml", "dims: 32"),
        ),
    ),
)

EXPERIMENT_BY_ID = {spec.experiment_id: spec for spec in EXPERIMENTS}
SOURCE_SRC_PATHS = {str(HERE / spec.source_dir / "src") for spec in EXPERIMENTS}
LOG_KEYS = (
    "loss",
    "forward_loss",
    "backward_loss",
    "reconstruction_loss",
    "ortho_loss",
    "latent_fwd_loss",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="write deterministic synthetic data for plot/layout work")
    parser.add_argument(
        "--experiment",
        choices=("all",) + tuple(EXPERIMENT_BY_ID),
        default="all",
        help="run one experiment row or all rows",
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-rollout-frames", type=int, default=320)
    parser.add_argument("--qq-quantiles", type=int, default=QQ_QUANTILES)
    parser.add_argument("--pick-n-trajs", type=int, default=None, help="debug cap for each split")
    parser.add_argument("--epoch-limit", type=int, default=None, help="debug cap applied to every selected run")
    parser.add_argument("--std-chunk-trajs", type=int, default=1)
    parser.add_argument("--individual-dir", type=Path, default=HERE / "producer_outputs")
    parser.add_argument("--no-individual", action="store_true", help="do not save per-row outputs")
    parser.add_argument("--no-deterministic", action="store_true")
    return parser.parse_args()


def selected_specs(experiment: str) -> list[ExperimentSpec]:
    if experiment == "all":
        return list(EXPERIMENTS)
    return [EXPERIMENT_BY_ID[experiment]]


def output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output
    return SMOKE_OUTPUT if args.smoke else FULL_OUTPUT


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(path, flush=True)


def save_figures(data_path: Path) -> None:
    from types import SimpleNamespace

    from plot_ar_further_attempts_data import default_output_path, plot_summary

    plot_args = SimpleNamespace(
        data=data_path,
        output=None,
        panel_width=2.15,
        panel_height=1.62,
        column_margin=0.62,
        row_margin=0.62,
        left_margin=1.78,
        right_margin=0.28,
        bottom_margin=0.58,
        top_margin=0.74,
        title_y=0.985,
        row_label_x=-0.34,
        line_width=1.05,
        font_size=7.2,
        title_font_size=8.2,
        label_font_size=8.0,
        legend_font_size=6.6,
        dpi=220,
        no_title=False,
    )
    plot_summary(data_path, default_output_path(data_path), plot_args)


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


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


def verify_source_snapshot(spec: ExperimentSpec) -> None:
    source = HERE / spec.source_dir
    if not source.exists():
        raise FileNotFoundError(f"missing source snapshot: {source}")
    for rel_path, marker in spec.source_markers:
        text = (source / rel_path).read_text()
        if marker not in text:
            raise RuntimeError(f"{spec.source_branch} marker {marker!r} missing from {rel_path}")


def reset_source_imports() -> None:
    sys.path[:] = [path for path in sys.path if path not in SOURCE_SRC_PATHS]
    for name in list(sys.modules):
        if name == "x01" or name.startswith("x01."):
            del sys.modules[name]


def load_source_modules(spec: ExperimentSpec):
    verify_source_snapshot(spec)
    reset_source_imports()
    source_src = str(HERE / spec.source_dir / "src")
    sys.path.insert(0, source_src)
    dataset_mod = importlib.import_module("x01.data.dataset")
    model_mod = importlib.import_module("x01.ar.models.koopman_ae_2d")
    loss_mod = importlib.import_module("x01.ar.models.loss_koopman")
    return dataset_mod.KMFlowDataset, model_mod.KoopmanAE2D, loss_mod.loss_koopman


def streaming_std(path: Path, dataset_cls, pick_n_trajs: int | None, chunk_trajs: int) -> float:
    data = np.load(path, mmap_mode="r")
    train_start, train_end = dataset_cls.split_bounds(data.shape[0], "train")
    if pick_n_trajs is not None:
        train_end = min(train_end, train_start + pick_n_trajs)
    chunk_trajs = max(1, chunk_trajs)

    total = 0
    sum_x = 0.0
    sum_x2 = 0.0
    for start in range(train_start, train_end, chunk_trajs):
        stop = min(train_end, start + chunk_trajs)
        chunk = np.asarray(data[start:stop], dtype=np.float32)
        total += chunk.size
        sum_x += float(chunk.sum(dtype=np.float64))
        sum_x2 += float(np.square(chunk, dtype=np.float64).sum(dtype=np.float64))

    mean = sum_x / total
    var = max(sum_x2 / total - mean * mean, 0.0)
    return float(np.sqrt(var))


def make_loader(
    dataset_cls,
    path: Path,
    split: str,
    train_std: float,
    batch_size: int,
    num_workers: int,
    pick_n_trajs: int | None,
    seed: int,
    device: torch.device,
) -> DataLoader:
    dataset = dataset_cls(
        str(path),
        split=split,
        mode="pairs",
        std=train_std,
        skip_n_frames=0,
        pick_n_trajs=pick_n_trajs,
    )
    generator = torch.Generator()
    generator.manual_seed(seed + {"train": 0, "val": 1, "test": 2}[split])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        generator=generator,
    )


def validation_trajectories(dataset, n_trajs: int | None, max_frames: int) -> list[torch.Tensor]:
    n = dataset.n_trajs if n_trajs is None else min(n_trajs, dataset.n_trajs)
    samples_per_traj = dataset.samples_per_traj
    frames_per_traj = min(max_frames, samples_per_traj + 1)
    pair_steps = frames_per_traj - 1
    trajectories = []
    for traj_idx in range(n):
        offset = traj_idx * samples_per_traj
        frames = [dataset[offset + t][0] for t in range(pair_steps)]
        frames.append(dataset[offset + pair_steps - 1][1])
        trajectories.append(torch.stack(frames))
    return trajectories


def build_model(model_cls, spec: ExperimentSpec, device: torch.device):
    return model_cls(
        in_channels=1,
        out_channels=1,
        dims=spec.dims,
        num_blocks=NUM_BLOCKS,
        num_heads=spec.num_heads,
        init_scale=INIT_SCALE,
    ).to(device)


def loss_call(loss_fn, model, x: torch.Tensor, y: torch.Tensor, spec: ExperimentSpec):
    candidates = {
        "lambda_recon": spec.lambda_recon,
        "lambda_ortho": spec.lambda_ortho,
        "lambda_bwd": spec.lambda_bwd,
        "lambda_fwd": spec.lambda_fwd,
        "lambda_latent_fwd": spec.lambda_latent_fwd,
    }
    params = inspect.signature(loss_fn).parameters
    kwargs = {key: value for key, value in candidates.items() if key in params and value is not None}
    return loss_fn(model, x, y, **kwargs)


def empty_totals() -> dict[str, float]:
    return {key: 0.0 for key in LOG_KEYS}


def add_log(totals: dict[str, float], log: dict[str, torch.Tensor]) -> None:
    for key in LOG_KEYS:
        if key in log:
            totals[key] += float(log[key].item())


def average_totals(totals: dict[str, float], log_counts: dict[str, int]) -> dict[str, float]:
    values = {}
    for key in LOG_KEYS:
        count = log_counts.get(key, 0)
        values[key] = totals[key] / count if count else np.nan
    return values


def update_counts(log_counts: dict[str, int], log: dict[str, torch.Tensor]) -> None:
    for key in LOG_KEYS:
        if key in log:
            log_counts[key] = log_counts.get(key, 0) + 1


def evaluate(model, loader: DataLoader, loss_fn, spec: ExperimentSpec, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = empty_totals()
    counts: dict[str, int] = {}
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            _, log = loss_call(loss_fn, model, x, y, spec)
            add_log(totals, log)
            update_counts(counts, log)
    return average_totals(totals, counts)


def forward_eigenvalue_stats(model) -> tuple[float, float, float]:
    K = model.dynamics.dynamics.detach().cpu().numpy()
    eig_abs = np.abs(np.linalg.eigvals(K))
    return float(eig_abs.min()), float(eig_abs.mean()), float(eig_abs.max())


def rollout(model, gt_frames: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    x0 = gt_frames[0:1].to(device)
    with torch.no_grad():
        z = model.encode(x0)
        pred = []
        for _ in range(gt_frames.shape[0]):
            pred.append(model.decode(z).cpu())
            z = model.dynamics(z)
    return torch.cat(pred, dim=0)


def temporal_autocorr(frames: np.ndarray) -> np.ndarray:
    frames = frames.astype(np.float64, copy=False)
    T = frames.shape[0]
    flat = frames.reshape(T, -1)
    R = np.empty(T, dtype=np.float64)
    for tau in range(T):
        n = T - tau
        R[tau] = (flat[:n] * flat[tau:]).sum(axis=1).mean()
    if abs(R[0]) > 1e-12:
        R /= R[0]
    return R.astype(np.float32)


def energy_spectrum(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T, H, W = frames.shape
    kx = np.fft.fftfreq(W) * W
    ky = np.fft.fftfreq(H) * H
    KX, KY = np.meshgrid(kx, ky)
    K_rad = np.round(np.sqrt(KX**2 + KY**2)).astype(int)
    power = np.abs(np.fft.fft2(frames)) ** 2 / (H * W) ** 2
    k_max = min(H, W) // 2
    k_bins = np.arange(1, k_max + 1)
    E = np.array([power[:, K_rad == k].mean() for k in k_bins], dtype=np.float32)
    return k_bins.astype(np.int32), E


def validation_diagnostics(
    model,
    gt_trajectories: list[torch.Tensor],
    device: torch.device,
    n_quantiles: int,
) -> dict[str, np.ndarray]:
    gt_R_list, pred_R_list = [], []
    gt_E_list, pred_E_list = [], []
    gt_vals, pred_vals = [], []
    k_bins = None

    for gt_frames in gt_trajectories:
        pred_frames = rollout(model, gt_frames, device)
        gt_np = gt_frames[:, 0].numpy().astype(np.float32)
        pred_np = pred_frames[:, 0].numpy().astype(np.float32)

        gt_R_list.append(temporal_autocorr(gt_np))
        pred_R_list.append(temporal_autocorr(pred_np))

        k_bins, gt_E = energy_spectrum(gt_np)
        _, pred_E = energy_spectrum(pred_np)
        gt_E_list.append(gt_E)
        pred_E_list.append(pred_E)

        gt_vals.append(gt_np.ravel())
        pred_vals.append(pred_np.ravel())

    gt_R = np.stack(gt_R_list)
    pred_R = np.stack(pred_R_list)
    gt_E = np.stack(gt_E_list)
    pred_E = np.stack(pred_E_list)
    quantiles = np.linspace(0, 100, n_quantiles, dtype=np.float32)
    gt_q = np.percentile(np.concatenate(gt_vals), quantiles).astype(np.float32)
    pred_q = np.percentile(np.concatenate(pred_vals), quantiles).astype(np.float32)

    return {
        "t_star": np.linspace(0, 1, gt_R.shape[1], dtype=np.float32),
        "autocorr_gt_mean": gt_R.mean(axis=0).astype(np.float32),
        "autocorr_gt_std": gt_R.std(axis=0).astype(np.float32),
        "autocorr_pred_mean": pred_R.mean(axis=0).astype(np.float32),
        "autocorr_pred_std": pred_R.std(axis=0).astype(np.float32),
        "k_bins": k_bins.astype(np.int32),
        "energy_gt_mean": gt_E.mean(axis=0).astype(np.float32),
        "energy_gt_std": gt_E.std(axis=0).astype(np.float32),
        "energy_pred_mean": pred_E.mean(axis=0).astype(np.float32),
        "energy_pred_std": pred_E.std(axis=0).astype(np.float32),
        "qq_quantiles": quantiles,
        "qq_gt": gt_q,
        "qq_pred": pred_q,
    }


def train_experiment(spec: ExperimentSpec, args: argparse.Namespace) -> dict[str, Any]:
    print(f"\n=== {spec.experiment_id}: {spec.source_branch} ===", flush=True)
    KMFlowDataset, KoopmanAE2D, loss_koopman = load_source_modules(spec)
    deterministic = not args.no_deterministic
    set_seed(spec.seed, deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = min(spec.epochs, args.epoch_limit) if args.epoch_limit else spec.epochs
    train_std = streaming_std(args.data, KMFlowDataset, args.pick_n_trajs, args.std_chunk_trajs)
    train_loader = make_loader(
        KMFlowDataset,
        args.data,
        "train",
        train_std,
        args.batch_size,
        args.num_workers,
        args.pick_n_trajs,
        spec.seed,
        device,
    )
    val_loader = make_loader(
        KMFlowDataset,
        args.data,
        "val",
        train_std,
        args.batch_size,
        args.num_workers,
        args.pick_n_trajs,
        spec.seed,
        device,
    )
    val_trajectories = validation_trajectories(
        val_loader.dataset,
        DIAGNOSTIC_TRAJECTORIES,
        args.max_rollout_frames,
    )

    model = build_model(KoopmanAE2D, spec, device)
    param_count = sum(param.numel() for param in model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict[str, Any]] = []
    eig_epochs = [0]
    eig_min, eig_mean, eig_max = [], [], []
    eig_values = forward_eigenvalue_stats(model)
    eig_min.append(eig_values[0])
    eig_mean.append(eig_values[1])
    eig_max.append(eig_values[2])

    print(f"source commit: {spec.source_commit}", flush=True)
    print(f"command      : {spec.command}", flush=True)
    print(f"device       : {device}", flush=True)
    print(f"seed         : {spec.seed}", flush=True)
    print(f"epochs       : {epochs}", flush=True)
    print(f"parameters   : {param_count:,}", flush=True)
    print(f"train std    : {train_std:.6f}", flush=True)
    print(f"train steps  : {len(train_loader)}", flush=True)
    print(f"val steps    : {len(val_loader)}", flush=True)

    t_run = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        totals = empty_totals()
        counts: dict[str, int] = {}

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss, log = loss_call(loss_koopman, model, x, y, spec)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            add_log(totals, log)
            update_counts(counts, log)

        train_values = average_totals(totals, counts)
        val_values = evaluate(model, val_loader, loss_koopman, spec, device)
        eig_values = forward_eigenvalue_stats(model)
        eig_epochs.append(epoch)
        eig_min.append(eig_values[0])
        eig_mean.append(eig_values[1])
        eig_max.append(eig_values[2])
        elapsed = time.perf_counter() - t0

        history.append(
            {
                "epoch": epoch,
                "train": train_values,
                "val": val_values,
                "elapsed": elapsed,
            }
        )
        print(
            f"epoch {epoch:03d}/{epochs} "
            f"train={train_values['loss']:.6f} "
            f"val={val_values['loss']:.6f} "
            f"Kmax={eig_values[2]:.6f} "
            f"t={elapsed:.1f}s",
            flush=True,
        )

    diagnostics = validation_diagnostics(model, val_trajectories, device, args.qq_quantiles)
    elapsed_total = time.perf_counter() - t_run

    result: dict[str, Any] = {
        "spec": spec,
        "final_epoch": epochs,
        "train_std": train_std,
        "param_count": param_count,
        "elapsed_seconds": elapsed_total,
        "loss_epochs": np.asarray([row["epoch"] for row in history], dtype=np.int32),
        "eig_epochs": np.asarray(eig_epochs, dtype=np.int32),
        "eig_forward_min": np.asarray(eig_min, dtype=np.float32),
        "eig_forward_mean": np.asarray(eig_mean, dtype=np.float32),
        "eig_forward_max": np.asarray(eig_max, dtype=np.float32),
    }
    for prefix in ("train", "val"):
        for key in LOG_KEYS:
            short_key = key.replace("_loss", "").replace("reconstruction", "recon")
            result[f"{prefix}_{short_key}"] = np.asarray(
                [row[prefix][key] for row in history],
                dtype=np.float32,
            )
    result.update(diagnostics)

    del model, optimizer, train_loader, val_loader, val_trajectories
    reset_source_imports()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def pad_1d(results: list[dict[str, Any]], key: str, dtype=np.float32, fill=np.nan) -> np.ndarray:
    max_len = max(len(result[key]) for result in results)
    out = np.full((len(results), max_len), fill, dtype=dtype)
    for idx, result in enumerate(results):
        values = np.asarray(result[key], dtype=dtype)
        out[idx, : len(values)] = values
    return out


def stack_1d(results: list[dict[str, Any]], key: str, dtype=np.float32) -> np.ndarray:
    return np.stack([np.asarray(result[key], dtype=dtype) for result in results])


def pack_results(results: list[dict[str, Any]], args: argparse.Namespace, smoke: bool) -> dict[str, Any]:
    specs = [result["spec"] for result in results]
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "smoke": smoke,
        "args": vars(args),
        "data": data_stamp(args.data),
        "experiments": [asdict(spec) for spec in specs],
    }

    arrays: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(metadata, default=str)),
        "experiment_ids": np.asarray([spec.experiment_id for spec in specs]),
        "experiment_labels": np.asarray([spec.label for spec in specs]),
        "source_dirs": np.asarray([spec.source_dir for spec in specs]),
        "source_branches": np.asarray([spec.source_branch for spec in specs]),
        "source_commits": np.asarray([spec.source_commit for spec in specs]),
        "commands": np.asarray([spec.command for spec in specs]),
        "final_epochs": np.asarray([result["final_epoch"] for result in results], dtype=np.int32),
        "seeds": np.asarray([spec.seed for spec in specs], dtype=np.int32),
        "dims": np.asarray([spec.dims for spec in specs], dtype=np.int32),
        "param_count": np.asarray([result["param_count"] for result in results], dtype=np.int64),
        "expected_param_count": np.asarray(
            [spec.expected_params if spec.expected_params is not None else -1 for spec in specs],
            dtype=np.int64,
        ),
        "train_std": np.asarray([result["train_std"] for result in results], dtype=np.float32),
        "elapsed_seconds": np.asarray([result["elapsed_seconds"] for result in results], dtype=np.float32),
        "loss_epochs": pad_1d(results, "loss_epochs", dtype=np.float32),
        "eig_epochs": pad_1d(results, "eig_epochs", dtype=np.float32),
        "eig_forward_min": pad_1d(results, "eig_forward_min"),
        "eig_forward_mean": pad_1d(results, "eig_forward_mean"),
        "eig_forward_max": pad_1d(results, "eig_forward_max"),
    }

    for prefix in ("train", "val"):
        for key in ("loss", "forward", "backward", "recon", "ortho", "latent_fwd"):
            arrays[f"{prefix}_{key}"] = pad_1d(results, f"{prefix}_{key}")

    for key in (
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
    ):
        arrays[key] = stack_1d(results, key)

    arrays["t_star"] = np.asarray(results[0]["t_star"], dtype=np.float32)
    arrays["k_bins"] = np.asarray(results[0]["k_bins"], dtype=np.int32)
    arrays["qq_quantiles"] = np.asarray(results[0]["qq_quantiles"], dtype=np.float32)
    return arrays


def make_smoke_result(spec: ExperimentSpec, row: int) -> dict[str, Any]:
    rng = np.random.default_rng(700 + row)
    final_epoch = spec.epochs
    loss_epochs = np.arange(1, final_epoch + 1, dtype=np.int32)
    eig_epochs = np.arange(0, final_epoch + 1, dtype=np.int32)
    t_star = np.linspace(0, 1, 160, dtype=np.float32)
    k_bins = np.arange(1, 33, dtype=np.int32)
    quantiles = np.linspace(0, 100, QQ_QUANTILES, dtype=np.float32)
    q_base = np.linspace(-2.6, 2.6, QQ_QUANTILES, dtype=np.float32)

    quality = [0.70, 0.76, 0.90][row]
    train_loss = (1.2 * np.exp(-loss_epochs / (5.5 + row)) + 0.10 + 0.02 * row).astype(np.float32)
    val_loss = (1.25 * np.exp(-loss_epochs / (6.3 + row)) + 0.16 - 0.02 * row).astype(np.float32)
    val_loss += rng.normal(0, 0.01, size=val_loss.shape).astype(np.float32)
    eig_progress = np.linspace(0, 1, len(eig_epochs), dtype=np.float32)
    gt_R = np.cos(2.5 * np.pi * t_star) * np.exp(-1.35 * t_star)
    pred_R = quality * gt_R + (1 - quality) * np.exp(-4.0 * t_star)
    gt_E = 0.025 * k_bins.astype(np.float32) ** -2.4
    pred_E = gt_E * (quality + (1 - quality) * np.exp(-k_bins / 10))

    result: dict[str, Any] = {
        "spec": spec,
        "final_epoch": final_epoch,
        "train_std": 1.0,
        "param_count": spec.expected_params or (7_500_000 + row * 400_000),
        "elapsed_seconds": 0.0,
        "loss_epochs": loss_epochs,
        "eig_epochs": eig_epochs,
        "eig_forward_min": (0.97 + 0.015 * eig_progress - 0.004 * row).astype(np.float32),
        "eig_forward_mean": (0.995 + 0.003 * eig_progress).astype(np.float32),
        "eig_forward_max": (1.035 - 0.020 * eig_progress + 0.004 * row).astype(np.float32),
        "t_star": t_star,
        "autocorr_gt_mean": gt_R.astype(np.float32),
        "autocorr_gt_std": (0.035 + 0.01 * np.sin(np.pi * t_star) ** 2).astype(np.float32),
        "autocorr_pred_mean": pred_R.astype(np.float32),
        "autocorr_pred_std": (0.045 + 0.015 * np.sin(np.pi * t_star) ** 2).astype(np.float32),
        "k_bins": k_bins,
        "energy_gt_mean": gt_E.astype(np.float32),
        "energy_gt_std": (0.10 * gt_E).astype(np.float32),
        "energy_pred_mean": pred_E.astype(np.float32),
        "energy_pred_std": (0.12 * pred_E).astype(np.float32),
        "qq_quantiles": quantiles,
        "qq_gt": q_base,
        "qq_pred": (quality * q_base + (1 - quality) * np.tanh(q_base)).astype(np.float32),
    }

    for prefix, total in (("train", train_loss), ("val", val_loss)):
        result[f"{prefix}_loss"] = total
        result[f"{prefix}_forward"] = (0.32 * total).astype(np.float32)
        result[f"{prefix}_backward"] = (0.22 * total).astype(np.float32)
        result[f"{prefix}_recon"] = (0.36 * total).astype(np.float32)
        result[f"{prefix}_ortho"] = (0.10 * total).astype(np.float32)
        if spec.lambda_latent_fwd is None:
            result[f"{prefix}_latent_fwd"] = np.full_like(total, np.nan, dtype=np.float32)
        else:
            result[f"{prefix}_latent_fwd"] = (0.18 * total).astype(np.float32)
    return result


def individual_output_path(args: argparse.Namespace, spec: ExperimentSpec, smoke: bool) -> Path:
    suffix = "_smoke" if smoke else ""
    return args.individual_dir / f"ar_further_attempts_{spec.experiment_id}{suffix}.npz"


def main() -> None:
    args = parse_args()
    specs = selected_specs(args.experiment)

    if args.smoke:
        results = [make_smoke_result(spec, idx) for idx, spec in enumerate(specs)]
    else:
        if not args.data.exists():
            raise FileNotFoundError(f"data file not found: {args.data}")
        results = []
        for spec in specs:
            result = train_experiment(spec, args)
            results.append(result)
            if not args.no_individual:
                save_npz(individual_output_path(args, spec, smoke=False), **pack_results([result], args, smoke=False))

    if args.smoke and not args.no_individual:
        for result in results:
            save_npz(individual_output_path(args, result["spec"], smoke=True), **pack_results([result], args, smoke=True))

    out = output_path(args)
    save_npz(out, **pack_results(results, args, smoke=args.smoke))
    save_figures(out)


if __name__ == "__main__":
    main()
