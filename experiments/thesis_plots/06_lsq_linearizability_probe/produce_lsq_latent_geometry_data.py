import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
X01_ROOT = HERE.parents[2]
PROJECT_ROOT = HERE.parents[3]
SMOKE_OUT = HERE / "lsq_latent_geometry_smoke.npz"
FULL_OUT = HERE / "lsq_latent_geometry_data.npz"

os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".matplotlib_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(X01_ROOT / "src"))

from x01.ar.models.koopman_ae_2d import KoopmanAE2D


CASES = [
    {
        "label": "N01",
        "N": 1,
        "data": PROJECT_ROOT / "kmflow_10unique.npy",
        "checkpoint": X01_ROOT / "checkpoints" / "ae_only_km10unique_n1_epoch0300.pt",
    },
    {
        "label": "N02",
        "N": 2,
        "data": PROJECT_ROOT / "kmflow_10unique.npy",
        "checkpoint": X01_ROOT / "checkpoints" / "ae_only_km10unique_n2_epoch0300.pt",
    },
    {
        "label": "N08",
        "N": 8,
        "data": PROJECT_ROOT / "kmflow_10unique.npy",
        "checkpoint": X01_ROOT / "checkpoints" / "ae_only_km10unique_trainN8_epoch0300.pt",
    },
    {
        "label": "N40",
        "N": 40,
        "data": PROJECT_ROOT / "kmflow_50unique.npy",
        "checkpoint": X01_ROOT / "checkpoints" / "ae_only_km50unique_n40_epoch0300.pt",
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--T", type=int, default=320)
    parser.add_argument("--max-pca-points", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dims", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, nargs="+", default=[4, 6, 6, 8])
    parser.add_argument("--num-heads", type=int, nargs="+", default=[1, 2, 4, 8])
    return parser.parse_args()


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(checkpoint, args, device):
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = KoopmanAE2D(
        in_channels=1,
        out_channels=1,
        dims=args.dims,
        num_blocks=tuple(args.num_blocks),
        num_heads=tuple(args.num_heads),
    )
    state = {key.replace("_orig_mod.", ""): value for key, value in ckpt["model"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"{checkpoint.name}: load_state_dict missing={missing} unexpected={unexpected}", flush=True)
    model.to(device)
    model.eval()
    return model, float(ckpt["train_std"]), int(ckpt.get("epoch", -1))


def train_split_stop(n_total):
    return int(0.8 * n_total)


def load_frames(case, train_std, args):
    data = np.load(case["data"], mmap_mode="r")
    n_train = train_split_stop(data.shape[0])
    if case["N"] > n_train:
        raise ValueError(f"{case['label']} requests {case['N']} trajectories, but train split has {n_train}")
    if args.T > data.shape[1]:
        raise ValueError(f"--T={args.T} exceeds available trajectory length {data.shape[1]}")

    frames = np.asarray(
        data[: case["N"], : args.T, :: args.downsample, :: args.downsample],
        dtype=np.float32,
    )
    frames = frames / train_std
    n_traj, n_time, height, width = frames.shape
    frames = frames.reshape(n_traj * n_time, 1, height, width)
    return torch.from_numpy(frames), n_traj, n_time


@torch.no_grad()
def encode_frames(model, frames, device, batch_size, label=""):
    encoded = []
    total = frames.shape[0]
    n_batches = (total + batch_size - 1) // batch_size
    desc = f"{label} encode" if label else "encode"
    for start in tqdm(range(0, total, batch_size), total=n_batches, desc=desc, leave=False):
        batch = frames[start : start + batch_size].to(device)
        encoded.append(model.encode(batch, mode="identity").detach().cpu().numpy())
    return np.concatenate(encoded, axis=0).astype(np.float32)


def pca_and_covariance(latents, n_traj, n_time, rng, max_pca_points):
    centered = latents.astype(np.float64) - latents.mean(axis=0, keepdims=True)
    denom = max(centered.shape[0] - 1, 1)
    cov = centered.T @ centered / denom
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]

    coords = centered @ eigvecs[:, :2]
    all_indices = np.arange(coords.shape[0])
    if coords.shape[0] > max_pca_points:
        keep = np.sort(rng.choice(all_indices, size=max_pca_points, replace=False))
    else:
        keep = all_indices

    z_by_traj = latents.reshape(n_traj, n_time, latents.shape[1])
    dz = z_by_traj[:, 1:] - z_by_traj[:, :-1]
    z0 = z_by_traj[:, :-1]
    displacement = np.linalg.norm(dz, axis=-1).reshape(-1)
    relative_displacement = (np.linalg.norm(dz, axis=-1) / (np.linalg.norm(z0, axis=-1) + 1e-12)).reshape(-1)
    latent_norm = np.linalg.norm(z_by_traj, axis=-1).reshape(-1)

    return {
        "cov_eigvals": eigvals.astype(np.float32),
        "cov_explained": (eigvals / max(float(eigvals.sum()), 1e-12)).astype(np.float32),
        "pca_coords": coords[keep].astype(np.float32),
        "pca_traj": (keep // n_time).astype(np.int32),
        "pca_time": (keep % n_time).astype(np.int32),
        "displacement_norms": displacement.astype(np.float32),
        "relative_displacement_norms": relative_displacement.astype(np.float32),
        "latent_norms": latent_norm.astype(np.float32),
    }


def add_summary(outputs, label, N, result):
    eigvals = result["cov_eigvals"].astype(np.float64)
    explained = result["cov_explained"].astype(np.float64)
    displacement = result["displacement_norms"].astype(np.float64)
    rel_displacement = result["relative_displacement_norms"].astype(np.float64)
    latent_norm = result["latent_norms"].astype(np.float64)

    total = float(eigvals.sum())
    outputs["summary_N"].append(N)
    outputs["summary_cov_participation_ratio"].append(float(total * total / max(float(np.square(eigvals).sum()), 1e-12)))
    outputs["summary_cov_top1_frac"].append(float(explained[:1].sum()))
    outputs["summary_cov_top10_frac"].append(float(explained[:10].sum()))
    outputs["summary_displacement_mean"].append(float(displacement.mean()))
    outputs["summary_displacement_median"].append(float(np.median(displacement)))
    outputs["summary_displacement_p95"].append(float(np.percentile(displacement, 95)))
    outputs["summary_relative_displacement_mean"].append(float(rel_displacement.mean()))
    outputs["summary_relative_displacement_median"].append(float(np.median(rel_displacement)))
    outputs["summary_latent_norm_mean"].append(float(latent_norm.mean()))
    outputs["summary_latent_norm_std"].append(float(latent_norm.std()))

    print(
        f"{label}: N={N}  PR={outputs['summary_cov_participation_ratio'][-1]:.2f}  "
        f"top10={outputs['summary_cov_top10_frac'][-1]:.3f}  "
        f"mean|dz|={outputs['summary_displacement_mean'][-1]:.4f}",
        flush=True,
    )


def make_smoke_data(args):
    rng = np.random.default_rng(args.seed)
    outputs = base_outputs(args, smoke=True)
    for case in CASES:
        label = case["label"]
        n_traj = case["N"]
        n_time = min(args.T, 40)
        latents = rng.normal(size=(n_traj * n_time, 32)).astype(np.float32)
        latents += np.linspace(0.0, 0.8, n_time, dtype=np.float32).repeat(n_traj).reshape(-1, 1)
        result = pca_and_covariance(latents, n_traj, n_time, rng, args.max_pca_points)
        store_case(outputs, label, case, result, n_traj, n_time, train_std=1.0, ckpt_epoch=-1)
        add_summary(outputs, label, n_traj, result)
    return outputs


def base_outputs(args, smoke):
    return {
        "metadata_json": json.dumps(
            {
                "smoke": smoke,
                "candidate_plots": [
                    "pca_projection",
                    "latent_displacement_distribution",
                    "latent_covariance_spectrum",
                ],
                "notes": (
                    "Latent spaces come from separately trained pure autoencoders. "
                    "PCA panels are visual diagnostics; covariance spectra and displacement norms "
                    "are safer to compare across N."
                ),
                "T": args.T,
                "downsample": args.downsample,
                "max_pca_points": args.max_pca_points,
            },
            indent=2,
        ),
        "summary_N": [],
        "summary_cov_participation_ratio": [],
        "summary_cov_top1_frac": [],
        "summary_cov_top10_frac": [],
        "summary_displacement_mean": [],
        "summary_displacement_median": [],
        "summary_displacement_p95": [],
        "summary_relative_displacement_mean": [],
        "summary_relative_displacement_median": [],
        "summary_latent_norm_mean": [],
        "summary_latent_norm_std": [],
    }


def store_case(outputs, label, case, result, n_traj, n_time, train_std, ckpt_epoch):
    outputs[f"{label}_N"] = np.asarray(n_traj, dtype=np.int32)
    outputs[f"{label}_T"] = np.asarray(n_time, dtype=np.int32)
    outputs[f"{label}_data_path"] = np.asarray(str(case["data"]))
    outputs[f"{label}_checkpoint_path"] = np.asarray(str(case["checkpoint"]))
    outputs[f"{label}_train_std"] = np.asarray(train_std, dtype=np.float32)
    outputs[f"{label}_ckpt_epoch"] = np.asarray(ckpt_epoch, dtype=np.int32)
    for key, value in result.items():
        outputs[f"{label}_{key}"] = value


def make_real_data(args):
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    outputs = base_outputs(args, smoke=False)

    total_t0 = time.perf_counter()
    per_case_timing = []
    for case in tqdm(CASES, desc="cases"):
        label = case["label"]
        t_case = time.perf_counter()

        t = time.perf_counter()
        print(f"{label}: loading {case['checkpoint'].name}", flush=True)
        model, train_std, ckpt_epoch = load_model(case["checkpoint"], args, device)
        t_load_ckpt = time.perf_counter() - t

        t = time.perf_counter()
        frames, n_traj, n_time = load_frames(case, train_std, args)
        t_load_frames = time.perf_counter() - t

        t = time.perf_counter()
        print(f"{label}: encoding {frames.shape[0]} frames from {case['data'].name}", flush=True)
        latents = encode_frames(model, frames, device, args.batch_size, label=label)
        t_encode = time.perf_counter() - t

        t = time.perf_counter()
        result = pca_and_covariance(latents, n_traj, n_time, rng, args.max_pca_points)
        t_stats = time.perf_counter() - t

        store_case(outputs, label, case, result, n_traj, n_time, train_std, ckpt_epoch)
        add_summary(outputs, label, n_traj, result)

        elapsed = time.perf_counter() - t_case
        per_case_timing.append((label, elapsed))
        print(
            f"{label}: done in {elapsed:.1f}s "
            f"(ckpt={t_load_ckpt:.1f}s, frames={t_load_frames:.1f}s, "
            f"encode={t_encode:.1f}s, stats={t_stats:.1f}s)",
            flush=True,
        )

    total = time.perf_counter() - total_t0
    print(f"\ntotal wall time: {total:.1f}s", flush=True)
    for label, elapsed in per_case_timing:
        print(f"  {label:5s} {elapsed:6.1f}s ({100 * elapsed / total:.1f}%)", flush=True)
    return outputs


def finalize(outputs):
    for key in list(outputs):
        if key.startswith("summary_") and isinstance(outputs[key], list):
            dtype = np.int32 if key == "summary_N" else np.float32
            outputs[key] = np.asarray(outputs[key], dtype=dtype)
    return outputs


def save_figures(data_path):
    from types import SimpleNamespace

    from plot_lsq_latent_geometry import build_figures, output_paths, save

    plot_args = SimpleNamespace(
        data=data_path,
        output_dir=data_path.parent,
        pca_width=6.8,
        pca_height=5.4,
        pca_point_size=3.0,
        pca_alpha=0.55,
        pca_wspace=0.30,
        pca_hspace=0.42,
        pca_colorbar_width=0.028,
        pca_colorbar_gap_scale=0.45,
        distribution_width=6.8,
        distribution_height=3.1,
        covariance_width=6.8,
        covariance_height=2.85,
        covariance_wspace=0.30,
        legend_length=2.4,
        legend_font_size=8.0,
        title_pad=8.0,
        max_rank=80,
    )
    figures = build_figures(plot_args)
    outs = output_paths(data_path)
    save(figures["pca"], outs["pca"])
    save(figures["displacement"], outs["displacement"])
    save(figures["covariance"], outs["covariance"])


def main():
    args = parse_args()
    outputs = make_smoke_data(args) if args.smoke else make_real_data(args)
    out = args.output if args.output is not None else (SMOKE_OUT if args.smoke else FULL_OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **finalize(outputs))
    print(out)
    save_figures(out)


if __name__ == "__main__":
    main()
