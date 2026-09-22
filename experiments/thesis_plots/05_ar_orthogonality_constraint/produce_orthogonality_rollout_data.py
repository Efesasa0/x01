import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
SOURCE = HERE / "source_copper_vortex_16"
DATA_PATH = ROOT.parent / "kmflow_highres.npy"
SMOKE_OUT = HERE / "orthogonality_rollout_smoke.npz"
FULL_OUT = HERE / "orthogonality_rollout_data.npz"

SOURCE_BRANCH = "copper-vortex-16"
SOURCE_COMMIT = "d94c12c88b34fe930620f1d2a006846f20ffa856"
WANDB_RUN = "x01_team/x01_wandb/l2gla88n"

MAX_EPOCHS = 100
DISPLAY_EPOCH = 80
MSE_EPOCHS = tuple(range(0, MAX_EPOCHS + 1, 10))
BATCH_SIZE = 32
LR = 1e-4
GRAD_CLIP = 1.0
LAMBDA_ORTH = 1e-2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(path)


def save_figures(data_path):
    from plot_orthogonality_rollout_data import load_data, output_path, plot_summary

    data = load_data(data_path)
    out_path = output_path(data_path)
    plot_args = SimpleNamespace(
        frame_column_margin=0.11,
        rollout_row_margin=0.675,
        rollout_diagnostics_margin=1.0,
        diagnostic_column_margin=1.0,
        diagnostic_height=2.2,
        colorbar_width=0.016,
        colorbar_height=1.0,
        colorbar_margin=0.05,
        title_closeness=0.97,
        frame_title_pad=3.0,
    )
    paths = plot_summary(data, out_path, plot_args)
    for path in paths:
        print(path)


def make_smoke_data():
    rng = np.random.default_rng(16)
    timesteps = np.arange(320, dtype=np.int32)
    y = np.linspace(-1, 1, 64, dtype=np.float32)
    x = np.linspace(-1, 1, 64, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    gt = np.empty((320, 64, 64), dtype=np.float32)
    for t in timesteps:
        phase = 2 * np.pi * t / 96
        gt[t] = np.sin(3 * X + phase) * np.cos(2 * Y - 0.25 * phase)

    mse_curves = []
    all_preds = []
    for epoch in MSE_EPOCHS:
        drift = 0.01 * (epoch / 20) * np.sin(2 * X - 3 * Y)
        noise = rng.normal(0, 0.01 * epoch / 20, size=gt.shape).astype(np.float32)
        rollout = (gt + drift + noise).astype(np.float32)
        mse_curves.append(np.square(rollout - gt).mean(axis=(1, 2)))
        all_preds.append(rollout)

    loss_epochs = np.arange(1, MAX_EPOCHS + 1, dtype=np.int32)
    train_loss = np.geomspace(1.2, 0.32, MAX_EPOCHS).astype(np.float32)
    val_loss = (np.geomspace(1.25, 0.46, MAX_EPOCHS) + 0.015 * np.sin(loss_epochs / 8)).astype(np.float32)
    eig_epochs = np.asarray(MSE_EPOCHS, dtype=np.int32)
    eig_progress = np.linspace(0, 1, len(eig_epochs), dtype=np.float32)

    return {
        "epochs": np.asarray(MSE_EPOCHS, dtype=np.int32),
        "display_epochs": np.asarray([DISPLAY_EPOCH], dtype=np.int32),
        "mse_epochs": np.asarray(MSE_EPOCHS, dtype=np.int32),
        "timesteps": timesteps,
        "gt": gt,
        "pred": np.asarray(all_preds, dtype=np.float32),
        "rollout_mse": np.asarray(mse_curves, dtype=np.float32),
        "eig_epochs": eig_epochs,
        "eig_forward_min": (0.995 + 0.0015 * eig_progress).astype(np.float32),
        "eig_forward_mean": (0.9998 + 0.0001 * eig_progress).astype(np.float32),
        "eig_forward_max": (1.006 - 0.0025 * eig_progress).astype(np.float32),
        "loss_epochs": loss_epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_std": np.asarray(1.0, dtype=np.float32),
    }


def verify_source_snapshot():
    train_text = (SOURCE / "train_ar.py").read_text()
    model_text = (SOURCE / "ar/models/koopman_ae_2d.py").read_text()
    loss_text = (SOURCE / "ar/models/loss_fn.py").read_text()
    required = (
        ("train_ar.py", train_text, "BATCH_SIZE = 32"),
        ("train_ar.py", train_text, "LR = 1e-4"),
        ("train_ar.py", train_text, "EPOCHS = 100"),
        ("train_ar.py", train_text, "GRAD_CLIP = 1.0"),
        ("koopman_ae_2d.py", model_text, "init_scale: float = 1.0"),
        ("loss_fn.py", loss_text, "LAMBDA_ORTH = 1e-2"),
        ("loss_fn.py", loss_text, "loss = forward_loss + reconstruction_loss + consist_loss + backward_loss + LAMBDA_ORTH * orth_loss"),
    )
    for name, text, marker in required:
        if marker not in text:
            raise RuntimeError(f"{SOURCE_BRANCH} source snapshot check failed: {marker!r} missing from {name}")


def setup_source():
    verify_source_snapshot()
    sys.path.insert(0, str(SOURCE))
    from ar.models.koopman_ae_2d import KoopmanAE2D
    from ar.models.loss_fn import loss_koopman
    from dataset import KMFlowDataset

    return KMFlowDataset, KoopmanAE2D, loss_koopman


def make_loader(dataset_cls, split, train_std, device):
    dataset = dataset_cls(str(DATA_PATH), split=split, normalize=True, std=train_std)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


def rollout(model, gt_frames, device, std):
    model.eval()
    x0 = gt_frames[0:1].to(device)

    with torch.no_grad():
        z = model.encode(x0)
        pred = []
        for _ in range(gt_frames.shape[0]):
            pred.append(model.decode(z).cpu())
            z = model.dynamics(z)

    pred = torch.cat(pred, dim=0)
    gt_np = (gt_frames[:, 0].cpu().numpy() * std).astype(np.float32)
    pred_np = (pred[:, 0].numpy() * std).astype(np.float32)
    return gt_np, pred_np


def build_val_trajectories(val_ds, traj_len=320):
    n_pairs_per_traj = traj_len - 1
    n_traj = len(val_ds) // n_pairs_per_traj
    trajs = []
    for k in range(n_traj):
        base = k * n_pairs_per_traj
        frames = [val_ds[base + t][0] for t in range(n_pairs_per_traj)]
        frames.append(val_ds[base + n_pairs_per_traj - 1][1])
        trajs.append(torch.stack(frames))
    return trajs


def rollout_val_set(model, val_trajs, device, std):
    per_traj_gt = []
    per_traj_pred = []
    for traj in val_trajs:
        gt_np, pred_np = rollout(model, traj, device, std)
        per_traj_gt.append(gt_np)
        per_traj_pred.append(pred_np)
    return per_traj_gt, per_traj_pred


def mean_rollout_mse(per_traj_gt, per_traj_pred):
    per_traj = np.stack([np.square(p - g).mean(axis=(1, 2)) for g, p in zip(per_traj_gt, per_traj_pred)])
    return per_traj.mean(axis=0)


def forward_eigenvalue_stats(model):
    K = model.dynamics.dynamics.detach().cpu().numpy()
    eig_abs = np.abs(np.linalg.eigvals(K))
    return float(eig_abs.min()), float(eig_abs.mean()), float(eig_abs.max())


def evaluate(model, loader, loss_fn, device):
    model.eval()
    totals = dict(loss=0.0, fwd=0.0, bwd=0.0, recon=0.0, consist=0.0, orth=0.0)
    count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            _, log = loss_fn(model, x, y)
            totals["loss"] += log["loss"].item()
            totals["fwd"] += log["forward_loss"].item()
            totals["bwd"] += log["backward_loss"].item()
            totals["recon"] += log["reconstruction_loss"].item()
            totals["consist"] += log["consist_loss"].item()
            totals["orth"] += log["orth_loss"].item()
            count += 1

    return {key: value / count for key, value in totals.items()}


def make_full_data():
    dataset_cls, model_cls, loss_fn = setup_source()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_std = float(np.load(DATA_PATH)[:32].std())

    train_loader = make_loader(dataset_cls, "train", train_std, device)
    val_loader = make_loader(dataset_cls, "val", train_std, device)
    val_ds = val_loader.dataset
    val_trajs = build_val_trajectories(val_ds)

    model = model_cls(in_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    history = []
    saved_gt = None
    saved_preds = []
    mse_epochs = []
    rollout_mse = []
    eig_epochs = [0]
    eig_min, eig_mean, eig_max = [], [], []
    eig_values = forward_eigenvalue_stats(model)
    eig_min.append(eig_values[0])
    eig_mean.append(eig_values[1])
    eig_max.append(eig_values[2])

    print(f"source branch: {SOURCE_BRANCH}", flush=True)
    print(f"source commit: {SOURCE_COMMIT}", flush=True)
    print(f"wandb run    : {WANDB_RUN}", flush=True)
    print(f"device       : {device}", flush=True)
    print(f"train std    : {train_std:.4f}", flush=True)
    print(f"train steps  : {len(train_loader)} / epoch", flush=True)
    print(f"val steps    : {len(val_loader)} / epoch", flush=True)

    per_traj_gt, per_traj_pred = rollout_val_set(model, val_trajs, device, train_std)
    saved_gt = per_traj_gt[0]
    mse_epochs.append(0)
    rollout_mse.append(mean_rollout_mse(per_traj_gt, per_traj_pred))
    saved_preds.append(per_traj_pred[0])

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        t0 = time.perf_counter()
        totals = dict(loss=0.0, fwd=0.0, bwd=0.0, recon=0.0, consist=0.0, orth=0.0)
        count = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss, log = loss_fn(model, x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            totals["loss"] += log["loss"].item()
            totals["fwd"] += log["forward_loss"].item()
            totals["bwd"] += log["backward_loss"].item()
            totals["recon"] += log["reconstruction_loss"].item()
            totals["consist"] += log["consist_loss"].item()
            totals["orth"] += log["orth_loss"].item()
            count += 1

        scheduler.step()
        train_values = {key: value / count for key, value in totals.items()}
        val_values = evaluate(model, val_loader, loss_fn, device)

        elapsed = time.perf_counter() - t0
        history.append((epoch, train_values, val_values, elapsed))

        print(
            f"epoch {epoch:03d}/{MAX_EPOCHS} "
            f"train={train_values['loss']:.6f} val={val_values['loss']:.6f} "
            f"orth={val_values['orth']:.6f} "
            f"lr={scheduler.get_last_lr()[0]:.2e} t={elapsed:.1f}s",
            flush=True,
        )

        if epoch in MSE_EPOCHS:
            per_traj_gt, per_traj_pred = rollout_val_set(model, val_trajs, device, train_std)
            mse_epochs.append(epoch)
            rollout_mse.append(mean_rollout_mse(per_traj_gt, per_traj_pred))
            saved_preds.append(per_traj_pred[0])
            eig_values = forward_eigenvalue_stats(model)
            eig_epochs.append(epoch)
            eig_min.append(eig_values[0])
            eig_mean.append(eig_values[1])
            eig_max.append(eig_values[2])

    loss_epochs = np.asarray([row[0] for row in history], dtype=np.int32)

    return {
        "epochs": np.asarray(mse_epochs, dtype=np.int32),
        "display_epochs": np.asarray([DISPLAY_EPOCH], dtype=np.int32),
        "mse_epochs": np.asarray(mse_epochs, dtype=np.int32),
        "timesteps": np.arange(saved_gt.shape[0], dtype=np.int32),
        "gt": saved_gt.astype(np.float32),
        "pred": np.asarray(saved_preds, dtype=np.float32),
        "rollout_mse": np.asarray(rollout_mse, dtype=np.float32),
        "eig_epochs": np.asarray(eig_epochs, dtype=np.int32),
        "eig_forward_min": np.asarray(eig_min, dtype=np.float32),
        "eig_forward_mean": np.asarray(eig_mean, dtype=np.float32),
        "eig_forward_max": np.asarray(eig_max, dtype=np.float32),
        "loss_epochs": loss_epochs,
        "train_loss": np.asarray([row[1]["loss"] for row in history], dtype=np.float32),
        "val_loss": np.asarray([row[2]["loss"] for row in history], dtype=np.float32),
        "train_std": np.asarray(train_std, dtype=np.float32),
    }


def main():
    args = parse_args()
    metadata = {
        "branch": SOURCE_BRANCH,
        "commit": SOURCE_COMMIT,
        "wandb_run": WANDB_RUN,
        "source": str(SOURCE.name),
        "data": str(DATA_PATH.name),
        "max_epochs": MAX_EPOCHS,
        "display_epoch": DISPLAY_EPOCH,
        "mse_epochs": MSE_EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "grad_clip": GRAD_CLIP,
        "init_scale": 1.0,
        "orthogonality_loss": True,
        "lambda_orth": LAMBDA_ORTH,
        "smoke": args.smoke,
    }

    if args.smoke:
        arrays = make_smoke_data()
        out = SMOKE_OUT
    else:
        arrays = make_full_data()
        out = FULL_OUT

    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    save_npz(out, **arrays)
    save_figures(out)


if __name__ == "__main__":
    main()
