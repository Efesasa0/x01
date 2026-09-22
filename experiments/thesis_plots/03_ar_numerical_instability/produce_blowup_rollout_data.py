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
SOURCE = HERE / "source_giddy_star_12"
DATA_PATH = ROOT.parent / "kmflow_highres.npy"
SMOKE_OUT = HERE / "blowup_rollout_smoke.npz"
FULL_OUT = HERE / "blowup_rollout_data.npz"

ROLLOUT_EPOCHS = tuple(range(10, 101, 10))
DISPLAY_EPOCHS = (20, 40, 80)
BATCH_SIZE = 32
LR = 1e-4
GRAD_CLIP = 1.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(path)


def save_figures(data_path):
    from plot_blowup_rollout_data import load_data, output_paths, plot_diagnostics, plot_rollouts

    data = load_data(data_path)
    diagnostics_out, rollouts_out = output_paths(data_path)
    plot_args = SimpleNamespace(
        frame_column_margin=0.11,
        rollout_row_margin=0.675,
        colorbar_width=0.016,
        colorbar_height=1.0,
        colorbar_margin=0.05,
        title_closeness=0.97,
        frame_title_pad=3.0,
    )
    diagnostics_paths = plot_diagnostics(data, diagnostics_out)
    rollout_paths = plot_rollouts(data, rollouts_out, plot_args)
    for path in (*diagnostics_paths, *rollout_paths):
        print(path)


def make_smoke_data():
    rng = np.random.default_rng(12)
    timesteps = np.arange(320)
    y = np.linspace(-1, 1, 64)
    x = np.linspace(-1, 1, 64)
    X, Y = np.meshgrid(x, y)
    gt = np.empty((320, 64, 64), dtype=np.float32)

    for t in timesteps:
        phase = 2 * np.pi * t / 80
        gt[t] = np.sin(3 * X + phase) * np.cos(2 * Y - 0.5 * phase)

    pred = []
    max_abs_vorticity = []
    eig_epochs = (0,) + ROLLOUT_EPOCHS
    eig_progress = np.linspace(0, 1, len(eig_epochs), dtype=np.float32)
    eig_min = 0.82 + 0.05 * eig_progress
    eig_mean = 0.95 + 0.10 * eig_progress
    eig_max = 1.05 + 0.55 * eig_progress

    for epoch in ROLLOUT_EPOCHS:
        drift = 1 + epoch / 60
        noise = rng.normal(0, epoch / 300, size=gt.shape).astype(np.float32)
        rollout = (gt * drift + noise).astype(np.float32)
        pred.append(rollout)
        max_abs_vorticity.append(float(np.nanmax(np.abs(rollout))))

    return {
        "epochs": np.asarray(ROLLOUT_EPOCHS, dtype=np.int32),
        "display_epochs": np.asarray(DISPLAY_EPOCHS, dtype=np.int32),
        "timesteps": timesteps.astype(np.int32),
        "gt": gt.astype(np.float32),
        "pred": np.asarray(pred, dtype=np.float32),
        "max_abs_vorticity": np.asarray(max_abs_vorticity, dtype=np.float32),
        "eig_epochs": np.asarray(eig_epochs, dtype=np.int32),
        "eig_forward_min": eig_min.astype(np.float32),
        "eig_forward_mean": eig_mean.astype(np.float32),
        "eig_forward_max": eig_max.astype(np.float32),
        "loss_epochs": np.arange(1, 101, dtype=np.int32),
        "train_loss": np.geomspace(1.4, 0.4, 100).astype(np.float32),
        "val_loss": np.r_[np.geomspace(1.5, 0.65, 45), np.geomspace(0.68, 8.0, 55)].astype(np.float32),
    }


def setup_old_source():
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
    T = gt_frames.shape[0]
    x0 = gt_frames[0:1].to(device)

    with torch.no_grad():
        z = model.encode(x0)
        pred = []
        for _ in range(T):
            pred.append(model.decode(z).cpu())
            z = model.dynamics(z)

    pred = torch.cat(pred, dim=0)
    gt_np = (gt_frames[:, 0] * std).numpy().astype(np.float32)
    pred_np = (pred[:, 0] * std).numpy().astype(np.float32)
    return gt_np, pred_np


def forward_eigenvalue_stats(model):
    K = model.dynamics.dynamics.detach().cpu().numpy()
    eig_abs = np.abs(np.linalg.eigvals(K))
    return float(eig_abs.min()), float(eig_abs.mean()), float(eig_abs.max())


def evaluate(model, loader, loss_fn, device):
    model.eval()
    totals = dict(loss=0.0, fwd=0.0, bwd=0.0, recon=0.0, consist=0.0)
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
            count += 1

    return {key: value / count for key, value in totals.items()}


def make_full_data():
    dataset_cls, model_cls, loss_fn = setup_old_source()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_std = float(np.load(DATA_PATH)[:32].std())

    train_loader = make_loader(dataset_cls, "train", train_std, device)
    val_loader = make_loader(dataset_cls, "val", train_std, device)
    val_ds = val_loader.dataset
    gt_probe = torch.stack([val_ds[t][0] for t in range(319)] + [val_ds[318][1]])

    model = model_cls(in_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(ROLLOUT_EPOCHS))

    history = []
    saved_gt = None
    saved_pred = []
    max_abs_vorticity = []
    eig_epochs = [0]
    eig_min, eig_mean, eig_max = [], [], []
    eig_values = forward_eigenvalue_stats(model)
    eig_min.append(eig_values[0])
    eig_mean.append(eig_values[1])
    eig_max.append(eig_values[2])

    for epoch in range(1, max(ROLLOUT_EPOCHS) + 1):
        model.train()
        t0 = time.perf_counter()
        totals = dict(loss=0.0, fwd=0.0, bwd=0.0, recon=0.0, consist=0.0)
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
            count += 1

        scheduler.step()
        train_values = {key: value / count for key, value in totals.items()}
        val_values = evaluate(model, val_loader, loss_fn, device)
        elapsed = time.perf_counter() - t0
        history.append((epoch, train_values, val_values, elapsed))

        print(
            f"epoch {epoch:03d}/{max(ROLLOUT_EPOCHS)} "
            f"train={train_values['loss']:.6f} val={val_values['loss']:.6f} "
            f"lr={scheduler.get_last_lr()[0]:.2e} t={elapsed:.1f}s",
            flush=True,
        )

        if epoch in ROLLOUT_EPOCHS:
            gt_np, pred_np = rollout(model, gt_probe, device, train_std)
            if saved_gt is None:
                saved_gt = gt_np
            saved_pred.append(pred_np)
            max_abs_vorticity.append(float(np.nanmax(np.abs(pred_np))))
            eig_values = forward_eigenvalue_stats(model)
            eig_epochs.append(epoch)
            eig_min.append(eig_values[0])
            eig_mean.append(eig_values[1])
            eig_max.append(eig_values[2])

    epochs = np.asarray([row[0] for row in history], dtype=np.int32)
    train_loss = np.asarray([row[1]["loss"] for row in history], dtype=np.float32)
    val_loss = np.asarray([row[2]["loss"] for row in history], dtype=np.float32)
    val_fwd = np.asarray([row[2]["fwd"] for row in history], dtype=np.float32)
    val_bwd = np.asarray([row[2]["bwd"] for row in history], dtype=np.float32)
    val_recon = np.asarray([row[2]["recon"] for row in history], dtype=np.float32)
    val_consist = np.asarray([row[2]["consist"] for row in history], dtype=np.float32)

    return {
        "epochs": np.asarray(ROLLOUT_EPOCHS, dtype=np.int32),
        "display_epochs": np.asarray(DISPLAY_EPOCHS, dtype=np.int32),
        "timesteps": np.arange(saved_gt.shape[0], dtype=np.int32),
        "gt": saved_gt.astype(np.float32),
        "pred": np.asarray(saved_pred, dtype=np.float32),
        "max_abs_vorticity": np.asarray(max_abs_vorticity, dtype=np.float32),
        "eig_epochs": np.asarray(eig_epochs, dtype=np.int32),
        "eig_forward_min": np.asarray(eig_min, dtype=np.float32),
        "eig_forward_mean": np.asarray(eig_mean, dtype=np.float32),
        "eig_forward_max": np.asarray(eig_max, dtype=np.float32),
        "loss_epochs": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_fwd": val_fwd,
        "val_bwd": val_bwd,
        "val_recon": val_recon,
        "val_consist": val_consist,
        "train_std": np.asarray(train_std, dtype=np.float32),
    }


def main():
    args = parse_args()
    metadata = {
        "branch": "giddy-star-12",
        "wandb_run": "x01_team/x01_wandb/d98wr6mq",
        "source": str(SOURCE.name),
        "data": str(DATA_PATH.name),
        "rollout_epochs": ROLLOUT_EPOCHS,
        "display_epochs": DISPLAY_EPOCHS,
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
