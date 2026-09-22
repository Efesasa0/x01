import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb
from ar.models.koopman_ae_2d import KoopmanAE2D
from ar.models.loss_fn import loss_koopman
from ar.viz import log_autocorr, log_energy_spectrum, log_images, log_qq, log_rollout
from dataset import KMFlowDataset

# Config
VIZ = True  # False: no image logging to wandb
VIZ_EVERY = 10  # log images every N epochs

DATA_PATH = "../kmflow_highres.npy"
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 100
GRAD_CLIP = 1.0
LOG_EVERY = 50  # steps
CKPT_EVERY = 10  # epochs
CKPT_DIR = "checkpoints/"


def make_loader(split: str, std: float) -> DataLoader:
    ds = KMFlowDataset(DATA_PATH, split=split, normalize=True, std=std)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="10 epochs, 1 step each; skips test eval")
    args = parser.parse_args()

    smoke = args.smoke
    epochs = 10 if smoke else EPOCHS
    log_every = 1 if smoke else LOG_EVERY

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    train_std = float(np.load(DATA_PATH)[:32].std())

    train_loader = make_loader("train", train_std)
    val_loader = make_loader("val", train_std)

    # Fixed probe: mid-trajectory val sample, same every epoch for comparability
    x_probe = val_loader.dataset[160][0].unsqueeze(0)  # would be the middle pair of first trajectory
    y_probe = val_loader.dataset[160][1].unsqueeze(0)

    # Full first val trajectory for rollout (320 frames)
    val_ds = val_loader.dataset
    gt_probe = torch.stack([val_ds[t][0] for t in range(319)] + [val_ds[318][1]])

    model = KoopmanAE2D(in_channels=1, out_channels=1).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters : {n_params:,}", flush=True)
    print(f"train std  : {train_std:.4f}", flush=True)
    print(f"train steps: {len(train_loader)} / epoch", flush=True)
    print(f"val steps  : {len(val_loader)} / epoch", flush=True)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    wandb.init(
        entity="x01_team",
        project="x01_wandb",
        config=dict(
            lr=LR, batch_size=BATCH_SIZE, epochs=epochs, grad_clip=GRAD_CLIP, params=n_params, smoke=smoke, viz=VIZ
        ),
    )

    os.makedirs(CKPT_DIR, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        totals = dict(loss=0.0, fwd=0.0, bwd=0.0, recon=0.0, consist=0.0, orth=0.0)
        n_t = 0

        for step, (x, y) in enumerate(train_loader, 1):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            loss, log = loss_koopman(model, x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            totals["loss"] += log["loss"].item()
            totals["fwd"] += log["forward_loss"].item()
            totals["bwd"] += log["backward_loss"].item()
            totals["recon"] += log["reconstruction_loss"].item()
            totals["consist"] += log["consist_loss"].item()
            totals["orth"] += log["orth_loss"].item()
            n_t = step

            if step % log_every == 0:
                print(
                    f"  [{epoch:3d}] step {step:4d}/{len(train_loader)}"
                    f"  loss={totals['loss']/n_t:.4f}"
                    f"  fwd={totals['fwd']/n_t:.4f}"
                    f"  bwd={totals['bwd']/n_t:.4f}"
                    f"  recon={totals['recon']/n_t:.4f}"
                    f"  consist={totals['consist']/n_t:.4f}"
                    f"  orth={totals['orth']/n_t:.4f}",
                    flush=True,
                )
                wandb.log(
                    {
                        "train/loss": totals["loss"] / n_t,
                        "train/fwd": totals["fwd"] / n_t,
                        "train/bwd": totals["bwd"] / n_t,
                        "train/recon": totals["recon"] / n_t,
                        "train/consist": totals["consist"] / n_t,
                        "train/orth": totals["orth"] / n_t,
                    }
                )

            if smoke:
                break

        scheduler.step()

        model.eval()
        val_totals = dict(loss=0.0, fwd=0.0, bwd=0.0, recon=0.0, consist=0.0, orth=0.0)
        n_v = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                _, vlog = loss_koopman(model, x, y)
                val_totals["loss"] += vlog["loss"].item()
                val_totals["fwd"] += vlog["forward_loss"].item()
                val_totals["bwd"] += vlog["backward_loss"].item()
                val_totals["recon"] += vlog["reconstruction_loss"].item()
                val_totals["consist"] += vlog["consist_loss"].item()
                val_totals["orth"] += vlog["orth_loss"].item()
                n_v += 1
                if smoke:
                    break

        elapsed = time.perf_counter() - t0
        print(
            f"epoch {epoch:3d}/{epochs}"
            f"  train={totals['loss']/n_t:.4f}"
            f"  val={val_totals['loss']/n_v:.4f}"
            f"  val_fwd={val_totals['fwd']/n_v:.4f}"
            f"  val_recon={val_totals['recon']/n_v:.4f}"
            f"  val_consist={val_totals['consist']/n_v:.4f}"
            f"  val_orth={val_totals['orth']/n_v:.4f}"
            f"  lr={scheduler.get_last_lr()[0]:.2e}"
            f"  t={elapsed:.1f}s",
            flush=True,
        )

        with torch.no_grad():
            K = model.dynamics.dynamics
            eig_mags = torch.linalg.eigvals(K.cpu()).abs()
            wandb.log(
                {
                    "epoch/K_eig_max": eig_mags.max().item(),
                    "epoch/K_eig_min": eig_mags.min().item(),
                    "epoch/K_eig_mean": eig_mags.mean().item(),
                }
            )

        wandb.log(
            {
                "epoch/train_loss": totals["loss"] / n_t,
                "epoch/val_loss": val_totals["loss"] / n_v,
                "epoch/val_fwd": val_totals["fwd"] / n_v,
                "epoch/val_recon": val_totals["recon"] / n_v,
                "epoch/val_consist": val_totals["consist"] / n_v,
                "epoch/val_orth": val_totals["orth"] / n_v,
                "epoch/lr": scheduler.get_last_lr()[0],
                "epoch": epoch,
            }
        )

        if VIZ and epoch % VIZ_EVERY == 0:
            log_images(model, x_probe, y_probe, device, epoch, train_std, tag="val")
            log_rollout(model, gt_probe, device, epoch, train_std, tag="val")

        if epoch % CKPT_EVERY == 0:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "train_std": train_std,
            }
            path = f"{CKPT_DIR}/koopman_ar_epoch{epoch:04d}.pt"
            torch.save(ckpt, path)
            print(f"  checkpoint saved to {path}", flush=True)

    # Test eval (once, at the end, skipped in smoke mode)
    if not smoke:
        test_loader = make_loader("test", train_std)
        model.eval()
        test_totals = dict(loss=0.0, fwd=0.0, bwd=0.0, recon=0.0, consist=0.0)
        rel_l2 = 0.0
        n_test = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                _, tlog = loss_koopman(model, x, y)
                test_totals["loss"] += tlog["loss"].item()
                test_totals["fwd"] += tlog["forward_loss"].item()
                test_totals["bwd"] += tlog["backward_loss"].item()
                test_totals["recon"] += tlog["reconstruction_loss"].item()
                test_totals["consist"] += tlog["consist_loss"].item()
                pred, _ = model(x, mode="forward")
                rel_l2 += (torch.norm(pred[0] - y) / torch.norm(y)).item()
                n_test += 1

        print(
            f"\ntest  loss={test_totals['loss']/n_test:.4f}" f"  rel_l2={rel_l2/n_test:.4f}",
            flush=True,
        )
        wandb.log(
            {
                "test/loss": test_totals["loss"] / n_test,
                "test/fwd": test_totals["fwd"] / n_test,
                "test/bwd": test_totals["bwd"] / n_test,
                "test/recon": test_totals["recon"] / n_test,
                "test/consist": test_totals["consist"] / n_test,
                "test/rel_l2": rel_l2 / n_test,
            }
        )

        if VIZ:
            test_ds = test_loader.dataset
            x_test_probe = test_ds[160][0].unsqueeze(0)
            y_test_probe = test_ds[160][1].unsqueeze(0)
            log_images(model, x_test_probe, y_test_probe, device, epochs, train_std, tag="test")
            gt_test_probe = torch.stack([test_ds[t][0] for t in range(319)] + [test_ds[318][1]])
            log_rollout(model, gt_test_probe, device, epochs, train_std, tag="test")
            gt_test_trajectories = [
                torch.stack([test_ds[i * 319 + t][0] for t in range(319)] + [test_ds[i * 319 + 318][1]])
                for i in range(4)
            ]
            log_autocorr(model, gt_test_trajectories, device, epochs, tag="test")
            log_energy_spectrum(model, gt_test_trajectories, device, epochs, tag="test")
            log_qq(model, gt_test_trajectories, device, epochs, tag="test")

    wandb.finish()


if __name__ == "__main__":
    main()
