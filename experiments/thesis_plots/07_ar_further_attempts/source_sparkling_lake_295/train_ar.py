import os
import time

import hydra
import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.distributed as torch_dist
import torch.optim as optim
from neuralop.models import FNO
from omegaconf import DictConfig
from physicsnemo.distributed.manager import DistributedManager
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import wandb
from x01.ar import (
    KoopmanAE2D,
    log_autocorr,
    log_energy_spectrum,
    log_images,
    log_qq,
    log_rollout,
    log_short_rollout,
    loss_fno,
    loss_koopman,
)
from x01.data import KMFlowDataset


def make_loader(
    data_path: str,
    split: str,
    std: float,
    batch_size: int,
    dist: DistributedManager,
    num_workers: int = 4,
    skip_n_frames: int = 0,
    pick_n_trajs: int = None,
) -> tuple[DataLoader, DistributedSampler | None]:
    ds = KMFlowDataset(
        data_path, split=split, mode="pairs", std=std, skip_n_frames=skip_n_frames, pick_n_trajs=pick_n_trajs
    )
    sampler = None
    if dist.distributed and split == "train":
        sampler = DistributedSampler(ds, num_replicas=dist.world_size, rank=dist.rank, shuffle=True)
    return (
        DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train" and sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train" and sampler is not None),  # Drops last uneven batch
        ),
        sampler,
    )


@hydra.main(version_base="1.3", config_path="conf", config_name="train_ar_config")
def main(cfg: DictConfig) -> None:
    smoke = cfg.training.smoke
    epochs = 2 if smoke else cfg.training.epochs
    log_every_n_steps = 1 if smoke else cfg.training.log_every_n_steps
    cfg_pick = cfg.data.get("pick_n_trajs", None)
    pick_n_trajs = 2 if smoke else cfg_pick

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfg.seed)

    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    rank0 = dist.rank == 0
    if rank0:
        print(f"device: {device}  world_size: {dist.world_size}  distributed: {dist.distributed}", flush=True)

    _full = np.load(cfg.data.path, mmap_mode="r")  # Loads the full data
    _n_train_end = KMFlowDataset.split_bounds(_full.shape[0], "train")[1]
    _std_end = pick_n_trajs if smoke else _n_train_end
    train_std = float(np.array(_full[:_std_end]).std())
    skip_n_frames = cfg.data.skip_trans_step if cfg.data.skip_trans else 0

    train_loader, train_sampler = make_loader(
        cfg.data.path,
        "train",
        train_std,
        cfg.training.batch_size,
        dist,
        cfg.training.num_workers,
        skip_n_frames,
        pick_n_trajs,
    )
    val_loader, _ = make_loader(
        cfg.data.path,
        "val",
        train_std,
        cfg.training.batch_size,
        dist,
        cfg.training.num_workers,
        skip_n_frames,
        pick_n_trajs,
    )

    # Full first trajectory for rollout — defaults to val; switch to train for overfit diagnostics.
    val_ds = val_loader.dataset
    probe_ds = train_loader.dataset if cfg.data.get("use_train_for_probe", False) else val_ds
    probe_T = probe_ds.samples_per_traj
    gt_probe = torch.stack([probe_ds[t][0] for t in range(probe_T)] + [probe_ds[probe_T - 1][1]])

    # Test loader + GT tensors hoisted out of end-of-training block so test plots
    # can be logged every cfg.viz.every epochs from inside the loop.
    test_ds_viz = None
    gt_test_probe = None
    gt_test_trajectories = None
    if rank0 and cfg.viz.enabled and not smoke:
        test_loader_viz, _ = make_loader(
            cfg.data.path,
            "test",
            train_std,
            cfg.training.batch_size,
            dist,
            cfg.training.num_workers,
            skip_n_frames,
            pick_n_trajs,
        )
        test_ds_viz = test_loader_viz.dataset
        test_T = test_ds_viz.samples_per_traj
        gt_test_probe = torch.stack([test_ds_viz[t][0] for t in range(test_T)] + [test_ds_viz[test_T - 1][1]])
        gt_test_trajectories = [
            torch.stack(
                [test_ds_viz[i * test_T + t][0] for t in range(test_T)] + [test_ds_viz[i * test_T + test_T - 1][1]]
            )
            for i in range(min(4, test_ds_viz.n_trajs))
        ]

    if cfg.ar.name == "koopman":
        model = KoopmanAE2D(
            in_channels=1,
            out_channels=1,
            dims=cfg.ar.dims,
            num_blocks=tuple(cfg.ar.num_blocks),
            num_heads=tuple(cfg.ar.num_heads),
            init_scale=cfg.ar.init_scale,
        ).to(device)
    if cfg.ar.name == "fno":
        model = FNO(
            n_modes=tuple(cfg.ar.n_modes),
            in_channels=1,
            out_channels=1,
            hidden_channels=cfg.ar.hidden_channels,
            n_layers=cfg.ar.n_layers,
            factorization=cfg.ar.factorization,
            rank=cfg.ar.rank,
            implementation=cfg.ar.implementation,
            fno_skip=cfg.ar.fno_skip,
            channel_mlp_dropout=cfg.ar.channel_mlp_dropout,
            channel_mlp_expansion=cfg.ar.channel_mlp_expansion,
            channel_mlp_skip=cfg.ar.channel_mlp_skip,
            lifting_channel_ratio=cfg.ar.lifting_channel_ratio,
            projection_channel_ratio=cfg.ar.projection_channel_ratio,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    if rank0:
        print(f"parameters : {n_params:,}", flush=True)
        print(f"train std  : {train_std:.4f}", flush=True)
        print(f"train steps: {len(train_loader)} / epoch", flush=True)
        print(f"val steps  : {len(val_loader)} / epoch", flush=True)

    if device.type == "cuda" and cfg.ar.name == "koopman":
        model = torch.compile(model, backend="aot_eager")
        if rank0:
            print("torch.compile: enabled (aot_eager)", flush=True)

    if dist.distributed:
        ddps = torch.cuda.Stream()
        with torch.cuda.stream(ddps):
            model = DistributedDataParallel(
                model,
                device_ids=[dist.local_rank],
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )
        torch.cuda.current_stream().wait_stream(ddps)
    base_model = model.module if dist.distributed else model

    if cfg.ar.name == "koopman":

        def predict_fn(x):
            pred, _ = base_model(x, mode="forward")
            return pred[0]

        def rollout_fn(x0, T):
            x0 = x0.to(device)
            z = base_model.encode(x0)
            frames = []
            for _ in range(T):
                frames.append(base_model.decode(z).cpu())
                z = base_model.dynamics(z)
            return torch.cat(frames)

    elif cfg.ar.name == "fno":

        def predict_fn(x):
            return base_model(x)

        def rollout_fn(x0, T):
            x = x0.to(device)
            frames = []
            for _ in range(T):
                x = base_model(x)
                frames.append(x.cpu())
            return torch.cat(frames)

    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)

    scheduler = None
    sched_type = cfg.training.scheduler
    if sched_type == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs)
    elif sched_type == "flat_decay":
        flat = epochs // 5
        decay = epochs - flat

        def _lr_lambda(last_epoch: int) -> float:
            if last_epoch < flat:
                return 1.0
            progress = (last_epoch - flat + 1) / decay
            return max(0.01, 1.0 - progress * 0.99)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    wandb_config = {
        "ar": cfg.ar.name,
        "lr": cfg.training.lr,
        "batch_size": cfg.training.batch_size,
        "epochs": epochs,
        "scheduler": sched_type,
        "grad_clip": cfg.training.grad_clip,
        "params": n_params,
        "smoke": smoke,
        "viz": cfg.viz.enabled,
        "skip_trans": cfg.data.skip_trans,
        "skip_trans_step": cfg.data.skip_trans_step,
    }
    if cfg.ar.name == "koopman":
        wandb_config.update(
            {
                "ar_dims": cfg.ar.dims,
                "ar_num_blocks": list(cfg.ar.num_blocks),
                "ar_num_heads": list(cfg.ar.num_heads),
                "init_scale": cfg.ar.init_scale,
                "lambda_recon": cfg.ar.lambda_recon,
                "lambda_ortho": cfg.ar.lambda_ortho,
                "lambda_bwd": cfg.ar.lambda_bwd,
                "lambda_fwd": cfg.ar.lambda_fwd,
                "lambda_latent_fwd": cfg.ar.lambda_latent_fwd,
            }
        )

    # Skip FNO logging
    if rank0:
        wandb.init(
            entity=cfg.wandb.entity, project=cfg.wandb.project, config=wandb_config, tags=list(cfg.wandb.tags) or None
        )
        os.makedirs(cfg.training.ckpt_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        t0 = time.perf_counter()
        totals = dict(loss=0.0, fwd=0.0, bwd=0.0, recon=0.0, ortho=0.0, latent_fwd=0.0)  # fwd/bwd/recon/ortho/latent_fwd only used for koopman
        n_t = 0

        for step, (x, y) in enumerate(train_loader, 1):

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            if cfg.ar.name == "koopman":
                loss, log = loss_koopman(
                    model,
                    x,
                    y,
                    lambda_recon=cfg.ar.lambda_recon,
                    lambda_ortho=cfg.ar.lambda_ortho,
                    lambda_bwd=cfg.ar.lambda_bwd,
                    lambda_fwd=cfg.ar.lambda_fwd,
                    lambda_latent_fwd=cfg.ar.lambda_latent_fwd,
                )
            elif cfg.ar.name == "fno":
                pred = model(x)
                loss, log = loss_fno(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            optimizer.step()

            step_loss = log["loss"].item()
            totals["loss"] += step_loss
            if cfg.ar.name == "koopman":
                totals["fwd"] += log["forward_loss"].item()
                totals["bwd"] += log["backward_loss"].item()
                totals["recon"] += log["reconstruction_loss"].item()
                totals["ortho"] += log["ortho_loss"].item()
                totals["latent_fwd"] += log["latent_fwd_loss"].item()
            n_t = step

            if rank0:
                if step % log_every_n_steps == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    if cfg.ar.name == "koopman":
                        print(
                            f"[{epoch:3d}/{epochs}] step {step:5d}"
                            f"  loss={step_loss:.4f}"
                            f"  fwd={log['forward_loss'].item():.4f}"
                            f"  bwd={log['backward_loss'].item():.4f}"
                            f"  recon={log['reconstruction_loss'].item():.4f}"
                            f"  ortho={log['ortho_loss'].item():.4f}"
                            f"  latent_fwd={log['latent_fwd_loss'].item():.4f}"
                            f"  lr={lr:.2e}",
                            flush=True,
                        )
                        wandb.log(
                            {
                                "train/loss": step_loss,
                                "train/fwd": log["forward_loss"].item(),
                                "train/bwd": log["backward_loss"].item(),
                                "train/recon": log["reconstruction_loss"].item(),
                                "train/ortho": log["ortho_loss"].item(),
                                "train/latent_fwd": log["latent_fwd_loss"].item(),
                                "train/lr": lr,
                            }
                        )
                    elif cfg.ar.name == "fno":
                        print(f"[{epoch:3d}/{epochs}] step {step:5d}  loss={step_loss:.4f}  lr={lr:.2e}", flush=True)
                        wandb.log({"train/loss": step_loss, "train/lr": lr})

            if smoke:
                break

        if rank0:
            model.eval()
            val_totals = dict(loss=0.0, fwd=0.0, bwd=0.0, recon=0.0, ortho=0.0, latent_fwd=0.0)
            n_v = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    if cfg.ar.name == "koopman":
                        _, vlog = loss_koopman(
                            base_model,
                            x,
                            y,
                            lambda_recon=cfg.ar.lambda_recon,
                            lambda_ortho=cfg.ar.lambda_ortho,
                            lambda_bwd=cfg.ar.lambda_bwd,
                            lambda_fwd=cfg.ar.lambda_fwd,
                            lambda_latent_fwd=cfg.ar.lambda_latent_fwd,
                        )
                        val_totals["fwd"] += vlog["forward_loss"].item()
                        val_totals["bwd"] += vlog["backward_loss"].item()
                        val_totals["recon"] += vlog["reconstruction_loss"].item()
                        val_totals["ortho"] += vlog["ortho_loss"].item()
                        val_totals["latent_fwd"] += vlog["latent_fwd_loss"].item()
                    elif cfg.ar.name == "fno":
                        pred = base_model(x)
                        _, vlog = loss_fno(pred, y)
                    val_totals["loss"] += vlog["loss"].item()
                    n_v += 1
                    if smoke:
                        break

            elapsed = time.perf_counter() - t0
            if cfg.ar.name == "koopman":
                print(
                    f"epoch {epoch:3d}/{epochs}"
                    f"  train={totals['loss']/n_t:.4f}"
                    f"  val={val_totals['loss']/n_v:.4f}"
                    f"  val_fwd={val_totals['fwd']/n_v:.4f}"
                    f"  val_recon={val_totals['recon']/n_v:.4f}"
                    f"  val_ortho={val_totals['ortho']/n_v:.4f}"
                    f"  val_latent_fwd={val_totals['latent_fwd']/n_v:.4f}"
                    f"  t={elapsed:.1f}s",
                    flush=True,
                )
            elif cfg.ar.name == "fno":
                print(
                    f"epoch {epoch:3d}/{epochs}"
                    f"  train={totals['loss']/n_t:.4f}"
                    f"  val={val_totals['loss']/n_v:.4f}"
                    f"  t={elapsed:.1f}s",
                    flush=True,
                )

            if cfg.ar.name == "koopman":
                with torch.no_grad():
                    K = base_model.dynamics.dynamics
                    eig_mags = torch.linalg.eigvals(K.cpu()).abs()
                    wandb.log(
                        {
                            "epoch/K_eig_max": eig_mags.max().item(),
                            "epoch/K_eig_min": eig_mags.min().item(),
                            "epoch/K_eig_mean": eig_mags.mean().item(),
                        }
                    )

            epoch_log = {
                "epoch/train_loss": totals["loss"] / n_t,
                "epoch/val_loss": val_totals["loss"] / n_v,
                "epoch": epoch,
            }
            if cfg.ar.name == "koopman":
                epoch_log.update(
                    {
                        "epoch/val_fwd": val_totals["fwd"] / n_v,
                        "epoch/val_recon": val_totals["recon"] / n_v,
                        "epoch/val_ortho": val_totals["ortho"] / n_v,
                        "epoch/val_latent_fwd": val_totals["latent_fwd"] / n_v,
                    }
                )
            wandb.log(epoch_log)

            if cfg.viz.enabled and epoch % cfg.viz.every == 0:
                log_images(predict_fn, val_ds, device, epoch, train_std, tag="val")
                log_rollout(rollout_fn, gt_probe, device, epoch, train_std, tag="val")
                log_short_rollout(rollout_fn, gt_probe, device, epoch, train_std, tag="val", with_recon=True)
                log_short_rollout(rollout_fn, gt_probe, device, epoch, train_std, tag="val", with_recon=False)

            if epoch % cfg.training.ckpt_every == 0:
                ckpt = {
                    "epoch": epoch,
                    "model": base_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "train_std": train_std,
                }
                path = f"{cfg.training.ckpt_dir}/{cfg.ar.name}_ar_epoch{epoch:04d}.pt"
                torch.save(ckpt, path)
                print(f"  checkpoint saved to {path}", flush=True)

        if scheduler is not None:
            scheduler.step()

        if dist.distributed:
            torch_dist.barrier()

    # Test eval (once, at the end, skipped in smoke mode, rank 0 only)
    if not smoke and rank0:
        test_loader, _ = make_loader(
            cfg.data.path, "test", train_std, cfg.training.batch_size, dist, cfg.training.num_workers, skip_n_frames
        )
        model.eval()
        test_totals = dict(loss=0.0)
        rel_l2 = 0.0
        n_test = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                if cfg.ar.name == "koopman":
                    _, tlog = loss_koopman(
                        base_model,
                        x,
                        y,
                        lambda_recon=cfg.ar.lambda_recon,
                        lambda_ortho=cfg.ar.lambda_ortho,
                        lambda_bwd=cfg.ar.lambda_bwd,
                        lambda_fwd=cfg.ar.lambda_fwd,
                        lambda_latent_fwd=cfg.ar.lambda_latent_fwd,
                    )
                    pred, _ = base_model(x, mode="forward")
                    pred = pred[0]
                elif cfg.ar.name == "fno":
                    pred = base_model(x)
                    _, tlog = loss_fno(pred, y)
                test_totals["loss"] += tlog["loss"].item()
                rel_l2 += (torch.norm(pred - y) / torch.norm(y)).item()
                n_test += 1

        print(
            f"\ntest  loss={test_totals['loss']/n_test:.4f}  rel_l2={rel_l2/n_test:.4f}",
            flush=True,
        )
        wandb.log({"test/loss": test_totals["loss"] / n_test, "test/rel_l2": rel_l2 / n_test})

        if cfg.viz.enabled:
            test_ds = test_loader.dataset
            log_images(predict_fn, test_ds, device, epochs, train_std, tag="test")
            test_T = test_ds.samples_per_traj
            gt_test_probe = torch.stack([test_ds[t][0] for t in range(test_T)] + [test_ds[test_T - 1][1]])
            log_rollout(rollout_fn, gt_test_probe, device, epochs, train_std, tag="test")
            log_short_rollout(rollout_fn, gt_test_probe, device, epochs, train_std, tag="test", with_recon=True)
            log_short_rollout(rollout_fn, gt_test_probe, device, epochs, train_std, tag="test", with_recon=False)
            gt_test_trajectories = [
                torch.stack([test_ds[i * test_T + t][0] for t in range(test_T)] + [test_ds[i * test_T + test_T - 1][1]])
                for i in range(min(4, test_ds.n_trajs))
            ]
            log_autocorr(rollout_fn, gt_test_trajectories, device, epochs, tag="test")
            log_energy_spectrum(rollout_fn, gt_test_trajectories, device, epochs, tag="test")
            log_qq(rollout_fn, gt_test_trajectories, device, epochs, tag="test")

    if rank0:
        wandb.finish()
    if dist.distributed:
        torch_dist.barrier()
        DistributedManager.cleanup()


if __name__ == "__main__":
    main()
