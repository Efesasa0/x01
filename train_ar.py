import os
import subprocess
import time
from datetime import datetime
from functools import partial

import hydra
import numpy as np
import torch
import torch.distributed as torch_dist
import torch.optim as optim
import wandb
from omegaconf import DictConfig
from physicsnemo.distributed.manager import DistributedManager
from torch.nn.parallel import DistributedDataParallel

from x01.ar import (
    KoopmanAE2D,
    benchmark_fps,
    log_autocorr,
    log_eig_spectrum,
    log_energy_spectrum,
    log_mse_vs_frame,
    log_qq,
    log_rollout,
    log_rollout_gif,
    log_short_rollout,
    loss_koopman,
)
from x01.data import load_full_trajectory, make_loader


@hydra.main(version_base="1.3", config_path="conf", config_name="train_ar_config")
def main(cfg: DictConfig) -> None:
    smoke = cfg.training.smoke
    epochs = 2 if smoke else cfg.training.epochs
    log_every_n_steps = 1 if smoke else cfg.training.log_every_n_steps
    if smoke:
        cfg.data.path = cfg.data.smoke_path
        n_train = n_val = n_test = 2
        T = 2
    else:
        n_train = cfg.data.n_train
        n_val = cfg.data.n_val
        n_test = cfg.data.n_test
        T = cfg.data.T
    T_in = cfg.data.T_in
    log_wandb = cfg.wandb.enabled
    inference_mode = str(cfg.inference_mode)
    if cfg.ar.pixel_rollout_steps < 1:
        raise ValueError(f"ar.pixel_rollout_steps must be >= 1, got {cfg.ar.pixel_rollout_steps}")
    if cfg.ar.latent_rollout_steps > cfg.ar.pixel_rollout_steps:
        raise ValueError(
            f"ar.latent_rollout_steps ({cfg.ar.latent_rollout_steps}) must not exceed"
            f" ar.pixel_rollout_steps ({cfg.ar.pixel_rollout_steps})"
        )

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfg.seed)

    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    rank0 = dist.rank == 0
    if rank0:
        try:
            branch = (
                subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL)
                .decode()
                .strip()
            )
        except Exception:
            branch = "?"
        print(f"run env: branch={branch}  cwd={os.getcwd()}", flush=True)
        print(f"device: {device}  world_size: {dist.world_size}  distributed: {dist.distributed}", flush=True)
        print(f"inference_mode: {inference_mode}  (used for viz rollouts)", flush=True)

        if cfg.save.enabled:
            iso = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            branch_slug = branch.replace("/", "-")
            parts = [iso, branch_slug]
            if cfg.save.run_tag:
                parts.append(cfg.save.run_tag)
            run_id = "smoke" if smoke else "__".join(parts)
            cfg.save.plot_dir = os.path.join(cfg.save.plot_dir, run_id)
            cfg.save.ckpt_dir = os.path.join(cfg.save.ckpt_dir, run_id)
            print(f"save   : plots -> {cfg.save.plot_dir}", flush=True)
            print(f"         ckpts -> {cfg.save.ckpt_dir}", flush=True)

    print("Loading Dataset...")
    t0 = time.perf_counter()
    _full = np.load(cfg.data.path, mmap_mode="r")  # Loads the full data
    print(f"Loaded Dataset. ({time.perf_counter() - t0:.2f}s)")
    if cfg.data.standardize:
        if cfg.data.use_precomputed_std:
            train_std = float(cfg.data.train_std)
            print(f"Using precomputed train_std={train_std:.4f}")
        else:
            print("Computing std...")
            t0 = time.perf_counter()
            train_std = float(np.array(_full[:n_train, T_in : T_in + T]).std())
            print(f"Computed std. ({time.perf_counter() - t0:.2f}s)")
    else:
        train_std = 1.0

    viz_T = 320

    print("Loading the Dataset for Training...")
    t0 = time.perf_counter()
    train_loader, train_sampler = make_loader(
        cfg.data.path,
        "train",
        train_std,
        cfg.training.batch_size,
        dist,
        n_train,
        n_val,
        n_test,
        T_in,
        T,
        cfg.training.num_workers,
        n_rollout=max(cfg.ar.pixel_rollout_steps, cfg.ar.latent_rollout_steps),
    )
    val_loader, _ = make_loader(
        cfg.data.path,
        "val",
        train_std,
        cfg.training.batch_size,
        dist,
        n_train,
        n_val,
        n_test,
        T_in,
        T,
        cfg.training.num_workers,
        n_rollout=max(cfg.ar.pixel_rollout_steps, cfg.ar.latent_rollout_steps),
    )
    print(f"Loaded Dataset for Training. ({time.perf_counter() - t0:.2f}s)")
    # Full first val trajectory for rollout
    gt_probe = load_full_trajectory(_full, n_train, viz_T, std=train_std)

    model = KoopmanAE2D(
        in_channels=1,
        out_channels=1,
        dims=cfg.ar.dims,
        num_blocks=tuple(cfg.ar.num_blocks),
        num_heads=tuple(cfg.ar.num_heads),
        dynamics_mode=cfg.ar.dynamics_mode,
        dynamics_rank=cfg.ar.dynamics_rank,
        steps=cfg.ar.pixel_rollout_steps,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    koop_params = sum(p.numel() for p in model.dynamics.parameters()) + sum(
        p.numel() for p in model.back_dynamics.parameters()
    )
    if rank0:
        print(f"parameters : {n_params:,}", flush=True)
        print(f"  encoder  : {enc_params:,}", flush=True)
        print(f"  koopman  : {koop_params:,}  (A + B)", flush=True)
        print(f"  decoder  : {dec_params:,}", flush=True)
        print(f"train std  : {train_std:.4f}", flush=True)
        print(f"pixel rollouts: {cfg.ar.pixel_rollout_steps} (Azencot multi-step supervision)", flush=True)
        print(f"latent rollouts: {cfg.ar.latent_rollout_steps} (Lusch latent-space supervision)", flush=True)
        print(f"train steps: {len(train_loader)} / epoch", flush=True)
        print(f"val steps  : {len(val_loader)} / epoch", flush=True)

    if device.type == "cuda":
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

    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    scheduler = None
    sched_type = cfg.training.scheduler
    if sched_type == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs)
    elif sched_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
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
        "params": n_params,
        "params_encoder": enc_params,
        "params_koopman": koop_params,
        "params_decoder": dec_params,
        "smoke": smoke,
        "viz": cfg.viz.enabled,
        "T_in": T_in,
        "T": T,
        "ar_dims": cfg.ar.dims,
        "ar_num_blocks": list(cfg.ar.num_blocks),
        "ar_num_heads": list(cfg.ar.num_heads),
        "lambda_fwd": cfg.ar.lambda_fwd,
        "lambda_recon": cfg.ar.lambda_recon,
        "lambda_consist": cfg.ar.lambda_consist,
        "lambda_bwd": cfg.ar.lambda_bwd,
        "lambda_latent_fwd": cfg.ar.lambda_latent_fwd,
        "lambda_latent_bwd": cfg.ar.lambda_latent_bwd,
        "latent_rollout_steps": cfg.ar.latent_rollout_steps,
        "lambda_ortho_a": cfg.ar.lambda_ortho_a,
        "lambda_ortho_b": cfg.ar.lambda_ortho_b,
        "lambda_consist_progressive": cfg.ar.lambda_consist_progressive,
        "loss_mode": cfg.ar.loss_mode,
        "lp_size_average": cfg.ar.lp_size_average,
        "hs_size_average": cfg.ar.hs_size_average,
        "mse_size_average": cfg.ar.mse_size_average,
        "dynamics_mode": cfg.ar.dynamics_mode,
        "dynamics_rank": cfg.ar.dynamics_rank,
        "pixel_rollout_steps": cfg.ar.pixel_rollout_steps,
        "inference_mode": inference_mode,
    }

    if rank0:
        if log_wandb:
            wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                config=wandb_config,
                tags=list(cfg.wandb.tags) or None,
            )
        if cfg.save.enabled:
            os.makedirs(cfg.save.ckpt_dir, exist_ok=True)
            os.makedirs(cfg.save.plot_dir, exist_ok=True)
        eig_history: dict[str, list[float]] = {
            "A_min": [],
            "A_mean": [],
            "A_max": [],
            "B_min": [],
            "B_mean": [],
            "B_max": [],
        }

    for epoch in range(1, epochs + 1):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        t0 = time.perf_counter()
        totals = dict(
            loss=0.0,
            fwd=0.0,
            bwd=0.0,
            recon=0.0,
            latent_fwd=0.0,
            latent_bwd=0.0,
            consist=0.0,
            ortho_a=0.0,
            ortho_b=0.0,
            consist_prog=0.0,
        )  # only used for koopman
        n_t = 0

        for step, (x, y) in enumerate(train_loader, 1):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            loss, log = loss_koopman(
                model,
                x,
                y,
                lambda_fwd=cfg.ar.lambda_fwd,
                lambda_recon=cfg.ar.lambda_recon,
                lambda_consist=cfg.ar.lambda_consist,
                lambda_bwd=cfg.ar.lambda_bwd,
                lambda_latent_fwd=cfg.ar.lambda_latent_fwd,
                lambda_latent_bwd=cfg.ar.lambda_latent_bwd,
                lambda_ortho_a=cfg.ar.lambda_ortho_a,
                lambda_ortho_b=cfg.ar.lambda_ortho_b,
                lambda_consist_progressive=cfg.ar.lambda_consist_progressive,
                loss_mode=cfg.ar.loss_mode,
                lp_size_average=cfg.ar.lp_size_average,
                hs_size_average=cfg.ar.hs_size_average,
                mse_size_average=cfg.ar.mse_size_average,
                latent_rollout_steps=cfg.ar.latent_rollout_steps,
            )
            loss.backward()
            optimizer.step()

            step_loss = log["loss"].item()
            totals["loss"] += step_loss
            totals["fwd"] += log["forward_loss"].item()
            totals["bwd"] += log["backward_loss"].item()
            totals["recon"] += log["reconstruction_loss"].item()
            totals["latent_fwd"] += log["latent_fwd_loss"].item()
            totals["latent_bwd"] += log["latent_bwd_loss"].item()
            totals["consist"] += log["consistency_loss"].item()
            totals["ortho_a"] += log["ortho_a_loss"].item()
            totals["ortho_b"] += log["ortho_b_loss"].item()
            totals["consist_prog"] += log["consist_progressive_loss"].item()
            n_t = step

            if rank0:
                if step % log_every_n_steps == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"[{epoch:3d}/{epochs}] step {step:5d}"
                        f"  loss={step_loss:.4f}"
                        f"  fwd={log['forward_loss'].item():.4f}"
                        f"  bwd={log['backward_loss'].item():.4f}"
                        f"  recon={log['reconstruction_loss'].item():.4f}"
                        f"  latent_fwd={log['latent_fwd_loss'].item():.4f}"
                        f"  latent_bwd={log['latent_bwd_loss'].item():.4f}"
                        f"  consist={log['consistency_loss'].item():.4f}"
                        f"  ortho_a={log['ortho_a_loss'].item():.4f}"
                        f"  ortho_b={log['ortho_b_loss'].item():.4f}"
                        f"  consist_prog={log['consist_progressive_loss'].item():.4f}"
                        f"  lr={lr:.2e}",
                        flush=True,
                    )
                    if log_wandb:
                        wandb.log(
                            {
                                "train/loss": step_loss,
                                "train/fwd": log["forward_loss"].item(),
                                "train/bwd": log["backward_loss"].item(),
                                "train/recon": log["reconstruction_loss"].item(),
                                "train/latent_fwd": log["latent_fwd_loss"].item(),
                                "train/latent_bwd": log["latent_bwd_loss"].item(),
                                "train/consist": log["consistency_loss"].item(),
                                "train/ortho_a": log["ortho_a_loss"].item(),
                                "train/ortho_b": log["ortho_b_loss"].item(),
                                "train/consist_prog": log["consist_progressive_loss"].item(),
                                "train/lr": lr,
                            }
                        )

            if smoke:
                break

        if rank0:
            model.eval()
            val_totals = dict(
                loss=0.0,
                fwd=0.0,
                bwd=0.0,
                recon=0.0,
                latent_fwd=0.0,
                latent_bwd=0.0,
                consist=0.0,
                ortho_a=0.0,
                ortho_b=0.0,
                consist_prog=0.0,
            )
            n_v = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    _, vlog = loss_koopman(
                        base_model,
                        x,
                        y,
                        lambda_fwd=cfg.ar.lambda_fwd,
                        lambda_recon=cfg.ar.lambda_recon,
                        lambda_consist=cfg.ar.lambda_consist,
                        lambda_bwd=cfg.ar.lambda_bwd,
                        lambda_latent_fwd=cfg.ar.lambda_latent_fwd,
                        lambda_latent_bwd=cfg.ar.lambda_latent_bwd,
                        lambda_ortho_a=cfg.ar.lambda_ortho_a,
                        lambda_ortho_b=cfg.ar.lambda_ortho_b,
                        lambda_consist_progressive=cfg.ar.lambda_consist_progressive,
                        loss_mode=cfg.ar.loss_mode,
                        lp_size_average=cfg.ar.lp_size_average,
                        hs_size_average=cfg.ar.hs_size_average,
                        mse_size_average=cfg.ar.mse_size_average,
                        latent_rollout_steps=cfg.ar.latent_rollout_steps,
                    )
                    val_totals["fwd"] += vlog["forward_loss"].item()
                    val_totals["bwd"] += vlog["backward_loss"].item()
                    val_totals["recon"] += vlog["reconstruction_loss"].item()
                    val_totals["latent_fwd"] += vlog["latent_fwd_loss"].item()
                    val_totals["latent_bwd"] += vlog["latent_bwd_loss"].item()
                    val_totals["consist"] += vlog["consistency_loss"].item()
                    val_totals["ortho_a"] += vlog["ortho_a_loss"].item()
                    val_totals["ortho_b"] += vlog["ortho_b_loss"].item()
                    val_totals["consist_prog"] += vlog["consist_progressive_loss"].item()
                    val_totals["loss"] += vlog["loss"].item()
                    n_v += 1
                    if smoke:
                        break

            elapsed = time.perf_counter() - t0
            raw_train_total = (
                totals["fwd"]
                + totals["recon"]
                + totals["consist"]
                + totals["bwd"]
                + totals["latent_fwd"]
                + totals["latent_bwd"]
                + totals["ortho_a"]
                + totals["ortho_b"]
                + totals["consist_prog"]
            ) / n_t
            print(
                f"epoch {epoch:3d}/{epochs}"
                f"  train_raw={raw_train_total:.4f}"
                f"  |  w_val={val_totals['loss'] / n_v:.4f}"
                f"  t={elapsed:.1f}s",
                flush=True,
            )
            print(
                f"             raw_val fwd={val_totals['fwd'] / n_v:.4f} recon={val_totals['recon'] / n_v:.4f}"
                f" consist={val_totals['consist'] / n_v:.4f} bwd={val_totals['bwd'] / n_v:.4f}"
                f" latent_fwd={val_totals['latent_fwd'] / n_v:.4f} latent_bwd={val_totals['latent_bwd'] / n_v:.4f}"
                f" ortho_a={val_totals['ortho_a'] / n_v:.4f} ortho_b={val_totals['ortho_b'] / n_v:.4f}"
                f" consist_prog={val_totals['consist_prog'] / n_v:.4f}",
                flush=True,
            )
            with torch.no_grad():
                A_eig = torch.linalg.eigvals(base_model.dynamics.matrix.cpu()).abs()
                B_eig = torch.linalg.eigvals(base_model.back_dynamics.matrix.cpu()).abs()
                eig_history["A_min"].append(A_eig.min().item())
                eig_history["A_mean"].append(A_eig.mean().item())
                eig_history["A_max"].append(A_eig.max().item())
                eig_history["B_min"].append(B_eig.min().item())
                eig_history["B_mean"].append(B_eig.mean().item())
                eig_history["B_max"].append(B_eig.max().item())
                A_unit_dev = (A_eig - 1.0).abs().mean().item()
                B_unit_dev = (B_eig - 1.0).abs().mean().item()
            print(f"             A_unit_dev={A_unit_dev:.4f}  B_unit_dev={B_unit_dev:.4f}", flush=True)

            plot_dir = cfg.save.plot_dir if (cfg.save.enabled and epoch % cfg.save.plot_every == 0) else None
            log_eig_spectrum(eig_history, epoch, save_dir=plot_dir)

            if log_wandb:
                wandb.log(
                    {
                        "epoch/train_loss": totals["loss"] / n_t,
                        "epoch/val_loss": val_totals["loss"] / n_v,
                        "epoch/val_fwd": val_totals["fwd"] / n_v,
                        "epoch/val_recon": val_totals["recon"] / n_v,
                        "epoch/val_consist": val_totals["consist"] / n_v,
                        "epoch/val_ortho_a": val_totals["ortho_a"] / n_v,
                        "epoch/val_ortho_b": val_totals["ortho_b"] / n_v,
                        "epoch/val_consist_prog": val_totals["consist_prog"] / n_v,
                        "epoch": epoch,
                    }
                )

            if cfg.viz.enabled and (log_wandb or cfg.save.enabled) and epoch % cfg.viz.every == 0:
                plot_dir = cfg.save.plot_dir if cfg.save.enabled else None
                rollout_fn = partial(base_model.rollout, inference_mode=inference_mode)
                log_rollout(rollout_fn, gt_probe, epoch, train_std, tag="val", save_dir=plot_dir)
                log_short_rollout(rollout_fn, gt_probe, epoch, train_std, tag="val", save_dir=plot_dir)
                log_rollout_gif(rollout_fn, gt_probe, epoch, train_std, tag="val", save_dir=plot_dir)
                gt_val_trajectories = [
                    load_full_trajectory(_full, n_train + i, viz_T, std=train_std) for i in range(n_val)
                ]
                log_autocorr(rollout_fn, gt_val_trajectories, epoch, tag="val", save_dir=plot_dir)
                log_energy_spectrum(rollout_fn, gt_val_trajectories, epoch, tag="val", save_dir=plot_dir)
                log_qq(rollout_fn, gt_val_trajectories, epoch, tag="val", save_dir=plot_dir)
                log_mse_vs_frame(rollout_fn, gt_val_trajectories, epoch, tag="val", save_dir=plot_dir)

            if cfg.save.enabled and (epoch % cfg.save.ckpt_every == 0 or (smoke and epoch == epochs)):
                ckpt = {
                    "epoch": epoch,
                    "model": base_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "train_std": train_std,
                    "config": {
                        "dims": cfg.ar.dims,
                        "num_blocks": list(cfg.ar.num_blocks),
                        "num_heads": list(cfg.ar.num_heads),
                        "dynamics_mode": cfg.ar.dynamics_mode,
                        "dynamics_rank": cfg.ar.dynamics_rank,
                        "pixel_rollout_steps": cfg.ar.pixel_rollout_steps,
                        "latent_rollout_steps": cfg.ar.latent_rollout_steps,
                        "standardize": cfg.data.standardize,
                    },
                }
                path = f"{cfg.save.ckpt_dir}/{cfg.ar.name}_ar_epoch{epoch:04d}.pt"
                torch.save(ckpt, path)
                print(f"  checkpoint saved to {path}", flush=True)

        if scheduler is not None:
            scheduler.step()

        if dist.distributed:
            torch_dist.barrier()

    # Test eval (once, at the end, skipped in smoke mode, rank 0 only)
    if not smoke and rank0:
        test_loader, _ = make_loader(
            cfg.data.path,
            "test",
            train_std,
            cfg.training.batch_size,
            dist,
            n_train,
            n_val,
            n_test,
            T_in,
            T,
            cfg.training.num_workers,
            n_rollout=cfg.ar.pixel_rollout_steps,
        )
        model.eval()
        test_totals = dict(loss=0.0)
        n_test = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                _, tlog = loss_koopman(
                    base_model,
                    x,
                    y,
                    lambda_fwd=cfg.ar.lambda_fwd,
                    lambda_recon=cfg.ar.lambda_recon,
                    lambda_consist=cfg.ar.lambda_consist,
                    lambda_bwd=cfg.ar.lambda_bwd,
                    lambda_latent_fwd=cfg.ar.lambda_latent_fwd,
                    lambda_latent_bwd=cfg.ar.lambda_latent_bwd,
                    lambda_ortho_a=cfg.ar.lambda_ortho_a,
                    lambda_ortho_b=cfg.ar.lambda_ortho_b,
                    lambda_consist_progressive=cfg.ar.lambda_consist_progressive,
                    loss_mode=cfg.ar.loss_mode,
                    lp_size_average=cfg.ar.lp_size_average,
                    hs_size_average=cfg.ar.hs_size_average,
                    mse_size_average=cfg.ar.mse_size_average,
                    latent_rollout_steps=cfg.ar.latent_rollout_steps,
                )
                test_totals["loss"] += tlog["loss"].item()
                n_test += 1

        print(
            f"\ntest  w_loss={test_totals['loss'] / n_test:.4f}",
            flush=True,
        )
        if log_wandb:
            wandb.log({"test/loss": test_totals["loss"] / n_test})

        if cfg.viz.enabled and (log_wandb or cfg.save.enabled):
            plot_dir = cfg.save.plot_dir if cfg.save.enabled else None
            test_ds = test_loader.dataset
            test_start = n_train + n_val
            gt_test_probe = load_full_trajectory(_full, test_start, viz_T, std=train_std)
            rollout_fn = partial(base_model.rollout, inference_mode=inference_mode)
            log_rollout(rollout_fn, gt_test_probe, epochs, train_std, tag="test", save_dir=plot_dir)
            log_short_rollout(rollout_fn, gt_test_probe, epochs, train_std, tag="test", save_dir=plot_dir)
            log_rollout_gif(rollout_fn, gt_test_probe, epochs, train_std, tag="test", save_dir=plot_dir)
            gt_test_trajectories = [
                load_full_trajectory(_full, test_start + i, viz_T, std=train_std) for i in range(test_ds.n_trajs)
            ]
            log_autocorr(rollout_fn, gt_test_trajectories, epochs, tag="test", save_dir=plot_dir)
            log_energy_spectrum(rollout_fn, gt_test_trajectories, epochs, tag="test", save_dir=plot_dir)
            log_qq(rollout_fn, gt_test_trajectories, epochs, tag="test", save_dir=plot_dir)
            log_mse_vs_frame(rollout_fn, gt_test_trajectories, epochs, tag="test", save_dir=plot_dir)

        benchmark_fps(
            lambda x, T: base_model.rollout(x, T, inference_mode=inference_mode),
            test_loader.dataset[0][0].unsqueeze(0),
        )

    if rank0 and log_wandb:
        wandb.finish()
    if dist.distributed:
        torch_dist.barrier()
        DistributedManager.cleanup()


if __name__ == "__main__":
    main()
