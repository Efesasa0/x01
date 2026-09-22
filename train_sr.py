import copy
import os
import subprocess
import time
from datetime import datetime

import hydra
import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.distributed as torch_dist
import torch.nn.functional as F
import torch.optim as optim
import wandb
from omegaconf import DictConfig
from physicsnemo.distributed.manager import DistributedManager
from torch.nn.parallel import DistributedDataParallel

from x01.data import make_loader
from x01.sr import DiffusionManager, UNet, log_energy_spectrum, log_samples, loss_ddpm


@torch.no_grad()
def ema_update(ema_model: torch.nn.Module, live_model: torch.nn.Module, decay: float) -> None:
    """In-place EMA: ema_p <- decay * ema_p + (1 - decay) * live_p for parameters; copy buffers."""
    for ema_p, live_p in zip(ema_model.parameters(), live_model.parameters(), strict=True):
        ema_p.mul_(decay).add_(live_p.detach(), alpha=1.0 - decay)
    for ema_b, live_b in zip(ema_model.buffers(), live_model.buffers(), strict=True):
        ema_b.copy_(live_b)


def synthesize_condition(x_hr: torch.Tensor, factor: int) -> torch.Tensor:
    """HR -> LR (stride subsample, matches KMFlowDataset's downsample) -> nearest upsample to HR.

    Conditional SR3-style: the result is fed to the UNet as `condition` (channel concat with
    noisy x_hr inside the model), so the model learns "noisy HR + blurry LR -> clean HR".
    At real inference the same shape of signal comes from a Koopman LR rollout upsampled.
    """
    x_lr = x_hr[:, :, ::factor, ::factor]
    return F.interpolate(x_lr, scale_factor=factor, mode="nearest")


@hydra.main(version_base="1.3", config_path="conf", config_name="train_sr_config")
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
    data_downsample = cfg.data.downsample
    image_resolution = cfg.sr.image_resolution
    log_wandb = cfg.wandb.enabled

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

    _full = np.load(cfg.data.path, mmap_mode="r")
    if cfg.data.standardize:
        train_std = float(np.array(_full[:n_train, T_in : T_in + T]).std())
    else:
        train_std = 1.0

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
        mode="frames",
        downsample=data_downsample,
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
        mode="frames",
        downsample=data_downsample,
    )

    val_ds = val_loader.dataset
    # 64 random i.i.d. frames from across the val split, used by log_energy_spectrum.
    # Seeded so the probe set is fixed across epochs (cleaner spectrum comparison).
    _probe_rng = np.random.RandomState(0)
    _probe_idx = sorted(_probe_rng.choice(len(val_ds), min(64, len(val_ds)), replace=False).tolist())
    gt_probe = torch.stack([val_ds[i] for i in _probe_idx])

    model = UNet(
        in_channels=cfg.sr.in_channels,
        out_channels=cfg.sr.out_channels,
        latent_dims=cfg.sr.latent_dims,
        channel_multipliers=tuple(cfg.sr.channel_multipliers),
        num_res_blocks=cfg.sr.num_res_blocks,
        attention_resolutions=tuple(cfg.sr.attention_resolutions),
        image_resolution=image_resolution,
        dropout_rate=cfg.sr.dropout_rate,
        resample_with_conv=cfg.sr.resample_with_conv,
    ).to(device)

    manager = DiffusionManager(
        beta_start=cfg.sr.beta_start,
        beta_end=cfg.sr.beta_end,
        num_diffusion_time_steps=cfg.sr.num_diffusion_time_steps,
        condition_dropout_rate=cfg.sr.condition_dropout_rate,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    if rank0:
        print(f"parameters : {n_params:,}", flush=True)
        print(f"train std  : {train_std:.4f}", flush=True)
        print(f"train steps: {len(train_loader)} / epoch", flush=True)
        print(f"val steps  : {len(val_loader)} / epoch", flush=True)

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

    # EMA shadow of the live UNet: deepcopy, eval mode, no grads. Updated every optimizer step
    # with a warmup-clipped decay so early-training noise doesn't poison the average.
    ema_model = copy.deepcopy(base_model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    def synthesize_sample_for_viz(hr_refs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """DDIM-sample conditioned on an LR-blurry version of the refs, using the EMA model.

        Returns `(condition, samples)` so viz can show the blurry guide alongside the refined
        sample. Both tensors are on CPU. Used for viz only.
        """
        cond = synthesize_condition(hr_refs.to(device), cfg.sr.condition_downsample_factor)
        x_noise = torch.randn_like(cond)
        samples = manager.infer_fn(
            x_noise,
            ema_model,
            inference_steps=cfg.sr.inference_steps,
            condition=cond,
            pbar=False,
        )
        return cond.cpu(), samples.cpu()

    def log_ema_vs_base_mse(refs: torch.Tensor, epoch: int, tag: str = "val") -> None:
        """Same refs, same cond, same x_noise -> sample with base and ema; log MSE vs refs."""
        refs_d = refs.to(device)
        cond = synthesize_condition(refs_d, cfg.sr.condition_downsample_factor)
        x_noise = torch.randn_like(cond)
        sample_base = manager.infer_fn(
            x_noise, base_model, inference_steps=cfg.sr.inference_steps, condition=cond, pbar=False
        )
        sample_ema = manager.infer_fn(
            x_noise, ema_model, inference_steps=cfg.sr.inference_steps, condition=cond, pbar=False
        )
        mse_base = (sample_base - refs_d).square().mean().item()
        mse_ema = (sample_ema - refs_d).square().mean().item()
        if log_wandb:
            wandb.log({f"{tag}/sample_mse_base": mse_base, f"{tag}/sample_mse_ema": mse_ema, "epoch": epoch})

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
        "sr": cfg.sr.name,
        "lr": cfg.training.lr,
        "batch_size": cfg.training.batch_size,
        "epochs": epochs,
        "scheduler": sched_type,
        "grad_clip": cfg.training.grad_clip,
        "ema_decay": cfg.training.ema_decay,
        "params": n_params,
        "smoke": smoke,
        "viz": cfg.viz.enabled,
        "latent_dims": cfg.sr.latent_dims,
        "channel_multipliers": list(cfg.sr.channel_multipliers),
        "num_res_blocks": cfg.sr.num_res_blocks,
        "attention_resolutions": list(cfg.sr.attention_resolutions),
        "image_resolution": image_resolution,
        "beta_start": cfg.sr.beta_start,
        "beta_end": cfg.sr.beta_end,
        "num_diffusion_time_steps": cfg.sr.num_diffusion_time_steps,
        "inference_steps": cfg.sr.inference_steps,
        "condition_dropout_rate": cfg.sr.condition_dropout_rate,
        "condition_downsample_factor": cfg.sr.condition_downsample_factor,
        "data_downsample": data_downsample,
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

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        t0 = time.perf_counter()
        totals = dict(loss=0.0)
        n_t = 0

        for step, x in enumerate(train_loader, 1):
            x = x.to(device)
            cond = synthesize_condition(x, cfg.sr.condition_downsample_factor)

            optimizer.zero_grad()
            loss, log = loss_ddpm(manager, model, x, condition=cond)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            optimizer.step()

            global_step += 1
            decay_t = min(cfg.training.ema_decay, (1 + global_step) / (10 + global_step))
            ema_update(ema_model, base_model, decay_t)

            step_loss = log["loss"].item()
            totals["loss"] += step_loss
            n_t = step

            if rank0 and step % log_every_n_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"[{epoch:3d}/{epochs}] step {step:5d}  loss={step_loss:.4f}  lr={lr:.2e}", flush=True)
                if log_wandb:
                    wandb.log({"train/loss": step_loss, "train/lr": lr})

            if smoke:
                break

        if rank0:
            model.eval()
            val_totals = dict(loss=0.0)
            n_v = 0
            with torch.no_grad():
                for x in val_loader:
                    x = x.to(device)
                    cond = synthesize_condition(x, cfg.sr.condition_downsample_factor)
                    _, vlog = loss_ddpm(manager, base_model, x, condition=cond)
                    val_totals["loss"] += vlog["loss"].item()
                    n_v += 1
                    if smoke:
                        break

            elapsed = time.perf_counter() - t0
            print(
                f"epoch {epoch:3d}/{epochs}"
                f"  train={totals['loss'] / n_t:.4f}"
                f"  val={val_totals['loss'] / n_v:.4f}"
                f"  t={elapsed:.1f}s",
                flush=True,
            )
            if log_wandb:
                wandb.log(
                    {
                        "epoch/train_loss": totals["loss"] / n_t,
                        "epoch/val_loss": val_totals["loss"] / n_v,
                        "epoch": epoch,
                    }
                )

            if cfg.viz.enabled and (log_wandb or cfg.save.enabled) and epoch % cfg.viz.every == 0:
                plot_dir = cfg.save.plot_dir if (cfg.save.enabled and epoch % cfg.save.plot_every == 0) else None
                log_samples(synthesize_sample_for_viz, val_ds, device, epoch, train_std, tag="val", save_dir=plot_dir)
                log_energy_spectrum(synthesize_sample_for_viz, gt_probe, device, epoch, tag="val", save_dir=plot_dir)
                log_ema_vs_base_mse(gt_probe, epoch, tag="val")

            if cfg.save.enabled and (epoch % cfg.save.ckpt_every == 0 or (smoke and epoch == epochs)):
                ckpt = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model": base_model.state_dict(),
                    "ema": ema_model.state_dict(),
                    "manager": manager.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "train_std": train_std,
                    "config": {
                        "in_channels": cfg.sr.in_channels,
                        "out_channels": cfg.sr.out_channels,
                        "latent_dims": cfg.sr.latent_dims,
                        "channel_multipliers": list(cfg.sr.channel_multipliers),
                        "num_res_blocks": cfg.sr.num_res_blocks,
                        "attention_resolutions": list(cfg.sr.attention_resolutions),
                        "image_resolution": image_resolution,
                        "dropout_rate": cfg.sr.dropout_rate,
                        "resample_with_conv": cfg.sr.resample_with_conv,
                        "condition_downsample_factor": cfg.sr.condition_downsample_factor,
                        "beta_start": cfg.sr.beta_start,
                        "beta_end": cfg.sr.beta_end,
                        "num_diffusion_time_steps": cfg.sr.num_diffusion_time_steps,
                        "inference_steps": cfg.sr.inference_steps,
                        "condition_dropout_rate": cfg.sr.condition_dropout_rate,
                        "standardize": cfg.data.standardize,
                    },
                }
                path = f"{cfg.save.ckpt_dir}/{cfg.sr.name}_sr_epoch{epoch:04d}.pt"
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
            mode="frames",
            downsample=data_downsample,
        )
        model.eval()
        test_totals = dict(loss=0.0)
        n_test = 0
        with torch.no_grad():
            for x in test_loader:
                x = x.to(device)
                cond = synthesize_condition(x, cfg.sr.condition_downsample_factor)
                _, tlog = loss_ddpm(manager, base_model, x, condition=cond)
                test_totals["loss"] += tlog["loss"].item()
                n_test += 1

        print(f"\ntest  loss={test_totals['loss'] / n_test:.4f}", flush=True)
        if log_wandb:
            wandb.log({"test/loss": test_totals["loss"] / n_test})

        if cfg.viz.enabled and (log_wandb or cfg.save.enabled):
            plot_dir = cfg.save.plot_dir if cfg.save.enabled else None
            test_ds = test_loader.dataset
            _test_probe_rng = np.random.RandomState(0)
            _test_probe_idx = sorted(
                _test_probe_rng.choice(len(test_ds), min(64, len(test_ds)), replace=False).tolist()
            )
            gt_test_probe = torch.stack([test_ds[i] for i in _test_probe_idx])
            log_samples(synthesize_sample_for_viz, test_ds, device, epochs, train_std, tag="test", save_dir=plot_dir)
            log_energy_spectrum(synthesize_sample_for_viz, gt_test_probe, device, epochs, tag="test", save_dir=plot_dir)

    if rank0 and log_wandb:
        wandb.finish()
    if dist.distributed:
        torch_dist.barrier()
        DistributedManager.cleanup()


if __name__ == "__main__":
    main()
