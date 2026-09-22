"""Generate KM flow dataset: runs the pseudo-spectral solver and saves snapshots."""

import os
import time

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from tqdm import tqdm

from x01.sim import KMFlowSolver, log_vorticity_image, log_vorticity_video


@hydra.main(version_base="1.3", config_path="conf", config_name="run_sim_config")
def main(cfg: DictConfig) -> None:
    # --- smoke overrides ---
    if cfg.smoke:
        cfg.record.n_trajectories = 6  # 2 train / 2 val / 2 test for the downstream smoke runs
        cfg.record.record_n_frames = 10
        cfg.record.record_step_interval = 10
        cfg.record.batch_size = 6
        cfg.record.warmup_time = 0.1  # short warmup so smoke stays smoke
        print("smoke=true: 6 trajectories x 10 frames x 10 steps", flush=True)

    log_wandb = cfg.wandb.enabled

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    fname = (
        "kmflow_smoke.npy"
        if cfg.smoke
        else (
            f"kmflow"
            f"_re{cfg.sim.re}"
            f"_r{cfg.sim.resolution}"
            f"_b{cfg.record.n_trajectories}"
            f"_f{cfg.record.record_n_frames}.npy"
        )
    )
    out_path = os.path.join(cfg.output.out_dir, fname)
    print(f"output: {out_path}", flush=True)

    if log_wandb:
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            tags=list(cfg.wandb.tags) or None,
            config={
                "re": cfg.sim.re,
                "alpha": cfg.sim.alpha,
                "tau": cfg.sim.tau,
                "dt": cfg.sim.dt,
                "resolution": cfg.sim.resolution,
                "n_trajectories": cfg.record.n_trajectories,
                "record_n_frames": cfg.record.record_n_frames,
                "record_step_interval": cfg.record.record_step_interval,
                "seed": cfg.seed,
                "smoke": cfg.smoke,
            },
        )

    solver = KMFlowSolver(
        N=cfg.sim.resolution,
        re=cfg.sim.re,
        alpha=cfg.sim.alpha,
        tau=cfg.sim.tau,
        dt=cfg.sim.dt,
        device=device,
        forcing_n=cfg.sim.forcing_n,
        drag=cfg.sim.drag,
    )
    warmup_steps = int(round(cfg.record.warmup_time / cfg.sim.dt))
    print(
        f"solver: N={cfg.sim.resolution}  re={cfg.sim.re}  dt={cfg.sim.dt}"
        f"  forcing_n={cfg.sim.forcing_n}  warmup_steps={warmup_steps}"
        f"  steps/frame={cfg.record.record_step_interval}"
        f"  total_frames={cfg.record.record_n_frames}",
        flush=True,
    )

    N = cfg.sim.resolution
    n_traj = cfg.record.n_trajectories
    n_frames = cfg.record.record_n_frames
    interval = cfg.record.record_step_interval
    batch_size = cfg.record.batch_size

    out = np.lib.format.open_memmap(out_path, mode="w+", dtype="float32", shape=(n_traj, n_frames, N, N))

    # process trajectories in batches — each batch runs fully in parallel on GPU
    traj_done = 0
    for batch_start in range(0, n_traj, batch_size):
        seeds = list(range(cfg.seed + batch_start, cfg.seed + min(batch_start + batch_size, n_traj)))
        B = len(seeds)
        solver.reset(seeds)
        t0 = time.perf_counter()

        # warmup: advance to statistical stationarity before recording starts
        if warmup_steps > 0:
            wpbar = tqdm(
                total=warmup_steps, desc=f"batch {batch_start // batch_size + 1} warmup", unit="step", leave=False
            )
            for _ in range(warmup_steps):
                solver.step()
                wpbar.update(1)
            wpbar.close()

        total_steps = n_frames * interval
        pbar = tqdm(total=total_steps, desc=f"batch {batch_start // batch_size + 1}", unit="step", leave=False)

        for frame in range(n_frames):
            for _ in range(interval):
                solver.step()
                pbar.update(1)
            pbar.set_postfix(frame=frame)

            omega = solver.get_vorticity()  # (B, N, N)
            out[batch_start : batch_start + B, frame] = omega.cpu().numpy()

            # log mean stats across the batch
            energy = 0.5 * omega.pow(2).mean().item()
            enstrophy = omega.pow(2).mean().item()
            if log_wandb:
                wandb.log(
                    {
                        "sim/energy": energy,
                        "sim/enstrophy": enstrophy,
                        "sim/frame": frame,
                        "sim/batch": batch_start // batch_size,
                    }
                )

        pbar.close()
        traj_done += B
        elapsed = time.perf_counter() - t0
        print(
            f"batch {batch_start // batch_size + 1}  "
            f"trajs {batch_start + 1}–{batch_start + B}/{n_traj}  t={elapsed:.1f}s",
            flush=True,
        )
        if log_wandb:
            wandb.log({"sim/traj_done": traj_done})
            # log image + video for the first trajectory in this batch
            log_vorticity_image(out[batch_start, -1], traj_idx=batch_start, frame_idx=n_frames - 1)
            log_vorticity_video(out[batch_start], traj_idx=batch_start)

    out.flush()
    print(f"saved: {out_path}  shape={out.shape}", flush=True)

    if log_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
