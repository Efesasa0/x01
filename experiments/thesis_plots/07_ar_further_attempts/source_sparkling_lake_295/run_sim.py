"""Generate KM flow dataset: runs the pseudo-spectral solver and saves snapshots."""

import os
import time

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

import wandb
from x01.sim import KMFlowSolver, log_vorticity_image, log_vorticity_video


def _energy_spectrum(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radial-binned 2D energy spectrum. frames: (T, H, W) → (k_bins, E)."""
    _, H, W = frames.shape
    kx = np.fft.fftfreq(W) * W
    ky = np.fft.fftfreq(H) * H
    KX, KY = np.meshgrid(kx, ky)
    K_rad = np.round(np.sqrt(KX**2 + KY**2)).astype(int)
    power = np.abs(np.fft.fft2(frames)) ** 2 / (H * W) ** 2
    k_max = min(H, W) // 2
    k_bins = np.arange(1, k_max + 1)
    E = np.array([power[:, K_rad == k].mean() for k in k_bins])
    return k_bins, E


@hydra.main(version_base="1.3", config_path="conf", config_name="run_sim_config")
def main(cfg: DictConfig) -> None:
    # --- smoke overrides ---
    if cfg.smoke:
        cfg.record.n_trajectories = 2
        cfg.record.n_frames = 10
        cfg.record.step_interval = 10
        cfg.record.batch_size = 2
        cfg.record.warmup_time = 0.1  # short warmup so smoke stays smoke
        print("smoke=true: 2 trajectories x 10 frames x 10 steps x 2 batches", flush=True)

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    fname = (
        f"kmflow"
        f"_re{cfg.sim.re}"
        f"_r{cfg.sim.resolution}"
        f"_b{cfg.record.n_trajectories}"
        f"_f{cfg.record.n_frames}"
        f"{'_smoke' if cfg.smoke else ''}.npy"
    )
    out_path = os.path.join(cfg.output.out_dir, fname)
    print(f"output: {out_path}", flush=True)

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
            "record_n_frames": cfg.record.n_frames,
            "record_step_interval": cfg.record.step_interval,
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
        f"  steps/frame={cfg.record.step_interval}"
        f"  total_frames={cfg.record.n_frames}",
        flush=True,
    )

    N = cfg.sim.resolution
    n_traj = cfg.record.n_trajectories
    n_frames = cfg.record.n_frames
    interval = cfg.record.step_interval
    batch_size = cfg.record.batch_size

    out = np.lib.format.open_memmap(out_path, mode="w+", dtype="float32", shape=(n_traj, n_frames, N, N))

    # process trajectories in batches — each batch runs fully in parallel on GPU
    traj_done = 0
    nan_frames: list[int] = []  # cumulative first-NaN frame index per dirty traj
    nan_seeds: list[int] = []  # cumulative seeds of dirty trajs (parallel to nan_frames)
    first_clean_idx: int | None = None  # first traj (global idx) with no NaN — for end-of-run spectrum
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

        pbar.close()
        traj_done += B
        elapsed = time.perf_counter() - t0
        print(
            f"batch {batch_start // batch_size + 1}  trajs {batch_start + 1}–{batch_start + B}/{n_traj}  t={elapsed:.1f}s",
            flush=True,
        )
        # first-NaN frame index + seed per traj in this batch (cumulative across batches)
        batch_data = out[batch_start : batch_start + B]  # (B, F, N, N)
        is_nan = np.isnan(batch_data).any(axis=(2, 3))  # (B, F)
        for b in range(B):
            if is_nan[b].any():
                nan_frames.append(int(np.argmax(is_nan[b])))
                nan_seeds.append(seeds[b])
            elif first_clean_idx is None:
                first_clean_idx = batch_start + b
        nan_table = wandb.Table(
            columns=["seed", "first_nan_frame"],
            data=[[s, f] for s, f in zip(nan_seeds, nan_frames)],
        )
        wandb.log(
            {
                "sim/traj_done": traj_done,
                "nan/first_nan_frame_hist": wandb.Histogram(nan_frames, num_bins=32) if nan_frames else None,
                "nan/total_dirty_trajs": len(nan_frames),
                "nan/dirty_trajs_table": nan_table,
            }
        )

        # log image + video for the first trajectory in this batch
        log_vorticity_image(out[batch_start, -1], traj_idx=batch_start, frame_idx=n_frames - 1)
        log_vorticity_video(out[batch_start], traj_idx=batch_start)

    out.flush()
    if cfg.output.save_dataset:
        print(f"saved: {out_path}  shape={out.shape}", flush=True)
    else:
        print(f"save_dataset=false — diagnostics only, no .npy retained", flush=True)

    # final-pass energy spectrum on the first clean trajectory (tracked during batch loop)
    if first_clean_idx is not None:
        traj = np.asarray(out[first_clean_idx])  # (F, N, N)
        k_bins, E = _energy_spectrum(traj)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.loglog(k_bins, E, color="steelblue", linewidth=1.5, label=f"traj {first_clean_idx}")
        ax.set_xlabel("k")
        ax.set_ylabel("E(k)")
        ax.set_title("energy spectrum — final clean trajectory")
        ax.legend(fontsize=9)
        fig.tight_layout()
        wandb.log({"sim/energy_spectrum": wandb.Image(fig)})
        plt.close(fig)
    else:
        print("no clean trajectory available for energy spectrum", flush=True)

    wandb.finish()

    if not cfg.output.save_dataset:
        del out
        os.remove(out_path)
        print(f"removed: {out_path}", flush=True)


if __name__ == "__main__":
    main()
