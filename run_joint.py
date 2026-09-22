import os
import subprocess
import time
from datetime import datetime

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf

from x01.ar import (
    KoopmanAE2D,
    log_autocorr,
    log_energy_spectrum,
    log_koopman_spectrum,
    log_latent_norm_drift,
    log_latent_trajectory_error,
    log_qq,
    log_rollout_gif,
)
from x01.sr import DiffusionManager, UNet


def _strip_compile_prefix(state_dict):
    """torch.compile wraps modules and prefixes their state_dict keys with '_orig_mod.'.
    Strip so we can load into an uncompiled model. No-op if the prefix isn't there."""
    return {k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k: v for k, v in state_dict.items()}


def _load_ar(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ckpt["config"]
    model = KoopmanAE2D(
        in_channels=1,
        out_channels=1,
        dims=c["dims"],
        num_blocks=tuple(c["num_blocks"]),
        num_heads=tuple(c["num_heads"]),
        dynamics_mode=c["dynamics_mode"],
        dynamics_rank=c.get("dynamics_rank"),
    ).to(device)
    model.load_state_dict(_strip_compile_prefix(ckpt["model"]))
    model.eval()
    return model, bool(c["standardize"]), float(ckpt["train_std"])


def _load_sr(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ckpt["config"]
    model = UNet(
        in_channels=c["in_channels"],
        out_channels=c["out_channels"],
        latent_dims=c["latent_dims"],
        channel_multipliers=tuple(c["channel_multipliers"]),
        num_res_blocks=c["num_res_blocks"],
        attention_resolutions=tuple(c["attention_resolutions"]),
        image_resolution=c["image_resolution"],
        dropout_rate=c["dropout_rate"],
        resample_with_conv=c["resample_with_conv"],
    ).to(device)
    state = ckpt.get("ema", ckpt["model"])
    model.load_state_dict(_strip_compile_prefix(state))
    model.eval()

    manager = DiffusionManager(
        beta_start=c["beta_start"],
        beta_end=c["beta_end"],
        num_diffusion_time_steps=c["num_diffusion_time_steps"],
        condition_dropout_rate=c["condition_dropout_rate"],
    ).to(device)
    manager.load_state_dict(ckpt["manager"])
    manager.eval()

    return (
        model,
        manager,
        int(c["condition_downsample_factor"]),
        int(c["inference_steps"]),
        bool(c["standardize"]),
        float(ckpt["train_std"]),
        int(c["image_resolution"]),
    )


def _sr_batch_refine(
    sr_model,
    manager,
    lr_frames: torch.Tensor,  # [N, 1, H_lr, W_lr] on cpu, raw scale
    upsample_factor: int,
    inference_steps: int,
    batch_size: int,
    sr_standardize: bool,
    sr_std: float,
    device: torch.device,
) -> torch.Tensor:
    """Upsample nearest -> per-batch SR sample. Returns HR frames [N, 1, H, W] on cpu, raw scale."""
    cond_hr = F.interpolate(lr_frames.to(device), scale_factor=upsample_factor, mode="nearest")
    if sr_standardize:
        cond_hr = cond_hr / sr_std

    out = []
    with torch.no_grad():
        for i in range(0, cond_hr.shape[0], batch_size):
            chunk = cond_hr[i : i + batch_size]
            x_noise = torch.randn_like(chunk)
            sample = manager.infer_fn(x_noise, sr_model, inference_steps=inference_steps, condition=chunk, pbar=False)
            out.append(sample.cpu())
    hr = torch.cat(out, dim=0)
    if sr_standardize:
        hr = hr * sr_std
    return hr


def _log_snapshot_row(
    hr_frames: torch.Tensor,  # [T, 1, H, W]
    timestamps: list,
    tag: str,
    save_dir: str | None,
) -> None:
    """Plot HR frames at each requested timestamp side-by-side; log to wandb / disk."""
    imgs = [hr_frames[t, 0].numpy() for t in timestamps]
    vmin = float(min(x.min() for x in imgs))
    vmax = float(max(x.max() for x in imgs))
    n = len(timestamps)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2))
    if n == 1:
        axes = [axes]
    for ax, img, t in zip(axes, imgs, timestamps, strict=True):
        ax.imshow(img, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(f"t={t}", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{tag} snapshots", fontsize=10)
    fig.tight_layout()
    if wandb.run is not None:
        wandb.log({f"{tag}/snapshots": wandb.Image(fig)})
    if save_dir is not None:
        fig.savefig(f"{save_dir}/{tag}_snapshots.png", dpi=120)
    plt.close(fig)


def _ar_rollout_lr(
    ar_model,
    x0_lr: torch.Tensor,  # [1, 1, H_lr, W_lr] raw scale, cpu ok
    steps: int,
    ar_standardize: bool,
    ar_std: float,
    device: torch.device,
    inference_mode: str,
) -> torch.Tensor:
    """AR rollout in LR space. Returns predicted LR frames [steps, 1, H_lr, W_lr] on cpu, raw scale."""
    x = x0_lr.to(device)
    if ar_standardize:
        x = x / ar_std
    with torch.no_grad():
        pred = ar_model.rollout(x, steps, inference_mode=inference_mode)
    if ar_standardize:
        pred = pred * ar_std
    return pred


def _ar_encode_lr(
    ar_model,
    x0_lr: torch.Tensor,
    ar_standardize: bool,
    ar_std: float,
    device: torch.device,
) -> torch.Tensor:
    """Encode a single LR frame into the model's latent (standardization aware)."""
    x = x0_lr.to(device)
    if ar_standardize:
        x = x / ar_std
    with torch.no_grad():
        return ar_model.encode(x)


def _ar_rollout_lr_from_z(
    ar_model,
    z0: torch.Tensor,
    steps: int,
    ar_standardize: bool,
    ar_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Strict Koopman continuation from a latent seed. Returns (LR frames raw scale, z_final on device)."""
    with torch.no_grad():
        pred, z_final = ar_model.rollout_from_z(z0, steps)
    if ar_standardize:
        pred = pred * ar_std
    return pred, z_final


@hydra.main(version_base="1.3", config_path="conf", config_name="run_joint_config")
def main(cfg: DictConfig) -> None:
    if cfg.smoke:
        cfg.data.path = cfg.data.smoke_path
        cfg.data.test_size = 2
        cfg.pipeline.part1_frames = 2
        cfg.pipeline.part2_frames = 0
        cfg.checkpoints.ar_path = cfg.checkpoints.ar_path or cfg.checkpoints.smoke_ar_path
        cfg.checkpoints.sr_path = cfg.checkpoints.sr_path or cfg.checkpoints.smoke_sr_path
        print("smoke=true: 2 test trajectories x 2 frames, part 1 only", flush=True)

    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_wandb = cfg.wandb.enabled
    inference_mode = str(cfg.inference_mode)

    try:
        branch = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        branch = "?"
    print(f"run env: branch={branch}  cwd={os.getcwd()}", flush=True)
    print(f"device : {device}", flush=True)
    print(f"inference_mode: {inference_mode}", flush=True)

    plot_dir = None
    if cfg.save.enabled:
        iso = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        branch_slug = branch.replace("/", "-")
        parts = [iso, branch_slug]
        if cfg.save.run_tag:
            parts.append(cfg.save.run_tag)
        run_id = "__".join(parts)
        plot_dir = os.path.join(cfg.save.plot_dir, run_id)
        os.makedirs(plot_dir, exist_ok=True)
        print(f"save   : plots -> {plot_dir}", flush=True)

    if log_wandb:
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=list(cfg.wandb.tags) or None,
        )

    assert cfg.checkpoints.ar_path, "checkpoints.ar_path must be set"
    assert cfg.checkpoints.sr_path, "checkpoints.sr_path must be set"
    print(f"loading AR from {cfg.checkpoints.ar_path}", flush=True)
    ar_model, ar_standardize, ar_std = _load_ar(cfg.checkpoints.ar_path, device)
    print(f"loading SR from {cfg.checkpoints.sr_path}", flush=True)
    (
        sr_model,
        manager,
        condition_downsample_factor,
        inference_steps,
        sr_standardize,
        sr_std,
        hr_resolution,
    ) = _load_sr(cfg.checkpoints.sr_path, device)

    lr_resolution = hr_resolution // condition_downsample_factor
    print(f"AR standardize={ar_standardize} train_std={ar_std:.4f}", flush=True)
    print(f"SR standardize={sr_standardize} train_std={sr_std:.4f}", flush=True)
    print(f"HR={hr_resolution}  LR={lr_resolution}  factor={condition_downsample_factor}", flush=True)

    full = np.load(cfg.data.path, mmap_mode="r")
    test_size = int(cfg.data.test_size)
    assert 0 < test_size <= full.shape[0], f"test_size {test_size} out of range [1, {full.shape[0]}]"
    test_start = full.shape[0] - test_size
    test_idxs = list(range(test_start, test_start + test_size))
    print(f"test trajectories: idx {test_start}..{test_idxs[-1]} (n={test_size})", flush=True)
    viz_j = test_size - 1
    viz_idx = test_idxs[viz_j]
    print(f"visualized trajectory: idx {viz_idx} frame 0 seed", flush=True)

    hr_check = np.array(full[test_idxs[0], 0], dtype=np.float32)
    assert hr_check.shape[0] == hr_resolution, f"dataset HR {hr_check.shape[0]} != SR image_resolution {hr_resolution}"

    x0_lr_list = []
    gt_hr_list = []
    for idx in test_idxs:
        hr_traj = np.array(full[idx], dtype=np.float32)  # (T_data, H, W)
        lr_frame0 = hr_traj[0, ::condition_downsample_factor, ::condition_downsample_factor]
        x0_lr_list.append(torch.from_numpy(lr_frame0).unsqueeze(0).unsqueeze(0))
        gt_hr_list.append(torch.from_numpy(hr_traj).unsqueeze(1))  # [T_data, 1, H, W]

    gt_lr_viz = gt_hr_list[viz_j][
        :, :, ::condition_downsample_factor, ::condition_downsample_factor
    ]  # [T_data, 1, H_lr, W_lr]
    gt_T = gt_lr_viz.shape[0]

    def _wrap_precomputed_list(preds_list):
        """Stateful callable that returns preds_list[i] on the i-th call, matching gt iteration order."""
        i = [0]

        def fn(x0, n_pred):
            pred = preds_list[i[0]][:n_pred]
            i[0] += 1
            return pred

        return fn

    def _sr_and_log(lr_pred_stage_list, n_steps, tag_stage, ar_elapsed):
        t_sr = time.perf_counter()
        lr_all = torch.cat(lr_pred_stage_list, dim=0)
        hr_all = _sr_batch_refine(
            sr_model,
            manager,
            lr_all,
            upsample_factor=condition_downsample_factor,
            inference_steps=inference_steps,
            batch_size=int(cfg.pipeline.sr_batch_size),
            sr_standardize=sr_standardize,
            sr_std=sr_std,
            device=device,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sr_elapsed = time.perf_counter() - t_sr

        hr_pred_stage_list = [hr_all[j * n_steps : (j + 1) * n_steps] for j in range(test_size)]

        total_frames = n_steps * test_size
        print(
            f"{tag_stage}: AR {ar_elapsed:.2f}s ({total_frames / ar_elapsed:.1f} fps)  "
            f"SR {sr_elapsed:.2f}s ({total_frames / sr_elapsed:.1f} fps)",
            flush=True,
        )
        if log_wandb:
            wandb.log(
                {
                    f"{tag_stage}/ar_seconds": ar_elapsed,
                    f"{tag_stage}/sr_seconds": sr_elapsed,
                    f"{tag_stage}/ar_fps": total_frames / ar_elapsed,
                    f"{tag_stage}/sr_fps": total_frames / sr_elapsed,
                }
            )
        return hr_pred_stage_list

    def _run_multi_traj_stage(x0_lr_stage_list, n_steps, tag_stage):
        # Sequential path: rollout in pixel space per traj (re-encode each step is baked into ar_model.rollout).
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_ar = time.perf_counter()
        lr_pred_stage_list = [
            _ar_rollout_lr(ar_model, x0_lr_stage_list[j], n_steps, ar_standardize, ar_std, device, inference_mode)
            for j in range(test_size)
        ]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ar_elapsed = time.perf_counter() - t_ar
        hr_pred_stage_list = _sr_and_log(lr_pred_stage_list, n_steps, tag_stage, ar_elapsed)
        return lr_pred_stage_list, hr_pred_stage_list

    def _run_multi_traj_stage_koopman(z_seed_list, n_steps, tag_stage):
        # Strict Koopman path: continue each traj's latent chain, no re-encoding at the boundary.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_ar = time.perf_counter()
        lr_pred_stage_list = []
        z_final_list = []
        for j in range(test_size):
            pred, z_end = _ar_rollout_lr_from_z(ar_model, z_seed_list[j], n_steps, ar_standardize, ar_std)
            lr_pred_stage_list.append(pred)
            z_final_list.append(z_end)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ar_elapsed = time.perf_counter() - t_ar
        hr_pred_stage_list = _sr_and_log(lr_pred_stage_list, n_steps, tag_stage, ar_elapsed)
        return lr_pred_stage_list, hr_pred_stage_list, z_final_list

    log_koopman_spectrum(ar_model)

    # Part 1: N1 AR predictions per traj (init is used only to seed; not counted as a prediction)
    N1 = int(cfg.pipeline.part1_frames)
    print(f"\n=== Part 1: {N1} predictions x {test_size} trajs ===", flush=True)

    z_state_list = None  # populated only in koopman mode; threaded across parts to skip re-encoding
    if inference_mode == "koopman":
        z_state_list = [
            _ar_encode_lr(ar_model, x0_lr_list[j], ar_standardize, ar_std, device) for j in range(test_size)
        ]
        part1_z_seed = z_state_list[viz_j]

    t_start = time.perf_counter()
    if inference_mode == "koopman":
        lr_pred_1_list, hr_pred_1_list, z_state_list = _run_multi_traj_stage_koopman(z_state_list, N1, "part1")
    else:
        lr_pred_1_list, hr_pred_1_list = _run_multi_traj_stage(x0_lr_list, N1, "part1")
    t_elapsed_1 = time.perf_counter() - t_start
    total_preds_1 = N1 * test_size
    fps_1 = total_preds_1 / t_elapsed_1
    print(f"part 1: {total_preds_1} predictions in {t_elapsed_1:.2f}s -> {fps_1:.2f} fps", flush=True)
    if log_wandb:
        wandb.log({"part1/fps": fps_1, "part1/frames": total_preds_1, "part1/seconds": t_elapsed_1})

    if cfg.viz.enabled:
        gt_part1_list = [gt[:N1] for gt in gt_hr_list]
        log_rollout_gif(
            _wrap_precomputed_list([hr_pred_1_list[viz_j]]),
            gt_part1_list[viz_j],
            epoch=0,
            std=1.0,
            tag="part1",
            save_dir=plot_dir,
            lr_pred_frames=lr_pred_1_list[viz_j],
        )
        log_autocorr(_wrap_precomputed_list(hr_pred_1_list), gt_part1_list, epoch=0, tag="part1", save_dir=plot_dir)
        log_energy_spectrum(
            _wrap_precomputed_list(hr_pred_1_list), gt_part1_list, epoch=0, tag="part1", save_dir=plot_dir
        )
        log_qq(_wrap_precomputed_list(hr_pred_1_list), gt_part1_list, epoch=0, tag="part1", save_dir=plot_dir)

    if inference_mode == "koopman":
        gt_slice_1 = gt_lr_viz[: N1 + 1] if gt_T >= N1 + 1 else None
        log_latent_norm_drift(
            ar_model,
            part1_z_seed,
            N1,
            tag="part1",
            save_dir=plot_dir,
            gt_lr_frames=gt_slice_1,
            ar_standardize=ar_standardize,
            ar_std=ar_std,
            device=device,
        )
        if gt_slice_1 is not None:
            log_latent_trajectory_error(
                ar_model,
                part1_z_seed,
                gt_slice_1,
                ar_standardize,
                ar_std,
                device,
                tag="part1",
                save_dir=plot_dir,
            )
        else:
            print(f"part1 latent_trajectory_error skipped: GT covers {gt_T} steps, need {N1 + 1}", flush=True)

    print("Part 1 done.", flush=True)

    # Part 2: continue to N2 total frames
    N2 = int(cfg.pipeline.part2_frames)
    if N2 == 0:
        print("\n=== Part 2: skipped (part2_frames=0) ===", flush=True)
        print("=== Part 3: skipped (requires part2 output) ===", flush=True)
        if log_wandb:
            wandb.finish()
        return
    assert N2 > N1, f"part2_frames ({N2}) must exceed part1_frames ({N1})"
    ar_steps_2 = N2 - N1
    print(f"\n=== Part 2: extending to {N2} frames ({ar_steps_2} new) x {test_size} trajs ===", flush=True)

    if inference_mode == "koopman":
        part2_z_seed = z_state_list[viz_j]
    else:
        x0_lr_part2_list = [lr_pred_1_list[j][-1:] for j in range(test_size)]

    t_start = time.perf_counter()
    if inference_mode == "koopman":
        lr_pred_2_list, hr_pred_2_list, z_state_list = _run_multi_traj_stage_koopman(z_state_list, ar_steps_2, "part2")
    else:
        lr_pred_2_list, hr_pred_2_list = _run_multi_traj_stage(x0_lr_part2_list, ar_steps_2, "part2")
    t_elapsed_2 = time.perf_counter() - t_start
    total_preds_2 = ar_steps_2 * test_size
    fps_2 = total_preds_2 / t_elapsed_2
    print(f"part 2: {total_preds_2} new predictions in {t_elapsed_2:.2f}s -> {fps_2:.2f} fps", flush=True)
    if log_wandb:
        wandb.log({"part2/fps": fps_2, "part2/frames": total_preds_2, "part2/seconds": t_elapsed_2})

    hr_all_2_list = [torch.cat([hr_pred_1_list[j], hr_pred_2_list[j]], dim=0) for j in range(test_size)]
    lr_all_2_viz = torch.cat([lr_pred_1_list[viz_j], lr_pred_2_list[viz_j]], dim=0)

    if cfg.viz.enabled:
        gt_part2_list = [gt[:N2] for gt in gt_hr_list]
        log_rollout_gif(
            _wrap_precomputed_list([hr_all_2_list[viz_j]]),
            gt_part2_list[viz_j],
            epoch=0,
            std=1.0,
            tag="part2",
            save_dir=plot_dir,
            lr_pred_frames=lr_all_2_viz,
        )
        log_autocorr(_wrap_precomputed_list(hr_all_2_list), gt_part2_list, epoch=0, tag="part2", save_dir=plot_dir)
        log_energy_spectrum(
            _wrap_precomputed_list(hr_all_2_list), gt_part2_list, epoch=0, tag="part2", save_dir=plot_dir
        )
        log_qq(_wrap_precomputed_list(hr_all_2_list), gt_part2_list, epoch=0, tag="part2", save_dir=plot_dir)

    if inference_mode == "koopman":
        gt_slice_2 = gt_lr_viz[N1 : N2 + 1] if gt_T >= N2 + 1 else None
        log_latent_norm_drift(
            ar_model,
            part2_z_seed,
            ar_steps_2,
            tag="part2",
            save_dir=plot_dir,
            gt_lr_frames=gt_slice_2,
            ar_standardize=ar_standardize,
            ar_std=ar_std,
            device=device,
        )
        if gt_slice_2 is not None:
            log_latent_trajectory_error(
                ar_model,
                part2_z_seed,
                gt_slice_2,
                ar_standardize,
                ar_std,
                device,
                tag="part2",
                save_dir=plot_dir,
            )
        else:
            print(f"part2 latent_trajectory_error skipped: GT covers {gt_T} steps, need {N2 + 1}", flush=True)

    print("Part 2 done.", flush=True)

    # Part 3: single-traj only (snapshots use the first traj; multi-traj rollout at 1024 frames is too expensive).
    N3 = int(cfg.pipeline.part3_frames)
    if N3 == 0:
        print("\n=== Part 3: skipped (part3_frames=0) ===", flush=True)
        if log_wandb:
            wandb.finish()
        return
    assert N3 > N2, f"part3_frames ({N3}) must exceed part2_frames ({N2})"
    ar_steps_3 = N3 - N2
    print(f"\n=== Part 3: extending to {N3} frames ({ar_steps_3} new) x 1 traj ===", flush=True)

    if inference_mode == "koopman":
        part3_z_seed = z_state_list[viz_j]
    else:
        x0_lr_part3 = lr_pred_2_list[viz_j][-1:]

    t_start = time.perf_counter()
    if inference_mode == "koopman":
        lr_pred_3, _ = _ar_rollout_lr_from_z(ar_model, part3_z_seed, ar_steps_3, ar_standardize, ar_std)
    else:
        lr_pred_3 = _ar_rollout_lr(ar_model, x0_lr_part3, ar_steps_3, ar_standardize, ar_std, device, inference_mode)
    hr_pred_3 = _sr_batch_refine(
        sr_model,
        manager,
        lr_pred_3,
        upsample_factor=condition_downsample_factor,
        inference_steps=inference_steps,
        batch_size=int(cfg.pipeline.sr_batch_size),
        sr_standardize=sr_standardize,
        sr_std=sr_std,
        device=device,
    )
    t_elapsed_3 = time.perf_counter() - t_start
    fps_3 = ar_steps_3 / t_elapsed_3
    print(f"part 3: {ar_steps_3} new predictions in {t_elapsed_3:.2f}s -> {fps_3:.2f} fps", flush=True)
    if log_wandb:
        wandb.log({"part3/fps": fps_3, "part3/frames": ar_steps_3, "part3/seconds": t_elapsed_3})

    if cfg.viz.enabled:
        snapshot_ts = [0, 200, 400, 600, 800, 1000]
        for t in snapshot_ts:
            assert 0 <= t < N3, f"snapshot t={t} out of range [0, {N3})"
        hr_all_3_viz = torch.cat([hr_all_2_list[viz_j], hr_pred_3], dim=0)  # [N3, 1, H, W]
        _log_snapshot_row(hr_all_3_viz, snapshot_ts, tag="part3", save_dir=plot_dir)

    if inference_mode == "koopman":
        gt_slice_3 = gt_lr_viz[N2 : N3 + 1] if gt_T >= N3 + 1 else None
        log_latent_norm_drift(
            ar_model,
            part3_z_seed,
            ar_steps_3,
            tag="part3",
            save_dir=plot_dir,
            gt_lr_frames=gt_slice_3,
            ar_standardize=ar_standardize,
            ar_std=ar_std,
            device=device,
        )
        if gt_slice_3 is not None:
            log_latent_trajectory_error(
                ar_model,
                part3_z_seed,
                gt_slice_3,
                ar_standardize,
                ar_std,
                device,
                tag="part3",
                save_dir=plot_dir,
            )
        else:
            print(f"part3 latent_trajectory_error skipped: GT covers {gt_T} steps, need {N3 + 1}", flush=True)

    print("Part 3 done.", flush=True)

    if log_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
