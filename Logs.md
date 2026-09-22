# Logs

Disclaimer: This document was generated with AI from my own git commit logs. I
can confirm everything here was done within the project life-cycle. It does not
include details for every experiment such as LS probe and sweeps.

## May 2026

- 2026-05-20 - Project setup with uv, physicsnemo from GitHub main, multi-platform lock for macOS CPU and Linux CUDA.
- 2026-05-23 - Ported the Koopman AE (AR module) from the Flax reference to PyTorch with a training loop and W&B logging.
- 2026-05-23 - Fixed aarch64 CUDA install on Isambard by switching torch to the cu126 wheel index.
- 2026-05-24 - Added AR eval panels: rollout, autocorrelation, energy spectrum, Q-Q. Logging eigenvalues of K.
- 2026-05-24 - Moved training config to Hydra, added seeding and a sweep script.
- 2026-05-27 - Fixed the paper loss and config groups, switched to constant LR to match the reference.
- 2026-05-28 - Added torch.compile and DDP to train_ar. Compile disabled for the FNO baseline after crashes on Isambard.
- 2026-05-29 - Fixed rank-0 val/test using the DDP wrapper instead of the inner model.
- 2026-05-30 - LR warmup scheduler, orthogonality and backward loss weights set to 0.1.
- 2026-05-31 - Skipping the first ~100 transient frames of each trajectory.

## June 2026

- 2026-06-01 - Moved to src layout. Added the Kolmogorov flow data generator under sim/.
- 2026-06-03 - Fixed trajectory blow-ups in the solver by porting the reference CN-Heun scheme, added linear drag.
- 2026-06-03 - Dataset NaN filter and plotting scripts. Split fractions now percentage based.
- 2026-06-04 - Backward dynamics can reuse a single K (transpose) instead of a separate matrix.
- 2026-06-09 - Removed dead parameters that were blocking DDP.
- 2026-06-10 - SR module: SR3-style conditional DDPM with UNet, EMA weights, on-the-fly LR conditions, DDIM sampling.
- 2026-06-25 - Pinned matplotlib below 3.11, the 3.11.0 wheel was missing backend_agg.

## July 2026

- 2026-07-04 - Cleanup pass: removed the FNO path and neuralop dependency, dead embeddings and aliases, unified make_loader for AR and SR.
- 2026-07-04 - Added rollout() on KoopmanAE2D, removed closures from train_ar.
- 2026-07-08 - Added relative LpLoss and StepLR. Replaced Frobenius orthogonality loss with a trace consistency loss.
- 2026-07-08 - Trajectory-level train/val/test splits, T_in/T windowing and optional standardisation.
- 2026-07-08 - Fixed plotters to include T_in + T frames at inference.
- 2026-07-10 - Inference is now encode -> K -> decode per step (sequential), plus GIF export.
- 2026-07-11 - Fixed AR test suite only using 4 trajectories for its statistical bands.
- 2026-07-12 - Eigenvalue distance to the unit circle for A and B, per-component parameter counts.
- 2026-07-13 - Zero-weighted loss terms are skipped. Added separate A/B orthogonality losses and progressive consistency (Azencot et al.).
- 2026-07-14 - Joint runner: AR and SR save their configs, run_joint.py chains them.
- 2026-07-14 - Fixed state_dict prefix when loading a compiled AR checkpoint.
- 2026-07-17 - Added Sobolev HsLoss as an alternative pixel loss.
- 2026-07-18 - Low-rank Koopman variant behind a dynamics_rank knob.
- 2026-07-22 - Inference mode knob: sequential vs strict koopman, for both train_ar and run_joint.
- 2026-07-25 - Latent drift, true-vs-model latent distance and singular value diagnostics in run_joint.
- 2026-07-25 - Fixed part transitions in joint koopman inference.
- 2026-07-30 - MSE pixel loss as default, multistep pixel rollout loss, latent rollout loss.

## August 2026

- 2026-08-06 - Joint plotting respects the SR validation split boundaries.
- 2026-08-15 - Started the thesis experiment folders (produce / plot per figure).
- 2026-08-18 - Thesis plots for dataset, AR failure modes, orthogonality, LSQ probe, SR and joint pipeline.
- 2026-08-20 - Three-stage AR capacity and loss coefficient sweep (alpha, beta, gamma).

## September 2026

- 2026-09-08 - Thesis submitted.
- 2026-09-22 - run_sim no longer requires a W&B login, logging is behind wandb.enabled.
- 2026-09-22 - train_ar fails early on invalid pixel/latent rollout steps instead of mid-training.
- 2026-09-22 - All datasets live in the parent dir. Smoke runs now chain: sim -> AR -> SR -> joint.
- 2026-09-22 - Plot data (.npz) for figures 03-11 ships with the repo. Stage C joint data trimmed to the 6 plotted frames.
- 2026-09-22 - Added ruff (format + lint, 120 cols). experiments/ excluded so the source snapshots stay as they ran.
