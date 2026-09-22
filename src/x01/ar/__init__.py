from x01.ar.diag import (
    log_encoder_manifold_residual,
    log_koopman_spectrum,
    log_latent_norm_drift,
    log_latent_trajectory_error,
)
from x01.ar.models import KoopmanAE2D, loss_koopman
from x01.ar.viz import (
    benchmark_fps,
    log_autocorr,
    log_eig_spectrum,
    log_energy_spectrum,
    log_mse_vs_frame,
    log_qq,
    log_rollout,
    log_rollout_gif,
    log_short_rollout,
)
