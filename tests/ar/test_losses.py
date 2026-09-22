from typing import Any

import pytest
import torch

from x01.ar.models.hs_loss import HsLoss
from x01.ar.models.koopman_ae_2d import KoopmanAE2D
from x01.ar.models.loss_koopman import loss_koopman
from x01.ar.models.lp_loss import LpLoss
from x01.ar.models.mse_loss import MseLoss

LOSS_KEYS = {
    "loss",
    "forward_loss",
    "backward_loss",
    "reconstruction_loss",
    "latent_fwd_loss",
    "latent_bwd_loss",
    "consistency_loss",
    "ortho_a_loss",
    "ortho_b_loss",
    "consist_progressive_loss",
}


@pytest.fixture
def xy():
    torch.manual_seed(0)
    return torch.randn(4, 1, 16, 16), torch.randn(4, 1, 16, 16)


# pixel losses


@pytest.mark.parametrize("loss_fn", [LpLoss(), HsLoss(), MseLoss()])
def test_pixel_loss_zero_on_identical_inputs(loss_fn, xy):
    x, _ = xy
    assert loss_fn(x, x).item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("loss_fn", [LpLoss(), HsLoss(), MseLoss()])
def test_pixel_loss_positive_on_different_inputs(loss_fn, xy):
    x, y = xy
    assert loss_fn(x, y).item() > 0.0


@pytest.mark.parametrize("loss_fn", [LpLoss(), HsLoss()])
def test_relative_losses_are_scale_invariant(loss_fn, xy):
    x, y = xy
    torch.testing.assert_close(loss_fn(3.0 * x, 3.0 * y), loss_fn(x, y))


def test_mse_matches_torch(xy):
    x, y = xy
    torch.testing.assert_close(MseLoss()(x, y), torch.nn.functional.mse_loss(x, y))


def test_size_average_sums_over_batch(xy):
    x, y = xy
    torch.testing.assert_close(LpLoss(size_average=False)(x, y), 4 * LpLoss(size_average=True)(x, y))


def test_hs_loss_rejects_non_4d():
    with pytest.raises(AssertionError):
        HsLoss()(torch.randn(4, 16, 16), torch.randn(4, 16, 16))


# loss_koopman


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return KoopmanAE2D(in_channels=1, out_channels=1, dims=4, steps=2)


@pytest.fixture
def batch():
    torch.manual_seed(1)
    return torch.randn(2, 1, 64, 64), torch.randn(2, 2, 1, 64, 64)


ALL_TERMS: dict[str, Any] = dict(
    lambda_latent_fwd=0.5,
    lambda_latent_bwd=0.5,
    lambda_ortho_a=0.1,
    lambda_ortho_b=0.1,
    lambda_consist_progressive=0.01,
    latent_rollout_steps=2,
)


@pytest.mark.parametrize("loss_mode", ["lp", "hs", "mse"])
def test_loss_koopman_keys_and_finite(model, batch, loss_mode):
    x, y = batch
    loss, log = loss_koopman(model, x, y, loss_mode=loss_mode, **ALL_TERMS)
    assert set(log) == LOSS_KEYS
    assert all(torch.isfinite(v) for v in log.values())
    assert log["loss"] is loss


def test_loss_koopman_is_weighted_sum(model, batch):
    x, y = batch
    terms = {
        "lambda_fwd": ("forward_loss", 1.0),
        "lambda_recon": ("reconstruction_loss", 0.7),
        "lambda_consist": ("consistency_loss", 0.1),
        "lambda_bwd": ("backward_loss", 0.2),
        "lambda_latent_fwd": ("latent_fwd_loss", 0.5),
        "lambda_latent_bwd": ("latent_bwd_loss", 0.3),
        "lambda_ortho_a": ("ortho_a_loss", 0.1),
        "lambda_ortho_b": ("ortho_b_loss", 0.05),
        "lambda_consist_progressive": ("consist_progressive_loss", 0.01),
    }
    weights: dict[str, Any] = {arg: w for arg, (_, w) in terms.items()}
    loss, log = loss_koopman(model, x, y, latent_rollout_steps=2, **weights)
    expected = sum(w * log[key] for key, w in terms.values())
    torch.testing.assert_close(loss, expected)


def test_disabled_terms_are_zero(model, batch):
    x, y = batch
    _, log = loss_koopman(model, x, y, lambda_recon=0.0, lambda_bwd=0.0, lambda_consist=0.0)
    for name in LOSS_KEYS - {"loss", "forward_loss"}:
        assert log[name].item() == 0.0


def test_ortho_loss_zero_at_orthogonal_init(batch):
    x, y = batch
    model = KoopmanAE2D(in_channels=1, out_channels=1, steps=2)
    _, log = loss_koopman(model, x, y, lambda_ortho_a=1.0, lambda_ortho_b=1.0)
    assert log["ortho_a_loss"].item() == pytest.approx(0.0, abs=1e-8)
    assert log["ortho_b_loss"].item() == pytest.approx(0.0, abs=1e-8)


def test_steps_must_match_rollout_targets(model, batch):
    x, y = batch
    with pytest.raises(AssertionError):
        loss_koopman(model, x, y[:, :1])


def test_unknown_loss_mode(model, batch):
    x, y = batch
    with pytest.raises(ValueError):
        loss_koopman(model, x, y, loss_mode="l1")


def test_gradients_reach_both_operators(batch):
    x, y = batch
    model = KoopmanAE2D(in_channels=1, out_channels=1, steps=2)
    loss, _ = loss_koopman(model, x, y)
    loss.backward()
    assert model.dynamics.dynamics.grad is not None
    assert model.back_dynamics.dynamics.grad is not None
    assert model.encoder[0].projection.weight.grad is not None
