import pytest
import torch

from ar.models.koopman_ae_2d import KoopmanAE2D
from ar.models.loss_fn import loss_koopman


@pytest.fixture(scope="module")
def model():
    return KoopmanAE2D(in_channels=1, out_channels=1)


@pytest.fixture(scope="module")
def xy():
    torch.manual_seed(0)
    return torch.randn(2, 1, 64, 64), torch.randn(2, 1, 64, 64)


# --- KoopmanAE2D ---


def test_forward_shape(model, xy):
    x, _ = xy
    out, out_id = model(x, mode="forward")
    assert out[0].shape == (2, 1, 64, 64)
    assert out_id[0].shape == (2, 1, 64, 64)


def test_backward_shape(model, xy):
    _, y = xy
    out, out_id = model(y, mode="backward")
    assert out[0].shape == (2, 1, 64, 64)
    assert out_id[0].shape == (2, 1, 64, 64)


def test_steps_length(xy):
    x, _ = xy
    m = KoopmanAE2D(in_channels=1, out_channels=1, steps=3)
    out, out_id = m(x, mode="forward")
    assert len(out) == 3
    assert len(out_id) == 1


def test_encode_shape(model, xy):
    x, _ = xy
    latent_size = model.latent_dim * 2 * 2
    assert model.encode(x).shape == (2, latent_size)
    assert model.encode(x, mode="forward").shape == (2, latent_size)
    assert model.encode(x, mode="backward").shape == (2, latent_size)


def test_decode_shape(model, xy):
    x, _ = xy
    z = model.encode(x)
    assert model.decode(z).shape == (2, 1, 64, 64)


def test_grid_axes(model):
    # gridx must vary along H (rows), gridy along W (columns) — the Flax reference had a bug
    # where both varied along rows.
    grid = model._get_grid(8, 8, 1, torch.device("cpu"))
    gridx = grid[0, 0]  # (H, W)
    gridy = grid[0, 1]  # (H, W)

    assert not torch.allclose(gridx[0], gridx[1]), "gridx should differ across rows"
    assert torch.allclose(gridx[:, 0], gridx[:, 1]), "gridx should be constant across columns"

    assert not torch.allclose(gridy[:, 0], gridy[:, 1]), "gridy should differ across columns"
    assert torch.allclose(gridy[0], gridy[1]), "gridy should be constant across rows"


def test_invalid_mode(model, xy):
    x, _ = xy
    with pytest.raises(ValueError):
        model(x, mode="invalid")


# --- loss_koopman ---


def test_loss_keys(model, xy):
    x, y = xy
    _, log = loss_koopman(model, x, y)
    assert set(log.keys()) == {"loss", "forward_loss", "backward_loss", "reconstruction_loss", "consist_loss"}


def test_loss_equals_sum(model, xy):
    x, y = xy
    loss, log = loss_koopman(model, x, y)
    expected = log["forward_loss"] + log["backward_loss"] + log["reconstruction_loss"] + log["consist_loss"]
    assert abs(loss.item() - expected.item()) < 1e-5


def test_loss_finite(model, xy):
    x, y = xy
    loss, log = loss_koopman(model, x, y)
    assert torch.isfinite(loss), "total loss is not finite"
    for k, v in log.items():
        assert torch.isfinite(v), f"{k} is not finite"


def test_loss_gradients(model, xy):
    x, y = xy
    loss, _ = loss_koopman(model, x, y)
    loss.backward()
    assert model.dynamics.dynamics.grad is not None
    assert model.back_dynamics.back_dynamics.grad is not None
