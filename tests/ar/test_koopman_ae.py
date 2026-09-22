import pytest
import torch

from x01.ar.models.koopman_ae_2d import KoopmanAE2D

# Default schedule (2, 4, 4) reduces 64x64 inputs to a 2x2 latent grid.
B, H, W = 2, 64, 64


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return KoopmanAE2D(in_channels=1, out_channels=1)


@pytest.fixture(scope="module")
def x():
    torch.manual_seed(1)
    return torch.randn(B, 1, H, W)


def test_latent_size(model):
    assert model.latent_dim == 32
    assert model.dynamics.matrix.shape == (128, 128)


@pytest.mark.parametrize("mode", ["forward", "backward"])
def test_forward_backward_shapes(model, x, mode):
    out, out_id = model(x, mode=mode)
    assert len(out) == 1 and out[0].shape == (B, 1, H, W)
    assert len(out_id) == 1 and out_id[0].shape == (B, 1, H, W)


def test_return_identity_false(model, x):
    _, out_id = model(x, mode="forward", return_identity=False)
    assert out_id == []


def test_steps_controls_prediction_count(x):
    model = KoopmanAE2D(in_channels=1, out_channels=1, steps=3)
    out, _ = model(x, mode="forward")
    assert len(out) == 3


def test_invalid_mode(model, x):
    with pytest.raises(ValueError):
        model(x, mode="sideways")


def test_encode_decode_shapes(model, x):
    z = model.encode(x)
    assert z.shape == (B, 128)
    assert model.encode(x, mode="forward").shape == (B, 128)
    assert model.encode(x, mode="backward").shape == (B, 128)
    assert model.decode(z).shape == (B, 1, H, W)


def test_forward_equals_decode_of_advanced_latent(model, x):
    with torch.no_grad():
        out, _ = model(x, mode="forward", return_identity=False)
        expected = model.decode(model.encode(x, mode="forward"))
    torch.testing.assert_close(out[0], expected)


def test_grid_axes(model):
    grid = model._get_grid(8, 8, 1, torch.device("cpu"))
    gridx, gridy = grid[0, 0], grid[0, 1]
    assert torch.allclose(gridx[:, 0], gridx[:, 1]) and not torch.allclose(gridx[0], gridx[1])
    assert torch.allclose(gridy[0], gridy[1]) and not torch.allclose(gridy[:, 0], gridy[:, 1])


@pytest.mark.parametrize("inference_mode", ["sequential", "koopman"])
def test_rollout_shape(model, x, inference_mode):
    frames = model.rollout(x[:1], T=4, inference_mode=inference_mode)
    assert frames.shape == (4, 1, H, W)
    assert torch.isfinite(frames).all()


def test_rollout_invalid_mode(model, x):
    with pytest.raises(ValueError):
        model.rollout(x[:1], T=2, inference_mode="magic")


def test_koopman_rollout_matches_rollout_from_z(model, x):
    frames = model.rollout(x[:1], T=3, inference_mode="koopman")
    with torch.no_grad():
        z0 = model.encode(x[:1])
    frames_z, z_final = model.rollout_from_z(z0, T=3)
    torch.testing.assert_close(frames, frames_z)
    assert z_final.shape == (1, 128)


def test_low_rank_variant(x):
    model = KoopmanAE2D(in_channels=1, out_channels=1, dynamics_rank=16)
    assert model.dynamics.matrix.shape == (16, 16)
    out, _ = model(x, mode="backward")
    assert out[0].shape == (B, 1, H, W)
