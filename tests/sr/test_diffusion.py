import numpy as np
import pytest
import torch

from x01.sr.blocks.unet import UNet
from x01.sr.models.diffusion_mngr import DiffusionManager
from x01.sr.models.loss_ddpm import loss_ddpm

T = 100


@pytest.fixture
def manager():
    return DiffusionManager(beta_start=1e-4, beta_end=0.02, num_diffusion_time_steps=T)


@pytest.fixture(scope="module")
def unet():
    torch.manual_seed(0)
    return UNet(
        in_channels=1,
        out_channels=1,
        latent_dims=32,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(8,),
        image_resolution=16,
        dropout_rate=0.0,
        resample_with_conv=True,
    )


def test_noise_schedule(manager):
    assert manager.betas.shape == (T,)
    assert manager.alphas_bar.shape == (T + 1,)
    assert manager.alphas_bar[0].item() == 1.0
    assert (manager.alphas_bar.diff() < 0).all()


def test_compute_alpha_is_shifted_by_one(manager):
    t = torch.tensor([0, T - 1])
    alpha = manager.compute_alpha(t)
    assert alpha.shape == (2, 1, 1, 1)
    torch.testing.assert_close(alpha.flatten(), manager.alphas_bar[[1, T]])
    torch.testing.assert_close(manager.compute_alpha(torch.tensor([-1])).flatten(), torch.ones(1))


def test_mask_loaded_from_file(tmp_path):
    mask = np.zeros((16, 16), dtype=np.float32)
    mask[4:12, 4:12] = 1.0
    np.save(tmp_path / "mask.npy", mask)
    manager = DiffusionManager(1e-4, 0.02, T, mask_path=str(tmp_path / "mask.npy"))
    assert manager.mask.shape == (1, 1, 16, 16)
    assert manager.mask.sum().item() == 64


def test_loss_ddpm_finite_and_differentiable(manager, unet):
    x = torch.randn(2, 1, 16, 16)
    loss, log = loss_ddpm(manager, unet, x, condition=torch.randn_like(x))
    assert torch.isfinite(loss) and loss.item() > 0
    assert log["loss"] is loss
    unet.zero_grad()
    loss.backward()
    assert unet.conv_out.weight.grad is not None


def test_condition_dropout_does_not_mutate_input(unet):
    manager = DiffusionManager(1e-4, 0.02, T, condition_dropout_rate=1.0)
    cond = torch.randn(2, 1, 16, 16)
    before = cond.clone()
    loss_ddpm(manager, unet, torch.randn(2, 1, 16, 16), condition=cond)
    torch.testing.assert_close(cond, before)


def test_ddim_inference_shape(manager, unet):
    x_init = torch.randn(2, 1, 16, 16)
    out = manager.infer_fn(x_init, unet.eval(), inference_steps=5, condition=x_init)
    assert out.shape == x_init.shape
    assert torch.isfinite(out).all()


def test_ddim_with_oracle_denoiser_recovers_x0():
    # infer_fn noises x_init to level T, so an oracle returning that exact noise
    # makes every deterministic DDIM step land back on x_init.
    manager = DiffusionManager(1e-4, 0.02, T)
    x0 = torch.randn(1, 1, 8, 8)
    torch.manual_seed(0)
    eps = torch.randn_like(x0)

    class Oracle(torch.nn.Module):
        def forward(self, x, t, condition=None):
            return eps

    torch.manual_seed(0)
    out = manager.infer_fn(x0, Oracle(), inference_steps=T)
    torch.testing.assert_close(out, x0, atol=1e-4, rtol=1e-4)
