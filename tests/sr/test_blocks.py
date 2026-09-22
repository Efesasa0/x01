from typing import Any

import pytest
import torch

from x01.sr.blocks.attention import ConvAttention
from x01.sr.blocks.embeddings import TimeStepEmbedding
from x01.sr.blocks.resnet import ResNetBlock
from x01.sr.blocks.sampler import Downsample, Upsample
from x01.sr.blocks.unet import UNet

UNET_CFG: dict[str, Any] = dict(
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


# embeddings


@pytest.mark.parametrize("dim", [8, 9])
def test_timestep_embedding_shape(dim):
    t = torch.arange(4)
    assert TimeStepEmbedding.timestep_embedding(t, dim).shape == (4, dim)


def test_timestep_embedding_at_zero_is_cos_one_sin_zero():
    emb = TimeStepEmbedding.timestep_embedding(torch.zeros(1), 8)
    torch.testing.assert_close(emb, torch.tensor([[1.0] * 4 + [0.0] * 4]))


def test_timestep_embedding_with_mlp_shape():
    model = TimeStepEmbedding(embedding_dim=8, with_mlp_block=True, mlp_hidden_dim=16)
    assert model(torch.arange(4)).shape == (4, 16)


# resnet, attention, samplers


@pytest.mark.parametrize(
    "in_c, out_c, conv_shortcut",
    [(32, 32, False), (32, 64, False), (32, 64, True)],
)
def test_resnet_shape(in_c, out_c, conv_shortcut):
    model = ResNetBlock(in_c, out_c, use_conv_shortcut=conv_shortcut, dropout=0.0, t_embed_channels=16)
    assert model(torch.randn(2, in_c, 8, 8), torch.randn(2, 16)).shape == (2, out_c, 8, 8)


def test_resnet_depends_on_time_embedding():
    torch.manual_seed(0)
    model = ResNetBlock(32, 32, use_conv_shortcut=False, dropout=0.0, t_embed_channels=16)
    x = torch.randn(1, 32, 8, 8)
    assert not torch.allclose(model(x, torch.zeros(1, 16)), model(x, torch.ones(1, 16)))


def test_conv_attention_is_residual_shape_preserving():
    model = ConvAttention(in_channels=32)
    x = torch.randn(2, 32, 8, 8)
    assert model(x).shape == x.shape


@pytest.mark.parametrize("with_conv", [True, False])
def test_sampler_shapes(with_conv):
    x = torch.randn(1, 8, 8, 8)
    assert Upsample(8, with_conv=with_conv)(x).shape == (1, 8, 16, 16)
    assert Downsample(8, with_conv=with_conv)(x).shape == (1, 8, 4, 4)


# unet


@pytest.mark.parametrize("resample_with_conv", [True, False])
def test_unet_shape(resample_with_conv):
    model = UNet(**{**UNET_CFG, "resample_with_conv": resample_with_conv}).eval()
    x = torch.randn(2, 1, 16, 16)
    t = torch.randint(0, 100, (2,))
    with torch.no_grad():
        assert model(x, t).shape == x.shape
        assert model(x, t, condition=torch.randn_like(x)).shape == x.shape


def test_unet_uses_condition():
    torch.manual_seed(0)
    model = UNet(**UNET_CFG).eval()
    x = torch.randn(1, 1, 16, 16)
    t = torch.tensor([10])
    with torch.no_grad():
        assert not torch.allclose(model(x, t), model(x, t, condition=torch.randn_like(x)))


def test_unet_rejects_wrong_resolution():
    model = UNet(**UNET_CFG)
    with pytest.raises(AssertionError):
        model(torch.randn(1, 1, 32, 32), torch.tensor([0]))
