import numpy as np
import torch

from x01.ar.blocks.samplers import Downsample, Upsample

GOLDEN = np.load("tests/ar/golden/samplers_golden.npz")


def test_downsample_shape():
    model = Downsample(in_features=16, downsample_factor=2)
    x = torch.randn(1, 16, 32, 32)
    assert model(x).shape == (1, 32, 16, 16)


def test_downsample_golden():
    W = torch.tensor(GOLDEN["W_ds"]).permute(3, 2, 0, 1)
    x = torch.tensor(GOLDEN["x_ds"]).permute(0, 3, 1, 2)
    expected = torch.tensor(GOLDEN["out_ds"]).permute(0, 3, 1, 2)

    model = Downsample(in_features=16, downsample_factor=2)
    model.input_conv.weight = torch.nn.Parameter(W)

    err = (model(x) - expected).abs().max().item()
    assert err < 1e-5, f"downsample golden error {err:.2e}"


def test_upsample_shape():
    model = Upsample(in_features=16, upsample_factor=2)
    x = torch.randn(1, 16, 16, 16)
    assert model(x).shape == (1, 8, 32, 32)


def test_upsample_golden():
    W = torch.tensor(GOLDEN["W_us"]).permute(3, 2, 0, 1)
    x = torch.tensor(GOLDEN["x_us"]).permute(0, 3, 1, 2)
    expected = torch.tensor(GOLDEN["out_us"]).permute(0, 3, 1, 2)

    model = Upsample(in_features=16, upsample_factor=2)
    model.input_conv.weight = torch.nn.Parameter(W)

    err = (model(x) - expected).abs().max().item()
    assert err < 1e-5, f"upsample golden error {err:.2e}"
