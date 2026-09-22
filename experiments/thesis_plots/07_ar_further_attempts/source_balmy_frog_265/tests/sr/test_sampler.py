import numpy as np
import torch

from x01.sr.blocks.sampler import Downsample, Upsample

GOLDEN = np.load("tests/sr/golden/sampler_golden.npz")


def test_upsample_shape_no_conv():
    model = Upsample(in_channels=8, with_conv=False)
    x = torch.randn(1, 8, 8, 8)
    assert model(x).shape == (1, 8, 16, 16)


def test_upsample_shape_with_conv():
    model = Upsample(in_channels=8, with_conv=True)
    x = torch.randn(1, 8, 8, 8)
    assert model(x).shape == (1, 8, 16, 16)


def test_downsample_shape_no_conv():
    model = Downsample(in_channels=8, with_conv=False)
    x = torch.randn(1, 8, 8, 8)
    assert model(x).shape == (1, 8, 4, 4)


def test_downsample_shape_with_conv():
    model = Downsample(in_channels=8, with_conv=True)
    x = torch.randn(1, 8, 8, 8)
    assert model(x).shape == (1, 8, 4, 4)


def test_upsample_golden_no_conv():
    x = torch.tensor(GOLDEN["x"]).permute(0, 3, 1, 2)
    expected = torch.tensor(GOLDEN["up_noconv_out"]).permute(0, 3, 1, 2)
    model = Upsample(in_channels=8, with_conv=False)
    with torch.no_grad():
        out = model(x)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


def test_upsample_golden_with_conv():
    W = torch.tensor(GOLDEN["W_up"]).permute(3, 2, 0, 1)
    b = torch.tensor(GOLDEN["b_up"])
    x = torch.tensor(GOLDEN["x"]).permute(0, 3, 1, 2)
    expected = torch.tensor(GOLDEN["up_conv_out"]).permute(0, 3, 1, 2)

    model = Upsample(in_channels=8, with_conv=True)
    model.conv.weight = torch.nn.Parameter(W)
    model.conv.bias = torch.nn.Parameter(b)
    with torch.no_grad():
        out = model(x)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-5)


def test_downsample_golden_with_conv():
    W = torch.tensor(GOLDEN["W_down"]).permute(3, 2, 0, 1)
    b = torch.tensor(GOLDEN["b_down"])
    x = torch.tensor(GOLDEN["x"]).permute(0, 3, 1, 2)
    expected = torch.tensor(GOLDEN["down_conv_out"]).permute(0, 3, 1, 2)

    model = Downsample(in_channels=8, with_conv=True)
    model.conv.weight = torch.nn.Parameter(W)
    model.conv.bias = torch.nn.Parameter(b)
    with torch.no_grad():
        out = model(x)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-5)
