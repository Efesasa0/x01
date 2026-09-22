import numpy as np
import torch

from x01.sr.blocks.resnet import ResNetBlock

GOLDEN = np.load("tests/sr/golden/resnet_golden.npz")


def test_resnet_shape_same_channels():
    model = ResNetBlock(in_channels=32, out_channels=32, use_conv_shortcut=False, dropout=0.0, t_embed_channels=16)
    x = torch.randn(1, 32, 8, 8)
    t = torch.randn(1, 16)
    assert model(x, t).shape == (1, 32, 8, 8)


def test_resnet_shape_conv_shortcut():
    model = ResNetBlock(in_channels=32, out_channels=64, use_conv_shortcut=True, dropout=0.0, t_embed_channels=16)
    x = torch.randn(1, 32, 8, 8)
    t = torch.randn(1, 16)
    assert model(x, t).shape == (1, 64, 8, 8)


def test_resnet_shape_1x1_shortcut():
    model = ResNetBlock(in_channels=32, out_channels=64, use_conv_shortcut=False, dropout=0.0, t_embed_channels=16)
    x = torch.randn(1, 32, 8, 8)
    t = torch.randn(1, 16)
    assert model(x, t).shape == (1, 64, 8, 8)


def test_resnet_golden():
    gn1_scale = torch.tensor(GOLDEN["gn1_scale"])
    gn1_bias = torch.tensor(GOLDEN["gn1_bias"])
    gn2_scale = torch.tensor(GOLDEN["gn2_scale"])
    gn2_bias = torch.tensor(GOLDEN["gn2_bias"])

    # JAX HWIO -> torch OIHW
    W_conv1 = torch.tensor(GOLDEN["W_conv1"]).permute(3, 2, 0, 1)
    W_conv2 = torch.tensor(GOLDEN["W_conv2"]).permute(3, 2, 0, 1)
    W_short = torch.tensor(GOLDEN["W_short"]).permute(3, 2, 0, 1)
    b_conv1 = torch.tensor(GOLDEN["b_conv1"])
    b_conv2 = torch.tensor(GOLDEN["b_conv2"])
    b_short = torch.tensor(GOLDEN["b_short"])

    # JAX Linear kernel (in, out) -> torch (out, in)
    W_temb = torch.tensor(GOLDEN["W_temb"]).T
    b_temb = torch.tensor(GOLDEN["b_temb"])

    # JAX NHWC -> torch NCHW
    x = torch.tensor(GOLDEN["x"]).permute(0, 3, 1, 2)
    expected = torch.tensor(GOLDEN["out"]).permute(0, 3, 1, 2)
    t = torch.tensor(GOLDEN["t"])

    model = ResNetBlock(in_channels=32, out_channels=64, use_conv_shortcut=False, dropout=0.0, t_embed_channels=16)

    model.norm1.weight = torch.nn.Parameter(gn1_scale)
    model.norm1.bias = torch.nn.Parameter(gn1_bias)
    model.norm2.weight = torch.nn.Parameter(gn2_scale)
    model.norm2.bias = torch.nn.Parameter(gn2_bias)

    model.conv1.weight = torch.nn.Parameter(W_conv1)
    model.conv1.bias = torch.nn.Parameter(b_conv1)
    model.conv2.weight = torch.nn.Parameter(W_conv2)
    model.conv2.bias = torch.nn.Parameter(b_conv2)
    model.shortcut.weight = torch.nn.Parameter(W_short)
    model.shortcut.bias = torch.nn.Parameter(b_short)

    model.t_embed_proj.weight = torch.nn.Parameter(W_temb)
    model.t_embed_proj.bias = torch.nn.Parameter(b_temb)

    model.eval()
    with torch.no_grad():
        out = model(x, t)

    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
