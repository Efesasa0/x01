import numpy as np
import torch

from x01.sr.blocks.attention import ConvAttention

GOLDEN = np.load("tests/sr/golden/attention_golden.npz")


def test_conv_attention_shape():
    model = ConvAttention(in_channels=32)
    x = torch.randn(1, 32, 8, 8)
    assert model(x).shape == (1, 32, 8, 8)


def test_conv_attention_golden():
    gn_scale = torch.tensor(GOLDEN["gn_scale"])
    gn_bias = torch.tensor(GOLDEN["gn_bias"])

    # JAX HWIO (kH, kW, in, out) -> torch OIHW (out, in, kH, kW)
    W_q = torch.tensor(GOLDEN["W_q"]).permute(3, 2, 0, 1)
    W_k = torch.tensor(GOLDEN["W_k"]).permute(3, 2, 0, 1)
    W_v = torch.tensor(GOLDEN["W_v"]).permute(3, 2, 0, 1)
    W_o = torch.tensor(GOLDEN["W_o"]).permute(3, 2, 0, 1)
    b_q = torch.tensor(GOLDEN["b_q"])
    b_k = torch.tensor(GOLDEN["b_k"])
    b_v = torch.tensor(GOLDEN["b_v"])
    b_o = torch.tensor(GOLDEN["b_o"])

    # JAX NHWC -> torch NCHW
    x = torch.tensor(GOLDEN["x"]).permute(0, 3, 1, 2)
    expected = torch.tensor(GOLDEN["out"]).permute(0, 3, 1, 2)

    model = ConvAttention(in_channels=32)
    model.norm.weight = torch.nn.Parameter(gn_scale)
    model.norm.bias = torch.nn.Parameter(gn_bias)
    model.q.weight = torch.nn.Parameter(W_q)
    model.q.bias = torch.nn.Parameter(b_q)
    model.k.weight = torch.nn.Parameter(W_k)
    model.k.bias = torch.nn.Parameter(b_k)
    model.v.weight = torch.nn.Parameter(W_v)
    model.v.bias = torch.nn.Parameter(b_v)
    model.o.weight = torch.nn.Parameter(W_o)
    model.o.bias = torch.nn.Parameter(b_o)

    with torch.no_grad():
        out = model(x)

    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
