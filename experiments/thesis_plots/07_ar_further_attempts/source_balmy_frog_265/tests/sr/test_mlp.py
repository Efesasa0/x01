import numpy as np
import torch

from x01.sr.blocks.mlp import MLPBlock

GOLDEN = np.load("tests/sr/golden/mlp_golden.npz")


def test_mlp_shape():
    model = MLPBlock(in_dim=4, hidden_dim=8, output_dim=4)
    x = torch.randn(2, 6, 4)
    assert model(x).shape == (2, 6, 4)


def test_mlp_golden():
    W1 = torch.tensor(GOLDEN["W1"]).T  # JAX (in, out) -> torch (out, in)
    b1 = torch.tensor(GOLDEN["b1"])
    W2 = torch.tensor(GOLDEN["W2"]).T
    b2 = torch.tensor(GOLDEN["b2"])
    x = torch.tensor(GOLDEN["x"])
    expected = torch.tensor(GOLDEN["out"])

    model = MLPBlock(in_dim=4, hidden_dim=8, output_dim=4)
    model.block[0].weight = torch.nn.Parameter(W1)
    model.block[0].bias = torch.nn.Parameter(b1)
    model.block[2].weight = torch.nn.Parameter(W2)
    model.block[2].bias = torch.nn.Parameter(b2)

    with torch.no_grad():
        out = model(x)

    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)
