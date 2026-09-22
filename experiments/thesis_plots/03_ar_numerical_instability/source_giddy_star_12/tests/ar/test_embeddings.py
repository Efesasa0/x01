import numpy as np
import pytest
import torch

from ar.blocks.embeddings import OverlapPatchEmbed

GOLDEN = np.load("tests/ar/golden/embeddings_golden.npz")


def test_shape():
    model = OverlapPatchEmbed(input_channels=3, embedding_dims=8, use_bias=False)
    x = torch.randn(1, 3, 16, 16)
    assert model(x).shape == (1, 8, 16, 16)


def test_golden():
    # Flax weight: (kH, kW, C_in, C_out) -> PyTorch: (C_out, C_in, kH, kW)
    W = torch.tensor(GOLDEN["W"]).permute(3, 2, 0, 1)
    # Flax input: NHWC -> PyTorch: NCHW
    x = torch.tensor(GOLDEN["x"]).permute(0, 3, 1, 2)
    # Flax output: NHWC -> PyTorch: NCHW
    expected = torch.tensor(GOLDEN["out"]).permute(0, 3, 1, 2)

    model = OverlapPatchEmbed(input_channels=3, embedding_dims=8, use_bias=False)
    model.projection.weight = torch.nn.Parameter(W)

    out = model(x)
    err = (out - expected).abs().max().item()
    assert err < 1e-5, f"golden error {err:.2e}"
