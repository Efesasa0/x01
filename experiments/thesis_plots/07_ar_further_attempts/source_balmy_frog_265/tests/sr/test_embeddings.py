import numpy as np
import torch

from x01.sr.blocks.embeddings import PatchEmbedding, SinePositionEmbedding2D, TimeStepEmbedding

GOLDEN = np.load("tests/sr/golden/embedding_golden.npz")


# ---------------- TimeStepEmbedding ----------------
def test_timestep_embedding_shape():
    model = TimeStepEmbedding(embedding_dim=8, with_mlp_block=False, mlp_hidden_dim=0)
    t = torch.arange(4, dtype=torch.float32)
    assert model(t).shape == (4, 8)


def test_timestep_embedding_with_mlp_shape():
    model = TimeStepEmbedding(embedding_dim=8, with_mlp_block=True, mlp_hidden_dim=16)
    t = torch.arange(4, dtype=torch.float32)
    assert model(t).shape == (4, 16)


def test_timestep_embedding_golden():
    W1 = torch.tensor(GOLDEN["ts_W1"]).T  # JAX (in, out) -> torch (out, in)
    b1 = torch.tensor(GOLDEN["ts_b1"])
    W2 = torch.tensor(GOLDEN["ts_W2"]).T
    b2 = torch.tensor(GOLDEN["ts_b2"])
    t = torch.tensor(GOLDEN["ts_t"])
    expected = torch.tensor(GOLDEN["ts_out"])

    model = TimeStepEmbedding(embedding_dim=8, with_mlp_block=True, mlp_hidden_dim=16)
    model.mlp[0].weight = torch.nn.Parameter(W1)
    model.mlp[0].bias = torch.nn.Parameter(b1)
    model.mlp[2].weight = torch.nn.Parameter(W2)
    model.mlp[2].bias = torch.nn.Parameter(b2)

    with torch.no_grad():
        out = model(t)

    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


# ---------------- PatchEmbedding ----------------
def test_patch_embedding_shape():
    model = PatchEmbedding(input_dim=2, embedding_dim=8, patch_size=2, use_bias=True)
    x = torch.randn(1, 2, 8, 8)
    assert model(x).shape == (1, 16, 8)


def test_patch_embedding_golden():
    # JAX kernel HWIO (kH, kW, in, out) -> torch OIHW (out, in, kH, kW)
    W = torch.tensor(GOLDEN["patch_W"]).permute(3, 2, 0, 1)
    b = torch.tensor(GOLDEN["patch_b"])
    # JAX input NHWC -> torch NCHW
    x = torch.tensor(GOLDEN["patch_x"]).permute(0, 3, 1, 2)
    expected = torch.tensor(GOLDEN["patch_out"])  # (B, N, D), no permute

    model = PatchEmbedding(input_dim=2, embedding_dim=8, patch_size=2, use_bias=True)
    model.conv.weight = torch.nn.Parameter(W)
    model.conv.bias = torch.nn.Parameter(b)

    with torch.no_grad():
        out = model(x)

    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


# ---------------- SinePositionEmbedding2D ----------------
def test_sine_position_embedding_shape():
    model = SinePositionEmbedding2D(embedding_dim=16, spatial_size=16)
    assert model().shape == (1, 16, 16)


def test_sine_position_embedding_golden():
    expected = torch.tensor(GOLDEN["sine_out"])
    model = SinePositionEmbedding2D(embedding_dim=16, spatial_size=16)
    with torch.no_grad():
        out = model()
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)
