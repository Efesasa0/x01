import numpy as np
import pytest
import torch

from x01.ar.blocks.dynamics import DynamicsBackBlock, DynamicsBlock

GOLDEN = np.load("tests/ar/golden/dynamics_golden.npz")

N = 64
INIT_SCALE = 1.0


@pytest.fixture
def blocks():
    forward = DynamicsBlock(N, N, INIT_SCALE)
    backward = DynamicsBackBlock(N, N, forward)
    return forward, backward


def test_forward_shape(blocks):
    forward, _ = blocks
    x = torch.ones(1, N)
    assert forward(x).shape == x.shape


def test_backward_shape(blocks):
    _, backward = blocks
    x = torch.ones(1, N)
    assert backward(x).shape == x.shape


def test_orthogonality(blocks):
    forward, _ = blocks
    K = forward.dynamics
    err = (K.T @ K - torch.eye(N)).abs().max().item()
    assert err < 1e-6, f"orthogonality error {err:.2e} exceeds 1e-6"


def test_back_dynamics_init(blocks):
    forward, backward = blocks
    K = forward.dynamics
    expected = torch.linalg.pinv(K.T)  # backward step is pseudo inverse of forward.
    err = (backward.back_dynamics - expected).abs().max().item()
    assert err < 1e-6, f"back_dynamics wiring error {err:.2e} exceeds 1e-6"


def test_golden_forward():
    K = torch.tensor(GOLDEN["K"])
    x = torch.tensor(GOLDEN["x"])
    expected = torch.tensor(GOLDEN["out_forward"])

    forward = DynamicsBlock(8, 8, 1.0)
    forward.dynamics = torch.nn.Parameter(K)

    out = forward(x)
    err = (out - expected).abs().max().item()
    assert err < 1e-6, f"golden forward error {err:.2e}"


def test_golden_backward():
    K = torch.tensor(GOLDEN["K"])
    x = torch.tensor(GOLDEN["x"])
    expected = torch.tensor(GOLDEN["out_backward"])

    forward = DynamicsBlock(8, 8, 1.0)
    forward.dynamics = torch.nn.Parameter(K)
    backward = DynamicsBackBlock(8, 8, forward)

    out = backward(forward(x))
    err = (out - expected).abs().max().item()
    assert err < 1e-6, f"golden backward error {err:.2e}"
