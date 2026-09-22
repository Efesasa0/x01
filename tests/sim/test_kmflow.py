import pytest
import torch

from x01.sim.kmflow import KMFlowSolver

N = 32


@pytest.fixture
def solver():
    return KMFlowSolver(N=N, re=1000.0, alpha=2.5, tau=7.0, dt=1e-3, device=torch.device("cpu"))


def test_reset_is_seed_deterministic(solver):
    solver.reset([0, 1])
    first = solver.get_vorticity().clone()
    solver.reset([0, 1])
    torch.testing.assert_close(solver.get_vorticity(), first)
    assert not torch.allclose(first[0], first[1])


def test_vorticity_shape_and_zero_mean(solver):
    solver.reset([0, 1, 2])
    w = solver.get_vorticity()
    assert w.shape == (3, N, N)
    torch.testing.assert_close(w.mean(dim=(1, 2)), torch.zeros(3), atol=1e-5, rtol=0)


def test_initial_state_is_dealiased(solver):
    solver.reset([0])
    assert (solver.omega_hat[0][solver.dealias == 0] == 0).all()


def test_steps_stay_finite_and_evolve(solver):
    solver.reset([0])
    w0 = solver.get_vorticity().clone()
    for _ in range(20):
        solver.step()
    w = solver.get_vorticity()
    assert torch.isfinite(w).all()
    assert not torch.allclose(w, w0)


def test_forcing_drives_quiescent_state():
    solver = KMFlowSolver(N=N, re=1000.0, alpha=2.5, tau=7.0, dt=1e-2, device=torch.device("cpu"), forcing_n=4)
    solver.omega_hat = torch.zeros(1, N, N // 2 + 1, dtype=torch.complex64)
    solver.step()
    w = solver.get_vorticity()[0]
    # Kolmogorov forcing -n cos(n y) is constant along x
    torch.testing.assert_close(w[0], w[N // 2])
    assert w.abs().max().item() > 0
