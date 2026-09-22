import torch
import torch.nn as nn


class DynamicsBlock(nn.Module):
    def __init__(self, latent_size: int):  # 32 to 32 mapper
        super().__init__()

        params = torch.randn(latent_size, latent_size)
        U, _, Vh = torch.linalg.svd(params)  # To get orthogonal matrices u and vh, to be energy preserving
        V = Vh.conj().T  # calculate the complex conjugate of a matrix to extract pure v
        self.dynamics = nn.Parameter(U @ V.T)  # creates orthogonal matrix

    @property
    def matrix(self) -> torch.Tensor:
        return self.dynamics

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x @ self.dynamics
        )  # goes forward prediction qt -> qt+1 by multiplying qt with Koopman matrix preserving channel


class DynamicsBackBlock(nn.Module):
    def __init__(self, omega: DynamicsBlock, mode: str = "separate"):
        super().__init__()
        if mode not in ("separate", "single"):
            raise ValueError(f"unknown dynamics mode: {mode!r} (allowed: 'separate', 'single')")
        self.mode = mode
        # keep reference without registering as submodule so omega's parameters
        # don't get double-counted in model.parameters() and DDP.
        object.__setattr__(self, "_omega", omega)
        if mode == "separate":
            self.dynamics = nn.Parameter(omega.dynamics.detach().clone())

    @property
    def matrix(self) -> torch.Tensor:
        if self.mode == "separate":
            return self.dynamics
        return self._omega.dynamics.T  # single: backward = A.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.matrix


class LowRankDynamicsBlock(nn.Module):
    """Low-rank Koopman: x -> up(down(x) @ core). Down/up projections + rank x rank core."""

    def __init__(self, latent_size: int, rank: int):
        super().__init__()
        assert 0 < rank < latent_size, f"rank {rank} must satisfy 0 < rank < latent_size {latent_size}"
        self.latent_size = latent_size
        self.rank = rank

        self.down = nn.Linear(latent_size, rank, bias=False)
        self.up = nn.Linear(rank, latent_size, bias=False)

        params = torch.randn(rank, rank)
        U, _, Vh = torch.linalg.svd(params)
        V = Vh.conj().T
        self.core = nn.Parameter(U @ V.T)  # orthogonal init in reduced space

    @property
    def matrix(self) -> torch.Tensor:
        return self.core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x) @ self.core)


class LowRankDynamicsBackBlock(nn.Module):
    """Backward twin of LowRankDynamicsBlock. Shares down/up with forward; only the reduced core differs."""

    def __init__(self, omega: LowRankDynamicsBlock, mode: str = "separate"):
        super().__init__()
        if mode not in ("separate", "single"):
            raise ValueError(f"unknown dynamics mode: {mode!r} (allowed: 'separate', 'single')")
        self.mode = mode
        object.__setattr__(self, "_omega", omega)
        if mode == "separate":
            self.core = nn.Parameter(omega.core.detach().clone())

    @property
    def matrix(self) -> torch.Tensor:
        if self.mode == "separate":
            return self.core
        return self._omega.core.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._omega.up(self._omega.down(x) @ self.matrix)
