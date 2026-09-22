import torch
import torch.nn as nn


class DynamicsBlock(nn.Module):
    def __init__(self, input_features: int, output_features: int, init_scale: float):  # 32 to 32 mapper
        super().__init__()
        if input_features != output_features:  # Koopman needs to be square matrix
            raise ValueError("DynamicsBlock expects input_features == output_features.")
        del output_features  # No longer needed since identical

        params = torch.randn(input_features, input_features) / init_scale
        U, _, Vh = torch.linalg.svd(params)  # To get orthogonal matrices u and vh, to be energy preserving
        V = Vh.conj().T  # calculate the complex conjugate of a matrix to extract pure v
        self.dynamics = nn.Parameter(U @ V.T * init_scale)  # creates orthogonal matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x @ self.dynamics
        )  # goes forward prediction qt -> qt+1 by multiplying qt with Koopman matrix preserving channel


class DynamicsBackBlock(nn.Module):
    def __init__(self, input_features: int, output_features: int, omega: DynamicsBlock):
        super().__init__()
        if input_features != output_features:
            raise ValueError("DynamicsBackBlock expects input_features == output_features.")
        del input_features, output_features

        # Backward operator is K^T (paper Eq. 11). No separate parameter — when K is
        # approximately orthogonal via the L_ortho penalty, K^T ≈ K^{-1}, so the
        # backward MSE term reuses K and couples both directions through one matrix.
        self._forward = omega

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self._forward.dynamics.T  # qt -> qt-1 via K^T
