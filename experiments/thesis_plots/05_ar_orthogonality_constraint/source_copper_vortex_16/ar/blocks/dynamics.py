import torch
import torch.nn as nn


class DynamicsBlock(nn.Module):
    def __init__(self, input_features: int, output_features: int, init_scale: float):  # 32 to 32 mapper
        super().__init__()
        if input_features != output_features:  # Koopman needs to be square matrix TODO: why?
            raise ValueError("DynamicsBlock expects input_features == output_features.")
        del output_features  # No longer needed since identical

        params = torch.randn(input_features, input_features) / init_scale
        U, _, Vh = torch.linalg.svd(params)  # To get orthogonal matrices u and vh, to be energy preserving
        V = Vh.mH  # To account for complex params. TODO: remove this and see if redundant
        self.dynamics = nn.Parameter(U @ V.mH * init_scale)  # creates orthogonal matrix TODO: remove mH too

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

        # Generalized inverse of the matrix, even if not inv or square, requires the Koopman matrix from the DynamicsBlock
        self.back_dynamics = nn.Parameter(torch.linalg.pinv(omega.dynamics.T))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.back_dynamics  # To go back in time qt -> qt-1, preserving channel dims
