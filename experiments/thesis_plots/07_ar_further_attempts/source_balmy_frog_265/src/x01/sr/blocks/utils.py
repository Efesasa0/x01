import math

import torch
import torch.nn as nn

# std of N(0, 1) truncated to [-2, 2]: sqrt(1 - 2a·phi(a) / (Phi(a) - Phi(-a))) at a=2.
# JAX's lecun_normal compensates for truncation shrinkage by dividing requested std by this.
_a = 2.0
_phi_a = math.exp(-_a * _a / 2) / math.sqrt(2 * math.pi)
_erf_a = math.erf(_a / math.sqrt(2))
_TRUNC_NORMAL_STDDEV_FACTOR = math.sqrt(1.0 - 2.0 * _a * _phi_a / _erf_a)


def variance_scaling_uniform_(tensor: torch.Tensor, scale: float = 1 / 3) -> None:
    fan_in = nn.init._calculate_correct_fan(tensor, "fan_in")
    limit = (3 * scale / fan_in) ** 0.5
    with torch.no_grad():
        tensor.uniform_(-limit, limit)


def lecun_normal_(tensor: torch.Tensor) -> None:
    fan_in = nn.init._calculate_correct_fan(tensor, "fan_in")
    std = math.sqrt(1.0 / fan_in) / _TRUNC_NORMAL_STDDEV_FACTOR
    with torch.no_grad():
        nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
