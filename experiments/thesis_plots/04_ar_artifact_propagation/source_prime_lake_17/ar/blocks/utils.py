import torch
import torch.nn as nn


# TODO: Why Lecun init? Kaiming is better with Relu.
def variance_scaling_uniform_(tensor: torch.Tensor, scale: float = 1 / 3) -> None:
    fan_in = nn.init._calculate_correct_fan(tensor, "fan_in")
    limit = (3 * scale / fan_in) ** 0.5
    with torch.no_grad():
        tensor.uniform_(-limit, limit)
