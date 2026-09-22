import torch


class MseLoss:
    def __init__(self, size_average: bool = True):
        self.size_average = size_average

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        num_examples = x.size(0)
        per_sample = ((x.reshape(num_examples, -1) - y.reshape(num_examples, -1)) ** 2).mean(dim=1)
        return torch.mean(per_sample) if self.size_average else torch.sum(per_sample)
