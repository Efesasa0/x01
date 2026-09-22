import torch


class LpLoss:
    def __init__(self, d: int = 2, p: int = 2, size_average: bool = True, reduction: bool = True):
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        num_examples = x.size(0)
        h = 1.0 / (x.size(1) - 1.0)
        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        if self.reduction:
            return torch.mean(all_norms) if self.size_average else torch.sum(all_norms)
        return all_norms

    def rel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        num_examples = x.size(0)
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            return torch.mean(diff_norms / y_norms) if self.size_average else torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rel(x, y)
