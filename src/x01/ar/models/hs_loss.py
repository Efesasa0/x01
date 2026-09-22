import torch


class HsLoss:
    """Sobolev H^s relative loss with per-frequency weighting.

    Expects inputs shaped [B, C, H, W] (channel-first, matching AR model I/O).
    Ported from uniflow/utilities.py:HsLoss.
    """

    def __init__(
        self,
        d: int = 2,
        p: int = 2,
        k: int = 1,
        a: list | None = None,
        group: bool = False,
        size_average: bool = True,
        reduction: bool = True,
    ):
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average
        self.a = a if a is not None else [1.0] * k

    def rel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        num_examples = x.shape[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            return torch.mean(diff_norms / y_norms) if self.size_average else torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Accept [B, C, H, W]; reshape to uniflow layout [B, H, W, C].
        assert x.dim() == 4 and y.dim() == 4, f"expected [B, C, H, W], got {x.shape}"
        x = x.permute(0, 2, 3, 1).contiguous()
        y = y.permute(0, 2, 3, 1).contiguous()

        nx = x.shape[1]
        ny = x.shape[2]
        k = self.k
        balanced = self.balanced
        a = self.a

        k_x = (
            torch.cat((torch.arange(start=0, end=nx // 2, step=1), torch.arange(start=-nx // 2, end=0, step=1)), 0)
            .reshape(nx, 1)
            .repeat(1, ny)
        )
        k_y = (
            torch.cat((torch.arange(start=0, end=ny // 2, step=1), torch.arange(start=-ny // 2, end=0, step=1)), 0)
            .reshape(1, ny)
            .repeat(nx, 1)
        )
        k_x = torch.abs(k_x).reshape(1, nx, ny, 1).to(x.device)
        k_y = torch.abs(k_y).reshape(1, nx, ny, 1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if not balanced:
            weight = 1
            if k >= 1:
                weight = weight + a[0] ** 2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight = weight + a[1] ** 2 * (k_x**4 + 2 * k_x**2 * k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x * weight, y * weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss = loss + self.rel(x * weight, y * weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2 * k_x**2 * k_y**2 + k_y**4)
                loss = loss + self.rel(x * weight, y * weight)
            loss = loss / (k + 1)

        return loss
