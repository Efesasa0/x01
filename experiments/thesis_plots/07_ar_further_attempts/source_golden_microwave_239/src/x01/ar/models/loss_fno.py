from typing import Dict, Tuple

import torch
from torch import Tensor


def loss_fno(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tuple[Tensor, Dict[str, Tensor]]:
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)

    diff_norm = torch.linalg.vector_norm(pred_flat - target_flat, ord=2, dim=1)
    target_norm = torch.linalg.vector_norm(target_flat, ord=2, dim=1)

    loss = (diff_norm / (target_norm + eps)).mean()

    return loss, {"loss": loss, "rel_l2": loss}
