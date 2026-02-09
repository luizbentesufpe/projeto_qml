from typing import Optional
import torch
import torch.nn.functional as F


def focal_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor,
                           alpha: float = 0.25, gamma: float = 2.0,
                           reduction: str = "mean", pos_weight: Optional[torch.Tensor] = None):
    logits = logits.view(-1)
    targets = targets.view(-1).float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    p = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, p, 1.0 - p)
    w = torch.where(targets > 0.5, torch.tensor(alpha, device=logits.device),
                    torch.tensor(1.0 - alpha, device=logits.device))
    loss = w * (1.0 - pt).pow(gamma) * bce
    if reduction == "mean": return loss.mean()
    if reduction == "sum": return loss.sum()
    return loss

