from typing import Tuple

import torch
import torch.nn as nn

from losses.negative_cosine_similarity import NegativeCosineSimilarity


class SimSiamLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = NegativeCosineSimilarity()

    def forward(
            self,
            x1: Tuple[torch.Tensor, torch.Tensor],
            x2: Tuple[torch.Tensor, torch.Tensor],
            reduction: str = 'mean'
    ) -> torch.Tensor:
        p1, z1 = x1
        p2, z2 = x2
        z1, z2 = z1.detach(), z2.detach()
        loss = 0.5 * (self.loss_fn(p1, z2) + self.loss_fn(p2, z1))
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        return loss
