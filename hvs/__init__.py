from typing import List, Tuple

import itertools as it
import torch
import torch.nn as nn
from torch import Tensor


@torch.no_grad()
def adversarial_crop_selection(
        images: List[Tensor],
        model: nn.Module,
        criterion: nn.Module,
) -> Tuple[Tuple[Tensor, ...], Tensor, Tensor]:

    bs, _, _, _ = images[0].shape
    device = images[0].device

    # prepare output tensors
    out1 = torch.zeros_like(images[0])
    out2 = torch.zeros_like(images[0])
    loss = torch.zeros(bs, device=device)
    selection = torch.zeros((2, bs), dtype=torch.uint8)

    # get features
    with torch.cuda.amp.autocast():
        features = model(torch.cat(images, dim=0)).chunk(len(images))

    # do adversarial crop selection
    for n, m in it.combinations(range(len(images)), 2):
        z1, z2 = features[n], features[m]
        with torch.cuda.amp.autocast():
            sim = criterion.acs_fwd(z1, z2)
            loss, indices = torch.stack((loss, sim)).max(dim=0)
            indices = indices.type(torch.bool)
        selection[0][indices] = n
        selection[1][indices] = m
        out1 = torch.where(indices[:, None, None, None], images[n], out1)
        out2 = torch.where(indices[:, None, None, None], images[m], out2)

    return (out1, out2), selection, loss
