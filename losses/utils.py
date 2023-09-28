from typing import Tuple

import torch
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all processes, supporting backward propagation.

    This code was taken and adapted from here:
    https://github.com/Spijkervet/SimCLR
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if dist.is_available() and dist.is_initialized():
            output = [torch.empty_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(output, x)
        else:
            output = [x]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads: torch.Tensor) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized():
            grad_out = grads[dist.get_rank()]
        else:
            grad_out = grads[0]
        return grad_out


def gather(x: torch.Tensor) -> torch.Tensor:
    """Gathers this tensor from all processes. Supports backprop."""
    return torch.cat(GatherLayer.apply(x), dim=0)
