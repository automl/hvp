import torch.nn as nn
from torch import Tensor

from methods.utils import prepare_backbone


class SimCLR(nn.Module):
    def __init__(
            self,
            backbone: nn.Module,
            proj_hidden_dim: int = 2048,
            out_dim: int = 128,
            num_layers: int = 2,
            use_bn: bool = True,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.backbone = backbone
        feat_dim = prepare_backbone(backbone)

        self.projector = SimCLRProjectionHead(feat_dim, proj_hidden_dim, out_dim, num_layers, use_bn)

    def forward(self, x: Tensor) -> Tensor:
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projector(h)
        return z


class SimCLRProjectionHead(nn.Module):
    """
    MLP projection head for SimCLR.
    https://github.com/google-research/simclr/blob/383d4143fd8cf7879ae10f1046a9baeb753ff438/tf2/model.py#L157C16-L157C16
    """
    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int = 2,
            use_bn: bool = True,
    ):
        super().__init__()
        # do not use bias for linear with bn
        layers = [nn.Linear(in_dim, hidden_dim, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        # do not use relu for the last layer
        layers.append(nn.Linear(hidden_dim, output_dim, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm1d(output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
