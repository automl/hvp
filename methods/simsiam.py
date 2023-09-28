import torch
import torch.nn as nn

from losses.simsiam_loss import SimSiamLoss


class SimSiam(nn.Module):
    def __init__(
            self,
            backbone: nn.Module,
            proj_hidden_dim: int = 2048,
            pred_hidden_dim: int = 512,
            out_dim: int = 2048,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.backbone = backbone
        embed_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.projector = SimSiamProjectionHead(
            embed_dim,
            proj_hidden_dim,
            out_dim
        )

        self.predictor = SimSiamPredictionHead(
            out_dim,
            pred_hidden_dim,
            out_dim
        )

        self.loss_fn = SimSiamLoss()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor = None):
        f1 = self.backbone(x1).flatten(start_dim=1)
        z1 = self.projector(f1)
        p1 = self.predictor(z1)
        out1 = (p1, z1)

        if x2 is None:
            return out1

        f2 = self.backbone(x2).flatten(start_dim=1)
        z2 = self.projector(f2)
        p2 = self.predictor(z2)

        return out1, (p2, z2)

    def training_step(self, batch, *args, **kwargs):
        x1, x2 = batch
        out1, out2 = self(x1, x2)
        loss = self.loss_fn(out1, out2)
        return loss

    def configure_optimizers(
            self,
            fix_pred_lr: bool = False,
            lr: float = 0.05,
            momentum: float = 0.9,
            weight_decay: float = 1e-4,
    ):
        if fix_pred_lr:
            optim_params = [{'params': [*self.backbone.parameters(), *self.projector.parameters()], 'fix_lr': False},
                            {'params': self.predictor.parameters(), 'fix_lr': True}]
        else:
            optim_params = self.parameters()
        return torch.optim.SGD(optim_params, lr=lr, momentum=momentum, weight_decay=weight_decay)


class SimSiamProjectionHead(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False)
        )

    def forward(self, x):
        return self.layers(x)


class SimSiamPredictionHead(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.layers(x)
