import torch
import torch.nn as nn


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """

    def __init__(self, base_encoder, dim=2048, pred_dim=512, proj_layer=3):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build an n-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        layers = [
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),
        ]
        for _ in range(proj_layer - 2):
            layers.extend([
                nn.Linear(prev_dim, prev_dim, bias=False),
                nn.BatchNorm1d(prev_dim),
                nn.ReLU(inplace=True),
            ])
        layers.extend([
            self.encoder.fc,
            nn.BatchNorm1d(dim, affine=False)
        ])
        self.encoder.fc = nn.Sequential(*layers)  # output layer
        self.encoder.fc[-2].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(pred_dim, dim))  # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1)  # NxC
        z2 = self.encoder(x2)  # NxC

        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC

        return p1, p2, z1.detach(), z2.detach()

    @torch.no_grad()
    def single_forward(self, x):
        z = self.encoder(x)
        p = self.predictor(z)
        return p, z

    @torch.no_grad()
    def conv1_layer(self, x):
        return self.encoder.conv1_layer(x)

    @torch.no_grad()
    def first_layer(self, x):
        return self.encoder.first_layer(x)

    @torch.no_grad()
    def second_layer(self, x):
        return self.encoder.second_layer(x)

    @torch.no_grad()
    def avgpool_layer(self, x):
        return self.encoder.avgpool_layer(x)
