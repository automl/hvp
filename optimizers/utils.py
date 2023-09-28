import torch
from optimizers.lars import LARS


def configure_optimizer(optimizer, parameters, lr, weight_decay, momentum) -> torch.optim.Optimizer:
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer == 'adamw':
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == 'sgd-nesterov':
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    elif optimizer == 'lars':
        optimizer = LARS(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise NotImplementedError(f'Optimizer {optimizer} not implemented')
    return optimizer
