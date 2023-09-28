import torch


def prepare_backbone(model: torch.nn.Module) -> int:
    if hasattr(model, "fc"):
        dim = model.fc.in_features
        model.fc = torch.nn.Identity()
    elif hasattr(model, "head"):
        dim = model.head.in_features
        model.head = torch.nn.Identity()
    else:
        raise ValueError("Model must have `fc` or `head` attribute.")
    return dim
