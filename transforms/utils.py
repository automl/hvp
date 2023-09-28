import torch

IMAGENET_NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
CIFAR_NORMALIZE = {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010]}


def custom_collate(batch):
    """
    A batch is a list of tuples (sample, target).
    Here: mini_batch = ({"images": [img1, img2, ...], "params": [param1, param2, ...]}, target)
    The samples are dicts with keys "images" and "params".
    """

    images = [[] for _ in range(len(batch[0][0]["images"]))]
    params = [[] for _ in range(len(batch[0][0]["params"]))] if "params" in batch[0][0] else None
    targets = []
    for mini_batch in batch:
        for i, img in enumerate(mini_batch[0]["images"]):
            images[i].append(img)
        if params is not None:
            for i, param in enumerate(mini_batch[0]["params"]):
                params[i].append(param)
        targets.append(mini_batch[1])
    images = [torch.stack(img) for img in images]
    targets = torch.tensor(targets)
    if params is not None:
        params = [torch.stack(p) for p in params]
    return images, params, targets
