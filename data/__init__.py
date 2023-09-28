import os
import sys

from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, FakeData


def make_dataset(
        root: str,
        dataset: str,
        train: bool,
        transform):
    if dataset.lower() == 'cifar10':
        return CIFAR10(root, download=True, train=train, transform=transform), 10
    elif dataset.lower() == 'cifar100':
        return CIFAR100(root, download=True, train=train, transform=transform), 100
    elif dataset.lower() == 'imagenet':
        root = os.path.join(root, 'train' if train else 'val')
        dataset = ImageFolder(root, transform=transform)
        return dataset, 1000
    elif dataset.lower() == 'test224':
        dataset = FakeData(size=1000, image_size=(3, 224, 224), num_classes=1000, transform=transform)
        return dataset, 1000
    print(f"Does not support dataset: {dataset}")
    sys.exit(1)
