import torch
import torch.nn as nn
import torchvision.transforms.functional as tf
from torchvision.transforms import Compose, RandomResizedCrop, ToTensor, Normalize

import numpy as np


def calc_iou(p1, p2):
    def get_dims(x):
        # input: top, left, height, width
        # out: top, bottom, left, right
        return x[0], x[0] + x[2], x[1], x[1] + x[3]

    def calc_overlap(x1, x2):
        t1, b1, l1, r1 = get_dims(x1)
        t2, b2, l2, r2 = get_dims(x2)
        h = max(0, min(b1, b2) - max(t1, t2))
        w = max(0, min(r1, r2) - max(l1, l2))
        return h * w

    def calc_area(x):
        h, w = x[-2:]
        return h * w

    A = calc_area(p1)
    B = calc_area(p2)
    o = calc_overlap(p1, p2)
    return o / (A + B - o)


class MSATransform(nn.Module):
    def __init__(self,
                 rrc: RandomResizedCrop,
                 total_epochs: int,
                 warmup_epochs: int = 0,
                 start_val: float = 0.5,
                 end_val: float = None,
                 schedule: str = 'linear',
                 transforms: Compose = None,
                 p: float = 0.0,
                 ):
        super().__init__()
        self.epoch = 0
        self.total_epochs = total_epochs

        if end_val is None:
            end_val = start_val
        if schedule == 'linear':
            warmup = np.linspace(1, 1, warmup_epochs)
            regular = np.linspace(start_val, end_val, total_epochs - warmup_epochs)
            self.schedule = np.concatenate((warmup, regular))
        elif schedule == 'cosine':
            warmup = np.linspace(1, 1, warmup_epochs)
            regular = cosine_scheduler(start_val, end_val, total_epochs - warmup_epochs, 1)
            self.schedule = np.concatenate((warmup, regular))
        self.start_val = start_val
        self.end_val = end_val
        self.p = p

        self.rrc = rrc
        self.transforms = transforms

    def forward(self, img):
        flag = False
        flag_rand = False
        for _ in range(512):
            p1 = self.rrc.get_params(img, self.rrc.scale, self.rrc.ratio)
            p2 = self.rrc.get_params(img, self.rrc.scale, self.rrc.ratio)

            if torch.rand(1) < self.p:
                flag_rand = True
                break
            if calc_iou(p1, p2) > self.schedule[self.epoch]:
                continue
            flag = True
            break

        _, height, width = tf.get_dimensions(img)
        img1 = tf.resized_crop(img, *p1, self.rrc.size, self.rrc.interpolation, antialias=self.rrc.antialias)
        img2 = tf.resized_crop(img, *p2, self.rrc.size, self.rrc.interpolation, antialias=self.rrc.antialias)

        img1, out1 = self.apply_transforms(img1)
        img2, out2 = self.apply_transforms(img2)

        params = [[height, width, *p1, flag, flag_rand], *out1], [[height, width, *p2, flag, flag_rand], *out2]

        return [img1, img2], params

    def set_epoch(self, epoch):
        self.epoch = epoch

    def apply_transforms(self, img):
        params = []
        for t in self.transforms.transforms:
            if isinstance(t, (ToTensor, Normalize)):
                img = t(img)
            else:
                img, p = t(img)
                params.append(p)
        return img, params


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep):
    iters = np.arange(epochs * niter_per_ep)
    if base_value < final_value:
        schedule = base_value + 0.5 * (final_value - base_value) * (1 + np.cos(np.pi * (iters / len(iters) + 1)))
        return schedule
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * (iters / len(iters))))
    return schedule
