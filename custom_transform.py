from typing import Sequence, Tuple

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter, GaussianBlur, InterpolationMode, RandomResizedCrop

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

CIFAR10_DEFAULT_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_DEFAULT_STD = (0.2023, 0.1994, 0.2010)


class TransformParams(object):
    def __init__(self,
                 crop_size: int = 224,
                 crop_scale: Tuple[float, float] = (0.2, 1.0),
                 interpolation: InterpolationMode = InterpolationMode.BILINEAR,
                 blur_prob: float = 0.5,
                 hflip_prob: float = 0.5,
                 mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
                 std: Sequence[float] = IMAGENET_DEFAULT_STD,
                 ):
        self.rrc = RandomResizedCrop(crop_size, scale=crop_scale, interpolation=interpolation)
        self.color_jitter = ColorJitter(0.4, 0.4, 0.4, 0.1)
        self.gray_prob = 0.2
        self.blur = GaussianBlur(9, (0.1, 2.0))
        self.blur_prob = blur_prob
        self.hflip_prob = hflip_prob
        self.mean = mean
        self.std = std

    def __call__(self, img):
        # RandomResizedCrop
        i, j, h, w = self.rrc.get_params(img, self.rrc.scale, self.rrc.ratio)
        _, height, width = F.get_dimensions(img)
        img = F.resized_crop(img, i, j, h, w, self.rrc.size, self.rrc.interpolation, antialias=self.rrc.antialias)
        params = [(height, width, i, j, h, w)]

        # RandomApply-ColorJitter
        if torch.rand(1) < 0.8:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
                self.color_jitter.get_params(self.color_jitter.brightness, self.color_jitter.contrast,
                                             self.color_jitter.saturation, self.color_jitter.hue)
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img = F.adjust_brightness(img, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img = F.adjust_contrast(img, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img = F.adjust_saturation(img, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img = F.adjust_hue(img, hue_factor)
            params.append((brightness_factor, contrast_factor, saturation_factor, hue_factor, fn_idx.tolist()))
        else:
            params.append(False)

        # RandomGrayscale
        if torch.rand(1) < self.gray_prob:
            num_output_channels, _, _ = F.get_dimensions(img)
            img = F.rgb_to_grayscale(img, num_output_channels=num_output_channels)
            params.append(True)
        else:
            params.append(False)

        # RandomApply-GaussianBlur
        if torch.rand(1) < self.blur_prob:
            params.append(False)
        else:
            sigma = self.blur.get_params(self.blur.sigma[0], self.blur.sigma[1])
            img = F.gaussian_blur(img, self.blur.kernel_size, [sigma, sigma])
            params.append(sigma)

        # RandomHorizontalFlip
        if torch.rand(1) < self.hflip_prob:
            img = F.hflip(img)
            params.append(True)
        else:
            params.append(False)

        img = F.to_tensor(img)
        img = F.normalize(img, self.mean, self.std, False)

        return img, params
