from typing import List, Optional, Tuple, Union

from PIL.Image import Image
from torch import Tensor

import transforms.custom_transforms as ct
from transforms.multi_view_transform import MultiViewTransform
from transforms.utils import IMAGENET_NORMALIZE


class SimCLRTransform(MultiViewTransform):
    def __init__(
            self,
            image_size: int = 224,
            cj_prob: float = 0.8,
            cj_strength: float = 1.0,
            cj_bright: float = 0.8,
            cj_contrast: float = 0.8,
            cj_sat: float = 0.8,
            cj_hue: float = 0.2,
            min_scale: float = 0.08,
            gray_prob: float = 0.2,
            blur_prob: float = 0.5,
            kernel_size: Optional[float] = 23,
            sigmas: Tuple[float, float] = (0.1, 2),
            hf_prob: float = 0.5,
            normalize: Union[None, dict] = IMAGENET_NORMALIZE,
            num_views: int = 2,
            return_params: bool = False
    ):
        view_transform = SimCLRViewTransform(
            image_size=image_size,
            cj_prob=cj_prob,
            cj_strength=cj_strength,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_sat=cj_sat,
            cj_hue=cj_hue,
            min_scale=min_scale,
            gray_prob=gray_prob,
            blur_prob=blur_prob,
            kernel_size=kernel_size,
            sigmas=sigmas,
            hf_prob=hf_prob,
            normalize=normalize,
        )

        super().__init__(transforms=[view_transform for _ in range(num_views)], return_params=return_params)


class SimCLRViewTransform:
    def __init__(
            self,
            image_size: int = 224,
            cj_prob: float = 0.8,
            cj_strength: float = 1.0,
            cj_bright: float = 0.8,
            cj_contrast: float = 0.8,
            cj_sat: float = 0.8,
            cj_hue: float = 0.2,
            min_scale: float = 0.08,
            gray_prob: float = 0.2,
            blur_prob: float = 0.5,
            kernel_size: Optional[float] = 23,
            sigmas: Tuple[float, float] = (0.1, 2),
            hf_prob: float = 0.5,
            normalize: Union[None, dict] = IMAGENET_NORMALIZE,
    ):
        color_jitter = ct.RandomColorJitter(
            brightness=cj_strength * cj_bright,
            contrast=cj_strength * cj_contrast,
            saturation=cj_strength * cj_sat,
            hue=cj_strength * cj_hue,
            p=cj_prob,
        )

        transform = [
            ct.RandomResizedCrop(size=image_size, scale=(min_scale, 1.0)),
            ct.RandomHorizontalFlip(p=hf_prob),
            color_jitter,
            ct.RandomGrayscale(p=gray_prob),
            ct.RandomGaussianBlur(kernel_size=kernel_size, sigma=sigmas, p=blur_prob),
            ct.ToTensor(),
        ]
        if normalize:
            transform += [ct.Normalize(mean=normalize["mean"], std=normalize["std"])]
        self.transform = ct.Compose(transform)

    def __call__(self, image: Union[Tensor, Image], return_params=False) -> Tensor | Tuple[Tensor, List]:
        return self.transform(image, return_params=return_params)
