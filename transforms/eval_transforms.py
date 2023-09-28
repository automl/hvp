from typing import Tuple, Union
import torchvision.transforms as tt
from transforms.utils import IMAGENET_NORMALIZE


class EvalTrainTransform:
    def __init__(
            self,
            img_size: int = 224,
            scale: Tuple[float, float] = (0.08, 1.0),
            interpolation: tt.InterpolationMode = tt.InterpolationMode.BICUBIC,
            hf_prob: float = 0.5,
            normalize: Union[None, dict] = IMAGENET_NORMALIZE,
    ):
        transform = [
            tt.RandomResizedCrop(img_size, scale, interpolation=interpolation),
            tt.RandomHorizontalFlip(p=hf_prob),
            tt.ToTensor(),
        ]
        if normalize:
            transform.append(tt.Normalize(**normalize))
        self.transform = tt.Compose(transform)

    def __call__(self, image):
        return self.transform(image)


class EvalValTransform:
    def __init__(
            self,
            img_size: int = 224,
            resize_size: int = 256,
            interpolation: tt.InterpolationMode = tt.InterpolationMode.BICUBIC,
            normalize: Union[None, dict] = IMAGENET_NORMALIZE,
    ):
        transform = [
            tt.Resize(resize_size, interpolation=interpolation),
            tt.CenterCrop(img_size),
            tt.ToTensor(),
        ]
        if normalize:
            transform.append(tt.Normalize(**normalize))
        self.transform = tt.Compose(transform)

    def __call__(self, image):
        return self.transform(image)
