import math
import numbers
import warnings
from collections.abc import Sequence
from typing import Tuple, List, Optional

import torch
from torch import Tensor

from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as ttf


class Compose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, return_params=False):
        params = []
        for t in self.transforms:
            out = t(img, return_params=return_params)
            if isinstance(out, tuple):
                img, *param = out
                params.extend(param)
            else:
                img = out
        if return_params:
            return img, Tensor(params)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor:

    def __call__(self, pic, *args, **kwargs):
        return ttf.to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class PILToTensor:

    def __call__(self, pic, *args, **kwargs):
        return ttf.pil_to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(torch.nn.Module):

    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor: Tensor, *args, **kwargs) -> Tensor:
        return ttf.normalize(tensor, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class RandomHorizontalFlip(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, return_params=False):
        param = 0.
        if torch.rand(1) < self.p:
            img = ttf.hflip(img)
            param = 1.
        if return_params:
            return img, param
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomResizedCrop(torch.nn.Module):

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(
            img: Tensor, scale: List[float], ratio: List[float]
    ) -> Tuple[int, int, int, int]:
        width, height = ttf.get_image_size(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img, return_params=False):
        width, height = ttf.get_image_size(img)
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = ttf.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        if return_params:
            return img, width, height, i, j, h, w
        return img

    def __repr__(self):
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class RandomColorJitter(torch.nn.Module):

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.5):
        super().__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        self.p = p

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness: Optional[List[float]],
                   contrast: Optional[List[float]],
                   saturation: Optional[List[float]],
                   hue: Optional[List[float]]
                   ) -> Tuple[Tensor, Optional[float], Optional[float], Optional[float], Optional[float]]:
        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def forward(self, img, return_params=False):
        params = 0, 0, 0, 0, 0, 0, 0, 0
        if torch.rand(1) < self.p:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
                self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img = ttf.adjust_brightness(img, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img = ttf.adjust_contrast(img, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img = ttf.adjust_saturation(img, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img = ttf.adjust_hue(img, hue_factor)

            params = *fn_idx.tolist(), brightness_factor, contrast_factor, saturation_factor, hue_factor

        if return_params:
            return img, *params
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


class RandomGrayscale(torch.nn.Module):

    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, img, return_params=False):
        num_output_channels = ttf.get_image_num_channels(img)
        param = 0.
        if torch.rand(1) < self.p:
            img = ttf.rgb_to_grayscale(img, num_output_channels=num_output_channels)
            param = 1.
        if return_params:
            return img, param
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={0})'.format(self.p)


class RandomGaussianBlur(torch.nn.Module):

    def __init__(self, kernel_size, sigma=(0.1, 2.0), p=0.5):
        super().__init__()
        self.kernel_size = _setup_size(kernel_size, "Kernel size should be a tuple/list of two integers")
        for ks in self.kernel_size:
            if ks <= 0 or ks % 2 == 0:
                raise ValueError("Kernel size value should be an odd and positive number.")

        if isinstance(sigma, numbers.Number):
            if sigma <= 0:
                raise ValueError("If sigma is a single number, it must be positive.")
            sigma = (sigma, sigma)
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            if not 0. < sigma[0] <= sigma[1]:
                raise ValueError("sigma values should be positive and of the form (min, max).")
        else:
            raise ValueError("sigma should be a single number or a list/tuple with length 2.")

        self.sigma = sigma
        self.p = p

    @staticmethod
    def get_params(sigma_min: float, sigma_max: float) -> float:
        return torch.empty(1).uniform_(sigma_min, sigma_max).item()

    def forward(self, img: Tensor, return_params: bool = False) -> tuple[Tensor, float] | Tensor:
        sigma = 0.
        if torch.rand(1).item() < self.p:
            sigma = self.get_params(self.sigma[0], self.sigma[1])
            img = ttf.gaussian_blur(img, self.kernel_size, [sigma, sigma])
        if return_params:
            return img, sigma
        return img

    def __repr__(self):
        s = '(kernel_size={}, '.format(self.kernel_size)
        s += 'sigma={})'.format(self.sigma)
        return self.__class__.__name__ + s


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


def _check_sequence_input(x, name, req_sizes):
    msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError("{} should be a sequence of length {}.".format(name, msg))
    if len(x) not in req_sizes:
        raise ValueError("{} should be sequence of length {}.".format(name, msg))


def _setup_angle(x, name, req_sizes=(2,)):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError("If {} is a single number, it must be positive.".format(name))
        x = [-x, x]
    else:
        _check_sequence_input(x, name, req_sizes)

    return [float(d) for d in x]


class RandomInvert(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, return_params=False):
        param = 0.
        if torch.rand(1).item() < self.p:
            img = ttf.invert(img)
            param = 1.
        if return_params:
            return img, param
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomSolarize(torch.nn.Module):

    def __init__(self, threshold, p=0.5):
        super().__init__()
        self.threshold = threshold
        self.p = p

    def forward(self, img, return_params=False):
        param = 0.
        if torch.rand(1).item() < self.p:
            img = ttf.solarize(img, self.threshold)
            param = 1.
        if return_params:
            return img, param
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(threshold={},p={})'.format(self.threshold, self.p)
