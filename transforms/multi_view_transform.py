from typing import List, Union, Dict

from PIL.Image import Image
from torch import Tensor


class MultiViewTransform:

    def __init__(self, transforms: List, return_params: bool = False):
        self.transforms = transforms
        self.return_params = return_params

    def __call__(self, image: Union[Tensor, Image]) -> Dict:  # Tuple[Any, Any] | List[Any]:
        out = [transform(image, return_params=self.return_params) for transform in self.transforms]
        if self.return_params:
            return {"images": [i[0] for i in out], "params": [i[1] for i in out]}
        return {"images": out}
