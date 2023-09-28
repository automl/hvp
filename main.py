from omegaconf import OmegaConf
from torch.backends import cudnn

import distributed as dist
import utils
import pretrain
from linear import eval_linear
from knn import eval_knn


if __name__ == '__main__':
    gpu_id, rank, world_size = dist.setup()
    print("git:\n  {}\n".format(utils.get_sha()))

    pretrain_cfg = OmegaConf.load('pretrain.yaml')
    utils.fix_random_seeds(pretrain_cfg.seed)
    cudnn.benchmark = True

    pretrain.main(
        **pretrain_cfg,
        gpu_id=gpu_id,
    )

    linear_cfg = OmegaConf.load('linear.yaml')
    eval_linear(
        **linear_cfg,
        gpu_id=gpu_id,
    )

    knn_cfg = OmegaConf.load('knn.yaml')
    eval_knn(
        **knn_cfg,
    )
