from torch.backends import cudnn
import distributed as dist

import pretrain
from linear import eval_linear, get_linear_args_parser
from knn import eval_knn, get_knn_args_parser


if __name__ == '__main__':
    gpu_id, rank, world_size = dist.setup()
    cudnn.benchmark = True

    pretrain_cfg = get_linear_args_parser().parse_args()
    pretrain_cfg.epochs = 3
    pretrain_cfg.warmup_epochs = 0
    pretrain_cfg.batch_size = 64
    pretrain_cfg.dataset = "test224"
    pretrain_cfg.output_dir = "exp"
    pretrain.main(
        **vars(pretrain_cfg),
        gpu_id=gpu_id,
    )

    linear_cfg = get_linear_args_parser().parse_args()
    linear_cfg.epochs = 3
    linear_cfg.batch_size = 128
    linear_cfg.dataset = "test224"
    linear_cfg.pretrained_weights = "exp/checkpoint.pth"
    linear_cfg.output_dir = "exp"
    eval_linear(
        **vars(linear_cfg),
        gpu_id=gpu_id,
    )

    knn_cfg = get_knn_args_parser().parse_args()
    knn_cfg.batch_size = 128
    knn_cfg.dataset = "test224"
    knn_cfg.pretrained_weights = "exp/checkpoint.pth"
    knn_cfg.output_dir = "exp"
    eval_knn(
        **vars(knn_cfg),
    )
