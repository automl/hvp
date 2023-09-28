from omegaconf import OmegaConf

import eval_linear
import eval_knn
import pretrain
import utils

if __name__ == "__main__":
    # load pretrain config
    cfg = OmegaConf.load('configs/test/_pretrain.yaml')

    # init distributed
    utils.distributed.init_distributed_mode(cfg)
    utils.fix_random_seeds(cfg.seed)

    print('STARTING PRETRAINING')
    pretrain.main(cfg)

    # load eval config
    eval_cfg = OmegaConf.load('configs/test/_eval_linear.yaml')
    eval_cfg.gpu = cfg.gpu
    eval_cfg.rank = cfg.rank
    eval_cfg.world_size = cfg.world_size

    # Test iNaturalist
    eval_cfg.dataset = "inat18"
    eval_cfg.data_path = "../"


    print('STARTING EVALUATION')
    eval_linear.main(eval_cfg)

    knn_cfg = OmegaConf.load('configs/test/_eval_knn.yaml')
    # copy dist parameters
    knn_cfg.gpu = cfg.gpu
    knn_cfg.rank = cfg.rank
    knn_cfg.world_size = cfg.world_size

    print('STARTING KNN EVALUATION')
    eval_knn.main(knn_cfg)
