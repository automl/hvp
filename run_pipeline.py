from omegaconf import OmegaConf

import eval_linear
import pretrain


if __name__ == "__main__":
    cfg = OmegaConf.load('pretrain.yaml')
    print('STARTING PRETRAINING')
    pretrain.main(cfg)

    eval_cfg = OmegaConf.load('eval_linear.yaml')
    # copy dist parameters
    eval_cfg.gpu = cfg.gpu
    eval_cfg.rank = cfg.rank
    eval_cfg.world_size = cfg.world_size
    print('STARTING EVALUATION')
    eval_linear.main(eval_cfg)
