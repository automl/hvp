# Beyond Random Augmentations: Pretraining with Hard Views

Other branches are available here (required for double-blind submission):
- [DINO branch](https://github.com/automl/pretraining-hard-views/tree/dino)
- [iBOT branch](https://github.com/automl/pretraining-hard-views/tree/ibot)
- [SimSiam Branch (main)](https://anonymous.4open.science/r/pretraining-hard-views/README.md)
- [SimCLR Branch](https://github.com/automl/pretraining-hard-views/tree/simclr)

## Setup:
```
conda env create -f environment.yaml
conda activate hvp
conda install -c conda-forge tensorboard
pip install omegaconf
```

## Download Model Files
(include pretraining, linear evaluation and finetuning checkpoints for both vanilla and hvp models)
- [DINO models](https://bit.ly/4dirXw1) (45G)
- [iBOT models](https://bit.ly/3WBEiGc) (11G)
- [SimSiam models](https://bit.ly/3WG2p5x) (20G)
- [SimCLR models](https://bit.ly/3LE64eL) (66G)
