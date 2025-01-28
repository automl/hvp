# Beyond Random Augmentations: Pretraining with Hard Views

Other branches are available here:
- [DINO branch](https://github.com/automl/hvp/tree/dino)
- [iBOT branch](https://github.com/automl/hvp/tree/ibot)
- [SimSiam Branch (main)](https://github.com/automl/hvp/)
- [SimCLR Branch](https://github.com/automl/hvp/tree/simclr)

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

## Citation
Please acknowledge the usage of this code by citing the following publication:

```
@inproceedings{ferreira-iclr25a,
  title        = {Beyond Random Augmentations: Pretraining with Hard Views},
  author       = {F. Ferreira and I. Rapant and J. Franke and F. Hutter},
  booktitle    = {The Thirteenth International Conference on Learning Representations},
  year         = {2025},
  URL          = {https://openreview.net/forum?id=AK1C55o4r7}
}
```
