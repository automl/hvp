# PLOT-SCRIPTS

## metrics folder structure
```
# parent folder
/work/dlclarge2/rapanti-MinSimAugment/metrics

1. source metrics json files
/work/dlclarge2/rapanti-MinSimAugment/metrics/json/...

2. converted metrics
# pandas dataframe pickle files - original content
/work/dlclarge2/rapanti-MinSimAugment/metrics/pickle/...

# for classification task
/work/dlclarge2/rapanti-MinSimAugment/metrics/pickle/params-for-classification/...
# with relative rrc-parameters
/work/dlclarge2/rapanti-MinSimAugment/metrics/pickle/params-for-classification-relative/...
```

## saved models folder structure
```
# parent folder
/work/dlclarge2/rapanti-MinSimAugment/saved-models

# atm only simsiam minsim and vanilla // S = {0, 1}
- .../saved-models/simsiam-minsim-resnet50-ImageNet-ep100-bs256-seed{S}
- .../saved-models/simsiam-vanilla-resnet50-ImageNet-ep100-bs256-seed{S}

# all checkpoints available
- .../saved-models/.../checkpoints.pth  # full training
- .../saved-models/.../checkpointsXXXX.pth  # checkpoints at epoch X
```
