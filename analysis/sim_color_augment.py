import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
import matplotlib.pyplot as plt

import builder
from data import MultiCropsTransform, make_pretrain_transform
from models.resnet import resnet50


if __name__ == "__main__":
    model = builder.SimSiam(
        resnet50,
        2048, 512, 3)
    pretrained_weights = r"C:\Users\ovi\Documents\SelfSupervisedLearning\exp_data\simsiam-minsim-collect_metrics-resnet50-ImageNet-ep100-bs256-select_cross-ncrops4-lr0.05-wd0.0001-mom0.9-seed0.checkpoint.pth"
    state_dict = torch.load(pretrained_weights, map_location="cpu")["model"]
    for k in list(state_dict.keys()):
        if k.startswith("module."):
            state_dict[k[len("module."):]] = state_dict[k]
        del state_dict[k]
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    model.cuda()
    model.eval()

    base_tranform = make_pretrain_transform()
    transform = MultiCropsTransform(base_tranform, 2)

    data_path = r"C:\Users\ovi\Documents\SelfSupervisedLearning\datasets\ImageNet\train"
    dataset = ImageFolder(data_path, transform=transform)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    criterion = nn.CosineSimilarity(dim=1).cuda()
    ssim = SSIM(reduction='none').cuda()

    out1 = []
    out2 = []
    for i, (images, _) in enumerate(data_loader):
        images = [img.cuda(non_blocking=True) for img in images]
        sim_aug_imgs = criterion(images[0].flatten(start_dim=1), images[1].flatten(start_dim=1))
        # sim_aug_imgs = ssim(images[0], images[1])
        with torch.no_grad():
            p1, p2, z1, z2 = model(*images)
            loss = criterion(p1, z2) + criterion(p2, z1)

        out1.extend(sim_aug_imgs.tolist())
        out2.extend(loss.tolist())
        # print(out1)
        # print(out2)
        # break

    plt.scatter(out1, out2)
    plt.show()
