from scipy.stats import pearsonr
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Normalize

import matplotlib.pyplot as plt

import builder
from models.resnet import resnet50

import transforms_p as tp


class SimSiamTransform(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rrc = tp.RandomResizedCrop(224, scale=(0.2, 1.0))
        self.norm = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def forward(self, x):
        img, p = self.rrc(x)
        return img, p


def calc_iou(p1, p2):
    def get_dims(x):
        # input: top, left, height, width
        # out: top, bottom, left, right
        return x[0], x[0] + x[2], x[1], x[1] + x[3]

    def calc_overlap(x1, x2):
        t1, b1, l1, r1 = get_dims(x1)
        t2, b2, l2, r2 = get_dims(x2)
        h = max(0, min(b1, b2) - max(t1, t2))
        w = max(0, min(r1, r2) - max(l1, l2))
        return h * w

    def calc_area(x):
        h, w = x[-2:]
        return h * w

    A = calc_area(p1)
    B = calc_area(p2)
    o = calc_overlap(p1, p2)
    return o / (A + B - o)


if __name__ == "__main__":
    torch.manual_seed(0)

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

    data_path = r"C:\Users\ovi\Documents\SelfSupervisedLearning\datasets\ImageNet\train"
    dataset = ImageFolder(data_path, transform=ToTensor())
    num_samples = len(dataset)

    samples = torch.randint(num_samples, (10,))

    # data_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=256,
    #     shuffle=False,
    #     num_workers=8,
    #     pin_memory=True,
    #     drop_last=True,
    # )

    criterion = nn.CosineSimilarity(dim=1).cuda()
    rrc = tp.RandomResizedCrop(224, scale=(0.2, 1.0))
    out_iou = []
    out_sim = []
    for index in samples:
        image = dataset[index][0].cuda()
        x1, x2, iou = [], [], []
        for _ in range(1000):
            image1, params1 = rrc(image)
            image2, params2 = rrc(image)

            iou.append(calc_iou(params1, params2))
            x1.append(image1)
            x2.append(image2)
        x1 = torch.stack(x1)
        x2 = torch.stack(x2)
        with torch.no_grad():
            z1, p1, z2, p2 = model(x1, x2)
            sim = (criterion(p1, z2) + criterion(p2, z1)) * 0.5

        out_sim.extend(sim.cpu().tolist())
        out_iou.extend(iou)

    print(pearsonr(out_iou, out_sim))
    plt.scatter(out_iou, out_sim)
    plt.show()
