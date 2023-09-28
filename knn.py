import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as nnf
import torchvision.models as torch_models
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets

import distributed as dist
import models.vision_transformer as vits
import transforms
import utils


def eval_knn(
        arch: str,
        arch_kwargs: Optional[dict | None],
        batch_size: int,
        dataset: str,
        data_path: str,
        nb_knn: list,
        temperature: float,
        pretrained_weights: str,
        ckp_key: str,
        num_workers: int,
        output_dir: str,
        dump_features: bool,
        use_cuda: bool,
) -> None:
    print("Running KNN evaluation...")
    utils.print_args(locals())

    if os.path.isfile(os.path.join(output_dir, "knn.log")):
        print("KNN evaluation has been done before. Skipping...")
        return

    if os.path.isfile(os.path.join(output_dir, "train_features.pth")):
        print("Loading saved features...")
        train_features = torch.load(Path(output_dir) / "train_features.pth", map_location="cpu")
        train_labels = torch.load(Path(output_dir) / "train_labels.pth", map_location="cpu")
        test_features = torch.load(Path(output_dir) / "test_features.pth", map_location="cpu")
        test_labels = torch.load(Path(output_dir) / "test_labels.pth", map_location="cpu")
        print("Features ready!")
    else:
        transform = transforms.EvalValTransform()
        train_dataset = ReturnIndexDataset(os.path.join(args.data_path, "train"), transform=transform)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            sampler=DistributedSampler(train_dataset, shuffle=False),
            drop_last=False,
        )

        val_dataset = ReturnIndexDataset(os.path.join(args.data_path, "val"), transform=transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        print(f"Data loaded with {len(train_dataset)} train and {len(val_dataset)} val images.")

        if arch_kwargs is None:
            arch_kwargs = {}

        if arch in torch_models.__dict__.keys():  # torchvision models
            model = torch_models.__dict__[arch](**arch_kwargs)
            model.fc = nn.Identity()
        elif arch in vits.__dict__.keys():  # vision transformer models
            model = vits.__dict__[arch](**arch_kwargs)
        else:
            raise NotImplementedError(f"Architecture {arch} not supported")

        model.cuda()
        utils.load_pretrained_weights(model, pretrained_weights, ckp_key)
        model.eval()
        print(f"Model {arch} built.")

        # ============ extract features ... ============
        print("Extracting features for train set...")
        train_features = extract_features(model, train_loader)
        print("Extracting features for val set...")
        test_features = extract_features(model, val_loader)
        print("Features are ready!")

        train_labels = torch.tensor([s[-1] for s in train_dataset.samples]).long()
        test_labels = torch.tensor([s[-1] for s in val_dataset.samples]).long()

        if dump_features and dist.is_main_process():
            print("Dumping features...")
            torch.save(train_features, Path(output_dir) / "train_features.pth")
            torch.save(train_labels, Path(output_dir) / "train_labels.pth")
            torch.save(test_features, Path(output_dir) / "test_features.pth")
            torch.save(test_labels, Path(output_dir) / "test_labels.pth")
            print("Features are dumped!")

    if use_cuda:
        train_features, train_labels = train_features.cuda(), train_labels.cuda()
        test_features, test_labels = test_features.cuda(), test_labels.cuda()
    else:
        if dist.is_enabled():
            print("Stopping knn evaluation because use_cuda is False and distributed is enabled.")
            return

    out = []
    if dist.is_main_process():
        for k in nb_knn:
            top1, top5 = knn_classifier(train_features, train_labels, test_features, test_labels, k, temperature)
            out.append({"k": k, "top1": top1, "top5": top5})
            print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")

        with (Path(output_dir) / "knn.log").open("a") as f:
            for line in out:
                f.write(json.dumps(line) + "\n")
    print("knn evaluation is done!")


@torch.no_grad()
def extract_features(model, data_loader):
    metric_logger = utils.MetricLogger(delimiter=" ")
    features = None
    for samples, index in metric_logger.log_every(data_loader, 100):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        feats = model(samples)
        feats = nnf.normalize(feats, dim=1, p=2)

        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            print(f"Storing features into tensor of shape {features.shape}")

            # get indexes from all processes
            y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
            y_l = list(y_all.unbind(0))
            y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
            y_all_reduce.wait()
            index_all = torch.cat(y_l)

            # share features between processes
            feats_all = torch.empty(
                dist.get_world_size(),
                feats.size(0),
                feats.size(1),
                dtype=feats.dtype,
                device=feats.device,
            )
            output_l = list(feats_all.unbind(0))
            output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
            output_all_reduce.wait()

            # update storage feature matrix
            if dist.get_rank() == 0:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, temperature, num_classes=1000):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[idx: min((idx + imgs_per_chunk), num_test_images), :]
        targets = test_labels[idx: min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(temperature).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx


def get_knn_args_parser():
    p = argparse.ArgumentParser("kNN eval")
    p.add_argument("--arch", default="resnet50", type=str, help="model architecture")
    p.add_argument("--arch-kwargs", default=None, type=utils.arg_dict, help="model architecture kwargs")
    p.add_argument("--batch-size", default=1024, type=int, help="batch size")
    p.add_argument("--dataset", default="imagenet", type=str, help="dataset name")
    p.add_argument("--data-path", default=None, type=str, help="path to data")
    p.add_argument("--nb-knn", default=[10, 20, 100, 200], type=list, help="number of nearest neighbors")
    p.add_argument("--temperature", default=0.07, type=float, help="temperature for similarity")
    p.add_argument("--pretrained-weights", default="checkpoint.pth", type=str, help="path to pretrained weights")
    p.add_argument("--ckp-key", default="model", type=str, help="key to load pretrained weights")
    p.add_argument("--num-workers", default=8, type=int, help="number of workers")
    p.add_argument("--output-dir", default=".", type=str, help="path to output directory")
    p.add_argument("--dump-features", default=False, type=utils.bool_flag, help="dump features to disk")
    p.add_argument("--use-cuda", default=False, type=utils.bool_flag, help="whether to use cuda for knn-classification")
    return p


if __name__ == "__main__":
    parser = get_knn_args_parser()
    args = parser.parse_args()

    local_rank, rank, world_size = dist.setup()

    eval_knn(**vars(args))
