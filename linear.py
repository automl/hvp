import argparse
import json
import os
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
import torchvision.models as torch_models
from torch.nn.functional import cross_entropy
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler

import data
import distributed as dist
# import models.vision_transformer as vits
import optimizers
import transforms
import utils
from models.linear_classifier import LinearClassifier


def eval_linear(
        arch: str,
        arch_kwargs: Optional[Dict | None],
        batch_size: int,
        epochs: int,
        optim: str,
        lr: float,
        dataset: str,
        data_path: str,
        output_dir: str,
        pretrained_weights: str,
        ckp_key: str,
        num_workers: int,
        val_freq: int,
        gpu_id: int,
) -> None:
    print("Running Linear evaluation...")
    utils.print_args(locals())

    batch_size_per_gpu = batch_size // dist.get_world_size()
    train_transform = transforms.EvalTrainTransform()
    train_dataset, num_classes = data.make_dataset(data_path, dataset, True, train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        num_workers=num_workers,
        pin_memory=True,
        sampler=DistributedSampler(train_dataset),
    )

    val_transform = transforms.EvalValTransform()
    val_dataset, _ = data.make_dataset(data_path, dataset, False, val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_per_gpu,
        num_workers=num_workers,
        pin_memory=True,
        sampler=SequentialSampler(val_dataset),
    )
    print(f"Data loaded with {len(train_dataset)} train and {len(val_dataset)} val images.")

    if arch_kwargs is None:
        arch_kwargs = {}

    if arch in torch_models.__dict__.keys():  # torchvision models
        model = torch_models.__dict__[arch](**arch_kwargs)
        embed_dim = model.fc.weight.shape[1]
        model.fc = nn.Identity()
    # elif arch in vits.__dict__.keys():  # vision transformer models
    #     model = vits.__dict__[arch](**arch_kwargs)
    #     embed_dim = model.embed_dim * (n_last_blocks + int(avgpool_patchtokens))
    #     model.fc = nn.Identity()
    else:
        raise NotImplementedError(f"Architecture {arch} not supported")

    model.cuda()
    utils.load_pretrained_weights(model, pretrained_weights, ckp_key)
    model.eval()
    print(f"Model {arch} built.")

    linear_classifier = LinearClassifier(embed_dim, num_classes)
    linear_classifier.cuda()
    linear_classifier = DDP(linear_classifier, device_ids=[gpu_id])

    optimizer = optimizers.configure_optimizer(
        optim,
        linear_classifier.parameters(),
        lr=lr * batch_size / 256.,
        momentum=0.9,
        weight_decay=0,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    utils.restart_from_checkpoint(
        os.path.join(output_dir, "linear_checkpoint.pth"),
        run_variables=to_restore,
        state_dict=linear_classifier,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    for epoch in range(start_epoch, epochs):
        train_loader.sampler.set_epoch(epoch)  # noqa

        train_stats = train_one_epoch(model, linear_classifier, optimizer, train_loader, epoch)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if epoch % val_freq == 0 or epoch == epochs - 1:
            test_stats = validate(val_loader, model, linear_classifier)
            print(f"Accuracy at epoch {epoch} of the network on the test images: {test_stats['acc1']:.1f}%")
            best_acc = max(best_acc, test_stats["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}

        if dist.is_main_process():
            with (Path(output_dir) / "linear.log").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": linear_classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(output_dir, "linear_checkpoint.pth"))
    print("Training of the supervised linear classifier on frozen features completed.\n"
          "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


def train_one_epoch(model, linear_classifier, optimizer, loader, epoch):
    linear_classifier.train()
    metric_logger = utils.MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for images, targets in metric_logger.log_every(loader, 20, header):
        images = images.cuda(non_blocking=True)
        target = targets.cuda(non_blocking=True)

        with torch.no_grad():
            output = model(images)
        logits = linear_classifier(output)

        # compute cross entropy loss
        loss = cross_entropy(logits, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(val_loader, model, linear_classifier):
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter=" ")
    header = 'Test:'
    for images, targets in metric_logger.log_every(val_loader, 20, header):
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        output = model(images)
        output = linear_classifier(output)
        loss = cross_entropy(output, targets)

        acc1, acc5 = utils.accuracy(output, targets, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def get_linear_args_parser():
    p = argparse.ArgumentParser(description="linear eval")
    p.add_argument("--arch", default="resnet50", type=str, help="Name of the architecture")
    p.add_argument("--arch_kwargs", default=None, type=utils.arg_dict, help="kwargs for the architecture")
    p.add_argument("--batch-size", default=4096, type=int, help="total batch size")
    p.add_argument("--epochs", default=90, type=int, help="number of total epochs to run")
    p.add_argument("--optim", default="sgd", type=str, help="optimizer")
    p.add_argument("--lr", default=0.1, type=float, help="learning rate")
    p.add_argument("--dataset", default="imagenet", type=str, help="dataset name")
    p.add_argument("--data-path", default=None, type=str, help="path to data")
    p.add_argument("--output-dir", default=".", type=str, help="path to save logs and checkpoints")
    p.add_argument("--pretrained-weights", default="checkpoint.pth", type=str, help="path to pretrained weights")
    p.add_argument("--ckp-key", default="model", type=str, help="key to load pretrained weights")
    p.add_argument("--num-workers", default=8, type=int, help="number of workers")
    p.add_argument("--val-freq", default=1, type=int, help="validation frequency (epochs)")
    return p


if __name__ == "__main__":
    parser = get_linear_args_parser()
    args = parser.parse_args()

    local_rank, rank, world_size = dist.setup()

    eval_linear(
        **vars(args),
        gpu_id=local_rank,
    )
