import argparse
import json
import os
import sys
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

import data
import utils
from models import resnet, resnet_cifar, vision_transformer as vits
from utils import distributed as dist, optimizers


def main(cfg):
    dist.init_distributed_mode(cfg) if not dist.is_enabled() else None
    cudnn.benchmark = True

    print("git:\n  {}\n".format(utils.get_sha()))
    print(OmegaConf.to_yaml(cfg))

    # prepare data
    if cfg.dataset == "CIFAR10":
        mean, std = data.CIFAR10_DEFAULT_MEAN, data.CIFAR10_DEFAULT_STD
    elif cfg.dataset == "Flowers102":
        mean, std = data.FLOWERS102_DEFAULT_MEAN, data.FLOWERS102_DEFAULT_STD
    else:
        mean, std = data.IMAGENET_DEFAULT_MEAN, data.IMAGENET_DEFAULT_STD

    val_transform = data.make_classification_val_transform(
        resize_size=cfg.resize_size,
        crop_size=cfg.crop_size,
        mean=mean,
        std=std,
    )

    val_data, cfg.num_labels = data.make_dataset(cfg.data_path, cfg.dataset, False, val_transform)

    batch_size_per_gpu = cfg.batch_size // dist.get_world_size()

    sampler = torch.utils.data.SequentialSampler(val_data)
    val_loader = torch.utils.data.DataLoader(
        val_data,
        sampler=sampler,
        batch_size=batch_size_per_gpu,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False
    )

    train_transform = data.make_classification_train_transform(
        crop_size=cfg.crop_size,
        mean=mean,
        std=std,
    )

    train_data, _ = data.make_dataset(cfg.data_path, cfg.dataset, True, train_transform)

    sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        sampler=sampler,
        batch_size=batch_size_per_gpu,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(train_data)} train and {len(val_data)} val images.")

    # create model
    print("=> creating model '{}'".format(cfg.arch))
    if cfg.arch in vits.__dict__.keys():
        model = vits.__dict__[cfg.arch](
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            num_classes=0,
        )
        embed_dim = model.embed_dim * (cfg.n_last_blocks + int(cfg.avgpool))
        model.fc = nn.Identity()
    elif cfg.arch in resnet_cifar.__dict__.keys():
        model = resnet_cifar.__dict__[cfg.arch](num_classes=cfg.num_labels)
        embed_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif cfg.arch in resnet.__dict__.keys():
        model = resnet.__dict__[cfg.arch](num_classes=cfg.num_labels)
        embed_dim = model.fc.in_features
        model.fc = nn.Identity()
    else:
        print(f"Unknown architecture: {cfg.arch}")
        sys.exit(1)
    model.cuda()

    # load weights to evaluate
    utils.load_pretrained_weights(model, cfg.pretrained, cfg.ckp_key)
    print(f"Model {cfg.arch} built.")
    print(model)

    # init the fc layer
    linear_classifier = LinearClassifier(embed_dim, num_labels=cfg.num_labels)
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[cfg.gpu])

    if cfg.finetune:
        for p in model.parameters():
            p.requires_grad = True
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu])
        model.train()
        params_to_optimize = list(model.parameters()) + list(linear_classifier.parameters())
    else:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        params_to_optimize = linear_classifier.parameters()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # infer learning rate
    init_lr = cfg.lr * cfg.batch_size / 256
    if cfg.optimizer == "lars":
        optimizer = optimizers.LARS(params_to_optimize, init_lr,
                                    momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_to_optimize, init_lr,
                                    momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_to_optimize, init_lr,
                                      weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs, eta_min=0)

    log_dir = os.path.join(cfg.output_dir, "tensorboard")
    board = SummaryWriter(log_dir) if dist.is_main_process() else None

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}

    checkpoint_name = "checkpoint.pth.tar" if cfg.dataset == "ImageNet" else f"checkpoint_{cfg.dataset}.pth.tar"
    if not cfg.finetune:
        checkpoint_name = f"checkpoint.pth.tar" if cfg.dataset == "ImageNet" else f"checkpoint_{cfg.dataset}_linear_eval.pth.tar"

    if cfg.finetune:
        # load classifier and backbone
        utils.restart_from_checkpoint(
            os.path.join(cfg.output_dir, checkpoint_name),
            run_variables=to_restore,
            model=model,
            linear_classifier=linear_classifier,
            optimizer=optimizer,
        )
    else:
        # only classifier needs to be loaded
        utils.restart_from_checkpoint(
            os.path.join(cfg.output_dir, checkpoint_name),
            run_variables=to_restore,
            linear_classifier=linear_classifier,
            optimizer=optimizer,
        )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    for epoch in range(start_epoch, cfg.epochs):
        train_loader.sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, cfg)

        # train for one epoch
        train_stats = train(train_loader, model, linear_classifier, criterion, optimizer, epoch, cfg, board)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        # evaluate on validation set
        if epoch % cfg.val_freq == 0 or epoch == cfg.epochs - 1:
            test_stats = validate(val_loader, model, linear_classifier, criterion, cfg)
            print(f"Accuracy at epoch {epoch} of the network on the {len(val_data)} test images: "
                  f"{test_stats['acc1']:.1f}%")
            # remember best acc@1 and save checkpoint
            best_acc = max(test_stats["acc1"], best_acc)
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}

            if board:
                board.add_scalar(tag="acc1", scalar_value=test_stats["acc1"], global_step=epoch)
                board.add_scalar(tag="acc5", scalar_value=test_stats["acc5"], global_step=epoch)
                board.add_scalar(tag="best-acc", scalar_value=best_acc, global_step=epoch)

        if dist.is_main_process():
            with (Path(cfg.output_dir) / "eval.log").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "linear_classifier": linear_classifier.state_dict(),
                "best_acc": best_acc,
            }
            path = os.path.join(cfg.output_dir, checkpoint_name)
            torch.save(save_dict, path)

    print("Training of the supervised linear classifier on frozen features completed.\n"
          "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


def train(loader, model, linear_classifier, criterion, optimizer, epoch, cfg, board):
    # switch to train mode
    linear_classifier.train()
    if cfg.finetune:
        model.train()
    else:
        model.eval()

    metric_logger = utils.MetricLogger(delimiter=" ")
    header = 'Epoch: [{}/{}]'.format(epoch, cfg.epochs)
    for it, (images, targets) in enumerate(metric_logger.log_every(loader, 10, header)):
        it = len(loader) * epoch + it

        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        # compute output
        if cfg.finetune:
            output = model(images)
        else:
            with torch.no_grad():
                if "vit" in cfg.arch:
                    intermediate_output = model.get_intermediate_layers(images, cfg.n_last_blocks)
                    output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                    if cfg.avgpool:
                        output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                        output = output.reshape(output.shape[0], -1)
                else:
                    output = model(images)

        output = linear_classifier(output)
        loss = criterion(output, targets)

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc1, acc5 = utils.accuracy(output, targets, topk=(1, 5))

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(acc1=acc1[0])
        metric_logger.update(acc5=acc5[0])

        if dist.is_main_process() and it % cfg.logger_freq:
            board.add_scalar(tag="eval acc1", scalar_value=acc1, global_step=it)
            board.add_scalar(tag="eval loss", scalar_value=loss.item(), global_step=it)
            board.add_scalar(tag="eval lr", scalar_value=optimizer.param_groups[0]["lr"], global_step=it)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(loader, model, linear_classifier, criterion, cfg):
    # switch to evaluate mode
    linear_classifier.eval()
    model.eval()

    metric_logger = utils.MetricLogger(delimiter=" ")
    header = 'Test:'
    with torch.no_grad():
        for i, (images, target) in enumerate(metric_logger.log_every(loader, 10, header)):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            if cfg.finetune:
                output = model(images)
            else:
                with torch.no_grad():
                    if "vit" in cfg.arch:
                        intermediate_output = model.get_intermediate_layers(images, cfg.n_last_blocks)
                        output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                        if cfg.avgpool:
                            output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                            output = output.reshape(output.shape[0], -1)
                    else:
                        output = model(images)

            output = linear_classifier(output)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # logging
            torch.cuda.synchronize()
            batch_size = images.size(0)
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


def adjust_learning_rate(optimizer, init_lr, epoch, cfg):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / cfg.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


def get_args_parser():
    p = argparse.ArgumentParser(description='PyTorch Eval-Linear ImageNet', add_help=False)
    p.add_argument('--dataset', default="ImageNet", type=str,
                   help='Specify dataset. (default: ImageNet)')
    p.add_argument('--data_path', type=str,
                   help='(root) path to dataset')
    p.add_argument('-a', '--arch', type=str,
                   help="Name of architecture to train (default: resnet50)")
    p.add_argument('--epochs', type=int,
                   help='number of total epochs to run (default: 90)')
    p.add_argument('-b', '--batch-size', type=int,
                   help='total-batch-size (default: 4096)')
    p.add_argument('--lr', type=float,
                   help='initial (base) learning rate (default: 0.1)')
    p.add_argument('--momentum', type=float,
                   help='momentum (default: 0.9)')
    p.add_argument('--wd', '--weight_decay', type=float, dest='weight_decay',
                   help='weight decay (default: 0.)')
    p.add_argument('--resize_size', type=int,
                   help="Resize size of images before center-crop (default: 256)")
    p.add_argument('--crop_size', type=int,
                   help="Size of center-crop (default: 224)")
    p.add_argument('--optimizer', type=str, choices=['adamw', 'sgd', 'lars'],
                   help="Optimizer (default: sqd)")
    p.add_argument('--finetune', type=utils.bool_flag,
                   help="")

    # additional configs:
    p.add_argument('--pretrained', default="checkpoint.pth", type=str,
                   help="path to simsiam pretrained checkpoint (default: checkpoint.pth)")
    p.add_argument('--output_dir', default=".", type=str,
                   help='Path to save logs and checkpoints (default: .)')
    p.add_argument('--ckp_key', default="model", type=str,
                   help='Checkpoint key (default: model)')
    p.add_argument('--val_freq', default=1, type=int,
                   help="Validate model every x epochs (default: 1)")
    p.add_argument('--logger_freq', default=50, type=int,
                   help="Log progress every x iterations to tensorboard (default: 50)")
    p.add_argument('--dist-url', default="env://", type=str,
                   help="url used to set up distributed training (default: env://)")
    p.add_argument('--dist-backend', default="nccl", type=str,
                   help="distributed backend (default: nccl)")
    p.add_argument('--num_workers', default=8, type=int,
                   help="number of data loading workers (default: 8)")
    p.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
            for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")

    return p


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
