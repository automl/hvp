import argparse
import json
import math
import os
import sys
import time
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from omegaconf import OmegaConf
from torch.utils.data import DistributedSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter

import data
from utils import distributed as dist
import builder
import select_crops
import custom_transform
import utils

from models import resnet_cifar, resnet, vision_transformer as vits
from torchsummary import summary


def custom_collate(batch):
    bs = len(batch[0][0][0])
    images = [torch.stack([item[0][0][n] for item in batch]) for n in range(bs)]
    params = [[item[0][1][n] for item in batch] for n in range(bs)]
    # target = [item[1] for item in batch]
    return images, params


def main(cfg):
    dist.init_distributed_mode(cfg) if not dist.is_enabled() else None
    cudnn.benchmark = True

    print(f"git:\n  {utils.get_sha()}\n")
    print(OmegaConf.to_yaml(cfg))

    if cfg.arch in vits.__dict__.keys():
        arch = vits.__dict__[cfg.arch]
        proj_layer = 3
        encoder_params = {
            "img_size": cfg.crop_size,
            "patch_size": cfg.patch_size,
            "drop_path_rate": cfg.drop_path_rate,
            "num_classes": cfg.dim
        }
    elif cfg.arch in resnet_cifar.__dict__.keys():
        arch = resnet_cifar.__dict__[cfg.arch]
        proj_layer = 2
        encoder_params = {
            "num_classes": cfg.dim,
            "zero_init_residual": True
        }
    elif cfg.arch in resnet.__dict__.keys():
        arch = resnet.__dict__[cfg.arch]
        proj_layer = 3
        encoder_params = {
            "num_classes": cfg.dim,
            "zero_init_residual": True
        }
    else:
        print(f"Unknown architecture: {cfg.arch}")
        sys.exit(1)

    model = builder.SimSiam(
        arch,
        dim=cfg.dim,
        pred_dim=cfg.pred_dim,
        proj_layer=proj_layer,
        encoder_params=encoder_params
    ).cuda()

    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu])

    init_lr = cfg.lr * cfg.batch_size / 256

    criterion = nn.CosineSimilarity(dim=1).cuda()

    if cfg.fix_pred_lr:
        optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    if cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(optim_params, init_lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(optim_params, init_lr, weight_decay=cfg.weight_decay)  # to use with ViTs

    fp16 = torch.cuda.amp.GradScaler() if cfg.fp16 else None

    # ============ preparing data ... ============
    if cfg.dataset == "CIFAR10":
        mean, std = data.CIFAR10_DEFAULT_MEAN, data.CIFAR10_DEFAULT_STD
    else:
        mean, std = data.IMAGENET_DEFAULT_MEAN, data.IMAGENET_DEFAULT_STD

    transform = custom_transform.TransformParams(
        crop_size=cfg.crop_size,
        crop_scale=cfg.crop_scale,
        blur_prob=cfg.blur_prob,
        hflip_prob=cfg.hflip_prob,
        mean=mean,
        std=std,
    )
    multi_crops_transform = data.MultiCropsTransform(transform, cfg.num_crops)
    dataset, _ = data.make_dataset(cfg.data_path, cfg.dataset, True, multi_crops_transform)

    sampler = DistributedSampler(dataset)
    cfg.batch_size_per_gpu = cfg.batch_size // dist.get_world_size()
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=cfg.batch_size_per_gpu,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate,
    )

    select_fn = select_crops.names[cfg.select_fn]

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0, "total_time": 0}
    utils.restart_from_checkpoint(
        os.path.join(cfg.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        model=model,
        optimizer=optimizer,
        fp16=fp16,
    )
    start_epoch = to_restore["epoch"]
    total_time = to_restore["total_time"]

    log_dir = os.path.join(cfg.output_dir, "tensorboard")
    board = SummaryWriter(log_dir) if dist.is_main_process() else None

    for epoch in range(start_epoch, cfg.epochs):
        loader.sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, cfg)

        start = time.time()
        train_stats, metrics = train(loader, model, criterion, optimizer, epoch, cfg, fp16, board, select_fn)
        total_time += int(time.time() - start)

        save_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "fp16": fp16.state_dict() if fp16 is not None else None,
            "total_time": total_time,
        }
        utils.save_on_master(save_dict, os.path.join(cfg.output_dir, 'checkpoint.pth'))
        if cfg.saveckp_freq and epoch and epoch % cfg.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(cfg.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if dist.is_main_process():
            with (Path(cfg.output_dir) / "pretrain.log").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        if dist.is_main_process():
            with (Path(cfg.output_dir) / "metrics.json").open("a") as f:
                f.write(json.dumps(metrics) + "\n")

    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train(loader, model, criterion, optimizer, epoch, cfg, fp16, board, select_fn):
    model.train()
    metrics = {
        "epoch": epoch,
        "loss": [],
        "lr": [],
        "selected": [],
        "params": [],
        "sample-loss": [],
    }
    metric_logger = utils.MetricLogger(delimiter=" ")
    header = 'Epoch: [{}/{}]'.format(epoch, cfg.epochs)
    for it, (images, params) in enumerate(metric_logger.log_every(loader, cfg.print_freq, header)):
        it = len(loader) * epoch + it  # global training iteration

        images = [im.cuda(non_blocking=True) for im in images]

        x1, x2, selected, sample_loss = select_fn(images, model, fp16)

        with torch.cuda.amp.autocast(fp16 is not None):
            p1, p2, z1, z2 = model(x1=x1, x2=x2)
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        optimizer.zero_grad()
        if fp16 is None:
            loss.backward()
            if cfg.clip_grad:
                _ = utils.clip_gradients(model, cfg.clip_grad)
            optimizer.step()
        else:
            fp16.scale(loss).backward()
            if cfg.clip_grad:
                fp16.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                _ = utils.clip_gradients(model, cfg.clip_grad)

            fp16.step(optimizer)
            fp16.update()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        metrics["loss"].append(loss.item())
        metrics["lr"].append(optimizer.param_groups[0]["lr"])
        if cfg.use_adv_metric and it % cfg.adv_metric_freq == 0:
            metrics["selected"].append(selected.tolist())
            metrics["params"].append(params)
            metrics["sample-loss"].append(sample_loss.tolist())

        if dist.is_main_process() and it % cfg.logger_freq == 0:
            board.add_scalar("training loss", loss.item(), it)
            board.add_scalar("training lr", optimizer.param_groups[0]["lr"], it)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, metrics


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


def get_args_parser():
    p = argparse.ArgumentParser("SimSiam", description='Pytorch Pretraining on ImageNet', add_help=False)
    p.add_argument('--dataset', type=str, default="ImageNet",
                   help='Specify dataset (default: ImageNet)')
    p.add_argument('--data_path', type=str,
                   help='(root) path to dataset')
    p.add_argument('-a', '--arch', type=str,
                   choices=["vit_tiny", "vit_small", "vit_base",
                            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                            "resnet18_cifar", "resnet34_cifar", "resnet50_cifar", "resnet101_cifar", "resnet152_cifar"],
                   help="Name of architecture to train (default: resnet50)")
    p.add_argument('--epochs', type=int,
                   help='number of total epochs to run (default: 100)')
    p.add_argument('-b', '--batch_size', type=int,
                   help='total batch-size (default: 512)')
    p.add_argument('--lr', type=float,
                   help='initial (base) learning rate (default: 0.05)')
    p.add_argument('--momentum', type=float,
                   help='momentum of SGD solver (default: 0.9)')
    p.add_argument('--wd', '--weight_decay', dest="weight_decay", type=float,
                   help='weight decay (default: 1e-4)')
    p.add_argument('--optimizer', default='sgd', type=str,
                        choices=['adamw', 'sgd'],
                        help="""Type of optimizer. We recommend using adamw with ViTs.""")

    # simsiam specific parameters:
    p.add_argument('--dim', type=int,
                   help='feature dimension (default: 2048)')
    p.add_argument('--pred_dim', type=int,
                   help='hidden dimension of the predictor (default: 512)')
    p.add_argument('--fix_pred_lr', type=utils.bool_flag,
                   help='Fix learning rate for the predictor (default: True)')

    # parameters for VitS
    p.add_argument('--patch_size', type=int, default=16,
                   help="Size in pixels of input square patches - default 16 (for 16x16 patches).")
    p.add_argument('--drop_path_rate', type=float,
                   help="stochastic depth rate. (default: 0.1)")

    # data augmentation parameters:
    p.add_argument("--crop_size", type=int,
                   help="Size of crops (default: 224)")
    p.add_argument("--crop_scale", type=float, nargs='+',
                   help="Scale range of the crops, relative to original image (default: 0.2 1.)")
    p.add_argument("--blur_prob", type=float,
                   help="Blur probability (default: 0.5)")
    p.add_argument("--hflip_prob", type=float,
                   help="Horizontal-Flip probability (default: 0.5)")

    # MinSim parameters:
    p.add_argument("--num_crops", default=2, type=int, help="Number of crops")
    p.add_argument("--select_fn", default="identity", type=str, choices=select_crops.names)

    # Misc
    p.add_argument('--fp16', default=True, type=utils.bool_flag,
                   help="Whether or not to use half precision for training. (default: True)")
    p.add_argument('--output_dir', type=str,
                   help='Path to save logs and checkpoints.')
    p.add_argument('--saveckp_freq', default=0, type=int,
                   help='Save checkpoint every x epochs.')
    p.add_argument('--seed', default=0, type=int,
                   help='Random seed.')
    p.add_argument('--num_workers', default=8, type=int,
                   help='Number of data loading workers per GPU.')
    p.add_argument("--dist_backend", default="nccl", type=str,
                   help="distributed backend (default: nccl)")
    p.add_argument("--dist_url", default="env://", type=str,
                   help="url used to set up distributed training")
    p.add_argument("--print_freq", default=10, type=int,
                   help="Print progress every x iterations (default: 10)")
    p.add_argument("--logger_freq", default=50, type=int,
                   help="Log progress every x iterations to tensorboard (default: 50)")
    p.add_argument("--use_adv_metric", default=True, type=utils.bool_flag,
                   help="Log advanced metrics: transforms params, crop selection, sample-loss, ... (default: False)")
    p.add_argument("--adv_metric_freq", default=10, type=int,
                   help="Log advanced metrics every x iterations (default: 100)")
    p.add_argument('--clip_grad', type=float, default=0.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")

    return p


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
