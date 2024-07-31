import argparse
import json
import math
import os
import sys
import time
import datetime
from collections import defaultdict
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

from torchvision import transforms
import transforms_p as tp

import data
import distributed as dist
import builder
import msatransform
import utils

import resnet_cifar
import resnet_imagenet


def custom_collate(batch):
    ncrops = len(batch[0][0][0])
    images = [torch.stack([item[0][0][n] for item in batch]) for n in range(ncrops)]
    params = [[item[0][1][n] for item in batch] for n in range(ncrops)]
    targets = [item[1] for item in batch]
    return images, params, targets


def main(cfg):
    dist.init_distributed_mode(cfg)
    utils.fix_random_seeds(cfg.seed)
    cudnn.benchmark = True

    print(f"git:\n  {utils.get_sha()}\n")
    print(OmegaConf.to_yaml(cfg))

    if cfg.arch in resnet_cifar.__dict__.keys():
        arch = resnet_cifar.__dict__[cfg.arch]
        proj_layer = 2
    elif cfg.arch in resnet_imagenet.__dict__.keys():
        arch = resnet_imagenet.__dict__[cfg.arch]
        proj_layer = 3
    else:
        print(f"Unknown architecture: {cfg.arch}")
        sys.exit(1)

    model = builder.SimSiam(
        arch,
        cfg.dim, cfg.pred_dim,
        proj_layer,
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

    optimizer = torch.optim.SGD(optim_params, init_lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    fp16 = torch.cuda.amp.GradScaler() if cfg.fp16 else None

    # ============ preparing data ... ============
    if cfg.dataset == "CIFAR10":
        mean, std = data.CIFAR10_DEFAULT_MEAN, data.CIFAR10_DEFAULT_STD
    else:
        mean, std = data.IMAGENET_DEFAULT_MEAN, data.IMAGENET_DEFAULT_STD

    rrc = transforms.RandomResizedCrop(cfg.crop_size, cfg.crop_scale)
    transform = transforms.Compose([
        tp.RandomColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
        tp.RandomGrayscale(p=0.2),
        tp.RandomGaussianBlur(9, (0.1, 2.0), p=cfg.blur_prob),
        tp.RandomHorizontalFlip(p=cfg.hflip_prob),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    msat = msatransform.MSATransform(
        rrc=rrc,
        total_epochs=cfg.epochs,
        warmup_epochs=cfg.msat_warmup,
        start_val=cfg.start_val,
        end_val=cfg.end_val,
        schedule=cfg.msat_schedule,
        transforms=transform,
        p=cfg.msat_prob
    )
    dataset, _ = data.make_dataset(cfg.data_path, cfg.dataset, True, msat)

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

    # select_fn = select_crops.names[cfg.select_fn]

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
        msat.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, cfg)

        start = time.time()
        train_stats, metrics = train(loader, model, criterion, optimizer, epoch, cfg, fp16, board)
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


def train(loader, model, criterion, optimizer, epoch, cfg, fp16, board):
    model.train()
    metrics = defaultdict(list)
    metrics["epoch"] = epoch
    metric_logger = utils.MetricLogger(delimiter=" ")
    header = 'Epoch: [{}/{}]'.format(epoch, cfg.epochs)
    for it, (images, params, _) in enumerate(metric_logger.log_every(loader, cfg.print_freq, header)):
        it = len(loader) * epoch + it  # global training iteration

        images = [im.cuda(non_blocking=True) for im in images]
        x1, x2 = images

        with torch.cuda.amp.autocast(fp16 is not None):
            p1, p2, z1, z2 = model(x1=x1, x2=x2)
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        optimizer.zero_grad()
        if fp16 is None:
            loss.backward()
            optimizer.step()
        else:
            fp16.scale(loss).backward()
            fp16.step(optimizer)
            fp16.update()

        # logging
        torch.cuda.synchronize()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        metrics["loss"].append(loss.item())
        metrics["lr"].append(optimizer.param_groups[0]["lr"])
        if dist.is_main_process() and cfg.use_adv_metric and it % cfg.adv_metric_freq == 0:
            metrics["params"].append(params)

        if dist.is_main_process() and it % cfg.log_freq == 0:
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

    # simsiam specific parameters:
    p.add_argument('--dim', type=int,
                   help='feature dimension (default: 2048)')
    p.add_argument('--pred_dim', type=int,
                   help='hidden dimension of the predictor (default: 512)')
    p.add_argument('--fix_pred_lr', type=utils.bool_flag,
                   help='Fix learning rate for the predictor (default: True)')

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
    # p.add_argument("--num_crops", default=2, type=int, help="Number of crops")
    # p.add_argument("--select_fn", default="identity", type=str, choices=select_crops.names)
    p.add_argument("--start_val", type=float, default=None,
                   help="Initial value of the MSATransform parameter for rejection sampling")
    p.add_argument("--end_val", type=float, default=None,
                   help="End value of the MSAT parameter")
    p.add_argument("--msat_schedule", type=str, choices=['linear', 'cosine'],
                   help="schedule type for MSAT value")
    p.add_argument("--msat_warmup", type=int, help='number of warmup epochs that turns off MSAT')
    p.add_argument("--msat_prob", type=float,
                   help="MSAT probability (default: 0.5)")

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
    p.add_argument("--log_freq", default=50, type=int,
                   help="Log progress every x iterations to tensorboard (default: 50)")
    p.add_argument("--use_adv_metric", default=True, type=utils.bool_flag,
                   help="Log advanced metrics: transforms params, crop selection, sample-loss, ... (default: False)")
    p.add_argument("--adv_metric_freq", default=10, type=int,
                   help="Log advanced metrics every x iterations (default: 100)")

    return p


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
