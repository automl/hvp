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
from builder import BarlowTwins
import select_crops
import custom_transform
import utils
from utils.optimizers import LARS
from data import Transform


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

    if not cfg.arch.startswith("resnet"):
        print(f"Unknown architecture: {cfg.arch}")
        sys.exit(1)

    model = BarlowTwins(cfg).cuda()
    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)

    parameters = [{'params': param_weights}, {'params': param_biases}]
    model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu])
    optimizer = LARS(parameters, lr=0, weight_decay=cfg.weight_decay,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)

    # init_lr = cfg.lr * cfg.batch_size / 256
    fp16 = torch.cuda.amp.GradScaler() if cfg.fp16 else None

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

    # ============ preparing data ... ============
    dataset, _ = data.make_dataset(cfg.data_path, cfg.dataset, train=True, transform=Transform())
    sampler = DistributedSampler(dataset)
    assert cfg.batch_size % dist.get_world_size() == 0
    cfg.batch_size_per_gpu = cfg.batch_size // cfg.grad_accum_steps // dist.get_world_size()
    print(f"{cfg.batch_size_per_gpu=}")
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=cfg.batch_size_per_gpu,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,  # required since select_crops_cross dives batch into chunk_size
        # collate_fn=custom_collate,
    )

    select_fn = select_crops.names[cfg.select_fn]

    log_dir = os.path.join(cfg.output_dir, "tensorboard")
    board = SummaryWriter(log_dir) if dist.is_main_process() else None

    for epoch in range(start_epoch, cfg.epochs):
        loader.sampler.set_epoch(epoch)

        start = time.time()
        train_stats, metrics = train(loader, model, optimizer, epoch, cfg, fp16, board, select_fn)
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


def train(loader, model, optimizer, epoch, cfg, fp16, board, select_fn):
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

    total_loss = 0
    for it, (images, _) in enumerate(metric_logger.log_every(loader, cfg.print_freq, header)):
        it = len(loader) * epoch + it  # global training iteration

        adjust_learning_rate(cfg, optimizer, loader, it)

        images = [im.cuda(non_blocking=True) for im in images]

        x1, x2, selected, sample_loss = select_fn(images, model, fp16, cfg)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(fp16 is not None):
            loss = model.forward(x1, x2)
            loss /= cfg.grad_accum_steps
            total_loss += loss.detach()

            if ((it+1) % cfg.grad_accum_steps == 0) or (it + 1 == len(loader)):
                if fp16 is None:
                    loss.backward()
                    optimizer.step()
                else:
                    fp16.scale(loss).backward()
                    fp16.step(optimizer)
                    fp16.update()

            # logging
            torch.cuda.synchronize()
            metric_logger.update(loss=total_loss.item())
            # metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            metrics["loss"].append(total_loss.item())
            # metrics["loss"].append(loss.item())
            metrics["lr"].append(optimizer.param_groups[0]["lr"])

            if dist.is_main_process() and it % cfg.logger_freq == 0:
                log_step = it // cfg.grad_accum_steps
                if cfg.use_adv_metric and not log_step % cfg.adv_metric_freq:
                    metrics["selected"].append(selected.tolist())
                    metrics["sample-loss"].append(sample_loss.tolist())
                #     metrics["params"].append(params)

                board.add_scalar("training loss", total_loss.item(), it)
                # board.add_scalar("training loss", loss.item(), it)
                board.add_scalar("training lr", optimizer.param_groups[0]["lr"], it)

        total_loss = 0

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, metrics


def adjust_learning_rate(cfg, optimizer, loader, step):
    max_steps = cfg.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = cfg.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * cfg.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * cfg.learning_rate_biases


def get_args_parser():
    p = argparse.ArgumentParser("Barlow Twins", description='Pytorch Pretraining on ImageNet', add_help=False)
    p.add_argument('--dataset', type=str, default="ImageNet",
                   help='Specify dataset (default: ImageNet)')
    p.add_argument('--data_path', type=str,
                   help='(root) path to dataset')
    p.add_argument('-a', '--arch', type=str,
                   choices=["vit_tiny", "vit_small", "vit_base",
                            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                            "resnet18_cifar", "resnet34_cifar", "resnet50_cifar", "resnet101_cifar", "resnet152_cifar"],
                   help="Name of architecture to train (default: resnet50)")
    p.add_argument('--epochs', type=int, default=300,
                   help='number of total epochs to run (default: 300)')
    p.add_argument('-b', '--batch_size', default=2048, type=int,
                   help='total batch-size (default: 2048)')

    p.add_argument('--wd', '--weight_decay', default=1e-6, dest="weight_decay", type=float,
                   help='weight decay (default: 1.5e-6)')

    # BT specific parameters:
    p.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                        help='weight on off-diagonal terms')
    p.add_argument('--projector', default='8192-8192-8192', type=str,
                        metavar='MLP', help='projector MLP')
    p.add_argument('--learning_rate_weights', default=0.2, type=float, metavar='LR',
                        help='base learning rate for weights')
    p.add_argument('--learning_rate_biases', default=0.0048, type=float, metavar='LR',
                        help='base learning rate for biases and batch norm parameters')

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
    p.add_argument("--print_freq", default=100, type=int,
                   help="Print progress every x iterations (default: 10)")
    p.add_argument("--logger_freq", default=50, type=int,
                   help="Log progress every x iterations to tensorboard (default: 50)")
    p.add_argument("--use_adv_metric", default=True, type=utils.bool_flag,
                   help="Log advanced metrics: transforms params, crop selection, sample-loss, ... (default: False)")
    p.add_argument("--adv_metric_freq", default=10, type=int,
                   help="Log advanced metrics every x iterations (default: 100)")

    p.add_argument("--grad_accum_steps", type=int,
                   help="Gradient accumulation. Effective batch size is given batch size (default: 1)"
                        "batch size per gpu = batch_size / grad_accum_steps / num_gpus")

    return p


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
