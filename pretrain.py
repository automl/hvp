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

from data import custom_transform
import data
import utils
from minsim import MinSim
from models import resnet, resnet_cifar, vision_transformer as vits
from utils import distributed as dist, optimizers
from utils.dino import MultiCropWrapper, DINOHead, DINOLoss


def custom_collate(batch):
    ncrops = len(batch[0][0][0])
    images = [torch.stack([item[0][0][n] for item in batch]) for n in range(ncrops)]
    params = [[item[0][1][n] for item in batch] for n in range(ncrops)]
    # target = [item[1] for item in batch]
    return images, params


def main(cfg):
    dist.init_distributed_mode(cfg) if not dist.is_enabled() else None
    cudnn.benchmark = True

    print(f"git:\n  {utils.get_sha()}\n")
    print(OmegaConf.to_yaml(cfg))

    # ============ preparing data ... ============
    if cfg.dataset == "CIFAR10":
        mean, std = data.CIFAR10_DEFAULT_MEAN, data.CIFAR10_DEFAULT_STD
    else:
        mean, std = data.IMAGENET_DEFAULT_MEAN, data.IMAGENET_DEFAULT_STD

    gt1 = custom_transform.TransformParams(
        crop_size=cfg.global_crops_size,
        crop_scale=cfg.global_crops_scale,
        blur_prob=1.0,
        hflip_prob=0.5,
        solarize_prob=0.0,
        mean=mean,
        std=std,
    )
    gt2 = custom_transform.TransformParams(
        crop_size=cfg.global_crops_size,
        crop_scale=cfg.global_crops_scale,
        blur_prob=0.1,
        hflip_prob=0.5,
        solarize_prob=0.2,
        mean=mean,
        std=std,
    )
    lt = custom_transform.TransformParams(
        crop_size=cfg.local_crops_size,
        crop_scale=cfg.local_crops_scale,
        blur_prob=0.5,
        hflip_prob=0.5,
        solarize_prob=0.0,
        mean=mean,
        std=std,
    )
    two_crops_transform = data.MultiCropsTransform(gt1, gt2, lt,
                                                   cfg.num_global_crops_loader, cfg.num_local_crops_loader)
    dataset, _ = data.make_dataset(cfg.data_path, cfg.dataset, True, two_crops_transform)

    sampler = DistributedSampler(dataset)
    batch_size_per_gpu = cfg.batch_size // cfg.grad_accum_steps // dist.get_world_size()
    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size_per_gpu,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    if cfg.arch in vits.__dict__.keys():
        student = vits.__dict__[cfg.arch](
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            drop_path_rate=cfg.drop_path_rate,
        )
        teacher = vits.__dict__[cfg.arch](img_size=cfg.img_size, patch_size=cfg.patch_size, )
        embed_dim = student.embed_dim
    elif cfg.arch in resnet_cifar.__dict__.keys():
        student = resnet_cifar.__dict__[cfg.arch]()
        teacher = resnet_cifar.__dict__[cfg.arch]()
        embed_dim = student.fc.weight.shape[1]
    elif cfg.arch in resnet.__dict__.keys():
        student = resnet.__dict__[cfg.arch]()
        teacher = resnet.__dict__[cfg.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknown architecture: {cfg.arch}")
        sys.exit(1)

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = MultiCropWrapper(student, DINOHead(
        embed_dim,
        cfg.out_dim,
        use_bn=cfg.use_bn_in_head,
        norm_last_layer=cfg.norm_last_layer,
    ))
    teacher = MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, cfg.out_dim, cfg.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[cfg.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[cfg.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {cfg.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        cfg.out_dim,
        2 + cfg.local_crops_number,  # total number of crops = 2 global crops + local_crops_number
        cfg.warmup_teacher_temp,
        cfg.teacher_temp,
        cfg.warmup_teacher_temp_epochs,
        cfg.epochs,
    ).cuda()

    params_groups = utils.get_params_groups(student)
    if cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif cfg.optimizer == "lars":
        optimizer = optimizers.LARS(params_groups)  # to use with convnet and large batches
    else:
        print("Unknown optimizer.")
        sys.exit(1)

    # for mixed precision training
    fp16 = torch.cuda.amp.GradScaler() if cfg.fp16 else None

    init_lr = cfg.lr * cfg.batch_size / 256.  # linear scaling rule
    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        init_lr,
        cfg.min_lr,
        cfg.epochs, len(data_loader),
        warmup_epochs=cfg.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        cfg.weight_decay,
        cfg.weight_decay_end,
        cfg.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(cfg.momentum_teacher, 1,
                                               cfg.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    select_fn = MinSim(cfg.select_fn, student, teacher, dino_loss, fp16,
                       cfg.num_global_crops_loader, cfg.num_local_crops_loader,
                       cfg.local_crops_number, cfg.limit_comparisons)

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0, "total_time": 0}
    utils.restart_from_checkpoint(
        os.path.join(cfg.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]
    total_time = to_restore["total_time"]

    log_dir = os.path.join(cfg.output_dir, "tensorboard")
    board = SummaryWriter(log_dir) if dist.is_main_process() else None

    for epoch in range(start_epoch, cfg.epochs):
        data_loader.sampler.set_epoch(epoch)

        start = time.time()
        train_stats, metrics = \
            train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader, optimizer,
                            lr_schedule, wd_schedule, momentum_schedule,
                            epoch, fp16, cfg, board, select_fn)
        total_time += int(time.time() - start)

        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'dino_loss': dino_loss.state_dict(),
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


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
                    fp16, cfg, board, select_fn):
    metrics = defaultdict(list)
    metrics["epoch"] = epoch
    metric_logger = utils.MetricLogger(delimiter=" ")
    header = 'Epoch: [{}/{}]'.format(epoch, cfg.epochs)
    total_loss = 0
    for it, (images, params) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # MinSim
        images, selected, sample_loss = select_fn(images, epoch)

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16 is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)
            loss /= cfg.grad_accum_steps
            total_loss += loss.detach()

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # backpropagation
        loss.backward() if fp16 is None else fp16.scale(loss).backward()

        if not (it + 1) % cfg.grad_accum_steps:
            grad_norms = None
            if fp16 is None:
                if cfg.clip_grad:
                    grad_norms = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.clip_grad)
                utils.cancel_gradients_last_layer(epoch, student, cfg.freeze_last_layer)
                optimizer.step()
            else:
                if cfg.clip_grad:
                    fp16.unscale_(optimizer)
                    grad_norms = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.clip_grad)
                utils.cancel_gradients_last_layer(epoch, student, cfg.freeze_last_layer)
                fp16.step(optimizer)
                fp16.update()

            dino_loss.update_center(teacher_output, cfg.grad_accum_steps)
            optimizer.zero_grad()

            # EMA update for the teacher
            with torch.no_grad():
                m = momentum_schedule[it]  # momentum parameter
                for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            torch.cuda.synchronize()

            # logging
            metric_logger.update(loss=total_loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

            metrics["training loss"].append(total_loss.item())
            metrics["training lr"].append(optimizer.param_groups[0]["lr"])
            metrics["training grad-norms"].append(grad_norms.item())
            if dist.is_main_process():
                log_step = it // cfg.grad_accum_steps
                if cfg.use_adv_metric and not log_step % cfg.adv_metric_freq:
                    metrics["selected"].append(selected.tolist())
                    metrics["params"].append(params)
                    metrics["sample-loss"].append(sample_loss.tolist())

                if not log_step % cfg.log_freq:
                    board.add_scalar("training loss", total_loss.item(), it)
                    board.add_scalar("training lr", optimizer.param_groups[0]["lr"], it)
                    board.add_scalar("training wd", optimizer.param_groups[0]["weight_decay"], it)
                    board.add_scalar("training grad-norms", grad_norms.item(), it)

            total_loss = 0

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, metrics


def get_args_parser():
    p = argparse.ArgumentParser("DINO", description='Pretraining for DINO', add_help=False)

    # Model parameters
    p.add_argument('-a', '--arch', type=str,
                   choices=["vit_tiny", "vit_small", "vit_base",
                            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                            "resnet18_cifar", "resnet34_cifar", "resnet50_cifar", "resnet101_cifar", "resnet152_cifar"],
                   help="Model architecture (default: vit_small)")
    p.add_argument('--img_size', type=int,
                   help="The standard input size. (default: 224)")
    p.add_argument('--patch_size', type=int,
                   help="Size in pixels of input square patches - default 16 (for 16x16 patches).")
    p.add_argument('--out_dim', type=int,
                   help="Dimensionality of the DINO head output. (default: 65536)")
    p.add_argument('--norm_last_layer', type=utils.bool_flag,
                   help="Whether or not to weight normalize the last layer of the DINO head. (default: True)")
    p.add_argument('--momentum_teacher', type=float,
                   help="Base EMA parameter for teacher update. (default: 0.996)")
    p.add_argument('--use_bn_in_head', type=utils.bool_flag,
                   help="Whether to use batch normalizations in projection head (default: False)")

    # Temperature teacher parameters
    p.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                   help="Initial value for the teacher temperature. (default: 0.04)")
    p.add_argument('--teacher_temp', type=float,
                   help="Final value (after linear warmup) of the teacher temperature. (default: 0.04)")
    p.add_argument('--warmup_teacher_temp_epochs', type=int,
                   help="Number of warmup epochs for the teacher temperature (default: 0).")

    # Training/Optimization parameters
    p.add_argument('--fp16', type=utils.bool_flag,
                   help="Whether or not to use half precision for training. (default: True)")
    p.add_argument('--weight_decay', type=float,
                   help="Initial value of the weight decay. (default: 0.04)")
    p.add_argument('--weight_decay_end', type=float,
                   help="Final value of the weight decay. (default: 0.4)")
    p.add_argument('--clip_grad', type=float,
                   help="Maximal parameter gradient norm if using gradient clipping. 0 for disabling. (default: 3.0)")
    p.add_argument('--epochs', type=int,
                   help='number of total epochs to run (default: 100)')
    p.add_argument('-b', '--batch_size', type=int,
                   help="total batch-size: (default: 512)")
    p.add_argument('--freeze_last_layer', type=int,
                   help="Number of epochs during which we keep the output layer fixed. (default: 1)")
    p.add_argument("--lr", type=float,
                   help="Learning rate at the end of linear warmup. (default: 0.0005)")
    p.add_argument("--warmup_epochs", type=int,
                   help="Number of epochs for the linear learning-rate warm up. (default: 10)")
    p.add_argument('--min_lr', type=float,
                   help="Target LR at the end of optimization. (default: 1e-6)")
    p.add_argument('--optimizer', type=str, choices=['adamw', 'sgd', 'lars'],
                   help="Type of optimizer. (default: adamw)")
    p.add_argument('--drop_path_rate', type=float,
                   help="stochastic depth rate. (default: 0.1)")

    # Multi-crop/Data-Augmentation parameters
    p.add_argument('--global_crops_scale', type=float, nargs='+',
                   help="Scale range of the cropped image before resizing, w.r.t. the original image. (default: 0.4 1)")
    p.add_argument('--local_crops_number', type=int,
                   help="Number of small local views to generate. Value 0 disables multi-crop training. (default: 8)")
    p.add_argument('--local_crops_scale', type=float, nargs='+',
                   help="Scale range of the cropped image before resizing. (default: 0.05 0.4)")
    p.add_argument("--global_crops_size", type=int,
                   help="Size of global crops (default: 224)")
    p.add_argument("--local_crops_size", type=int,
                   help="Size of local crops (default: 96)")

    # MinSim parameters:
    p.add_argument("--select_fn", type=str, choices=["identity", "cross"],
                   help="Select function for MinSim (default: identity)")
    p.add_argument('--num_global_crops_loader', type=int,
                   help="Number of global views to generate per image in the loader. (default: 2)")
    p.add_argument('--num_local_crops_loader', type=int,
                   help="Number of local views to generate per image in the loader. (default: 8)")
    p.add_argument('--limit_comparisons', type=int,
                   help="""Limit the number of comparisons; implemented as number of combinations for local crops.
                   Default is 0, which turns off the limit. (default: 0)""")

    # Misc
    p.add_argument('--dataset', type=str, default="ImageNet",
                   help='Specify dataset (default: ImageNet)')
    p.add_argument('--data_path', type=str,
                   help='(root) path to dataset')
    p.add_argument("--grad_accum_steps", type=int,
                   help="Gradient accumulation. Effective batch size is given batch size (default: 1)"
                        "batch size per gpu = batch_size / grad_accum_steps / num_gpus")
    p.add_argument('--output_dir', type=str,
                   help='Path to save logs and checkpoints. (default: .)')
    p.add_argument('--saveckp_freq', type=int,
                   help='Save checkpoint every x epochs. (default: 0)')
    p.add_argument('--seed', type=int,
                   help='Random seed. (default: 0)')
    p.add_argument('--num_workers', type=int,
                   help='Number of data loading workers per GPU. (default: 8)')
    p.add_argument("--dist_backend", type=str,
                   help="distributed backend (default: nccl)")
    p.add_argument("--dist_url", type=str,
                   help="url used to set up distributed training (default: env://)")
    p.add_argument("--print_freq", type=int,
                   help="Print progress every x iterations (default: 10)")
    p.add_argument("--log_freq", type=int,
                   help="Log progress every x iterations to tensorboard (default: 50)")
    p.add_argument("--use_adv_metric", type=utils.bool_flag,
                   help="Log advanced metrics: transforms params, crop selection, sample-loss, ... (default: True)")
    p.add_argument("--adv_metric_freq", type=int,
                   help="Log advanced metrics every x iterations (default: 10)")
    return p


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
