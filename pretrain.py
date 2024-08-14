import argparse
import datetime
import itertools as itert
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import yaml
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import models as torchvision_models

import data
import utils
from data import custom_transform
from models import resnet, resnet_cifar
from models import vision_transformer as vits
from utils import distributed as dist
from utils import optimizers
from utils.dino import DINOHead, DINOLoss, MultiCropWrapper

torchvision_archs = sorted(
    name
    for name in torchvision_models.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(torchvision_models.__dict__[name])
)


def custom_collate(batch):
    ncrops = len(batch[0][0][0])
    images = [torch.stack([item[0][0][n] for item in batch]) for n in range(ncrops)]
    params = [[item[0][1][n] for item in batch] for n in range(ncrops)]
    # target = [item[1] for item in batch]
    return images, params


def index_generator(args):
    num_local_crops = args.local_crops_number
    num_crops_loader = args.num_global_crops_loader
    num_local_crops_loader = args.num_local_crops_loader
    limit = args.limit_comparisons
    if limit is not None:
        for _ in range(limit):
            yield [
                random.choice(range(0, num_crops_loader, 2)),
                random.choice(range(1, num_crops_loader, 2)),
                *random.sample(
                    range(num_crops_loader, num_crops_loader + num_local_crops_loader),
                    num_local_crops,
                ),
            ]
    else:
        for i in itert.product(
            range(0, num_crops_loader, 2),
            range(1, num_crops_loader, 2),
            itert.combinations(
                range(
                    num_crops_loader,
                    num_crops_loader + num_local_crops_loader,
                ),
                num_local_crops,
            ),
        ):
            yield i


@torch.no_grad()
def hvs(images, student, teacher, criterion, epoch, args):
    bs = images[0].size(0)
    device = student.device

    score = torch.zeros(bs, device=device)
    selected = torch.zeros(
        (2 + args.local_crops_number, bs), dtype=torch.uint8, device=device
    )
    out = [torch.empty_like(images[0]) for _ in range(2)] + [
        torch.empty_like(images[-1]) for _ in range(args.local_crops_number)
    ]

    with torch.autocast("cuda"):
        teacher_output = teacher(images[: args.num_global_crops_loader])
        student_output = student(images)
        student_output, teacher_output = criterion.prepare_outputs(
            student_output, teacher_output, epoch
        )
        student_output, teacher_output = (
            student_output.chunk(len(images)),
            teacher_output.chunk(args.num_global_crops_loader),
        )
        teacher_output = teacher(images[: args.num_global_crops_loader])
        student_output = student(images)
        student_output, teacher_output = criterion.prepare_outputs(
            student_output, teacher_output, epoch
        )
        student_output, teacher_output = (
            student_output.chunk(len(images)),
            teacher_output.chunk(args.num_global_crops_loader),
        )

    for idx in index_generator(args):
        _teacher_out = [teacher_output[x] for x in idx[:2]]
        _student_out = [student_output[x] for x in idx]
        with torch.autocast("cuda"):
            sim = criterion.select_forward(_student_out, _teacher_out)  # sample-loss
            score, indices = torch.stack((score, sim)).max(dim=0)
            indices = indices.type(torch.bool)

        for n, ids in enumerate(idx):
            selected[n][indices] = ids

    for n in range(2):
        for m in range(args.num_global_crops_loader):
            out[n] = torch.where(
                (selected[n] == m)[:, None, None, None], images[m], out[n]
            )
    for n in range(2, len(out)):
        for m in range(args.num_global_crops_loader, len(images)):
            out[n] = torch.where(
                (selected[n] == m)[:, None, None, None], images[m], out[n]
            )
    return out, selected, score


def main(args: argparse.Namespace):
    dist.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    print(f"\ngit:\n  {utils.get_sha()}\n")
    print(*[f"{k}: {str(v)}" for k, v in sorted(dict(vars(args)).items())], sep="\n")

    # ============ preparing data ... ============
    gt1 = custom_transform.TransformParams(
        crop_size=args.global_crops_size,
        crop_scale=args.global_crops_scale,
        blur_prob=1.0,
        hflip_prob=0.5,
        solarize_prob=0.0,
    )
    gt2 = custom_transform.TransformParams(
        crop_size=args.global_crops_size,
        crop_scale=args.global_crops_scale,
        blur_prob=0.1,
        hflip_prob=0.5,
        solarize_prob=0.2,
    )
    lt = custom_transform.TransformParams(
        crop_size=args.local_crops_size,
        crop_scale=args.local_crops_scale,
        blur_prob=0.5,
        hflip_prob=0.5,
        solarize_prob=0.0,
    )
    multi_crops_transform = data.MultiCropsTransform(
        gt1, gt2, lt, args.num_global_crops_loader, args.num_local_crops_loader
    )
    dataset = data.make_dataset(
        args.data_path, args.dataset, True, multi_crops_transform
    )

    sampler = DistributedSampler(dataset)
    batch_size_per_gpu = (
        args.batch_size // args.grad_accum_steps // dist.get_world_size()
    )
    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate,
    )
    print(f"Data loaded: there are {len(dataset)} images.")  # type: ignore

    # ============ building student and teacher networks ... ============
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            img_size=args.img_size,
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,
        )
        teacher = vits.__dict__[args.arch](
            img_size=args.img_size,
            patch_size=args.patch_size,
        )
        embed_dim = student.embed_dim
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load(
            "facebookresearch/xcit:main",
            args.arch,
            pretrained=False,
            drop_path_rate=args.drop_path_rate,
        )
        teacher = torch.hub.load(
            "facebookresearch/xcit:main", args.arch, pretrained=False
        )
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    elif args.arch in resnet_cifar.__dict__.keys():
        student = resnet_cifar.__dict__[args.arch]()
        teacher = resnet_cifar.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    elif args.arch in resnet.__dict__.keys():
        student = resnet.__dict__[args.arch]()
        teacher = resnet.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknown architecture: {args.arch}")
        sys.exit(1)

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = MultiCropWrapper(
        student,
        DINOHead(
            embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        ),
    )
    teacher = MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        2
        + args.local_crops_number,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # type: ignore
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(  # type: ignore
            params_groups, lr=0, momentum=0.9
        )  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = optimizers.LARS(
            params_groups
        )  # to use with convnet and large batches
    else:
        print("Unknown optimizer.")
        sys.exit(1)

    # for mixed precision training
    fp16 = torch.GradScaler(enabled=args.fp16)

    init_lr = args.lr * args.batch_size / 256.0  # linear scaling rule
    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        init_lr,
        args.min_lr,
        args.epochs,
        len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(
        args.momentum_teacher, 1, args.epochs, len(data_loader)
    )
    print("Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0, "total_time": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]
    total_time = to_restore["total_time"]

    log_dir = os.path.join(args.output_dir, "tensorboard")
    board = None
    if dist.is_main_process() and args.log_freq > 0:
        board = SummaryWriter(log_dir)

    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)  # type: ignore

        start = time.time()
        train_stats = train_one_epoch(
            student,
            teacher,
            teacher_without_ddp,
            dino_loss,
            data_loader,
            optimizer,
            lr_schedule,
            wd_schedule,
            momentum_schedule,
            epoch,
            fp16,
            args,
            board,
        )
        total_time += int(time.time() - start)

        save_dict = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "dino_loss": dino_loss.state_dict(),
            "epoch": epoch + 1,
            "total_time": total_time,
            "fp16_scaler": fp16.state_dict(),
        }
        utils.save_on_master(save_dict, os.path.join(args.output_dir, "checkpoint.pth"))
        if args.saveckp_freq and epoch and epoch % args.saveckp_freq == 0:
            utils.save_on_master(
                save_dict, os.path.join(args.output_dir, f"checkpoint{epoch:04}.pth")
            )
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }
        if dist.is_main_process():
            with (Path(args.output_dir) / "pretrain.log").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def train_one_epoch(
    student,
    teacher,
    teacher_without_ddp,
    dino_loss,
    data_loader,
    optimizer,
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    epoch,
    fp16,
    args,
    board,
):
    metric_logger = utils.MetricLogger(delimiter=" ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)
    total_loss = 0
    for it, (images, _) in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # MinSim
        if args.use_hvp:
            images, _, _ = hvs(
                images, student, teacher_without_ddp, dino_loss, epoch, args
            )

        # teacher and student forward passes + compute dino loss
        with torch.autocast("cuda", enabled=args.fp16):
            teacher_output = teacher(
                images[:2]
            )  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)
            loss /= args.grad_accum_steps
            total_loss += loss.detach()

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)  # type: ignore
            sys.exit(1)

        # backpropagation
        loss.backward() if fp16 is None else fp16.scale(loss).backward()

        if not (it + 1) % args.grad_accum_steps:
            grad_norm = None
            if fp16 is None:
                if args.clip_grad:
                    grad_norm = clip_grad_norm_(student.parameters(), args.clip_grad)
                utils.cancel_gradients_last_layer(
                    epoch, student, args.freeze_last_layer
                )
                optimizer.step()
            else:
                if args.clip_grad:
                    fp16.unscale_(optimizer)
                    grad_norm = clip_grad_norm_(student.parameters(), args.clip_grad)
                utils.cancel_gradients_last_layer(
                    epoch, student, args.freeze_last_layer
                )
                fp16.step(optimizer)
                fp16.update()

            dino_loss.update_center(teacher_output, args.grad_accum_steps)
            optimizer.zero_grad()

            # EMA update for the teacher
            with torch.no_grad():
                m = momentum_schedule[it]  # momentum parameter
                for param_q, param_k in zip(
                    student.module.parameters(), teacher_without_ddp.parameters()
                ):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            torch.cuda.synchronize()

            # logging
            metric_logger.update(loss=total_loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

            if board is not None:
                log_step = it // args.grad_accum_steps
                if not log_step % args.log_freq:
                    board.add_scalar("training loss", total_loss.item(), it)
                    board.add_scalar("training lr", optimizer.param_groups[0]["lr"], it)
                    board.add_scalar(
                        "training wd",
                        optimizer.param_groups[0]["weight_decay"],
                        it,
                    )
                    if grad_norm is not None:
                        board.add_scalar("training grad-norms", grad_norm.item(), it)

            total_loss = 0

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def get_args_parser():
    p = argparse.ArgumentParser(
        "DINO", description="Pretraining for DINO", add_help=False
    )

    # Model parameters
    p.add_argument(
        "-a",
        "--arch",
        type=str,
        choices=["vit_tiny", "vit_small", "vit_base", "xcit", "deit_tiny", "deit_small"]
        + torchvision_archs
        + torch.hub.list("facebookresearch/xcit:main"),
        default="vit_small",
        help="Model architecture (default: vit_small)",
    )
    p.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="The standard input size. (default: 224)",
    )
    p.add_argument(
        "--patch_size",
        type=int,
        default=16,
        help="Size in pixels of input square patches - default 16 (for 16x16 patches).",
    )
    p.add_argument(
        "--out_dim",
        type=int,
        default=65536,
        help="Dimensionality of the DINO head output. (default: 65536)",
    )
    p.add_argument(
        "--norm_last_layer",
        type=utils.bool_flag,
        default=True,
        help="Whether or not to weight normalize the last layer of the DINO head. (default: True)",
    )
    p.add_argument(
        "--momentum_teacher",
        type=float,
        default=0.996,
        help="Base EMA parameter for teacher update. (default: 0.996)",
    )
    p.add_argument(
        "--use_bn_in_head",
        type=utils.bool_flag,
        default=False,
        help="Whether to use batch normalizations in projection head (default: False)",
    )

    # Temperature teacher parameters
    p.add_argument(
        "--warmup_teacher_temp",
        default=0.04,
        type=float,
        help="Initial value for the teacher temperature. (default: 0.04)",
    )
    p.add_argument(
        "--teacher_temp",
        type=float,
        default=0.04,
        help="Final value (after linear warmup) of the teacher temperature. (default: 0.04)",
    )
    p.add_argument(
        "--warmup_teacher_temp_epochs",
        type=int,
        default=0,
        help="Number of warmup epochs for the teacher temperature (default: 0).",
    )

    # Training/Optimization parameters
    p.add_argument(
        "--fp16",
        type=utils.bool_flag,
        default=True,
        help="Whether or not to use half precision for training. (default: True)",
    )
    p.add_argument(
        "--weight_decay",
        type=float,
        default=0.04,
        help="Initial value of the weight decay. (default: 0.04)",
    )
    p.add_argument(
        "--weight_decay_end",
        type=float,
        default=0.4,
        help="Final value of the weight decay. (default: 0.4)",
    )
    p.add_argument(
        "--clip_grad",
        type=float,
        default=3.0,
        help="Maximal parameter gradient norm if using gradient clipping. 0 for disabling. (default: 3.0)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of total epochs to run (default: 100)",
    )
    p.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=512,
        help="total batch-size: (default: 512)",
    )
    p.add_argument(
        "--freeze_last_layer",
        type=int,
        default=1,
        help="Number of epochs during which we keep the output layer fixed. (default: 1)",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=0.0005,
        help="Learning rate at the end of linear warmup. (default: 0.0005)",
    )
    p.add_argument(
        "--warmup_epochs",
        type=int,
        default=10,
        help="Number of epochs for the linear learning-rate warm up. (default: 10)",
    )
    p.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Target LR at the end of optimization. (default: 1e-6)",
    )
    p.add_argument(
        "--optimizer",
        type=str,
        choices=["adamw", "sgd", "lars"],
        default="adamw",
        help="Type of optimizer. (default: adamw)",
    )
    p.add_argument(
        "--drop_path_rate",
        type=float,
        default=0.1,
        help="stochastic depth rate. (default: 0.1)",
    )

    # Multi-crop/Data-Augmentation parameters
    p.add_argument(
        "--global_crops_scale",
        type=float,
        nargs="+",
        default=(0.4, 1),
        help="Scale range of the cropped image before resizing, w.r.t. the original image. (default: 0.4 1)",
    )
    p.add_argument(
        "--local_crops_number",
        type=int,
        default=8,
        help="Number of small local views to generate. Value 0 disables multi-crop training. (default: 8)",
    )
    p.add_argument(
        "--local_crops_scale",
        type=float,
        nargs="+",
        default=(0.05, 0.4),
        help="Scale range of the cropped image before resizing. (default: 0.05 0.4)",
    )
    p.add_argument(
        "--global_crops_size",
        type=int,
        default=224,
        help="Size of global crops (default: 224)",
    )
    p.add_argument(
        "--local_crops_size",
        type=int,
        default=96,
        help="Size of local crops (default: 96)",
    )

    # HVS parameters:
    p.add_argument(
        "--use_hvp",
        type=utils.bool_flag,
        default=False,
        help="Use Hard View Pretraining. (default: False)",
    )
    p.add_argument(
        "--num_global_crops_loader",
        type=int,
        default=2,
        help="Number of global views to generate per image in the loader. (default: 2)",
    )
    p.add_argument(
        "--num_local_crops_loader",
        type=int,
        default=8,
        help="Number of local views to generate per image in the loader. (default: 8)",
    )
    p.add_argument(
        "--limit_comparisons",
        type=int,
        default=0,
        help="""Limit the number of comparisons; implemented as number of combinations for local crops. Default is 0, which turns off the limit. (default: 0)""",
    )

    # Misc
    p.add_argument(
        "--dataset",
        type=str,
        default="imagenet",
        help="Specify dataset (default: imagenet)",
    )
    p.add_argument(
        "--data_path",
        type=str,
        default="path/to/dataset",
        help="(root) path to dataset",
    )
    p.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation. Effective batch size is given batch size (default: 1)"
        "batch size per gpu = batch_size / grad_accum_steps / num_gpus",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Path to save logs and checkpoints. (default: .)",
    )
    p.add_argument(
        "--saveckp_freq",
        type=int,
        default=50,
        help="Save checkpoint every x epochs. (default: 50)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. (default: None)",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of data loading workers per GPU. (default: 8)",
    )
    p.add_argument(
        "--dist_backend",
        type=str,
        default="nccl",
        help="distributed backend (default: nccl)",
    )
    p.add_argument(
        "--dist_url",
        type=str,
        default="env://",
        help="url used to set up distributed training (default: env://)",
    )
    p.add_argument(
        "--print_freq",
        type=int,
        default=10,
        help="Print progress every x iterations (default: 10)",
    )
    p.add_argument(
        "--log_freq",
        type=int,
        default=0,
        help="Log progress every x iterations to tensorboard (default: 0)",
    )
    return p


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    if Path("pretrain.yaml").exists():
        with open("pretrain.yaml", "r") as f:
            yaml_args = yaml.safe_load(f)
        for k, v in yaml_args.items():
            setattr(args, k, v)
            
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
