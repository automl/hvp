import argparse
import datetime
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
import torchvision.models as torch_models
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa
from torch.utils.data import DataLoader, DistributedSampler

import data
import distributed as dist
import losses
# import models.vision_transformer as vits
import optimizers
import utils
from hvs import adversarial_crop_selection
from methods import SimCLR
from transforms import SimCLRTransform, custom_collate


GRAD_ACCUM_STEPS = 1
LOG_FREQ = 0
MAX_EPOCHS = -1


def main(
        arch: str = "resnet50",
        arch_kwargs: dict = None,
        proj_hidden_dim: int = 2048,
        out_dim: int = 128,
        use_bn_in_head: bool = True,
        optim: str = "lars",
        lr: float = 0.3,
        sqrt_lr: bool = False,
        batch_size: int = 4096,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        warmup_epochs: int = 10,
        use_hvs: bool = False,
        num_views: int = 2,
        use_fp16: bool = True,
        num_workers: int = 8,
        data_path: str = "path/to/dataset",
        dataset: str = "imagenet",
        output_dir: str = ".",
        log_freq: int = 10,
        save_ckp_freq: int = 0,
        grad_accum_steps: int = 1,
        gpu_id: int = 0,
        **kwargs,
):
    print("Running pretraining ...")
    utils.print_args(locals())

    global LOG_FREQ, GRAD_ACCUM_STEPS, MAX_EPOCHS
    GRAD_ACCUM_STEPS = grad_accum_steps
    LOG_FREQ = log_freq
    MAX_EPOCHS = epochs

    if arch_kwargs is None:
        arch_kwargs = {}

    # ============ building network ... ============
    if arch in torch_models.__dict__.keys():  # torchvision models
        backbone = torch_models.__dict__[arch](**arch_kwargs)
    # elif arch in vits.__dict__.keys():  # vision transformer models
    #     backbone = vits.__dict__[arch](**arch_kwargs)
    else:
        raise NotImplementedError(f"Architecture {arch} not supported")

    model = SimCLR(
        backbone,
        proj_hidden_dim=proj_hidden_dim,
        out_dim=out_dim,
        use_bn=use_bn_in_head,
    ).cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(model):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[gpu_id])
    print(f"SimCLR network created with {arch} backbone.")

    # ============ preparing data ... ============
    batch_size_per_gpu = batch_size // dist.get_world_size() // GRAD_ACCUM_STEPS
    train_dataset, _ = data.make_dataset(data_path, dataset, True, SimCLRTransform(num_views=num_views))
    data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        num_workers=num_workers,
        pin_memory=True,
        sampler=DistributedSampler(train_dataset),
        collate_fn=custom_collate,
        drop_last=True,
    )
    print(f"Data loaded with {len(train_dataset)} train images.")

    # ============ building loss ... ============
    ntx_loss = losses.NTXentLoss(temperature=0.1, gather_distributed=dist.is_enabled())

    # ============ building optimizer ... ============
    if sqrt_lr:
        init_lr = lr * math.sqrt(batch_size)  # square root scaling rule
    else:
        init_lr = lr * batch_size / 256.  # linear scaling rule
    optimizer = optimizers.configure_optimizer(
        optim,
        model.parameters(),
        lr=init_lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        init_lr,
        1e-6,
        epochs,
        len(data_loader),
        warmup_epochs=warmup_epochs,
    )

    # Gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    # # ============ preparing logging ... ============
    logger = utils.CSVLogger(os.path.join(output_dir, "pretrain.csv"))

    print("Preparation done!")

    # ============ optionally resume training ... ============
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    to_restore = {"epoch": 0, "total_time": 0}
    utils.restart_from_checkpoint(
        os.path.join(output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
    )
    start_epoch = to_restore["epoch"]
    total_time = to_restore["total_time"]

    print("Starting training !")
    for epoch in range(start_epoch, epochs):
        data_loader.sampler.set_epoch(epoch)  # noqa

        # ============ training one epoch ... ============
        start = time.time()
        train_stats, metrics = \
            train_one_epoch(model, ntx_loss, data_loader, optimizer, lr_schedule, scaler, epoch, logger, use_hvs)
        total_time += int(time.time() - start)

        # ============ writing logs ... ============
        save_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "total_time": total_time,
        }
        utils.save_on_master(save_dict, os.path.join(output_dir, 'checkpoint.pth'))
        if save_ckp_freq and epoch and epoch % save_ckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(output_dir, f'checkpoint{epoch:04}.pth'))

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if dist.is_main_process():
            with (Path(output_dir) / "pretrain.log").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            with (Path(output_dir) / "metrics.json").open("a") as f:
                f.write(json.dumps(metrics) + "\n")
        logger.flush()

    logger.close()
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Pretraining finished. Total training time: {total_time_str}\n")


def train_one_epoch(
        model,
        criterion,
        data_loader,
        optimizer,
        lr_schedule,
        scaler,
        epoch,
        logger,
        use_hvs,
):
    metrics = defaultdict(list)
    metrics["epoch"] = epoch
    metric_logger = utils.MetricLogger(delimiter=" ")
    header = 'Epoch: [{}/{}]'.format(epoch, MAX_EPOCHS)

    for it, batch in enumerate(metric_logger.log_every(data_loader, 100, header)):
        it = len(data_loader) * epoch + it  # global training iteration

        for i, param_group in enumerate(optimizer.param_groups):  # adjust learning rate
            param_group["lr"] = lr_schedule[it]

        images, params, _ = batch
        images = [img.cuda(non_blocking=True) for img in images]

        # hard view selection
        if use_hvs:
            images, selection, _ = adversarial_crop_selection(images, model, criterion)

        # regular training
        with torch.cuda.amp.autocast(scaler is not None):
            z1, z2 = model(torch.cat(images)).chunk(len(images))
            loss = criterion(z1, z2)
            loss /= GRAD_ACCUM_STEPS

        loss.backward() if scaler is None else scaler.scale(loss).backward()

        if not (it + 1) % GRAD_ACCUM_STEPS:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

            # logging
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item() * GRAD_ACCUM_STEPS)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            step = (it + 1) // GRAD_ACCUM_STEPS
            logger.log("loss", loss.item(), step)
            logger.log("lr", optimizer.param_groups[0]["lr"], step)
            if not step % LOG_FREQ:
                metrics["step"].append(it)
                # metrics["params"].append(params.tolist())

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, metrics


def get_pretrain_args_parser():
    p = argparse.ArgumentParser("SSL Pretraining")
    # Model parameters
    p.add_argument("--arch", default="resnet50", type=str, help="model architecture")
    p.add_argument("--arch-kwargs", default=None, type=utils.arg_dict, help="architecture kwargs")
    p.add_argument("--proj-hidden-dim", default=2048, type=int, help="hidden dimension of projection head")
    p.add_argument("--out-dim", default=128, type=int, help="output dimension of projection head")
    p.add_argument("--use-bn-in-head", default=True, type=utils.bool_flag,
                   help="whether to use batch norm in projection head")
    # Optimization parameters
    p.add_argument("--batch-size", default=4096, type=int, help="batch size")
    p.add_argument("--optim", default="lars", type=str, help="optimizer")
    p.add_argument("--lr", default=0.3, type=float, help="learning rate")
    p.add_argument("--sqrt-lr", default=False, type=utils.bool_flag, help="scale lr by square root of batch size")
    p.add_argument("--momentum", default=0.9, type=float, help="momentum")
    p.add_argument("--weight-decay", default=1e-4, type=float, help="weight decay")
    p.add_argument("--epochs", default=100, type=int, help="number of epochs")
    p.add_argument("--warmup-epochs", default=10, type=int, help="number of warmup epochs")
    p.add_argument("--grad-accum-steps", default=1, type=int, help="gradient accumulation steps")
    p.add_argument("--use-fp16", default=True, type=utils.bool_flag, help="use mixed precision training")
    # ACS parameters
    p.add_argument("--use-hvs", default=False, type=utils.bool_flag, help="whether to use hard view selection")
    p.add_argument("--num_views", default=2, type=int, help="number of views")
    # Misc
    p.add_argument("--dataset", default="imagenet", type=str, help="dataset name")
    p.add_argument("--data-path", default=None, type=str, help="path to data")
    p.add_argument("--num-workers", default=8, type=int, help="number of workers")
    p.add_argument("--output-dir", default=".", type=str, help="path to output directory")
    p.add_argument("--log-freq", default=100, type=int, help="logging frequency")
    p.add_argument("--save-ckp-freq", default=0, type=int, help="save checkpoint frequency")
    p.add_argument("--seed", default=0, type=int, help="random seed.")
    return p


if __name__ == "__main__":
    parser = get_pretrain_args_parser()
    args = parser.parse_args()

    local_rank, rank, world_size = dist.setup()
    main(
        **vars(args),
        gpu_id=local_rank
    )
