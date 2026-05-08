# This file contains original contributions licensed under CC BY-NC-ND 4.0
# (Copyright (c) 2026 Kieran Carrigg), and is adapted from the DyT
# repository (https://github.com/jiachenzhu/DyT), licensed under the
# MIT License (Copyright (c) 2025 Jiachen Zhu).
"""Training and evaluation loops for GP-LayerNorm experiments."""

import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from . import helpers


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler=None,
    max_norm: float = 0,
    model_ema=None,
    mixup_fn=None,
    log_writer=None,
    wandb_logger=None,
    start_steps: int = 0,
    lr_schedule_values=None,
    wd_schedule_values=None,
    num_training_steps_per_epoch: Optional[int] = None,
    update_freq: int = 1,
    use_amp: bool = False,
) -> dict:
    """
    Train the model for one epoch.

    Args:
        model: Model to train.
        criterion: Classification loss function.
        data_loader: Training data loader.
        optimizer: Optimizer.
        device: Training device.
        epoch: Current epoch index (for logging).
        loss_scaler: AMP grad scaler (NativeScalerWithGradNormCount), or None for fp32.
        max_norm: Gradient clipping norm (0 = disabled).
        model_ema: Optional EMA model to update after each step.
        mixup_fn: Optional Mixup/CutMix function.
        log_writer: Unused (kept for API compatibility).
        wandb_logger: Optional WandbLogger instance for batch-level logging.
        start_steps: Global step offset for LR/WD schedule indexing.
        lr_schedule_values: Per-step LR schedule array, or None.
        wd_schedule_values: Per-step WD schedule array, or None.
        num_training_steps_per_epoch: Steps per epoch (defaults to len(data_loader)).
        update_freq: Gradient accumulation steps.
        use_amp: Enable automatic mixed precision.

    Returns:
        Dict of averaged training metrics.
    """
    model.train(True)
    if num_training_steps_per_epoch is None:
        num_training_steps_per_epoch = len(data_loader)

    metric_logger = helpers.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", helpers.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("min_lr", helpers.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, 10, header)
    ):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step

        # Apply per-step LR/WD schedules if provided
        if (lr_schedule_values is not None or wd_schedule_values is not None)                 and data_iter_step % update_freq == 0:
            for pg in optimizer.param_groups:
                if lr_schedule_values is not None:
                    pg["lr"] = lr_schedule_values[it] * pg.get("lr_scale", 1.0)
                if wd_schedule_values is not None and pg["weight_decay"] > 0:
                    pg["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        else:
            output = model(samples)
            loss = criterion(output, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss = loss / update_freq
        update_now = (data_iter_step + 1) % update_freq == 0

        if use_amp:
            is_second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
            grad_norm = loss_scaler(
                loss, optimizer,
                clip_grad=max_norm if max_norm and max_norm > 0 else None,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=update_now,
            )
            if update_now:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else:
            loss.backward()
            if update_now:
                if max_norm and max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None

        torch.cuda.synchronize()

        class_acc = (output.max(-1)[-1] == targets).float().mean() if mixup_fn is None else None
        metric_logger.update(loss=loss_value)
        if class_acc is not None:
            metric_logger.update(class_acc=class_acc)

        min_lr = min(pg["lr"] for pg in optimizer.param_groups)
        max_lr = max(pg["lr"] for pg in optimizer.param_groups)
        metric_logger.update(lr=max_lr, min_lr=min_lr)

        wd = next(
            (pg["weight_decay"] for pg in optimizer.param_groups if pg["weight_decay"] > 0),
            None,
        )
        metric_logger.update(weight_decay=wd)

        if use_amp and grad_norm is not None:
            metric_logger.update(grad_norm=grad_norm)

        if wandb_logger:
            wandb_logger._wandb.log({
                "Rank-0 Batch Wise/train_loss": loss_value,
                "Rank-0 Batch Wise/train_max_lr": max_lr,
                "Rank-0 Batch Wise/train_min_lr": min_lr,
                "Rank-0 Batch Wise/global_train_step": it,
            })

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, use_amp: bool = False) -> dict:
    """
    Evaluate model accuracy and loss on a validation set.

    Args:
        data_loader: Validation data loader.
        model: Model to evaluate.
        device: Evaluation device.
        use_amp: Enable automatic mixed precision.

    Returns:
        Dict with keys 'loss', 'acc1', 'acc5'.
    """
    from timm.utils import accuracy

    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = helpers.MetricLogger(delimiter="  ")
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, "Test:"):
        images, target = batch[0], batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    print(
        f"* Acc@1 {metric_logger.acc1.global_avg:.3f}  "
        f"Acc@5 {metric_logger.acc5.global_avg:.3f}  "
        f"loss {metric_logger.loss.global_avg:.3f}"
    )
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_logit_distill(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    loss_scaler=None,
    use_amp: bool = False,
    temp: float = 4.0,
    alpha: float = 0.5,
) -> dict:
    """
    Train for one epoch using logit-based knowledge distillation.

    The blended loss is: (1 - alpha) * CE(student, labels) + alpha * KL(student || teacher).

    Args:
        student_model: Model being trained.
        teacher_model: Frozen teacher model providing soft targets.
        criterion: Cross-entropy loss for the hard-label term.
        optimizer: Optimizer for the student.
        data_loader: Training data loader.
        device: Training device.
        epoch: Current epoch index (for logging).
        loss_scaler: AMP grad scaler, or None for fp32.
        use_amp: Enable automatic mixed precision.
        temp: Softmax temperature for soft targets.
        alpha: Weight on the KD term (0 = pure CE, 1 = pure KD).

    Returns:
        Dict of averaged training metrics (loss, ce_loss, kd_loss).
    """
    student_model.train()
    teacher_model.eval()

    metric_logger = helpers.MetricLogger(delimiter="  ")
    metric_logger.add_meter("loss", helpers.SmoothedValue(window_size=20, fmt="{value:.4f}"))
    metric_logger.add_meter("ce_loss", helpers.SmoothedValue(window_size=20, fmt="{value:.4f}"))
    metric_logger.add_meter("kd_loss", helpers.SmoothedValue(window_size=20, fmt="{value:.4f}"))
    header = f"Epoch: [{epoch}] (distill)"

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                teacher_logits = teacher_model(samples)

        with torch.cuda.amp.autocast(enabled=use_amp):
            student_logits = student_model(samples)
            ce_loss = criterion(student_logits, targets)
            kd_loss = F.kl_div(
                F.log_softmax(student_logits / temp, dim=1),
                F.softmax(teacher_logits / temp, dim=1),
                reduction="batchmean",
            ) * (temp ** 2)
            loss = (1.0 - alpha) * ce_loss + alpha * kd_loss

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, parameters=student_model.parameters())
        else:
            loss.backward()
            optimizer.step()

        metric_logger.update(
            loss=loss_value, ce_loss=ce_loss.item(), kd_loss=kd_loss.item()
        )

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
