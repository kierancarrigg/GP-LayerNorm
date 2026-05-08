# This file contains original contributions licensed under CC BY-NC-ND 4.0
# (Copyright (c) 2026 Kieran Carrigg), and portions adapted from the
# DyT repository (https://github.com/jiachenzhu/DyT), licensed under the
# MIT License (Copyright (c) 2025 Jiachen Zhu).
#
# MIT-licensed adapted components: NativeScalerWithGradNormCount,
# get_grad_norm_(), and SmoothedValue property methods (median, avg,
# global_avg, max, value).
# All other code is original and licensed under CC BY-NC-ND 4.0.
"""Shared utilities: metric tracking, distributed helpers, AMP scaler, W&B logging."""

import math
import os
import time
import datetime
from collections import defaultdict, deque
from typing import Iterable, Optional

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------

import argparse as _argparse


def str2bool(v) -> bool:
    """Convert a string CLI argument to a Python bool.

    Accepts yes/no, true/false, t/f, y/n, 1/0 (case-insensitive).
    Intended for use as the ``type=`` argument in argparse.

    Args:
        v: Value to convert (passthrough if already bool).

    Returns:
        Parsed boolean value.

    Raises:
        argparse.ArgumentTypeError: If the value cannot be interpreted as bool.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise _argparse.ArgumentTypeError('Boolean value expected.')


# ---------------------------------------------------------------------------
# Metric tracking
# ---------------------------------------------------------------------------

class SmoothedValue:
    """Tracks a series of values; exposes sliding-window and global statistics."""

    def __init__(self, window_size: int = 20, fmt: str = "{median:.4f} ({global_avg:.4f})"):
        self._window = deque(maxlen=window_size)
        self._total = 0.0
        self._count = 0
        self._fmt = fmt

    def update(self, value: float, n: int = 1) -> None:
        self._window.append(value)
        self._count += n
        self._total += value * n

    def synchronize_between_processes(self) -> None:
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self._count, self._total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        self._count = int(t[0].item())
        self._total = t[1].item()

    @property
    def median(self) -> float:
        return float(torch.tensor(list(self._window)).median().item())

    @property
    def avg(self) -> float:
        return float(torch.tensor(list(self._window), dtype=torch.float32).mean().item())

    @property
    def global_avg(self) -> float:
        return self._total / self._count

    @property
    def max(self) -> float:
        return max(self._window)

    @property
    def value(self) -> float:
        return self._window[-1]

    def __str__(self) -> str:
        return self._fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    """Aggregates named metrics and logs training progress to stdout."""

    def __init__(self, delimiter: str = "\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            if not isinstance(v, (float, int)):
                raise TypeError(f"Metric '{k}' must be float or int, got {type(v)}")
            self.meters[k].update(v)

    def __getattr__(self, attr: str):
        if attr in ("meters", "delimiter"):
            raise AttributeError(attr)
        if attr in self.meters:
            return self.meters[attr]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{attr}'")

    def __str__(self) -> str:
        return self.delimiter.join(f"{k}: {v}" for k, v in self.meters.items())

    def synchronize_between_processes(self) -> None:
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name: str, meter: SmoothedValue) -> None:
        self.meters[name] = meter

    def log_every(self, iterable, print_freq: int, header: str = "") -> Iterable:
        """Yield items from iterable, printing a progress line every print_freq steps."""
        total = len(iterable)
        width = len(str(total))
        MB = 1024.0 ** 2
        t_start = time.time()
        t_end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")

        for i, obj in enumerate(iterable):
            data_time.update(time.time() - t_end)
            yield obj
            iter_time.update(time.time() - t_end)
            if i % print_freq == 0 or i == total - 1:
                eta = str(datetime.timedelta(seconds=int(iter_time.global_avg * (total - i))))
                parts = [
                    header,
                    f"[{i:{width}d}/{total}]",
                    f"eta: {eta}",
                    str(self),
                    f"time: {iter_time}",
                    f"data: {data_time}",
                ]
                if torch.cuda.is_available():
                    parts.append(f"max mem: {torch.cuda.max_memory_allocated() / MB:.0f}")
                print(self.delimiter.join(parts))
            t_end = time.time()

        elapsed = time.time() - t_start
        print(f"{header} Total time: {str(datetime.timedelta(seconds=int(elapsed)))} ({elapsed / total:.4f} s / it)")


# ---------------------------------------------------------------------------
# Distributed utilities
# ---------------------------------------------------------------------------

def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def save_on_master(*args, **kwargs) -> None:
    if is_main_process():
        torch.save(*args, **kwargs)


def _suppress_non_master_prints() -> None:
    """Silence print output on all non-master processes."""
    import builtins
    _print = builtins.print

    def _filtered(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_main_process() or force:
            _print(*args, **kwargs)

    builtins.print = _filtered


def init_distributed_mode(args) -> None:
    """Initialise torch distributed from environment variables or SLURM."""
    if getattr(args, "dist_on_itp", False):
        args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        args.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        args.gpu = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        args.dist_url = "tcp://{}:{}".format(
            os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"]
        )
        os.environ.update({
            "LOCAL_RANK": str(args.gpu),
            "RANK": str(args.rank),
            "WORLD_SIZE": str(args.world_size),
        })
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
        os.environ.update({
            "RANK": str(args.rank),
            "LOCAL_RANK": str(args.gpu),
            "WORLD_SIZE": str(args.world_size),
        })
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}, gpu {args.gpu}", flush=True)
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()
    _suppress_non_master_prints()


# ---------------------------------------------------------------------------
# AMP and gradient utilities
# ---------------------------------------------------------------------------

class NativeScalerWithGradNormCount:
    """Wraps torch.cuda.amp.GradScaler and returns gradient norm after each step."""

    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        clip_grad: Optional[float] = None,
        parameters=None,
        create_graph: bool = False,
        update_grad: bool = True,
    ) -> Optional[torch.Tensor]:
        self._scaler.scale(loss).backward(create_graph=create_graph)
        norm = None
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    """Compute total gradient norm across a collection of parameters."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    device = grads[0].device
    if norm_type == float("inf"):
        return max(g.detach().abs().max().to(device) for g in grads)
    norms = torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads])
    return torch.norm(norms, norm_type)


# ---------------------------------------------------------------------------
# W&B logger
# ---------------------------------------------------------------------------

class WandbLogger:
    """Minimal W&B wrapper for epoch-level metric logging."""

    def __init__(self, args):
        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            raise ImportError("wandb is not installed. Run: pip install wandb")
        if self._wandb.run is None:
            self._wandb.init(
                project=args.project,
                name=getattr(args, "wandb_run_name", None),
                tags=[args.wandb_tag] if getattr(args, "wandb_tag", None) else None,
                config=vars(args),
            )

    def log_epoch_metrics(self, metrics: dict, commit: bool = True) -> None:
        """Log a dict of train_*/test_* metrics indexed by epoch."""
        metrics = dict(metrics)
        self._wandb.summary["n_parameters"] = metrics.pop("n_parameters", None)
        epoch = metrics.pop("epoch", None)
        if epoch is not None:
            self._wandb.log({"epoch": epoch}, commit=False)
        for k, v in metrics.items():
            ns = "Global Train" if "train" in k else "Global Test"
            self._wandb.log({f"{ns}/{k}": v}, commit=False)
        self._wandb.log({})
