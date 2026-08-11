"""Training script for finetuning a pretrained ViT with DynamicTanh (DyT) layers.

Replaces LayerNorm modules with DyT equivalents, then finetunes on ImageNet using
one of four training modes (affine, alpha, alpha_affine, full). Supports logit
distillation from a frozen LN teacher, per-parameter-group learning rates,
mixed-precision training, and distributed training via torchrun.
"""
import os
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import time
import datetime

from timm.models import create_model
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import build_dataset, train_one_epoch, evaluate, train_one_epoch_logit_distill, str2bool
import utils
import wandb

from baselines.dyt import convert_ln_to_dyt, DynamicTanh


def set_trainable_dyt_params(model, mode: str):
    """
    mode:
      - 'alpha'       : train only alpha
      - 'alpha_affine': train alpha + (weight,bias)
      - 'affine'      : train only (weight,bias)
      - 'full'        : train the full model
    """
    if not mode == "full":
        for p in model.parameters():
            p.requires_grad = False

        for m in model.modules():
            if isinstance(m, DynamicTanh):
                if mode == "alpha":
                    m.alpha.requires_grad = True
                elif mode == "alpha_affine":
                    m.alpha.requires_grad = True
                    m.weight.requires_grad = True
                    m.bias.requires_grad = True
                elif mode == "affine":
                    m.weight.requires_grad = True
                    m.bias.requires_grad = True



def get_dyt_param_groups(model, lr_alpha, lr_affine, weight_decay):
    """Build optimizer param groups for DyT-only training modes.

    Reads requires_grad flags set by set_trainable_dyt_params and assigns
    separate learning rates to alpha scalars and affine (weight/bias) params.
    Alpha params always have weight_decay=0.

    Args:
        model: Model with DynamicTanh modules.
        lr_alpha: Learning rate for alpha parameters.
        lr_affine: Learning rate for weight/bias parameters.
        weight_decay: Weight decay applied to affine params.

    Returns:
        List of param group dicts for torch.optim.
    """
    alpha_params = []
    affine_params = []

    for m in model.modules():
        if isinstance(m, DynamicTanh):
            if m.alpha.requires_grad:
                alpha_params.append(m.alpha)
            if m.weight.requires_grad:
                affine_params.append(m.weight)
            if m.bias.requires_grad:
                affine_params.append(m.bias)

    param_groups = []
    if alpha_params:
        param_groups.append({
            "params": alpha_params,
            "lr": lr_alpha,
            "weight_decay": 0.0,   # don’t WD scalars
        })
    if affine_params:
        param_groups.append({
            "params": affine_params,
            "lr": lr_affine,
            "weight_decay": weight_decay,  # ok to regularize weight/bias lightly
        })

    return param_groups

def get_full_param_groups(model, lr_backbone, wd_backbone, lr_alpha, lr_affine, wd_dyt):
    """Build optimizer param groups for full model finetuning.

    Separates parameters into four groups: backbone (with weight decay),
    backbone no-WD (1D params and biases), DyT alpha (no WD), and DyT affine.
    DyT params are identified by module type rather than requires_grad flags.

    Args:
        model: Model with DynamicTanh modules.
        lr_backbone: Learning rate for backbone parameters.
        wd_backbone: Weight decay for backbone parameters.
        lr_alpha: Learning rate for DyT alpha scalars.
        lr_affine: Learning rate for DyT weight/bias params.
        wd_dyt: Weight decay for DyT affine params.

    Returns:
        List of param group dicts for torch.optim.
    """
    alpha_params, affine_params, backbone_params = [], [], []
    backbone_no_wd = []  # biases + 1D params (common practice)

    # Detect DyT params by module type:
    dyt_param_ids = set()
    for m in model.modules():
        if isinstance(m, DynamicTanh):
            for p in [m.alpha, m.weight, m.bias]:
                dyt_param_ids.add(id(p))

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in dyt_param_ids:
            # DyT params
            if name.endswith(".alpha"):
                alpha_params.append(p)
            else:
                affine_params.append(p)
        else:
            # Backbone params
            if p.ndim == 1 or name.endswith(".bias"):
                backbone_no_wd.append(p)
            else:
                backbone_params.append(p)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": wd_backbone})
    if backbone_no_wd:
        groups.append({"params": backbone_no_wd, "lr": lr_backbone, "weight_decay": 0.0})

    if alpha_params:
        groups.append({"params": alpha_params, "lr": lr_alpha, "weight_decay": 0.0})
    if affine_params:
        groups.append({"params": affine_params, "lr": lr_affine, "weight_decay": wd_dyt})

    return groups

def get_args_parser():
    """Build the argument parser for the DyT finetuning script.

    Returns:
        argparse.ArgumentParser with all training, model, dataset,
        distillation, distributed, and logging arguments.
    """
    p = argparse.ArgumentParser("DyT minimal finetune", add_help=False)

    # core
    p.add_argument("--model", default="vit_base_patch16_224", type=str)
    p.add_argument("--epochs", default=20, type=int)
    p.add_argument("--batch_size", default=512, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--lr_alpha", default=None, type=float, help="LR for DyT alpha params (overrides --lr if set)")
    p.add_argument("--lr_affine", default=None, type=float, help="LR for DyT weight/bias params (overrides --lr if set)")
    p.add_argument("--lr_backbone", default=None, type=float,
               help="LR for all non-DyT params (overrides --lr if set).")
    p.add_argument("--wd_backbone", default=0.02, type=float,
               help="Weight decay for backbone params.")
    p.add_argument("--max_norm", default=None, type=float)
    p.add_argument("--lr_scheduler", default="cosine", type=str, choices=["cosine", "none"],
                   help="LR schedule: 'cosine' anneals the LR to zero over --epochs, 'none' keeps it constant")
    p.add_argument("--weight_decay", default=0.0, type=float)
    p.add_argument("--train_mode", default="alpha_affine", choices=["alpha", "alpha_affine", "affine", "full"])
    p.add_argument("--eval_only", type=str2bool, default=False)
    p.add_argument("--drop_path", default=0.0, type=float)
    p.add_argument("--input_size", default=224, type=int)
    p.add_argument("--output_dir", default="", type=str)

    # Architecture & Distillation
    p.add_argument("--use_dyt", type=str2bool, default=True, help="Convert LN to DyT. Set to False for pure ViT-B baseline.")
    p.add_argument("--distill_logit", type=str2bool, default=False, help="Enable logit-based Knowledge Distillation")
    p.add_argument("--distill_temp", type=float, default=4.0, help="Temperature for softening teacher logits")
    p.add_argument("--distill_alpha", type=float, default=0.5, help="Weight for distillation loss")

    # dataset / transforms (build_dataset expects these; training uses more)
    p.add_argument("--data_path", required=True, type=str)
    p.add_argument("--data_set", default="IMNET", choices=["IMNET", "image_folder"])
    p.add_argument("--eval_data_path", default=None, type=str)
    p.add_argument("--nb_classes", default=1000, type=int)
    p.add_argument("--imagenet_default_mean_and_std", type=str2bool, default=True)
    p.add_argument("--crop_pct", type=float, default=None)

    # training transform args (used by datasets.py when is_train=True)
    # keep defaults matching main.py
    p.add_argument("--color_jitter", type=float, default=0.4)
    p.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1")
    p.add_argument("--train_interpolation", type=str, default="bicubic")
    p.add_argument("--reprob", type=float, default=0.25)
    p.add_argument("--remode", type=str, default="pixel")
    p.add_argument("--recount", type=int, default=1)

    # runtime
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--num_workers", default=8, type=int)
    p.add_argument("--pin_mem", type=str2bool, default=True)
    p.add_argument("--use_amp", type=str2bool, default=True)

    # wandb (optional)
    p.add_argument("--enable_wandb", type=str2bool, default=False)
    p.add_argument("--project", default="DyT-finetune", type=str)
    p.add_argument("--wandb_run_name", default=None, type=str)
    p.add_argument("--wandb_tag", default=None, type=str)
    # distributed training parameters
    p.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    p.add_argument('--local_rank', default=-1, type=int)
    p.add_argument('--dist_on_itp', type=str2bool, default=False)
    p.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return p


def main(args):
    """Run DyT finetuning.

    Loads a pretrained ViT, converts LayerNorm to DyT, configures trainable
    parameters according to args.train_mode, builds an AdamW optimizer with
    optional per-group learning rates, and runs the training loop with
    per-epoch evaluation and checkpoint saving.

    Args:
        args: Parsed arguments from get_args_parser().
    """
    utils.init_distributed_mode(args)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True  # training: faster

    # datasets
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.enable_wandb:
        wandb_logger = utils.WandbLogger(args)
    else:
        wandb_logger = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
    else:
        data_loader_val = None

    # model: start from pretrained timm (your baseline path)
    model = create_model(
        args.model,
        pretrained=True,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
    )

    # replace LN -> DyT
    if args.use_dyt:
        model = convert_ln_to_dyt(model)

        set_trainable_dyt_params(model, args.train_mode)
    else:
        print("\n--- Running Pure ViT-B Baseline (No DyT Conversion) ---")
        if args.train_mode != "full":
            print("Forcing train_mode='full' since we are training standard ViT-B.")
            args.train_mode = "full"
        for p in model.parameters():
            p.requires_grad = True

    model.to(device)

    # --- LOGIT DISTILLATION SETUP ---
    teacher_model = None
    if args.distill_logit:
        print("\n--- Setting up Logit Distillation ---")
        teacher_model = create_model(
            args.model,
            pretrained=True,
            num_classes=args.nb_classes,
        )
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        teacher_model.to(device)
        print("Teacher model loaded and frozen.\n")

    # Decide LRs (Fall back to global --lr if specific ones aren't provided)
    lr_alpha   = args.lr_alpha   if args.lr_alpha   is not None else args.lr
    lr_affine  = args.lr_affine  if args.lr_affine  is not None else args.lr
    lr_backbone = args.lr_backbone if args.lr_backbone is not None else args.lr

    if args.train_mode == "full":
        param_groups = get_full_param_groups(
            model,
            lr_backbone=lr_backbone,
            wd_backbone=args.wd_backbone,
            lr_alpha=lr_alpha,
            lr_affine=lr_affine,
            wd_dyt=args.weight_decay,
        )
        print(f"FULL finetune groups: lr_backbone={lr_backbone}, wd_backbone={args.wd_backbone}, lr_affine={lr_affine}, lr_alpha={lr_alpha}")
    else:
        param_groups = get_dyt_param_groups(model, lr_alpha, lr_affine, args.weight_decay)
        print(f"DyT-only mode ({args.train_mode}): lr_alpha={lr_alpha}, lr_affine={lr_affine}")
        print(f"Number of trainable DyT params: {sum(p.numel() for g in param_groups for p in g['params'])}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay, lr=args.lr)

    # LR schedule (stepped once per epoch, after training)
    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    criterion = torch.nn.CrossEntropyLoss()

    # AMP scaler
    loss_scaler = utils.NativeScalerWithGradNormCount() if args.use_amp else None

    max_accuracy = 0.0
    # train loop
    if not args.eval_only:
        # --- INITIAL EVALUATION (EPOCH 0) ---
        print("Performing initial zero-shot evaluation (Epoch 0)...")
        model.eval()
        init_test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
        print(f"Initial val_acc1: {init_test_stats['acc1']:.3f}%")
        
        if wandb_logger:
            log_stats = {f'test_{k}': v for k, v in init_test_stats.items()}
            log_stats['epoch'] = 0 
            wandb_logger.log_epoch_metrics(log_stats)

        start_time = time.time()
        for epoch in range(1, args.epochs+1):
            model.train()
            print(f"Starting epoch {epoch}...")
            if args.distill_logit:
                train_stats = train_one_epoch_logit_distill(
                    student_model=model,
                    teacher_model=teacher_model,
                    criterion=criterion,
                    optimizer=optimizer,
                    data_loader=data_loader_train,
                    device=device,
                    epoch=epoch,
                    loss_scaler=loss_scaler,
                    use_amp=args.use_amp,
                    temp=args.distill_temp,
                    alpha=args.distill_alpha
                )
            else:
                train_stats = train_one_epoch(
                    model=model,
                    criterion=criterion,
                    data_loader=data_loader_train,
                    optimizer=optimizer,
                    device=device,
                    epoch=epoch,
                    loss_scaler=loss_scaler,
                    max_norm=args.max_norm,
                    model_ema=None,
                    mixup_fn=None,
                    log_writer=None,
                    wandb_logger=wandb_logger,
                    start_steps=epoch * len(data_loader_train),
                    lr_schedule_values=None,
                    wd_schedule_values=None,
                    num_training_steps_per_epoch=len(data_loader_train),
                    update_freq=1,
                    use_amp=args.use_amp,
                )

            if scheduler is not None:
                scheduler.step()

            model.eval()
            test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)

            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }

            if utils.is_main_process():
                torch.save(ckpt, os.path.join(args.output_dir, "dyt_finetune_last.pt"))
                if test_stats['acc1'] > max_accuracy:
                    print(f"*** New best accuracy! {max_accuracy:.2f}% -> {test_stats['acc1']:.2f}% ***")
                    max_accuracy = test_stats['acc1']
                    torch.save(ckpt, os.path.join(args.output_dir, "dyt_finetune_best.pt"))

            print(f"[epoch {epoch}] train_loss={train_stats.get('loss', None)}  "
                f"val_acc1={test_stats['acc1']:.3f}  val_acc5={test_stats['acc5']:.3f}  val_loss={test_stats['loss']:.4f}")

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,}
            if wandb_logger:
                wandb_logger.log_epoch_metrics(log_stats)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'Training time {total_time_str}')
    else:
        model.eval()
        test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
        print(f"[eval only] val_acc1={test_stats['acc1']:.3f}  val_acc5={test_stats['acc5']:.3f}  val_loss={test_stats['loss']:.4f}")
        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch':0,}
        if wandb_logger:
            wandb_logger.log_epoch_metrics(log_stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("DyT minimal finetune", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)