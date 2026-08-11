"""GP fine-tuning script: fine-tune a ViT with evolved LayerNorm replacements on ImageNet.

Loads a dynamically generated evolved_layers_seed_N module from gp/layers/, injects the
evolved expressions into a pretrained ViT, then fine-tunes either the full model or just
the normalization affine parameters. Supports logit-distillation from a frozen teacher.
"""
import argparse
import datetime
import importlib
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from timm.models import create_model

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from utils import (
    build_dataset, train_one_epoch, evaluate,
    train_one_epoch_logit_distill, NativeScalerWithGradNormCount, str2bool,
)
import utils

def set_trainable_gp_params(model, mode: str, EvolvedLayerClass):
    """Configure which parameters require gradients.

    Args:
        model: The ViT model with injected evolved layers.
        mode: Training mode. ``'affine'`` freezes everything except the weight/bias
            of EvolvedLayer and nn.LayerNorm modules; ``'full'`` leaves all parameters
            trainable.
        EvolvedLayerClass: The EvolvedLayer class from the dynamically loaded module.
    """
    print(f"Setting trainable parameters mode: {mode}")
    
    if mode == "full":
        return

    if mode == "affine":
        # 1. Freeze EVERYTHING first
        for p in model.parameters():
            p.requires_grad = False

        # 2. Unfreeze only the normalization parameters
        count = 0
        for m in model.modules():
            if isinstance(m, (EvolvedLayerClass, nn.LayerNorm)):
                if m.weight is not None:
                    m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.requires_grad = True
                count += 1
        print(f" -> Unfroze parameters for {count} normalization layers.")

def get_args_parser():
    """Build the argument parser for the GP fine-tuning script.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser('Evolved GP Fine-tuning', add_help=False)
    
    # --- Training Parameters ---
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_scheduler', default='cosine', type=str, choices=['cosine', 'none'],
                        help="LR schedule: 'cosine' anneals the LR to zero over --epochs, 'none' keeps it constant")
    parser.add_argument('--weight_decay', default=0.05, type=float)
    # --- Model Parameters ---
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--drop', type=float, default=0.0, help='Dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, help='Drop path rate (default: 0.1)')

    # --- GP Specific ---
    parser.add_argument('--gp_seed', default=1, type=int, help="Which GP seed to load evolved layers from (1-5)")
    parser.add_argument('--train_mode', default='affine', type=str, choices=['affine', 'full'],
                        help='affine: train only norm weights/biases. full: train everything.')
    parser.add_argument('--distill_logit', type=str2bool, default=False,
                        help='Enable Logit-based Knowledge Distillation')
    parser.add_argument('--distill_temp', type=float, default=4.0,
                        help='Temperature for softening teacher logits')
    parser.add_argument('--distill_alpha', type=float, default=0.5,
                        help='Weight for distillation loss (1.0 = pure KD, 0.0 = pure CE)')

    # --- Optimizer ---
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        choices=['sgd', 'adamw'],
                        help='Optimizer (default: "adamw")')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')

    # --- Data / System ---
    parser.add_argument('--data_path', default='/path/to/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--data_set', default='IMNET', choices=['IMNET', 'image_folder'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True)

    # --- Distributed / Amp ---
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--use_amp', type=str2bool, default=True, help="Use PyTorch AMP (Automatic Mixed Precision)")
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)

    # --- Augmentation & Dataset Params ---
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". (default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--crop_pct', type=float, default=None)

    # --- Evaluation ---
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    
    # --- Logging ---
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--enable_wandb', type=str2bool, default=False)
    parser.add_argument('--project', default='GP-ViT-Finetune', type=str)
    parser.add_argument('--wandb_run_name', default=None, type=str)
    parser.add_argument('--wandb_tag', default=None, type=str, help="Tag for W&B run")

    return parser


def main(args):
    """Fine-tune a ViT with injected evolved LayerNorm replacements.

    Dynamically loads the evolved layer module for the requested GP seed, injects the
    evolved expressions into a pretrained model, configures which parameters are trained,
    optionally sets up a frozen teacher for logit distillation, then runs the training
    loop with per-epoch validation, saving last and best checkpoints to output_dir.

    Args:
        args: Parsed argument namespace from get_args_parser.
    """
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # 1. Load Dataset
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # 2. Create Model
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=True,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
    )
    
    # 3. APPLY EVOLUTION (Dynamically Loaded)
    print(f"Loading Evolved Layers for Seed: {args.gp_seed}...")

    layers_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "layers")
    if layers_dir not in sys.path:
        sys.path.append(layers_dir)

    module_name = f"evolved_layers_seed_{args.gp_seed}"
    try:
        evolved_module = importlib.import_module(module_name)
        apply_evolution = evolved_module.apply_evolution
        EvolvedLayer = evolved_module.EvolvedLayer
    except ImportError:
        print(f"❌ Error: Could not find {module_name}.py! Did you run generate_model_code.py?")
        return

    print("Injecting Evolved Layers...")
    model = apply_evolution(model, verbose=True)

    print("\n--- Verifying Injected Equations ---")
    for name, module in model.named_modules():
        if isinstance(module, EvolvedLayer) and module.__doc__:
            print(f"{name}: {module.__doc__.split('Equation:')[-1].strip()}")
    print("------------------------------------\n")

    # 4. Set Trainable Parameters
    set_trainable_gp_params(model, args.train_mode, EvolvedLayer)

    model.to(device)
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'number of trainable params: {n_parameters}')

    # --- 4b. LOGIT DISTILLATION SETUP ---
    teacher_model = None
    if args.distill_logit:
        print("\n--- Setting up Logit Distillation ---")
        # Create the frozen Teacher (Standard ViT, NO evolution)
        teacher_model = create_model(
            args.model,
            pretrained=True,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
        )
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        teacher_model.to(device)
        print("Teacher model loaded and frozen.\n")

    # Optimizer
    optimizer = create_optimizer(args, model)

    # LR schedule (stepped once per epoch, after training)
    scheduler = None
    if args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss scaling
    loss_scaler = NativeScalerWithGradNormCount() if args.use_amp else None
    criterion = torch.nn.CrossEntropyLoss()

    # WandB
    if args.enable_wandb and utils.is_main_process():
        wandb_logger = utils.WandbLogger(args)
    else:
        wandb_logger = None

    # --- EVALUATION ONLY ---
    if args.eval:
        print("Starting Zero-Shot Evaluation...")
        test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        if wandb_logger:
            log_stats = {f'test_{k}': v for k, v in test_stats.items()}
            log_stats['epoch'] = 0
            wandb_logger.log_epoch_metrics(log_stats)
        return

    # --- INITIAL EVALUATION (EPOCH 0) ---
    print("Performing initial zero-shot evaluation (Epoch 0)...")
    init_test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
    print(f"Initial accuracy: {init_test_stats['acc1']:.1f}%")
    
    if wandb_logger:
        log_stats = {f'test_{k}': v for k, v in init_test_stats.items()}
        log_stats['epoch'] = args.start_epoch  # This logs as 0
        wandb_logger.log_epoch_metrics(log_stats)
    
    # --- TRAINING LOOP ---
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch+1, args.epochs+1):
        if args.distill_logit:
            # Run our custom Logit Distillation loop
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
                max_norm=None,
                model_ema=None,
                mixup_fn=None,
                log_writer=None,
                wandb_logger=wandb_logger,
                start_steps=epoch * len(data_loader_train),
                lr_schedule_values=None,
                wd_schedule_values=None,
                num_training_steps_per_epoch=len(data_loader_train),
                update_freq=1,
                use_amp=args.use_amp
            )

        if scheduler is not None:
            scheduler.step()

        if args.output_dir:
            # 1. Always save the LAST epoch
            last_ckpt_path = os.path.join(args.output_dir, 'checkpoint-last.pth')
            utils.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }, last_ckpt_path)

        test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        
        # 2. Check if this is the BEST epoch and save
        if args.output_dir and test_stats['acc1'] > max_accuracy:
            print(f"*** New best accuracy! {max_accuracy:.2f}% -> {test_stats['acc1']:.2f}% ***")
            max_accuracy = test_stats['acc1']
            best_ckpt_path = os.path.join(args.output_dir, 'checkpoint-best.pth')
            utils.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }, best_ckpt_path)
        
        if wandb_logger:
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
            }
            wandb_logger.log_epoch_metrics(log_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')

def create_optimizer(args, model):
    """Construct the optimizer from args.

    Args:
        args: Parsed argument namespace; uses ``args.opt``, ``args.lr``,
            ``args.weight_decay``, and ``args.momentum`` (SGD only).
        model: The model; only parameters with ``requires_grad=True`` are included.

    Returns:
        Configured optimizer instance.

    Raises:
        ValueError: If ``args.opt`` is not ``'sgd'`` or ``'adamw'``.
    """
    opt_lower = args.opt.lower()
    parameters = [p for p in model.parameters() if p.requires_grad]
    
    if opt_lower == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evolved GP training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    print("\n" + "="*60)
    print("Configuration:")
    print("="*60)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    main(args)