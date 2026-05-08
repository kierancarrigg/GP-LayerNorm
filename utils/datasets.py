# This file contains original contributions licensed under CC BY-NC-ND 4.0
# (Copyright (c) 2026 Kieran Carrigg), and is adapted from the DyT
# repository (https://github.com/jiachenzhu/DyT), licensed under the
# MIT License (Copyright (c) 2025 Jiachen Zhu).
"""Dataset loading for ImageNet training and evaluation."""

import os
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)


def build_dataset(is_train: bool, args) -> tuple:
    """
    Build a dataset for training or evaluation.

    Args:
        is_train: Whether to build the training split (vs. validation).
        args: Namespace with fields: data_set, data_path, input_size,
              imagenet_default_mean_and_std, crop_pct, and augmentation args.

    Returns:
        (dataset, num_classes)
    """
    transform = build_transform(is_train, args)

    if args.data_set == "IMNET":
        split = "train" if is_train else "val"
        dataset = datasets.ImageFolder(
            os.path.join(args.data_path, split), transform=transform
        )
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
    else:
        raise ValueError(f"Unknown dataset: {args.data_set!r}")

    return dataset, nb_classes


def build_transform(is_train: bool, args) -> transforms.Compose:
    """
    Build the image transform pipeline.

    Args:
        is_train: Whether to apply training augmentations.
        args: Namespace with input_size, imagenet_default_mean_and_std,
              color_jitter, aa, train_interpolation, reprob, remode, recount,
              crop_pct.

    Returns:
        Composed transform pipeline.
    """
    use_default_stats = getattr(args, "imagenet_default_mean_and_std", True)
    mean = IMAGENET_DEFAULT_MEAN if use_default_stats else IMAGENET_INCEPTION_MEAN
    std = IMAGENET_DEFAULT_STD if use_default_stats else IMAGENET_INCEPTION_STD

    if is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=getattr(args, "color_jitter", 0.4),
            auto_augment=getattr(args, "aa", "rand-m9-mstd0.5-inc1"),
            interpolation=getattr(args, "train_interpolation", "bicubic"),
            re_prob=getattr(args, "reprob", 0.25),
            re_mode=getattr(args, "remode", "pixel"),
            re_count=getattr(args, "recount", 1),
            mean=mean,
            std=std,
        )
        return transform

    # Validation transforms
    t = []
    if args.input_size >= 384:
            # Full-size resize without crop for large resolution inputs
            t.append(
                transforms.Resize(
                    (args.input_size, args.input_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                )
            )
    else:
        crop_pct = getattr(args, "crop_pct", None) or (224 / 256)
        size = int(args.input_size / crop_pct)
        t.append(transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC))
        t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
