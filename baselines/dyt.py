"""
DynamicTanh (DyT) implementation for normalization-free transformers.

This implementation is adapted from the DyT repository:
https://github.com/jiachenzhu/DyT

Original paper: "Transformers without Normalization" (CVPR 2025)
by Jiachen Zhu, Xinlei Chen, Kaiming He, Yann LeCun, and Zhuang Liu

Licensed under MIT License (Copyright 2025 Jiachen Zhu)

DyT is an element-wise operation defined as: DyT(x) = tanh(α·x), where α is
a learnable scalar. It replaces LayerNorm in transformers while maintaining
similar performance.

We include this as a baseline for comparison with our GP-evolved normalization
replacement functions.
"""

import torch
import torch.nn as nn
from timm.layers import LayerNorm2d


class DynamicTanh(nn.Module):
    """
    DynamicTanh layer for replacing normalization in transformers.
    
    Applies: output = tanh(α * x) * weight + bias
    where α is a learnable scalar, and weight/bias are affine parameters.
    
    Args:
        normalized_shape: Shape of the normalized dimensions (e.g., embedding dim)
        channels_last: Whether input is in channels-last format (True) or channels-first (False)
        alpha_init_value: Initial value for the α parameter (default: 0.5)
    """
    
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        # Learnable scaling parameter α
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        
        # Affine transformation parameters (like LayerNorm)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        """
        Forward pass: tanh(α * x) * weight + bias
        
        Args:
            x: Input tensor
        
        Returns:
            Transformed tensor with same shape as input
        """
        x = torch.tanh(self.alpha * x)
        
        if self.channels_last:
            # For standard transformers (batch, seq, dim)
            x = x * self.weight + self.bias
        else:
            # For ConvNets (batch, channels, height, width)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        
        return x

    def extra_repr(self):
        return (f"normalized_shape={self.normalized_shape}, "
                f"alpha_init_value={self.alpha_init_value}, "
                f"channels_last={self.channels_last}")


def convert_ln_to_dyt(module, prefix=""):
    """
    Recursively convert all LayerNorm layers in a model to DynamicTanh.
    
    Args:
        module: PyTorch module to convert
        prefix: Internal prefix for tracking layer names (used in recursion)
    
    Returns:
        Module with all LayerNorm layers replaced by DynamicTanh
    """
    module_output = module
    
    if isinstance(module, nn.LayerNorm):
        # Create DynamicTanh layer with same shape as original LayerNorm
        module_output = DynamicTanh(
            module.normalized_shape, 
            not isinstance(module, LayerNorm2d), 
            alpha_init_value=0.5
        )
        
        # Copy affine parameters from original LayerNorm
        module_output.weight.data = module.weight.data.clone()
        if module.bias is not None:
            module_output.bias.data = module.bias.data.clone()

    # Recursively convert child modules
    for name, child in module.named_children():
        child_prefix = f"{prefix}{name}."
        module_output.add_module(
            name, 
            convert_ln_to_dyt(child, prefix=child_prefix)
        )
    
    del module
    return module_output
