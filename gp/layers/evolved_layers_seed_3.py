# Evolved GP expressions for ViT-B/16 - Seed 3
# These layers replace LayerNorm in all 24 normalization positions of ViT-B/16.
# Use apply_evolution() to inject these layers into a timm ViT-B model.

import torch
import torch.nn as nn
import math

# --- 1. Robust Operator Definitions ---
# These match the logic used during GP evolution (JAX -> PyTorch)
# We use isinstance checks to safely handle both Tensors and raw Python floats
# without causing CUDA/CPU device mismatch errors.


def clip(a):
    if isinstance(a, torch.Tensor):
        return torch.clamp(a, -5.0, 5.0)
    return max(-5.0, min(a, 5.0))

    
def sigmoid(a):
    if isinstance(a, torch.Tensor):
        return torch.sigmoid(a)
    return 1.0 / (1.0 + math.exp(-a))
    
# --- 2. The Base Replacement Class ---
class EvolvedLayer(nn.Module):
    def __init__(self, original_ln):
        super().__init__()
        # We preserve the original affine weights (gamma/beta) if they exist
        if original_ln.elementwise_affine:
            self.weight = nn.Parameter(original_ln.weight.clone())
            self.bias = nn.Parameter(original_ln.bias.clone())
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward")

# --- 3. Evolved Layer Implementations ---

class Blocks0Norm1(EvolvedLayer):
    """
    Original: blocks.0.norm1
    Equation: -0.517*x + 2.09*clip(x)
    """
    def forward(self, x):
        x_norm = -0.517*x + 2.09*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks0Norm2(EvolvedLayer):
    """
    Original: blocks.0.norm2
    Equation: 0.687*clip(2.35*x)
    """
    def forward(self, x):
        x_norm = 0.687*clip(2.35*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks1Norm1(EvolvedLayer):
    """
    Original: blocks.1.norm1
    Equation: 0.672*clip(2.37*x)
    """
    def forward(self, x):
        x_norm = 0.672*clip(2.37*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks1Norm2(EvolvedLayer):
    """
    Original: blocks.1.norm2
    Equation: 2.54*tanh(clip(x) + 0.0437)
    """
    def forward(self, x):
        x_norm = 2.54*torch.tanh(clip(x) + 0.0437)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks2Norm1(EvolvedLayer):
    """
    Original: blocks.2.norm1
    Equation: -0.555*x + clip(clip(2.7*x)) + 0.116
    """
    def forward(self, x):
        x_norm = -0.555*x + clip(clip(2.7*x)) + 0.116
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks2Norm2(EvolvedLayer):
    """
    Original: blocks.2.norm2
    Equation: 2.5*tanh(clip(x)) + 0.136
    """
    def forward(self, x):
        x_norm = 2.5*torch.tanh(clip(x)) + 0.136
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks3Norm1(EvolvedLayer):
    """
    Original: blocks.3.norm1
    Equation: 4.59*tanh((x + 0.069)*sigmoid((-0.92*x + clip(x))*(x + 0.0536)))
    """
    def forward(self, x):
        x_norm = 4.59*torch.tanh((x + 0.069)*sigmoid((-0.92*x + clip(x))*(x + 0.0536)))
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks3Norm2(EvolvedLayer):
    """
    Original: blocks.3.norm2
    Equation: -1.78*x + 2.67*clip(x) + 2.67*tanh(0.574*tanh(x)) + 0.151
    """
    def forward(self, x):
        x_norm = -1.78*x + 2.67*clip(x) + 2.67*torch.tanh(0.574*torch.tanh(x)) + 0.151
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks4Norm1(EvolvedLayer):
    """
    Original: blocks.4.norm1
    Equation: -0.762*x + clip(2*x) + 1.21*tanh(x) + 0.177
    """
    def forward(self, x):
        x_norm = -0.762*x + clip(2*x) + 1.21*torch.tanh(x) + 0.177
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks4Norm2(EvolvedLayer):
    """
    Original: blocks.4.norm2
    Equation: -0.107*x + 2.88*tanh(x) + 0.199
    """
    def forward(self, x):
        x_norm = -0.107*x + 2.88*torch.tanh(x) + 0.199
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks5Norm1(EvolvedLayer):
    """
    Original: blocks.5.norm1
    Equation: 1.57*clip(-0.516*x + clip(2*x) + 0.144)
    """
    def forward(self, x):
        x_norm = 1.57*clip(-0.516*x + clip(2*x) + 0.144)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks5Norm2(EvolvedLayer):
    """
    Original: blocks.5.norm2
    Equation: -0.649*x + 2.74*clip(x) + 0.224
    """
    def forward(self, x):
        x_norm = -0.649*x + 2.74*clip(x) + 0.224
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks6Norm1(EvolvedLayer):
    """
    Original: blocks.6.norm1
    Equation: 2.57*tanh(clip(x)) + 0.241
    """
    def forward(self, x):
        x_norm = 2.57*torch.tanh(clip(x)) + 0.241
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks6Norm2(EvolvedLayer):
    """
    Original: blocks.6.norm2
    Equation: 2.43*tanh(x) + 0.248
    """
    def forward(self, x):
        x_norm = 2.43*torch.tanh(x) + 0.248
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks7Norm1(EvolvedLayer):
    """
    Original: blocks.7.norm1
    Equation: 2.36*tanh(clip(x)) + 0.238
    """
    def forward(self, x):
        x_norm = 2.36*torch.tanh(clip(x)) + 0.238
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks7Norm2(EvolvedLayer):
    """
    Original: blocks.7.norm2
    Equation: -0.0474*x + 0.725*clip(x) + clip(x + 0.174) + 0.000736
    """
    def forward(self, x):
        x_norm = -0.0474*x + 0.725*clip(x) + clip(x + 0.174) + 0.000736
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks8Norm1(EvolvedLayer):
    """
    Original: blocks.8.norm1
    Equation: 0.519*clip(3.05*x + 0.394)
    """
    def forward(self, x):
        x_norm = 0.519*clip(3.05*x + 0.394)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks8Norm2(EvolvedLayer):
    """
    Original: blocks.8.norm2
    Equation: -0.174*x + 1.61*clip(x) + 0.202
    """
    def forward(self, x):
        x_norm = -0.174*x + 1.61*clip(x) + 0.202
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks9Norm1(EvolvedLayer):
    """
    Original: blocks.9.norm1
    Equation: -0.0192*x + 0.473*clip(2*x + 0.355) + 0.473*tanh(x)
    """
    def forward(self, x):
        x_norm = -0.0192*x + 0.473*clip(2*x + 0.355) + 0.473*torch.tanh(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks9Norm2(EvolvedLayer):
    """
    Original: blocks.9.norm2
    Equation: 0.563*clip(clip(0.859*clip(2.42*x))) + 0.185
    """
    def forward(self, x):
        x_norm = 0.563*clip(clip(0.859*clip(2.42*x))) + 0.185
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks10Norm1(EvolvedLayer):
    """
    Original: blocks.10.norm1
    Equation: -0.0301*x + clip(x) + 0.17
    """
    def forward(self, x):
        x_norm = -0.0301*x + clip(x) + 0.17
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks10Norm2(EvolvedLayer):
    """
    Original: blocks.10.norm2
    Equation: clip(-0.12*x) + clip(x) + 0.15
    """
    def forward(self, x):
        x_norm = clip(-0.12*x) + clip(x) + 0.15
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks11Norm1(EvolvedLayer):
    """
    Original: blocks.11.norm1
    Equation: -0.055*x + clip(0.474*x + 0.318*tanh(x + 0.11)) + 0.0894
    """
    def forward(self, x):
        x_norm = -0.055*x + clip(0.474*x + 0.318*torch.tanh(x + 0.11)) + 0.0894
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks11Norm2(EvolvedLayer):
    """
    Original: blocks.11.norm2
    Equation: 0.552*clip(0.84*x - 0.901) + 0.545
    """
    def forward(self, x):
        x_norm = 0.552*clip(0.84*x - 0.901) + 0.545
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Norm(EvolvedLayer):
    """
    Original: norm
    Equation: 0.961*x*sigmoid(-0.000302*x**2 + 0.0052*x)
    """
    def forward(self, x):
        x_norm = 0.961*x*sigmoid(-0.000302*x**2 + 0.0052*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

# --- 4. Injection Helper ---
def apply_evolution(model, verbose=True):
    """
    Automatically swaps all LayerNorms in the model with their evolved counterparts.
    """
    import sys
    current_module = sys.modules[__name__]
    replaced_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            # Construct the expected class name: blocks.0.norm1 -> Blocks0Norm1
            clean_name = name.replace('.', '_').title().replace('_', '')
            
            if hasattr(current_module, clean_name):
                EvolvedClass = getattr(current_module, clean_name)
                new_layer = EvolvedClass(module)
                
                # Swap in the model
                parts = name.rsplit('.', 1)
                if len(parts) > 1:
                    parent_name, child_name = parts
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                    child_name = name
                    
                setattr(parent, child_name, new_layer)
                if verbose: print(f"Swapped {name} -> {clean_name}")
                replaced_count += 1
                
    print(f"Total layers replaced: {replaced_count}")
    return model
