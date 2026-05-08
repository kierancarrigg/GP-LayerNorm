# Evolved GP expressions for ViT-B/16 - Seed 5
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
    Equation: 2.51*clip(-0.299*x) + 2.51*clip(clip(0.927*x))
    """
    def forward(self, x):
        x_norm = 2.51*clip(-0.299*x) + 2.51*clip(clip(0.927*x))
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks0Norm2(EvolvedLayer):
    """
    Original: blocks.0.norm2
    Equation: 0.692*clip(2.33*x)
    """
    def forward(self, x):
        x_norm = 0.692*clip(2.33*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks1Norm1(EvolvedLayer):
    """
    Original: blocks.1.norm1
    Equation: -0.129*x + clip(1.73*x)
    """
    def forward(self, x):
        x_norm = -0.129*x + clip(1.73*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks1Norm2(EvolvedLayer):
    """
    Original: blocks.1.norm2
    Equation: 2.54*tanh(clip(x) + 0.0421)
    """
    def forward(self, x):
        x_norm = 2.54*torch.tanh(clip(x) + 0.0421)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks2Norm1(EvolvedLayer):
    """
    Original: blocks.2.norm1
    Equation: -0.555*x + clip(2.67*x) + 0.112
    """
    def forward(self, x):
        x_norm = -0.555*x + clip(2.67*x) + 0.112
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks2Norm2(EvolvedLayer):
    """
    Original: blocks.2.norm2
    Equation: 2.5*tanh(x + 0.0581)
    """
    def forward(self, x):
        x_norm = 2.5*torch.tanh(x + 0.0581)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks3Norm1(EvolvedLayer):
    """
    Original: blocks.3.norm1
    Equation: 2.62*tanh(x + 0.0674)
    """
    def forward(self, x):
        x_norm = 2.62*torch.tanh(x + 0.0674)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks3Norm2(EvolvedLayer):
    """
    Original: blocks.3.norm2
    Equation: -0.694*x + clip(2.01*x + 0.172) + tanh(x)
    """
    def forward(self, x):
        x_norm = -0.694*x + clip(2.01*x + 0.172) + torch.tanh(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks4Norm1(EvolvedLayer):
    """
    Original: blocks.4.norm1
    Equation: -0.675*x + clip(clip(clip(2.37*x))) + sigmoid(2.77*x) - 0.341
    """
    def forward(self, x):
        x_norm = -0.675*x + clip(clip(clip(2.37*x))) + sigmoid(2.77*x) - 0.341
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks4Norm2(EvolvedLayer):
    """
    Original: blocks.4.norm2
    Equation: -0.103*x + 2.66*tanh(x) + 0.218
    """
    def forward(self, x):
        x_norm = -0.103*x + 2.66*torch.tanh(x) + 0.218
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks5Norm1(EvolvedLayer):
    """
    Original: blocks.5.norm1
    Equation: -0.636*x + clip(2.15*x + 0.229) + tanh(x)
    """
    def forward(self, x):
        x_norm = -0.636*x + clip(2.15*x + 0.229) + torch.tanh(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks5Norm2(EvolvedLayer):
    """
    Original: blocks.5.norm2
    Equation: -0.758*x + 3.02*clip(x) + 0.224
    """
    def forward(self, x):
        x_norm = -0.758*x + 3.02*clip(x) + 0.224
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks6Norm1(EvolvedLayer):
    """
    Original: blocks.6.norm1
    Equation: 2.58*tanh(clip(x)) + 0.239
    """
    def forward(self, x):
        x_norm = 2.58*torch.tanh(clip(x)) + 0.239
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks6Norm2(EvolvedLayer):
    """
    Original: blocks.6.norm2
    Equation: -0.202*x + clip(2.19*x) + 0.234
    """
    def forward(self, x):
        x_norm = -0.202*x + clip(2.19*x) + 0.234
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks7Norm1(EvolvedLayer):
    """
    Original: blocks.7.norm1
    Equation: 2.36*tanh(clip(x)) + 0.241
    """
    def forward(self, x):
        x_norm = 2.36*torch.tanh(clip(x)) + 0.241
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks7Norm2(EvolvedLayer):
    """
    Original: blocks.7.norm2
    Equation: clip(2*x) - 0.953*clip(0.262*x - 0.216)
    """
    def forward(self, x):
        x_norm = clip(2*x) - 0.953*clip(0.262*x - 0.216)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks8Norm1(EvolvedLayer):
    """
    Original: blocks.8.norm1
    Equation: tanh(x) + tanh(x + 0.269)
    """
    def forward(self, x):
        x_norm = torch.tanh(x) + torch.tanh(x + 0.269)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks8Norm2(EvolvedLayer):
    """
    Original: blocks.8.norm2
    Equation: -0.104*x + clip(1.58*x + 0.225)
    """
    def forward(self, x):
        x_norm = -0.104*x + clip(1.58*x + 0.225)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks9Norm1(EvolvedLayer):
    """
    Original: blocks.9.norm1
    Equation: -0.0312*x + clip(1.28*x) + 0.186
    """
    def forward(self, x):
        x_norm = -0.0312*x + clip(1.28*x) + 0.186
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks9Norm2(EvolvedLayer):
    """
    Original: blocks.9.norm2
    Equation: 1.59*tanh(clip(x))
    """
    def forward(self, x):
        x_norm = 1.59*torch.tanh(clip(x))
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks10Norm1(EvolvedLayer):
    """
    Original: blocks.10.norm1
    Equation: -0.0313*x + clip(clip(clip(x))) + 0.195
    """
    def forward(self, x):
        x_norm = -0.0313*x + clip(clip(clip(x))) + 0.195
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks10Norm2(EvolvedLayer):
    """
    Original: blocks.10.norm2
    Equation: -0.0355*x + clip(0.911*x + 0.168)
    """
    def forward(self, x):
        x_norm = -0.0355*x + clip(0.911*x + 0.168)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks11Norm1(EvolvedLayer):
    """
    Original: blocks.11.norm1
    Equation: -0.0477*x + clip(0.603*x + 0.081)
    """
    def forward(self, x):
        x_norm = -0.0477*x + clip(0.603*x + 0.081)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks11Norm2(EvolvedLayer):
    """
    Original: blocks.11.norm2
    Equation: (sigmoid(clip(0.132*x)*sigmoid(-0.0179*x)) - 0.039)*clip(x)
    """
    def forward(self, x):
        x_norm = (sigmoid(clip(0.132*x)*sigmoid(-0.0179*x)) - 0.039)*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Norm(EvolvedLayer):
    """
    Original: norm
    Equation: x*(-0.0134*x*clip(0.0539*clip(x)) + 0.48)
    """
    def forward(self, x):
        x_norm = x*(-0.0134*x*clip(0.0539*clip(x)) + 0.48)
        
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
