# Evolved GP expressions for ViT-B/16 - Seed 1
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
    Equation: -0.522*x + 2.11*clip(x)
    """
    def forward(self, x):
        x_norm = -0.522*x + 2.11*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks0Norm2(EvolvedLayer):
    """
    Original: blocks.0.norm2
    Equation: 0.698*clip(2.31*x)
    """
    def forward(self, x):
        x_norm = 0.698*clip(2.31*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks1Norm1(EvolvedLayer):
    """
    Original: blocks.1.norm1
    Equation: 0.668*clip(2.38*x)
    """
    def forward(self, x):
        x_norm = 0.668*clip(2.38*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks1Norm2(EvolvedLayer):
    """
    Original: blocks.1.norm2
    Equation: 0.576*clip(3.64*x)
    """
    def forward(self, x):
        x_norm = 0.576*clip(3.64*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks2Norm1(EvolvedLayer):
    """
    Original: blocks.2.norm1
    Equation: -0.84*x + 1.31*clip(2*x) + 0.548*tanh(tanh(tanh(x)) + 0.227)
    """
    def forward(self, x):
        x_norm = -0.84*x + 1.31*clip(2*x) + 0.548*torch.tanh(torch.tanh(torch.tanh(x)) + 0.227)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks2Norm2(EvolvedLayer):
    """
    Original: blocks.2.norm2
    Equation: 2.5*tanh(clip(x)) + 0.135
    """
    def forward(self, x):
        x_norm = 2.5*torch.tanh(clip(x)) + 0.135
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks3Norm1(EvolvedLayer):
    """
    Original: blocks.3.norm1
    Equation: -0.141*x + 2.9*tanh(x) + 0.158
    """
    def forward(self, x):
        x_norm = -0.141*x + 2.9*torch.tanh(x) + 0.158
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks3Norm2(EvolvedLayer):
    """
    Original: blocks.3.norm2
    Equation: -0.687*x + clip(1.99*x) + clip(clip(tanh(x)) + 0.154)
    """
    def forward(self, x):
        x_norm = -0.687*x + clip(1.99*x) + clip(clip(torch.tanh(x)) + 0.154)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks4Norm1(EvolvedLayer):
    """
    Original: blocks.4.norm1
    Equation: (x + 0.0719)*(-0.307*x*tanh(2.99*x) + 2.46)
    """
    def forward(self, x):
        x_norm = (x + 0.0719)*(-0.307*x*torch.tanh(2.99*x) + 2.46)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks4Norm2(EvolvedLayer):
    """
    Original: blocks.4.norm2
    Equation: 2.66*tanh(clip(x)) + 0.208
    """
    def forward(self, x):
        x_norm = 2.66*torch.tanh(clip(x)) + 0.208
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks5Norm1(EvolvedLayer):
    """
    Original: blocks.5.norm1
    Equation: -0.488*x + clip(2.85*x) + 0.217
    """
    def forward(self, x):
        x_norm = -0.488*x + clip(2.85*x) + 0.217
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks5Norm2(EvolvedLayer):
    """
    Original: blocks.5.norm2
    Equation: -0.247*x + clip(2.44*x) + 0.233
    """
    def forward(self, x):
        x_norm = -0.247*x + clip(2.44*x) + 0.233
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks6Norm1(EvolvedLayer):
    """
    Original: blocks.6.norm1
    Equation: 2.58*tanh(x) + 0.24
    """
    def forward(self, x):
        x_norm = 2.58*torch.tanh(x) + 0.24
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks6Norm2(EvolvedLayer):
    """
    Original: blocks.6.norm2
    Equation: -0.192*x + clip(2.17*x) + 0.222
    """
    def forward(self, x):
        x_norm = -0.192*x + clip(2.17*x) + 0.222
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks7Norm1(EvolvedLayer):
    """
    Original: blocks.7.norm1
    Equation: 2.35*tanh(clip(x)) + 0.239
    """
    def forward(self, x):
        x_norm = 2.35*torch.tanh(clip(x)) + 0.239
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks7Norm2(EvolvedLayer):
    """
    Original: blocks.7.norm2
    Equation: 0.862*clip(x) + 0.862*clip(-0.077*x + clip(x)) + 0.172
    """
    def forward(self, x):
        x_norm = 0.862*clip(x) + 0.862*clip(-0.077*x + clip(x)) + 0.172
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks8Norm1(EvolvedLayer):
    """
    Original: blocks.8.norm1
    Equation: tanh(x) + tanh(x + 0.27)
    """
    def forward(self, x):
        x_norm = torch.tanh(x) + torch.tanh(x + 0.27)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks8Norm2(EvolvedLayer):
    """
    Original: blocks.8.norm2
    Equation: clip(x) + clip(-0.555*x + clip(x)) + 0.225
    """
    def forward(self, x):
        x_norm = clip(x) + clip(-0.555*x + clip(x)) + 0.225
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks9Norm1(EvolvedLayer):
    """
    Original: blocks.9.norm1
    Equation: 0.541*clip(clip(2.35*x)) + 0.174
    """
    def forward(self, x):
        x_norm = 0.541*clip(clip(2.35*x)) + 0.174
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks9Norm2(EvolvedLayer):
    """
    Original: blocks.9.norm2
    Equation: clip(0.471*clip(2.58*x)) + 0.212
    """
    def forward(self, x):
        x_norm = clip(0.471*clip(2.58*x)) + 0.212
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks10Norm1(EvolvedLayer):
    """
    Original: blocks.10.norm1
    Equation: -0.0307*x + clip(x) + 0.178
    """
    def forward(self, x):
        x_norm = -0.0307*x + clip(x) + 0.178
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks10Norm2(EvolvedLayer):
    """
    Original: blocks.10.norm2
    Equation: clip(-0.106*x) + clip(x) + 0.154
    """
    def forward(self, x):
        x_norm = clip(-0.106*x) + clip(x) + 0.154
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks11Norm1(EvolvedLayer):
    """
    Original: blocks.11.norm1
    Equation: -0.0337*x + 0.597*clip(x) + 0.0932
    """
    def forward(self, x):
        x_norm = -0.0337*x + 0.597*clip(x) + 0.0932
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks11Norm2(EvolvedLayer):
    """
    Original: blocks.11.norm2
    Equation: clip(x*tanh(sigmoid(-0.0401*x*tanh(x))))
    """
    def forward(self, x):
        x_norm = clip(x*torch.tanh(sigmoid(-0.0401*x*torch.tanh(x))))
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Norm(EvolvedLayer):
    """
    Original: norm
    Equation: x*tanh(sigmoid(clip(-0.000308*x**2)))
    """
    def forward(self, x):
        x_norm = x*torch.tanh(sigmoid(clip(-0.000308*x**2)))
        
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
