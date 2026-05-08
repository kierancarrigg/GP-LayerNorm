# Evolved GP expressions for ViT-B/16 - Seed 2
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
    Equation: -0.526*x + 2.11*clip(x)
    """
    def forward(self, x):
        x_norm = -0.526*x + 2.11*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks0Norm2(EvolvedLayer):
    """
    Original: blocks.0.norm2
    Equation: 0.695*clip(2.32*x)
    """
    def forward(self, x):
        x_norm = 0.695*clip(2.32*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks1Norm1(EvolvedLayer):
    """
    Original: blocks.1.norm1
    Equation: 0.674*clip(2.36*x)
    """
    def forward(self, x):
        x_norm = 0.674*clip(2.36*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks1Norm2(EvolvedLayer):
    """
    Original: blocks.1.norm2
    Equation: 2.55*tanh(x + 0.0424)
    """
    def forward(self, x):
        x_norm = 2.55*torch.tanh(x + 0.0424)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks2Norm1(EvolvedLayer):
    """
    Original: blocks.2.norm1
    Equation: -0.556*x + clip(2.68*x + 0.13)
    """
    def forward(self, x):
        x_norm = -0.556*x + clip(2.68*x + 0.13)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks2Norm2(EvolvedLayer):
    """
    Original: blocks.2.norm2
    Equation: 2.5*tanh(x) + 0.13
    """
    def forward(self, x):
        x_norm = 2.5*torch.tanh(x) + 0.13
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks3Norm1(EvolvedLayer):
    """
    Original: blocks.3.norm1
    Equation: 2.62*tanh(x + 0.0656)
    """
    def forward(self, x):
        x_norm = 2.62*torch.tanh(x + 0.0656)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks3Norm2(EvolvedLayer):
    """
    Original: blocks.3.norm2
    Equation: -0.413*x + 3.72*tanh(x) - 0.413*tanh(2*x*(x + 1.48)) + 0.205
    """
    def forward(self, x):
        x_norm = -0.413*x + 3.72*torch.tanh(x) - 0.413*torch.tanh(2*x*(x + 1.48)) + 0.205
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks4Norm1(EvolvedLayer):
    """
    Original: blocks.4.norm1
    Equation: -0.747*x + clip(2.16*x) + tanh(x) + 0.167
    """
    def forward(self, x):
        x_norm = -0.747*x + clip(2.16*x) + torch.tanh(x) + 0.167
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks4Norm2(EvolvedLayer):
    """
    Original: blocks.4.norm2
    Equation: 2.65*tanh(clip(x)) + 0.212
    """
    def forward(self, x):
        x_norm = 2.65*torch.tanh(clip(x)) + 0.212
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks5Norm1(EvolvedLayer):
    """
    Original: blocks.5.norm1
    Equation: -0.494*x + clip(2.84*x) + 0.223
    """
    def forward(self, x):
        x_norm = -0.494*x + clip(2.84*x) + 0.223
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks5Norm2(EvolvedLayer):
    """
    Original: blocks.5.norm2
    Equation: -0.674*x + 2.78*clip(x) + 0.195
    """
    def forward(self, x):
        x_norm = -0.674*x + 2.78*clip(x) + 0.195
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks6Norm1(EvolvedLayer):
    """
    Original: blocks.6.norm1
    Equation: 2.58*tanh(clip(x)) + 0.24
    """
    def forward(self, x):
        x_norm = 2.58*torch.tanh(clip(x)) + 0.24
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks6Norm2(EvolvedLayer):
    """
    Original: blocks.6.norm2
    Equation: 2.43*tanh(x) + 0.244
    """
    def forward(self, x):
        x_norm = 2.43*torch.tanh(x) + 0.244
        
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
    Equation: 0.598*clip(2.92*x) + tanh((0.499 - 0.0953*x)*sigmoid(0.45*clip(x)))
    """
    def forward(self, x):
        x_norm = 0.598*clip(2.92*x) + torch.tanh((0.499 - 0.0953*x)*sigmoid(0.45*clip(x)))
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks8Norm1(EvolvedLayer):
    """
    Original: blocks.8.norm1
    Equation: 0.508*clip(3.22*x) + 0.214
    """
    def forward(self, x):
        x_norm = 0.508*clip(3.22*x) + 0.214
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks8Norm2(EvolvedLayer):
    """
    Original: blocks.8.norm2
    Equation: -0.113*x + 1.02*clip(clip(x) + tanh(0.79*clip(clip(x) + 0.107) + 0.19))
    """
    def forward(self, x):
        x_norm = -0.113*x + 1.02*clip(clip(x) + torch.tanh(0.79*clip(clip(x) + 0.107) + 0.19))
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks9Norm1(EvolvedLayer):
    """
    Original: blocks.9.norm1
    Equation: 0.492*clip(clip(2.39*x + 0.248*tanh(x)) + 0.398)
    """
    def forward(self, x):
        x_norm = 0.492*clip(clip(2.39*x + 0.248*torch.tanh(x)) + 0.398)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks9Norm2(EvolvedLayer):
    """
    Original: blocks.9.norm2
    Equation: 0.487*clip(x + clip(1.41*x)) + 0.193
    """
    def forward(self, x):
        x_norm = 0.487*clip(x + clip(1.41*x)) + 0.193
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks10Norm1(EvolvedLayer):
    """
    Original: blocks.10.norm1
    Equation: clip(x) + clip(0.213 - 0.0305*x) - 0.0381
    """
    def forward(self, x):
        x_norm = clip(x) + clip(0.213 - 0.0305*x) - 0.0381
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks10Norm2(EvolvedLayer):
    """
    Original: blocks.10.norm2
    Equation: clip(x) + clip(clip(-0.117*x) + 0.153)
    """
    def forward(self, x):
        x_norm = clip(x) + clip(clip(-0.117*x) + 0.153)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks11Norm1(EvolvedLayer):
    """
    Original: blocks.11.norm1
    Equation: -0.0285*x + 0.469*clip(x) + 0.469*clip(tanh(tanh(0.513*x))) + 0.103
    """
    def forward(self, x):
        x_norm = -0.0285*x + 0.469*clip(x) + 0.469*clip(torch.tanh(torch.tanh(0.513*x))) + 0.103
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks11Norm2(EvolvedLayer):
    """
    Original: blocks.11.norm2
    Equation: 0.918*clip(x)*sigmoid((x + clip(-0.92*x))*sigmoid(-0.0602*x))
    """
    def forward(self, x):
        x_norm = 0.918*clip(x)*sigmoid((x + clip(-0.92*x))*sigmoid(-0.0602*x))
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Norm(EvolvedLayer):
    """
    Original: norm
    Equation: clip(0.687*clip(0.596*x - 1.2) + 0.911)
    """
    def forward(self, x):
        x_norm = clip(0.687*clip(0.596*x - 1.2) + 0.911)
        
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
