# Evolved GP expressions for ViT-B/16 - Seed 4
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
    Equation: -0.522*x + 2.1*clip(x)
    """
    def forward(self, x):
        x_norm = -0.522*x + 2.1*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks0Norm2(EvolvedLayer):
    """
    Original: blocks.0.norm2
    Equation: -0.696*clip(-2.32*x)
    """
    def forward(self, x):
        x_norm = -0.696*clip(-2.32*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks1Norm1(EvolvedLayer):
    """
    Original: blocks.1.norm1
    Equation: 0.667*clip(2.38*x)
    """
    def forward(self, x):
        x_norm = 0.667*clip(2.38*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks1Norm2(EvolvedLayer):
    """
    Original: blocks.1.norm2
    Equation: -0.284*x + clip(2.41*x) + 0.0921
    """
    def forward(self, x):
        x_norm = -0.284*x + clip(2.41*x) + 0.0921
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks2Norm1(EvolvedLayer):
    """
    Original: blocks.2.norm1
    Equation: -0.556*x + clip(2.7*x) + 0.115
    """
    def forward(self, x):
        x_norm = -0.556*x + clip(2.7*x) + 0.115
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks2Norm2(EvolvedLayer):
    """
    Original: blocks.2.norm2
    Equation: 2.49*tanh(x) + 0.129
    """
    def forward(self, x):
        x_norm = 2.49*torch.tanh(x) + 0.129
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks3Norm1(EvolvedLayer):
    """
    Original: blocks.3.norm1
    Equation: 2.63*tanh(x + 0.0649)
    """
    def forward(self, x):
        x_norm = 2.63*torch.tanh(x + 0.0649)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks3Norm2(EvolvedLayer):
    """
    Original: blocks.3.norm2
    Equation: -0.679*x + clip(2*x + 0.163) + tanh(x)
    """
    def forward(self, x):
        x_norm = -0.679*x + clip(2*x + 0.163) + torch.tanh(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks4Norm1(EvolvedLayer):
    """
    Original: blocks.4.norm1
    Equation: -0.75*x + clip(2.14*x) + tanh(x) + 0.165
    """
    def forward(self, x):
        x_norm = -0.75*x + clip(2.14*x) + torch.tanh(x) + 0.165
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks4Norm2(EvolvedLayer):
    """
    Original: blocks.4.norm2
    Equation: 2.64*tanh(clip(x)) + 0.208
    """
    def forward(self, x):
        x_norm = 2.64*torch.tanh(clip(x)) + 0.208
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks5Norm1(EvolvedLayer):
    """
    Original: blocks.5.norm1
    Equation: -0.513*x + clip(clip(2.86*x)) + 0.232
    """
    def forward(self, x):
        x_norm = -0.513*x + clip(clip(2.86*x)) + 0.232
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks5Norm2(EvolvedLayer):
    """
    Original: blocks.5.norm2
    Equation: -0.717*x + 2.82*clip(x) + 0.206
    """
    def forward(self, x):
        x_norm = -0.717*x + 2.82*clip(x) + 0.206
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks6Norm1(EvolvedLayer):
    """
    Original: blocks.6.norm1
    Equation: 2.57*tanh(clip(x)) + 0.239
    """
    def forward(self, x):
        x_norm = 2.57*torch.tanh(clip(x)) + 0.239
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks6Norm2(EvolvedLayer):
    """
    Original: blocks.6.norm2
    Equation: -0.19*x + clip(2.21*x) + 0.222
    """
    def forward(self, x):
        x_norm = -0.19*x + clip(2.21*x) + 0.222
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks7Norm1(EvolvedLayer):
    """
    Original: blocks.7.norm1
    Equation: 2.36*tanh(x) + 0.235
    """
    def forward(self, x):
        x_norm = 2.36*torch.tanh(x) + 0.235
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks7Norm2(EvolvedLayer):
    """
    Original: blocks.7.norm2
    Equation: -1.89*tanh(0.28*x - clip(x) + 0.28*tanh(tanh(x)) - 0.13) + tanh(clip(x))
    """
    def forward(self, x):
        x_norm = -1.89*torch.tanh(0.28*x - clip(x) + 0.28*torch.tanh(torch.tanh(x)) - 0.13) + torch.tanh(clip(x))
        
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
    Equation: -0.13*x + clip(x) + 0.735*tanh(x) + 0.224
    """
    def forward(self, x):
        x_norm = -0.13*x + clip(x) + 0.735*torch.tanh(x) + 0.224
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks9Norm1(EvolvedLayer):
    """
    Original: blocks.9.norm1
    Equation: 2.34*tanh(x - 0.464*tanh(x)) + 0.217
    """
    def forward(self, x):
        x_norm = 2.34*torch.tanh(x - 0.464*torch.tanh(x)) + 0.217
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks9Norm2(EvolvedLayer):
    """
    Original: blocks.9.norm2
    Equation: 0.486*clip(2.37*x) + 0.188
    """
    def forward(self, x):
        x_norm = 0.486*clip(2.37*x) + 0.188
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks10Norm1(EvolvedLayer):
    """
    Original: blocks.10.norm1
    Equation: 0.485*clip(clip(2*x)) + sigmoid(clip(sigmoid(-0.0469*x))) - 0.433
    """
    def forward(self, x):
        x_norm = 0.485*clip(clip(2*x)) + sigmoid(clip(sigmoid(-0.0469*x))) - 0.433
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks10Norm2(EvolvedLayer):
    """
    Original: blocks.10.norm2
    Equation: 0.908*clip(x + 0.167) + 0.908*clip(clip(-0.0376*x))
    """
    def forward(self, x):
        x_norm = 0.908*clip(x + 0.167) + 0.908*clip(clip(-0.0376*x))
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks11Norm1(EvolvedLayer):
    """
    Original: blocks.11.norm1
    Equation: clip(0.645*clip(-0.0443*x + clip(0.91*x) + 0.127))
    """
    def forward(self, x):
        x_norm = clip(0.645*clip(-0.0443*x + clip(0.91*x) + 0.127))
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks11Norm2(EvolvedLayer):
    """
    Original: blocks.11.norm2
    Equation: 0.618*clip(0.732*x - 1.93) + 1.25
    """
    def forward(self, x):
        x_norm = 0.618*clip(0.732*x - 1.93) + 1.25
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Norm(EvolvedLayer):
    """
    Original: norm
    Equation: 1.14 - 0.656*clip(1.62 - 0.621*x)
    """
    def forward(self, x):
        x_norm = 1.14 - 0.656*clip(1.62 - 0.621*x)
        
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
