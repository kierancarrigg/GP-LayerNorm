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

def neg(a):
    if isinstance(a, torch.Tensor):
        return torch.neg(a)
    return -a

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
    Equation: -0.207*x + 1.11*clip(x)
    """
    def forward(self, x):
        x_norm = -0.207*x + 1.11*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks0Norm2(EvolvedLayer):
    """
    Original: blocks.0.norm2
    Equation: -0.358*x + 1.28*clip(1.07*x)
    """
    def forward(self, x):
        x_norm = -0.358*x + 1.28*clip(1.07*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks1Norm1(EvolvedLayer):
    """
    Original: blocks.1.norm1
    Equation: -0.194*x + 1.11*clip(x)
    """
    def forward(self, x):
        x_norm = -0.194*x + 1.11*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks1Norm2(EvolvedLayer):
    """
    Original: blocks.1.norm2
    Equation: 0.552*clip(1.93*x)
    """
    def forward(self, x):
        x_norm = 0.552*clip(1.93*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks2Norm1(EvolvedLayer):
    """
    Original: blocks.2.norm1
    Equation: clip(1.06*x)
    """
    def forward(self, x):
        x_norm = clip(1.06*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks2Norm2(EvolvedLayer):
    """
    Original: blocks.2.norm2
    Equation: clip(1.08*x)
    """
    def forward(self, x):
        x_norm = clip(1.08*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks3Norm1(EvolvedLayer):
    """
    Original: blocks.3.norm1
    Equation: -0.486*x + 1.57*clip(x)
    """
    def forward(self, x):
        x_norm = -0.486*x + 1.57*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks3Norm2(EvolvedLayer):
    """
    Original: blocks.3.norm2
    Equation: -0.122*x + clip(1.22*x)
    """
    def forward(self, x):
        x_norm = -0.122*x + clip(1.22*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks4Norm1(EvolvedLayer):
    """
    Original: blocks.4.norm1
    Equation: -0.0846*x + clip(x)
    """
    def forward(self, x):
        x_norm = -0.0846*x + clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks4Norm2(EvolvedLayer):
    """
    Original: blocks.4.norm2
    Equation: -0.736*x + 1.81*clip(x)
    """
    def forward(self, x):
        x_norm = -0.736*x + 1.81*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks5Norm1(EvolvedLayer):
    """
    Original: blocks.5.norm1
    Equation: -0.0865*x + clip(x)
    """
    def forward(self, x):
        x_norm = -0.0865*x + clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks5Norm2(EvolvedLayer):
    """
    Original: blocks.5.norm2
    Equation: 0.495*clip(2.2*x)
    """
    def forward(self, x):
        x_norm = 0.495*clip(2.2*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks6Norm1(EvolvedLayer):
    """
    Original: blocks.6.norm1
    Equation: -0.639*x + 1.74*clip(x)
    """
    def forward(self, x):
        x_norm = -0.639*x + 1.74*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks6Norm2(EvolvedLayer):
    """
    Original: blocks.6.norm2
    Equation: 0.502*clip(2.27*x)
    """
    def forward(self, x):
        x_norm = 0.502*clip(2.27*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks7Norm1(EvolvedLayer):
    """
    Original: blocks.7.norm1
    Equation: clip(neg(-1.15*x))
    """
    def forward(self, x):
        x_norm = clip(neg(-1.15*x))
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks7Norm2(EvolvedLayer):
    """
    Original: blocks.7.norm2
    Equation: clip(1.12*x)
    """
    def forward(self, x):
        x_norm = clip(1.12*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks8Norm1(EvolvedLayer):
    """
    Original: blocks.8.norm1
    Equation: clip(1.18*x)
    """
    def forward(self, x):
        x_norm = clip(1.18*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks8Norm2(EvolvedLayer):
    """
    Original: blocks.8.norm2
    Equation: -0.145*x + 1.28*clip(x)
    """
    def forward(self, x):
        x_norm = -0.145*x + 1.28*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks9Norm1(EvolvedLayer):
    """
    Original: blocks.9.norm1
    Equation: 0.564*clip(2*x)
    """
    def forward(self, x):
        x_norm = 0.564*clip(2*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks9Norm2(EvolvedLayer):
    """
    Original: blocks.9.norm2
    Equation: clip(1.12*x)
    """
    def forward(self, x):
        x_norm = clip(1.12*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks10Norm1(EvolvedLayer):
    """
    Original: blocks.10.norm1
    Equation: clip(1.16*x)
    """
    def forward(self, x):
        x_norm = clip(1.16*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks10Norm2(EvolvedLayer):
    """
    Original: blocks.10.norm2
    Equation: -0.157*x + 1.28*clip(x)
    """
    def forward(self, x):
        x_norm = -0.157*x + 1.28*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks11Norm1(EvolvedLayer):
    """
    Original: blocks.11.norm1
    Equation: clip(1.13*x)
    """
    def forward(self, x):
        x_norm = clip(1.13*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks11Norm2(EvolvedLayer):
    """
    Original: blocks.11.norm2
    Equation: -0.093*x + clip(x)
    """
    def forward(self, x):
        x_norm = -0.093*x + clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks12Norm1(EvolvedLayer):
    """
    Original: blocks.12.norm1
    Equation: 0.925*clip(x)
    """
    def forward(self, x):
        x_norm = 0.925*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks12Norm2(EvolvedLayer):
    """
    Original: blocks.12.norm2
    Equation: -0.0512*x + clip(x)
    """
    def forward(self, x):
        x_norm = -0.0512*x + clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks13Norm1(EvolvedLayer):
    """
    Original: blocks.13.norm1
    Equation: 0.872*clip(x)
    """
    def forward(self, x):
        x_norm = 0.872*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks13Norm2(EvolvedLayer):
    """
    Original: blocks.13.norm2
    Equation: -0.03*x + clip(x)
    """
    def forward(self, x):
        x_norm = -0.03*x + clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks14Norm1(EvolvedLayer):
    """
    Original: blocks.14.norm1
    Equation: 0.493*clip(1.88*x)
    """
    def forward(self, x):
        x_norm = 0.493*clip(1.88*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks14Norm2(EvolvedLayer):
    """
    Original: blocks.14.norm2
    Equation: -0.795*clip(neg(x))
    """
    def forward(self, x):
        x_norm = -0.795*clip(neg(x))
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks15Norm1(EvolvedLayer):
    """
    Original: blocks.15.norm1
    Equation: 0.451*clip(1.93*x)
    """
    def forward(self, x):
        x_norm = 0.451*clip(1.93*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks15Norm2(EvolvedLayer):
    """
    Original: blocks.15.norm2
    Equation: 0.748*clip(neg(neg(x)))
    """
    def forward(self, x):
        x_norm = 0.748*clip(neg(neg(x)))
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks16Norm1(EvolvedLayer):
    """
    Original: blocks.16.norm1
    Equation: 0.718*clip(x)
    """
    def forward(self, x):
        x_norm = 0.718*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks16Norm2(EvolvedLayer):
    """
    Original: blocks.16.norm2
    Equation: 0.691*clip(x)
    """
    def forward(self, x):
        x_norm = 0.691*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks17Norm1(EvolvedLayer):
    """
    Original: blocks.17.norm1
    Equation: 0.67*clip(x)
    """
    def forward(self, x):
        x_norm = 0.67*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks17Norm2(EvolvedLayer):
    """
    Original: blocks.17.norm2
    Equation: 0.661*clip(x)
    """
    def forward(self, x):
        x_norm = 0.661*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks18Norm1(EvolvedLayer):
    """
    Original: blocks.18.norm1
    Equation: 0.633*clip(x)
    """
    def forward(self, x):
        x_norm = 0.633*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks18Norm2(EvolvedLayer):
    """
    Original: blocks.18.norm2
    Equation: 0.609*clip(x)
    """
    def forward(self, x):
        x_norm = 0.609*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks19Norm1(EvolvedLayer):
    """
    Original: blocks.19.norm1
    Equation: 0.583*clip(x)
    """
    def forward(self, x):
        x_norm = 0.583*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks19Norm2(EvolvedLayer):
    """
    Original: blocks.19.norm2
    Equation: 0.574*clip(x)
    """
    def forward(self, x):
        x_norm = 0.574*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks20Norm1(EvolvedLayer):
    """
    Original: blocks.20.norm1
    Equation: 0.534*clip(x)
    """
    def forward(self, x):
        x_norm = 0.534*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks20Norm2(EvolvedLayer):
    """
    Original: blocks.20.norm2
    Equation: 0.52*clip(x)
    """
    def forward(self, x):
        x_norm = 0.52*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks21Norm1(EvolvedLayer):
    """
    Original: blocks.21.norm1
    Equation: 0.46*clip(x)
    """
    def forward(self, x):
        x_norm = 0.46*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks21Norm2(EvolvedLayer):
    """
    Original: blocks.21.norm2
    Equation: 0.442*clip(x)
    """
    def forward(self, x):
        x_norm = 0.442*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks22Norm1(EvolvedLayer):
    """
    Original: blocks.22.norm1
    Equation: 0.398*clip(x)
    """
    def forward(self, x):
        x_norm = 0.398*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks22Norm2(EvolvedLayer):
    """
    Original: blocks.22.norm2
    Equation: 0.392*clip(x)
    """
    def forward(self, x):
        x_norm = 0.392*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks23Norm1(EvolvedLayer):
    """
    Original: blocks.23.norm1
    Equation: 0.375*clip(x)
    """
    def forward(self, x):
        x_norm = 0.375*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks23Norm2(EvolvedLayer):
    """
    Original: blocks.23.norm2
    Equation: 0.352*clip(x)
    """
    def forward(self, x):
        x_norm = 0.352*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Norm(EvolvedLayer):
    """
    Original: norm
    Equation: 0.357*clip(x)
    """
    def forward(self, x):
        x_norm = 0.357*clip(x)
        
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
