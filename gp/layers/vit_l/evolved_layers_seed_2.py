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
    Equation: -0.176*x + clip(1.1*x)
    """
    def forward(self, x):
        x_norm = -0.176*x + clip(1.1*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks0Norm2(EvolvedLayer):
    """
    Original: blocks.0.norm2
    Equation: -0.18*x + clip(x)
    """
    def forward(self, x):
        x_norm = -0.18*x + clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks1Norm1(EvolvedLayer):
    """
    Original: blocks.1.norm1
    Equation: -0.191*x + 1.11*clip(x)
    """
    def forward(self, x):
        x_norm = -0.191*x + 1.11*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks1Norm2(EvolvedLayer):
    """
    Original: blocks.1.norm2
    Equation: -0.297*x + clip(1.36*x)
    """
    def forward(self, x):
        x_norm = -0.297*x + clip(1.36*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks2Norm1(EvolvedLayer):
    """
    Original: blocks.2.norm1
    Equation: -0.082*x + clip(x)
    """
    def forward(self, x):
        x_norm = -0.082*x + clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks2Norm2(EvolvedLayer):
    """
    Original: blocks.2.norm2
    Equation: -0.189*x + 1.27*clip(x)
    """
    def forward(self, x):
        x_norm = -0.189*x + 1.27*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks3Norm1(EvolvedLayer):
    """
    Original: blocks.3.norm1
    Equation: -0.543*clip(-2.01*x)
    """
    def forward(self, x):
        x_norm = -0.543*clip(-2.01*x)
        
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
    Equation: clip(neg(-1.1*x))
    """
    def forward(self, x):
        x_norm = clip(neg(-1.1*x))
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks4Norm2(EvolvedLayer):
    """
    Original: blocks.4.norm2
    Equation: -0.73*x + 1.8*clip(x)
    """
    def forward(self, x):
        x_norm = -0.73*x + 1.8*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks5Norm1(EvolvedLayer):
    """
    Original: blocks.5.norm1
    Equation: -0.098*x + clip(1.19*x)
    """
    def forward(self, x):
        x_norm = -0.098*x + clip(1.19*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks5Norm2(EvolvedLayer):
    """
    Original: blocks.5.norm2
    Equation: -0.721*x + 1.8*clip(x)
    """
    def forward(self, x):
        x_norm = -0.721*x + 1.8*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks6Norm1(EvolvedLayer):
    """
    Original: blocks.6.norm1
    Equation: -0.642*x + 1.74*clip(x)
    """
    def forward(self, x):
        x_norm = -0.642*x + 1.74*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks6Norm2(EvolvedLayer):
    """
    Original: blocks.6.norm2
    Equation: 1.86*clip(clip(-0.394*x) + clip(clip(x)))
    """
    def forward(self, x):
        x_norm = 1.86*clip(clip(-0.394*x) + clip(clip(x)))
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks7Norm1(EvolvedLayer):
    """
    Original: blocks.7.norm1
    Equation: clip(1.15*x)
    """
    def forward(self, x):
        x_norm = clip(1.15*x)
        
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
    Equation: -0.142*x + 1.26*clip(x)
    """
    def forward(self, x):
        x_norm = -0.142*x + 1.26*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks9Norm1(EvolvedLayer):
    """
    Original: blocks.9.norm1
    Equation: 0.583*clip(2*x)
    """
    def forward(self, x):
        x_norm = 0.583*clip(2*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks9Norm2(EvolvedLayer):
    """
    Original: blocks.9.norm2
    Equation: -0.221*x + 1.34*clip(x)
    """
    def forward(self, x):
        x_norm = -0.221*x + 1.34*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks10Norm1(EvolvedLayer):
    """
    Original: blocks.10.norm1
    Equation: clip(1.15*x)
    """
    def forward(self, x):
        x_norm = clip(1.15*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks10Norm2(EvolvedLayer):
    """
    Original: blocks.10.norm2
    Equation: -0.16*x + 1.29*clip(x)
    """
    def forward(self, x):
        x_norm = -0.16*x + 1.29*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks11Norm1(EvolvedLayer):
    """
    Original: blocks.11.norm1
    Equation: 0.545*clip(2.04*x)
    """
    def forward(self, x):
        x_norm = 0.545*clip(2.04*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks11Norm2(EvolvedLayer):
    """
    Original: blocks.11.norm2
    Equation: -0.122*x + 1.19*clip(x)
    """
    def forward(self, x):
        x_norm = -0.122*x + 1.19*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks12Norm1(EvolvedLayer):
    """
    Original: blocks.12.norm1
    Equation: -0.0377*x + clip(x)
    """
    def forward(self, x):
        x_norm = -0.0377*x + clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks12Norm2(EvolvedLayer):
    """
    Original: blocks.12.norm2
    Equation: -0.0514*x + clip(x)
    """
    def forward(self, x):
        x_norm = -0.0514*x + clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks13Norm1(EvolvedLayer):
    """
    Original: blocks.13.norm1
    Equation: -0.0194*x + clip(x)
    """
    def forward(self, x):
        x_norm = -0.0194*x + clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks13Norm2(EvolvedLayer):
    """
    Original: blocks.13.norm2
    Equation: -0.0301*x + clip(x)
    """
    def forward(self, x):
        x_norm = -0.0301*x + clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks14Norm1(EvolvedLayer):
    """
    Original: blocks.14.norm1
    Equation: 0.826*clip(x)
    """
    def forward(self, x):
        x_norm = 0.826*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks14Norm2(EvolvedLayer):
    """
    Original: blocks.14.norm2
    Equation: 0.796*clip(x)
    """
    def forward(self, x):
        x_norm = 0.796*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks15Norm1(EvolvedLayer):
    """
    Original: blocks.15.norm1
    Equation: clip(-0.141*x) + clip(x)
    """
    def forward(self, x):
        x_norm = clip(-0.141*x) + clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks15Norm2(EvolvedLayer):
    """
    Original: blocks.15.norm2
    Equation: 0.468*clip(1.75*x)
    """
    def forward(self, x):
        x_norm = 0.468*clip(1.75*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks16Norm1(EvolvedLayer):
    """
    Original: blocks.16.norm1
    Equation: 0.714*clip(x)
    """
    def forward(self, x):
        x_norm = 0.714*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks16Norm2(EvolvedLayer):
    """
    Original: blocks.16.norm2
    Equation: 0.456*clip(1.64*x)
    """
    def forward(self, x):
        x_norm = 0.456*clip(1.64*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks17Norm1(EvolvedLayer):
    """
    Original: blocks.17.norm1
    Equation: 0.672*clip(x)
    """
    def forward(self, x):
        x_norm = 0.672*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks17Norm2(EvolvedLayer):
    """
    Original: blocks.17.norm2
    Equation: 0.455*clip(1.55*x)
    """
    def forward(self, x):
        x_norm = 0.455*clip(1.55*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks18Norm1(EvolvedLayer):
    """
    Original: blocks.18.norm1
    Equation: 0.639*clip(x)
    """
    def forward(self, x):
        x_norm = 0.639*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks18Norm2(EvolvedLayer):
    """
    Original: blocks.18.norm2
    Equation: 0.61*clip(x)
    """
    def forward(self, x):
        x_norm = 0.61*clip(x)
        
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
    Equation: 0.536*clip(x)
    """
    def forward(self, x):
        x_norm = 0.536*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks20Norm2(EvolvedLayer):
    """
    Original: blocks.20.norm2
    Equation: 0.519*clip(x)
    """
    def forward(self, x):
        x_norm = 0.519*clip(x)
        
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
    Equation: 0.449*clip(x)
    """
    def forward(self, x):
        x_norm = 0.449*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks22Norm1(EvolvedLayer):
    """
    Original: blocks.22.norm1
    Equation: -0.396*neg(clip(x))
    """
    def forward(self, x):
        x_norm = -0.396*neg(clip(x))
        
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
    Equation: 0.369*clip(x)
    """
    def forward(self, x):
        x_norm = 0.369*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks23Norm2(EvolvedLayer):
    """
    Original: blocks.23.norm2
    Equation: 0.35*clip(x)
    """
    def forward(self, x):
        x_norm = 0.35*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Norm(EvolvedLayer):
    """
    Original: norm
    Equation: 0.359*clip(x)
    """
    def forward(self, x):
        x_norm = 0.359*clip(x)
        
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
