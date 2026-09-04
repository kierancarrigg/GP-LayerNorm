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
    Equation: -0.2*x + 1.11*clip(x)
    """
    def forward(self, x):
        x_norm = -0.2*x + 1.11*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks0Norm2(EvolvedLayer):
    """
    Original: blocks.0.norm2
    Equation: -0.383*x + 1.4*clip(x) - 0.0197
    """
    def forward(self, x):
        x_norm = -0.383*x + 1.4*clip(x) - 0.0197
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks1Norm1(EvolvedLayer):
    """
    Original: blocks.1.norm1
    Equation: -0.153*x + clip(x)
    """
    def forward(self, x):
        x_norm = -0.153*x + clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks1Norm2(EvolvedLayer):
    """
    Original: blocks.1.norm2
    Equation: -0.393*x + clip(x) + 0.284*clip(1.63*x)
    """
    def forward(self, x):
        x_norm = -0.393*x + clip(x) + 0.284*clip(1.63*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks2Norm1(EvolvedLayer):
    """
    Original: blocks.2.norm1
    Equation: -0.0825*x + clip(x)
    """
    def forward(self, x):
        x_norm = -0.0825*x + clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks2Norm2(EvolvedLayer):
    """
    Original: blocks.2.norm2
    Equation: -0.187*x + 1.26*clip(x)
    """
    def forward(self, x):
        x_norm = -0.187*x + 1.26*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks3Norm1(EvolvedLayer):
    """
    Original: blocks.3.norm1
    Equation: -0.304*x + clip(1.4*x)
    """
    def forward(self, x):
        x_norm = -0.304*x + clip(1.4*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks3Norm2(EvolvedLayer):
    """
    Original: blocks.3.norm2
    Equation: -0.115*x + clip(x)
    """
    def forward(self, x):
        x_norm = -0.115*x + clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks4Norm1(EvolvedLayer):
    """
    Original: blocks.4.norm1
    Equation: clip(1.09*x)
    """
    def forward(self, x):
        x_norm = clip(1.09*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks4Norm2(EvolvedLayer):
    """
    Original: blocks.4.norm2
    Equation: 0.924*clip(x)
    """
    def forward(self, x):
        x_norm = 0.924*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks5Norm1(EvolvedLayer):
    """
    Original: blocks.5.norm1
    Equation: clip(1.09*x)
    """
    def forward(self, x):
        x_norm = clip(1.09*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks5Norm2(EvolvedLayer):
    """
    Original: blocks.5.norm2
    Equation: -0.394*x + clip(1.48*x)
    """
    def forward(self, x):
        x_norm = -0.394*x + clip(1.48*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks6Norm1(EvolvedLayer):
    """
    Original: blocks.6.norm1
    Equation: -0.637*x + 1.74*clip(x)
    """
    def forward(self, x):
        x_norm = -0.637*x + 1.74*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks6Norm2(EvolvedLayer):
    """
    Original: blocks.6.norm2
    Equation: -0.491*x + 1.62*clip(x)
    """
    def forward(self, x):
        x_norm = -0.491*x + 1.62*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks7Norm1(EvolvedLayer):
    """
    Original: blocks.7.norm1
    Equation: clip(1.14*x)
    """
    def forward(self, x):
        x_norm = clip(1.14*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks7Norm2(EvolvedLayer):
    """
    Original: blocks.7.norm2
    Equation: -0.0887*x + clip(x)
    """
    def forward(self, x):
        x_norm = -0.0887*x + clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks8Norm1(EvolvedLayer):
    """
    Original: blocks.8.norm1
    Equation: clip(1.17*x)
    """
    def forward(self, x):
        x_norm = clip(1.17*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks8Norm2(EvolvedLayer):
    """
    Original: blocks.8.norm2
    Equation: 0.512*clip(2.22*x)
    """
    def forward(self, x):
        x_norm = 0.512*clip(2.22*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks9Norm1(EvolvedLayer):
    """
    Original: blocks.9.norm1
    Equation: -1.35*neg(-0.15*x + clip(x))
    """
    def forward(self, x):
        x_norm = -1.35*neg(-0.15*x + clip(x))
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks9Norm2(EvolvedLayer):
    """
    Original: blocks.9.norm2
    Equation: clip(1.1*x)
    """
    def forward(self, x):
        x_norm = clip(1.1*x)
        
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
    Equation: -0.156*x + 1.28*clip(x)
    """
    def forward(self, x):
        x_norm = -0.156*x + 1.28*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks11Norm1(EvolvedLayer):
    """
    Original: blocks.11.norm1
    Equation: clip(1.11*x)
    """
    def forward(self, x):
        x_norm = clip(1.11*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks11Norm2(EvolvedLayer):
    """
    Original: blocks.11.norm2
    Equation: -0.0937*x + clip(x)
    """
    def forward(self, x):
        x_norm = -0.0937*x + clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks12Norm1(EvolvedLayer):
    """
    Original: blocks.12.norm1
    Equation: 0.926*clip(x)
    """
    def forward(self, x):
        x_norm = 0.926*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks12Norm2(EvolvedLayer):
    """
    Original: blocks.12.norm2
    Equation: 0.523*clip(1.94*x)
    """
    def forward(self, x):
        x_norm = 0.523*clip(1.94*x)
        
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
    Equation: -0.0302*x + clip(x)
    """
    def forward(self, x):
        x_norm = -0.0302*x + clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks14Norm1(EvolvedLayer):
    """
    Original: blocks.14.norm1
    Equation: 0.492*clip(1.88*x)
    """
    def forward(self, x):
        x_norm = 0.492*clip(1.88*x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks14Norm2(EvolvedLayer):
    """
    Original: blocks.14.norm2
    Equation: 0.798*clip(x)
    """
    def forward(self, x):
        x_norm = 0.798*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks15Norm1(EvolvedLayer):
    """
    Original: blocks.15.norm1
    Equation: 0.772*clip(x)
    """
    def forward(self, x):
        x_norm = 0.772*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks15Norm2(EvolvedLayer):
    """
    Original: blocks.15.norm2
    Equation: 0.747*clip(x)
    """
    def forward(self, x):
        x_norm = 0.747*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks16Norm1(EvolvedLayer):
    """
    Original: blocks.16.norm1
    Equation: 0.715*clip(x)
    """
    def forward(self, x):
        x_norm = 0.715*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks16Norm2(EvolvedLayer):
    """
    Original: blocks.16.norm2
    Equation: 0.692*clip(x)
    """
    def forward(self, x):
        x_norm = 0.692*clip(x)
        
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
    Equation: 0.637*clip(x)
    """
    def forward(self, x):
        x_norm = 0.637*clip(x)
        
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
    Equation: 0.584*clip(x)
    """
    def forward(self, x):
        x_norm = 0.584*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks19Norm2(EvolvedLayer):
    """
    Original: blocks.19.norm2
    Equation: 0.573*clip(x)
    """
    def forward(self, x):
        x_norm = 0.573*clip(x)
        
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
    Equation: 0.509*clip(x)
    """
    def forward(self, x):
        x_norm = 0.509*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks21Norm1(EvolvedLayer):
    """
    Original: blocks.21.norm1
    Equation: 0.459*clip(x)
    """
    def forward(self, x):
        x_norm = 0.459*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks21Norm2(EvolvedLayer):
    """
    Original: blocks.21.norm2
    Equation: 0.45*clip(x)
    """
    def forward(self, x):
        x_norm = 0.45*clip(x)
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm

class Blocks22Norm1(EvolvedLayer):
    """
    Original: blocks.22.norm1
    Equation: 0.402*clip(x)
    """
    def forward(self, x):
        x_norm = 0.402*clip(x)
        
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
