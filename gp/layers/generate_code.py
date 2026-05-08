"""Code generation script: convert GP results CSV into per-seed PyTorch evolved-layer files.

Reads a GP results CSV produced by gp/evolution/main.py, selects the best expression
per layer for each seed, and writes one evolved_layers_seed_N.py file per seed into
the specified output directory. Each generated file contains EvolvedLayer subclasses
and an apply_evolution() helper for injecting them into a pretrained ViT.
"""
import argparse
import os
import re

import pandas as pd

# --- The File Header (Boilerplate for the generated file) ---
FILE_HEADER = '''import torch
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
'''

# --- Translation Logic: String -> PyTorch ---
def translate_to_torch(expr):
    """Translate a GP expression string into PyTorch-compatible code.

    Maps bare ``tanh(`` calls to ``torch.tanh(`` so they use PyTorch's built-in;
    ``clip`` and ``sigmoid`` are left as-is since they use the custom helpers
    defined in FILE_HEADER.

    Args:
        expr: Expression string, e.g. ``"clip(2*x + tanh(x))"``.

    Returns:
        Translated expression string, e.g. ``"clip(2*x + torch.tanh(x))"``.
        Returns ``"x"`` if expr is NaN.
    """
    if pd.isna(expr): return "x"

    # Only tanh needs the torch. prefix; clip and sigmoid use our custom helpers
    torch_ops = [
        ('tanh(', 'torch.tanh('),
    ]
    
    clean_expr = expr
    for old, new in torch_ops:
        clean_expr = clean_expr.replace(old, new)
        
    return clean_expr

def get_layer_sort_key(name):
    """Return a numeric sort key so layers are ordered by block depth, norm1 before norm2."""
    nums = re.findall(r'\d+', name)
    if not nums:
        return 999
    return int(nums[0]) * 10 + (1 if 'norm2' in name else 0)

def main():
    """Parse arguments and generate one evolved_layers_seed_N.py file per GP seed."""
    parser = argparse.ArgumentParser("Generate evolved layer PyTorch files from GP results CSV")
    parser.add_argument('--input_csv', required=True, type=str, help="Path to the GP results CSV file")
    parser.add_argument('--output_dir', required=True, type=str, help="Directory to write evolved_layers_seed_N.py files")
    args = parser.parse_args()

    INPUT_CSV = args.input_csv

    print(f"Reading results from {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV, on_bad_lines='skip')
    except FileNotFoundError:
        print(f"❌ Error: Could not find {INPUT_CSV}")
        return

    if 'Run_Seed' not in df.columns:
        print("❌ Error: Could not find a 'Run_Seed' column in the CSV.")
        print(f"Available columns are: {list(df.columns)}")
        return

    unique_seeds = df['Run_Seed'].unique()
    print(f"[*] Found {len(unique_seeds)} unique seeds: {unique_seeds}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[*] Ensured output directory exists: {args.output_dir}\n")

    injector_code = '''
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
'''

    # Loop through every seed and generate a dedicated PyTorch file
    for seed in unique_seeds:
        # 1. Filter the dataframe for ONLY this seed
        seed_df = df[df['Run_Seed'] == seed].copy()
        
        # 2. (Optional safeguard) If a single seed logged multiple times per layer, 
        # take the best Val_MSE within this specific seed to prevent duplicates
        seed_df = seed_df.loc[seed_df.groupby('Layer')['Val_MSE'].idxmin()].copy()
        
        # 3. Sort nicely by layer depth
        seed_df['sort_idx'] = seed_df['Layer'].apply(get_layer_sort_key)
        seed_df = seed_df.sort_values('sort_idx')
        
        output_filename = os.path.join(args.output_dir, f'evolved_layers_seed_{seed}.py')
        
        # 5. Write the file
        with open(output_filename, 'w') as f:
            f.write(FILE_HEADER)
            
            count = 0
            for _, row in seed_df.iterrows():
                layer_name = row['Layer']
                raw_expr = row['Expression']
                
                class_name = layer_name.replace('.', '_').title().replace('_', '')
                torch_code = translate_to_torch(raw_expr)
                
                class_def = f'''
class {class_name}(EvolvedLayer):
    """
    Original: {layer_name}
    Equation: {raw_expr}
    """
    def forward(self, x):
        x_norm = {torch_code}
        
        if self.weight is not None:
            return x_norm * self.weight + self.bias
        return x_norm
'''
                f.write(class_def)
                count += 1
                
            # Add the injector helper at the bottom of each file
            f.write(injector_code)
            
        print(f"✅ Generated {output_filename} ({count} layers)")

if __name__ == "__main__":
    main()