"""Code generation script: convert GP results CSV into per-seed PyTorch evolved-layer files.

Reads a GP results CSV produced by gp/evolution/main.py, selects one expression
per layer for each seed, and writes one evolved_layers_seed_N.py file per seed into
the specified output directory. Each generated file contains EvolvedLayer subclasses
and an apply_evolution() helper for injecting them into a pretrained ViT.

Two selection strategies are available (--strategy):

  kneedle  Pick the knee point of the per-layer (FLOPs, MSE) Pareto front, trading
           accuracy against hardware cost. This is the default.
  min_mse  Pick the lowest-MSE solution regardless of cost, reproducing the
           original selection method.

FLOPs are recomputed from the expression string rather than read from the CSV, so
this works on results files written before main.py gained its FLOPs column.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import sympy

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

# --- Selection: FLOP cost and Pareto knee point ---
def calculate_flops_sympy(expr_str, clip_cost=0.0):
    """Calculate the FLOP cost of an expression by walking its SymPy AST.

    Parsing the expression rather than string-matching avoids miscounting negative
    constants and scientific notation.

    Note that clip defaults to 0 FLOPs here: it is a bounds check with no arithmetic,
    so 0 is its true hardware cost. Evolution deliberately charges it 1.0 instead
    (see gp/evolution/operators.py) to stop GP stacking redundant clips for free;
    that regularisation should not leak into the reported cost.

    Args:
        expr_str: Expression string, e.g. ``"clip(2*x + tanh(x))"``.
        clip_cost: FLOP cost to attribute to each clip call.

    Returns:
        Total FLOP cost as a float. Returns 99999.0 for blank or unparseable
        expressions so they sort to the expensive end and are never selected.
    """
    if pd.isna(expr_str) or str(expr_str).strip() == "":
        return 99999.0

    try:
        expr = sympy.parse_expr(str(expr_str))
    except Exception:
        return 99999.0

    cost = 0.0
    for node in sympy.preorder_traversal(expr):
        if isinstance(node, sympy.Add):
            cost += (len(node.args) - 1) * 1.0
        elif isinstance(node, sympy.Mul):
            cost += (len(node.args) - 1) * 1.0
            # SymPy represents negation as multiplication by -1, which costs nothing
            if sympy.Integer(-1) in node.args:
                cost -= 1.0
        elif isinstance(node, sympy.Pow):
            if isinstance(node.args[1], sympy.Integer) and node.args[1] > 0:
                cost += (int(node.args[1]) - 1) * 1.0
        elif node.func.__name__ == 'tanh':
            cost += 23.0
        elif node.func.__name__ == 'sigmoid':
            cost += 22.0
        elif node.func.__name__ == 'clip':
            cost += clip_cost

    return max(0.0, cost)

def get_optimal_by_kneedle(df):
    """Return the knee point of a Pareto front using the Kneedle algorithm.

    Normalises FLOPs and MSE to [0, 1], draws a chord between the cheapest and most
    accurate endpoints, and picks the point furthest from that chord — the elbow
    beyond which extra FLOPs buy little accuracy.

    Args:
        df: Pareto front vertices, sorted by ascending FLOPs, with 'Calc_FLOPs'
            and 'Val_MSE' columns.

    Returns:
        The selected row.
    """
    if len(df) <= 2:
        return df.iloc[-1]

    df = df.copy()
    f_range = df['Calc_FLOPs'].max() - df['Calc_FLOPs'].min()
    m_range = df['Val_MSE'].max() - df['Val_MSE'].min()
    df['n_F'] = (df['Calc_FLOPs'] - df['Calc_FLOPs'].min()) / (f_range if f_range > 0 else 1)
    df['n_M'] = (df['Val_MSE'] - df['Val_MSE'].min()) / (m_range if m_range > 0 else 1)

    p_a = df[['n_F', 'n_M']].iloc[0].values
    p_b = df[['n_F', 'n_M']].iloc[-1].values

    distances = []
    for _, row in df.iterrows():
        p_c = row[['n_F', 'n_M']].values
        vec_ab, vec_ac = p_b - p_a, p_c - p_a
        # Perpendicular distance from the point to the chord, via the 2D cross product
        distances.append(
            np.abs(vec_ab[0] * vec_ac[1] - vec_ab[1] * vec_ac[0]) / np.linalg.norm(vec_ab)
        )
    df['dist'] = distances

    return df.loc[df['dist'].idxmax()]

def select_per_layer(seed_df, strategy):
    """Select one expression per layer for a single seed.

    Args:
        seed_df: Rows for one seed, with a 'Calc_FLOPs' column already populated.
        strategy: Either 'kneedle' or 'min_mse'.

    Returns:
        A DataFrame with one row per layer.
    """
    selected_rows = []

    for layer in seed_df['Layer'].unique():
        layer_df = seed_df[seed_df['Layer'] == layer].copy()

        if strategy == 'min_mse':
            selected_rows.append(layer_df.loc[layer_df['Val_MSE'].idxmin()])
            continue

        # Build the Pareto front: cheapest first, keeping the best MSE at each cost,
        # then enforce monotonically decreasing MSE so only genuine trade-offs remain.
        pareto_df = layer_df.sort_values(['Calc_FLOPs', 'Val_MSE']).reset_index(drop=True)
        pareto_df = pareto_df.drop_duplicates(subset=['Calc_FLOPs'], keep='first')
        pareto_df['cummin_MSE'] = pareto_df['Val_MSE'].cummin()
        vertices = pareto_df.drop_duplicates(subset=['cummin_MSE'], keep='first').copy().reset_index(drop=True)

        selected_rows.append(get_optimal_by_kneedle(vertices))

    return pd.DataFrame(selected_rows)

def main():
    """Parse arguments and generate one evolved_layers_seed_N.py file per GP seed."""
    parser = argparse.ArgumentParser("Generate evolved layer PyTorch files from GP results CSV")
    parser.add_argument('--input_csv', required=True, type=str, help="Path to the GP results CSV file")
    parser.add_argument('--output_dir', required=True, type=str, help="Directory to write evolved_layers_seed_N.py files")
    parser.add_argument('--strategy', type=str, default='kneedle', choices=['kneedle', 'min_mse'],
                        help="Per-layer selection: 'kneedle' knee point of the FLOPs/MSE Pareto front "
                             "(default), or 'min_mse' lowest MSE regardless of cost")
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

    # Recomputed from the expression so older CSVs without a FLOPs column still work
    print("[*] Calculating FLOP complexity for all expressions...")
    df['Calc_FLOPs'] = df['Expression'].apply(calculate_flops_sympy)

    unique_seeds = df['Run_Seed'].unique()
    print(f"[*] Found {len(unique_seeds)} unique seeds: {unique_seeds}")
    print(f"[*] Selection strategy: {args.strategy}\n")

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

        # 2. Select one expression per layer using the chosen strategy
        seed_df = select_per_layer(seed_df, args.strategy)

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
            
        print(f"✅ Generated {output_filename} ({count} layers using {args.strategy})")

if __name__ == "__main__":
    main()