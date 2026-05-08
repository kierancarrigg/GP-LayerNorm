"""GP evolution entry point: evolve LayerNorm replacement functions for each ViT-B layer.

Loads pre-extracted LN input-output mappings from extract_mappings.py, initialises a
Kozax GeneticProgramming strategy, and runs num_seeds independent evolution runs per
layer. Results (Pareto-front expressions with train/val MSE) are written to a CSV file.
Supports resuming interrupted runs by skipping layers already fully recorded in the CSV.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

from kozax.genetic_programming import GeneticProgramming

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from gp.evolution.fitness import LNMappingFitness
from gp.evolution.operators import get_operator_list
from utils import str2bool

def get_args():
    """Build the argument parser for the GP evolution script.

    Returns:
        Parsed argument namespace.
    """
    p = argparse.ArgumentParser("LayerNorm Mapping with Genetic Programming")
    p.add_argument('--seed', default=42, type=int, help="Base seed for data splitting")
    p.add_argument('--num_seeds', default=5, type=int, help="Number of independent GP runs per layer")
    p.add_argument('--data_file', default='ln_mappings.npz', type=str)
    p.add_argument('--output_csv', default='gp_pareto_results.csv', type=str, help="Where to save the results")
    
    # GP args
    p.add_argument('--num_generations', default=50, type=int)
    p.add_argument('--population_size', default=500, type=int)
    p.add_argument('--num_populations', default=1, type=int)
    p.add_argument('--max_init_depth', default=4, type=int)
    p.add_argument('--max_nodes', default=20, type=int)
    p.add_argument('--tournament_size', default=7, type=int)
    p.add_argument('--penalty_weight', default=0.005, type=float)
    p.add_argument('--complexity_objective', type=str2bool, default=True)
    p.add_argument('--constant_optimization_method', type=str, default='gradient',
                   help="Constant optimisation method: None to disable, 'gradient' to enable (default). Matches kozax 0.0.13 API.")
    p.add_argument('--constant_optimization_steps', type=int, default=1)
    return p.parse_args()

def main():
    """Run GP evolution across all LayerNorm layers.

    Initialises a single GeneticProgramming instance (compiled once by JAX JIT),
    then iterates over each layer in the .npz data file, running num_seeds independent
    evolution runs per layer. Each run performs a 90/10 train/val split, evolves a
    Pareto front of expressions trading off MSE against complexity, and appends all
    front members to the output CSV. Layers already present in an existing CSV are
    skipped automatically.
    """
    args = get_args()
    print("\n" + "="*60)
    print("Configuration:")
    print("="*60)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # Setup JAX and Reproducibility
    np.random.seed(args.seed)
    key = jr.PRNGKey(args.seed)
    device_type = "gpu" if jax.default_backend() == "gpu" else "cpu"
    print(f"JAX running on: {device_type.upper()}")

    # Load pre-extracted data
    print(f"Loading data from {args.data_file}...")
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Could not find {args.data_file}. Did you run extract_mappings.py?")
    
    data = np.load(args.data_file)
    
    # Extract unique layer names
    all_layer_names = sorted(list(set([k.rsplit('_', 1)[0] for k in data.files])))
    
    # --- RESUME LOGIC (Optional: Checks for existing CSV) ---
    layers_to_process = []
    if os.path.exists(args.output_csv):
        print(f"Found existing results at {args.output_csv}. Checking for completed layers...")
        try:
            with open(args.output_csv, 'r') as f:
                content = f.read()
            for ln in all_layer_names:
                # Count rows where this layer is the first CSV field (exact match)
                completed_count = sum(1 for line in content.splitlines() if line.split(',')[0] == ln)
                if completed_count >= args.num_seeds:
                    print(f"  Skipping {ln} (Already completed)")
                else:
                    layers_to_process.append(ln)
        except Exception:
            layers_to_process = all_layer_names
    else:
        layers_to_process = all_layer_names
    
    if not layers_to_process:
        print("All layers appear to be finished!")
        return

    print(f"Layers remaining to process: {len(layers_to_process)}\n")

    # Initialize GP once outside all loops to avoid recompiling JAX kernels
    print("Initializing Global GP Strategy (Compiling JAX kernels)...")
    
    operator_list = get_operator_list(use_probs=True)

    print("\n" + "="*60)
    print("GP Operators:")
    print("="*60)
    for op_name, _, arity, prob in operator_list:
        print(f"  {op_name}: arity={arity}, prob={prob:.2f}")
    print("="*60 + "\n")
    
    variable_list = ["x"]
    fitness_fn = LNMappingFitness(penalty_weight=args.penalty_weight)
    
    # Initialize the strategy instance ONCE
    # JAX will compile the .fit() method the first time it's called
    gp = GeneticProgramming(
        num_generations=args.num_generations,
        population_size=args.population_size,
        num_populations=args.num_populations,
        fitness_function=fitness_fn,
        operator_list=operator_list,
        variable_list=variable_list,
        layer_sizes=jnp.array([1]),
        max_init_depth=args.max_init_depth,
        max_nodes=args.max_nodes,
        tournament_size=args.tournament_size,
        device_type=device_type,
        constant_sd=1.0,
        complexity_objective=args.complexity_objective,
        constant_optimization_method=args.constant_optimization_method,
        constant_optimization_steps=args.constant_optimization_steps
    )

    mode = 'a' if os.path.exists(args.output_csv) else 'w'
    
    with open(args.output_csv, mode=mode, newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Only write header if new file
        if mode == 'w':
            writer.writerow(['Layer', 'Run_Seed', 'Complexity_Rank', 'Train_MSE', 'Val_MSE', 'Expression'])

        # 1. Outer Loop: Layers
        for layer_name in layers_to_process:
            print(f"\n{'='*60}\nEvolving Layer: {layer_name}\n{'='*60}")
            layer_start_time = time.time()
            
            # Load and format data
            x_np = data[f"{layer_name}_x"]
            y_np = data[f"{layer_name}_y"]
            
            N = x_np.shape[0]
            x_all = jnp.asarray(x_np).reshape(N, 1)
            y_all = jnp.asarray(y_np).reshape(N, 1)

            # Train/Val Split
            rng = np.random.default_rng(args.seed)
            perm = rng.permutation(N)
            n_train = int(0.9 * N)
            train_idx, val_idx = perm[:n_train], perm[n_train:]
            train_data = (x_all[train_idx], y_all[train_idx])
            val_data = (x_all[val_idx], y_all[val_idx])
            
            # 2. Inner Loop: Seeds
            for seed_idx in range(args.num_seeds):
                print(f"  --> Starting Run {seed_idx + 1}/{args.num_seeds}...")
                seed_start_time = time.time()
                
                # Generate unique JAX subkey
                key, subkey = jr.split(key)
                
                # Reuse the compiled gp instance; verbose=0 suppresses Kozax per-generation output
                gp.fit(subkey, train_data, verbose=0)
                
                # Extract results
                pareto_fitness, pareto_solutions = gp.pareto_front
                
                best_val_mse_for_run = float('inf')
                
                for rank, (train_mse, prog) in enumerate(zip(pareto_fitness, pareto_solutions)):
                    expr_str = gp.expression_to_string(prog)
                    val_mse = float(fitness_fn(prog, val_data, gp.tree_evaluator))
                    
                    if val_mse < best_val_mse_for_run:
                        best_val_mse_for_run = val_mse
                    
                    writer.writerow([
                        layer_name, seed_idx + 1, rank,
                        float(train_mse), val_mse, expr_str
                    ])
                
                csv_file.flush()
                
                seed_duration = time.time() - seed_start_time
                print(f"      Run {seed_idx + 1} Complete. Best Val MSE: {best_val_mse_for_run:.6f} | Time: {seed_duration:.2f}s")

            layer_duration = time.time() - layer_start_time
            print(f"Layer {layer_name} Finished. Total Time: {layer_duration/60:.2f} minutes")

    print(f"\nEvolution complete! All results saved to {args.output_csv}")

if __name__ == "__main__":
    main()