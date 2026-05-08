import jax
import jax.numpy as jnp
from kozax.fitness_functions.base_fitness_function import BaseFitnessFunction

class LNMappingFitness(BaseFitnessFunction):
    """
    Fitness function for evolving LayerNorm replacement functions.
    
    Evaluates how well a candidate function f(x) matches the target
    LayerNorm input-output mapping.
    """
    
    def __init__(self, penalty_weight: float = 0.01):
        """
        Initializes the fitness function.
        
        Args:
            penalty_weight (float): How strongly to penalize exploding outputs on out-of-distribution extremes.
        """
        self.penalty_weight = penalty_weight
    
    def __call__(self, candidate, data, tree_evaluator):
        """
        Evaluate fitness of a candidate function.
        
        Args:
            candidate: GP tree representing a function
            data: Tuple of (x_inputs, y_targets) where:
                  - x_inputs: LN layer inputs (1D array)
                  - y_targets: LN layer pre-affine outputs (1D array)
            tree_evaluator: Function to evaluate GP tree on data
        
        Returns:
            fitness: Scalar loss (MSE between pre-affine predictions and targets)
        """
        # Step 1: Unpack the data
        x_inputs, y_targets = data
        
        # Step 2: Evaluate the candidate function on all inputs
        # vmap applies tree_evaluator to each x value in parallel
        predictions = jax.vmap(tree_evaluator, in_axes=(None, 0))(candidate, x_inputs)
        mse = jnp.mean((predictions - y_targets) ** 2)
        
        # Step 3: Dynamically calculate informed OOD anchors
        max_real_val = jnp.max(jnp.abs(x_inputs))
        
        # Step 4: Evaluate and penalize high values
        anchor = max_real_val * 2.0
        extreme_inputs = jnp.array([[-anchor], [anchor]])
        extreme_preds = jax.vmap(tree_evaluator, in_axes=(None, 0))(candidate, extreme_inputs)
        extreme_penalty = jnp.mean(extreme_preds ** 2)

        # Step 5: Combine MSE with extreme penalty
        total_loss = mse + (self.penalty_weight * extreme_penalty)
        
        return total_loss