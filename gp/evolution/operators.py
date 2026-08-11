import jax
import jax.numpy as jnp
from jax import Array

def add(a: Array, b: Array) -> Array:
    return jnp.add(a, b)

def mul(a: Array, b: Array) -> Array:
    return jnp.multiply(a, b)

def neg(a: Array) -> Array:
    return jnp.negative(a)

def tanh(a: Array) -> Array:
    return jnp.tanh(a)

def clip(a: Array) -> Array:
    return jnp.clip(a, -5.0, 5.0)

def sigmoid(a: Array) -> Array:
    return jax.nn.sigmoid(a)


def get_operator_list(
    *,
    use_probs: bool = True,
    probs: dict[str, float] | None = None,
) -> list[dict]:
    """
    Return Kozax operator dictionaries.
    Each dict: {"string": name, "fn": function, "arity": arity, "prob": probability, "flops": flops}

    The "flops" entry is the per-operator cost Kozax uses as the complexity
    objective, replacing a plain node count.

    Note on clip: its true cost is 0 FLOPs (a bounds check, no arithmetic), and
    that is the cost used when reporting FLOPs for a solution. It is charged 1.0
    here purely as a regulariser during evolution — at a true cost of 0.0 GP
    stacked redundant clips for free, since they were complexity-neutral.
    """
    ops = [
        {"string": "+", "fn": add, "arity": 2, "flops": 1.0},
        {"string": "*", "fn": mul, "arity": 2, "flops": 1.0},
        {"string": "neg", "fn": neg, "arity": 1, "flops": 0.0},
        {"string": "tanh", "fn": tanh, "arity": 1, "flops": 23.0},
        {"string": "sigmoid", "fn": sigmoid, "arity": 1, "flops": 22.0},
        {"string": "clip", "fn": clip, "arity": 1, "flops": 1.0},
    ]

    if not use_probs:
        # Dummy probability keeps the dict shape consistent for Kozax
        for op in ops:
            op["prob"] = 0.0
        return ops

    # default weights (bias toward simple algebra + mild nonlinearity)
    default_probs = {
        "+": 0.28,
        "*": 0.26,
        "neg": 0.08,
        "tanh": 0.20,
        "sigmoid": 0.10,
        "clip": 0.08,
    }
    if probs is not None:
        default_probs.update(probs)

    for op in ops:
        op["prob"] = float(default_probs.get(op["string"], 0.1))

    return ops