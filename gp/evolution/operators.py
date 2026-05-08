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
) -> list[tuple]:
    """
    Return Kozax operator tuples.
    Each tuple: (name, function, arity[, probability])
    """
    ops = [
        ("+", add, 2),
        ("*", mul, 2),
        ("neg", neg, 1),
        ("tanh", tanh, 1),
        ("sigmoid", sigmoid, 1),
        ("clip", clip, 1),
    ]

    if not use_probs:
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

    return [(name, fn, arity, float(default_probs.get(name, 0.1)))
            for (name, fn, arity) in ops]