"""nn_controller.py
Defines a neural network controller for the cart-pole system.
The controller is a simple multilayer perceptron mapping the
5D state [x, cos(theta), sin(theta), x_dot, theta_dot] to a
scalar force.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx


class CartPoleNN(eqx.Module):
    """Multi-layer perceptron policy for cart-pole control."""

    layers: list[eqx.Module]
    activations: list = eqx.static_field()

    def __init__(self, key: jax.Array, in_dim: int = 5,
                 hidden_dims: tuple[int, ...] = (64, 64), out_dim: int = 1) -> None:
        keys = jax.random.split(key, len(hidden_dims) + 1)
        self.layers = []
        input_size = in_dim
        for i, h in enumerate(hidden_dims):
            self.layers.append(eqx.nn.Linear(input_size, h, key=keys[i]))
            input_size = h
        self.layers.append(eqx.nn.Linear(input_size, out_dim, key=keys[-1]))
        self.activations = [jax.nn.relu for _ in hidden_dims]

    def __call__(self, state: jnp.ndarray, t: float | None = None) -> jnp.ndarray:
        """Evaluate the policy. The time argument is ignored."""
        x = state
        for layer, activation in zip(self.layers[:-1], self.activations):
            x = activation(layer(x))
        return self.layers[-1](x)[0]

