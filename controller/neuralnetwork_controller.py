# controller/neuralnetwork_controller.py
# Description: This module implements a neural network controller for a cart-pole system.
# It uses a multi-layer perceptron to compute the control force based on the current state.
# It also includes functions to compute the trajectory cost and train the controller using gradient descent.

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random


class CartPolePolicy(eqx.Module):
    """
    A simple multi-layer perceptron for cart-pole swing-up.
    Maps [x, cos(theta), sin(theta), x_dot, theta_dot] to a single force output.
    """
    layers: list
    activations: list = eqx.static_field()

    def __init__(
        self,
        key,
        in_dim: int = 5,
        hidden_dims=(64, 64),
        out_dim: int = 1
    ):
        """
        hidden_dims: tuple specifying the size of each hidden layer, e.g. (64, 128).
        """
        # We'll create multiple Linear layers and store them in self.layers
        keys = jax.random.split(key, num=len(hidden_dims) + 1)
        all_layers = []
        # First layer
        in_size = in_dim
        for i, hdim in enumerate(hidden_dims):
            all_layers.append(eqx.nn.Linear(in_size, hdim, key=keys[i]))
            in_size = hdim
        # Final layer
        all_layers.append(eqx.nn.Linear(in_size, out_dim, key=keys[-1]))
        self.layers = all_layers

        # Activation for hidden layers (we'll just use relu for each)
        self.activations = [jax.nn.relu for _ in hidden_dims]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass. x is shape (5,).
        Returns a scalar (float) representing the force.
        """
        for layer, activation in zip(self.layers[:-1], self.activations):
            x = layer(x)
            x = activation(x)
        # Final layer has no activation
        x = self.layers[-1](x)
        return x[0]  # Return as a scalar