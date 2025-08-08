# controller/neuralnetwork_controller.py
import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom


class MLP(eqx.Module):
    """
    A simple 2-layer MLP with hidden_sizes and Tanh activation.
    """
    layers: list  # List of eqx.nn.Linear modules
    activation: callable = eqx.static_field()
    # hidden_sizes=[64, 64]
    def __init__(self, in_size=5, hidden_sizes=[64, 64], out_size=1, key=None):
        keys = jrandom.split(key, len(hidden_sizes) + 1)

        # Create MLP layers
        self.layers = []
        prev_size = in_size
        for h in hidden_sizes:
            self.layers.append(eqx.nn.Linear(in_features=prev_size, out_features=h, key=keys[0]))
            keys = keys[1:]
            prev_size = h

        # Final output layer
        self.layers.append(eqx.nn.Linear(in_features=prev_size, out_features=out_size, key=keys[0]))
        # Activation function
        self.activation = jnn.gelu

    def __call__(self, x):
        # Forward pass through the network
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)  # No activation on the output layer
        return x.squeeze(-1)  # Shape: ()
