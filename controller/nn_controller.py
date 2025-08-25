"""
controller/nn_controller.py

Deterministic, JAX-friendly neural network controller operating on the
5-state representation [x, cos(theta), sin(theta), xdot, thdot].

- Equinox MLP backbone with tanh activations
- Deterministic initialisation via `init_weights(seed)`
- JIT-ready forward pass and batched support via base `Controller`
- Optimized for 5-state only (no conversion overhead)
- Action bounded to [-u_max, +u_max] for stability
"""

from __future__ import annotations
from typing import Iterable, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import equinox as eqx

from controller.base import Controller

__all__ = ["NNController", "init_weights"]


class Tanh(eqx.Module):
    """Activation wrapper compatible with Equinox Sequential API."""
    def __call__(self, x, *, key=None):  # key ignored intentionally
        return jax.nn.tanh(x)


def _build_mlp(in_dim: int, hidden_dims: Sequence[int], *, key: jax.Array) -> eqx.Module:
    """Construct a simple MLP with tanh activations and scalar output."""
    layers: list = []
    dims = [in_dim, *hidden_dims, 1]
    k = key
    for i in range(len(dims) - 1):
        k, sub = jax.random.split(k)
        layers.append(eqx.nn.Linear(dims[i], dims[i + 1], key=sub))
        if i < len(dims) - 2:
            layers.append(Tanh())
    return eqx.nn.Sequential(layers)


def init_weights(seed: int = 0, hidden_dims: Sequence[int] = (256, 256, 128, 128)) -> eqx.Module:
    """Deterministically initialise an Equinox MLP for 5D input and 1D output.
    
    Larger architecture for better swing-up performance:
    - 4 hidden layers instead of 2
    - 256->256->128->128 instead of 64->64
    - Better capacity for complex control policies
    """
    key = jax.random.PRNGKey(seed)
    return _build_mlp(5, hidden_dims, key=key)


@dataclass(frozen=True)
class NNController(Controller):
    """Neural-network controller with Equinox backbone.

    Attributes:
        net: Equinox module mapping R^5 -> R
        u_max: Maximum control force magnitude (action bound)
    """

    net: eqx.Module
    u_max: float = 10.0  # Default action bound

    @classmethod
    def init(
        cls,
        *,
        seed: int = 0,
        hidden_dims: Sequence[int] = (256, 256, 128, 128),  # Larger default architecture
        key: jax.Array | None = None,
        u_max: float = 10.0,
    ) -> "NNController":
        """Create controller with deterministic weights and action bound."""
        if key is not None:
            # Backward compatibility: accept explicit PRNG key
            net = _build_mlp(5, hidden_dims, key=key)
        else:
            net = init_weights(seed, hidden_dims)
        return cls(net=net, u_max=u_max)

    def _force(self, state: jnp.ndarray, _t: float) -> jnp.ndarray:
        """Compute NN force for 5-state representation, bounded to [-u_max, +u_max]."""
        # Direct computation without state conversion checks
        raw_force = jnp.squeeze(self.net(state, key=None))
        # Bound action using tanh scaling: u_max * tanh(raw_force / u_max)
        return self.u_max * jax.nn.tanh(raw_force / self.u_max)


