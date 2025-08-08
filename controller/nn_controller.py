"""
controller/nn_controller.py

Deterministic, JAX-friendly neural network controller operating on the
5-state representation [x, cos(theta), sin(theta), xdot, thdot]. If a 4-state
[x, theta, xdot, thdot] vector is passed, it is up-cast to 5-state.

- Equinox MLP backbone with tanh activations
- Deterministic initialisation via `init_weights(seed)`
- JIT-ready forward pass and batched support via base `Controller`
"""

from __future__ import annotations
from typing import Iterable, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import equinox as eqx

from controller.base import Controller
from env.helpers import four_to_five

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


def init_weights(seed: int = 0, hidden_dims: Sequence[int] = (64, 64)) -> eqx.Module:
    """Deterministically initialise an Equinox MLP for 5D input and 1D output."""
    key = jax.random.PRNGKey(seed)
    return _build_mlp(5, hidden_dims, key=key)


@dataclass(frozen=True)
class NNController(Controller):
    """Neural-network controller with Equinox backbone.

    Attributes:
        net: Equinox module mapping R^5 -> R
    """

    net: eqx.Module

    @classmethod
    def init(
        cls,
        *,
        seed: int = 0,
        hidden_dims: Sequence[int] = (64, 64),
        key: jax.Array | None = None,
    ) -> "NNController":
        """Create controller with deterministic weights."""
        if key is not None:
            # Backward compatibility: accept explicit PRNG key
            net = _build_mlp(5, hidden_dims, key=key)
        else:
            net = init_weights(seed, hidden_dims)
        return cls(net=net)

    def _force(self, state: jnp.ndarray, _t: float) -> jnp.ndarray:
        """Compute NN force; up-casts 4-state to 5-state if needed."""
        if state.shape[-1] == 4:
            s5 = four_to_five(state)
        else:
            s5 = state
        # Avoid passing PRNG keys to pure functions inside Sequential; our Tanh ignores key
        return jnp.squeeze(self.net(s5, key=None))


