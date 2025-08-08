"""LQR controller training utilities."""

from __future__ import annotations
import jax.numpy as jnp

from controller.lqr_controller import LQRController
from env.cartpole import CartPoleParams


def train_lqr_controller(
    params: CartPoleParams = CartPoleParams(),
    Q: jnp.ndarray | None = None,
    R: jnp.ndarray | None = None,
) -> LQRController:
    """Create LQR controller from linearised cart-pole."""
    if Q is None:
        Q = jnp.diag(jnp.array([10.0, 10.0, 1.0, 1.0]))
    if R is None:
        R = jnp.array([[0.1]])
    return LQRController.from_linearisation(params, Q, R)
