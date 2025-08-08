"""Shared utilities for linear controller training."""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Target upright state: [x=0, cos(theta)=1, sin(theta)=0, x_dot=0, theta_dot=0]
TARGET = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0])


@jax.jit
def _trajectory_cost_impl(traj: jnp.ndarray, K: jnp.ndarray, Q: jnp.ndarray, R: float, dt: float) -> jnp.ndarray:
    """Compute trajectory cost for a single rollout."""
    err = traj - TARGET
    forces = -(err @ K)
    state_cost = jnp.einsum("ij,jk,ik->i", err, Q, err)
    ctrl_cost = R * jnp.square(forces)
    return dt * jnp.sum(state_cost + ctrl_cost)


__all__ = ["TARGET", "_trajectory_cost_impl"]

