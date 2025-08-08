"""Minimal gradient-descent training for linear controllers."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from controller.linear_controller import LinearController
from env.closedloop import simulate
from env.cartpole import CartPoleParams
from lib.cost_functions import create_cost_matrices
from lib.training_utils import TrainingHistory

from ._common import TARGET, _trajectory_cost_impl


@dataclass
class BasicTrainingConfig:
    """Configuration for basic training."""

    learning_rate: float = 0.01
    num_iterations: int = 100
    trajectory_length: float = 3.0


def _make_loss_fn(ts: jnp.ndarray, y0: jnp.ndarray, Q: jnp.ndarray, R: float, params: CartPoleParams):
    dt = float(ts[1] - ts[0])

    def loss(K: jnp.ndarray) -> jnp.ndarray:
        def ctrl(y, t):
            err = y - TARGET
            return jnp.clip(-(err @ K), -100.0, 100.0)

        sol = simulate(ctrl, params, (ts[0], ts[-1]), ts, y0)
        bad = ~jnp.all(jnp.isfinite(sol.ys))
        cost = _trajectory_cost_impl(sol.ys, K, Q, R, dt)
        return jnp.where(bad, jnp.inf, cost)

    return loss


def train_linear_controller(
    initial_K: jnp.ndarray,
    initial_state: jnp.ndarray,
    config: BasicTrainingConfig = BasicTrainingConfig(),
    *,
    Q: jnp.ndarray | None = None,
    params: CartPoleParams = CartPoleParams(),
):
    """Train linear controller using vanilla gradient descent."""

    if initial_K is None:
        raise ValueError("initial_K is required for basic training")

    if Q is None:
        Q = create_cost_matrices()

    dt = 0.02
    ts = jnp.arange(0.0, config.trajectory_length + dt / 2, dt)

    loss_fn = _make_loss_fn(ts, initial_state, Q, 0.1, params)
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    K = initial_K
    history = TrainingHistory()

    initial_cost = loss_fn(K)
    if not jnp.isfinite(initial_cost):
        print("Invalid initial cost")
        return LinearController(K=initial_K), history

    for _ in range(config.num_iterations):
        cost, grads = value_and_grad(K)
        K = K - config.learning_rate * grads
        history.update(cost, K)

    controller = LinearController(K=K)
    return controller, history


__all__ = ["BasicTrainingConfig", "train_linear_controller"]

