"""
lib/training/linear_training.py
Unified training utilities for linear controllers with optional advanced features.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as random
import optax
import time

from controller.linear_controller import LinearController
from controller.lqr_controller import LQRController, _linearise
from env.cartpole import CartPoleParams
from env.closedloop import simulate_batch
from lib.cost_functions import create_cost_matrices
from lib.training_utils import TrainingHistory

from ._common import TARGET, _trajectory_cost_impl


@dataclass
class LinearTrainingConfig:
    """Configuration for linear controller training with optional advanced features."""

    learning_rate: float = 0.01
    num_iterations: int = 100
    trajectory_length: float = 3.0
    batch_size: int = 1
    perturb_std: float = 0.0
    lr_schedule: str = "none"  # 'none' | 'cosine' | 'step'
    stability_weight: float = 0.0
    seed: int = 0
    lqr_warm_start: bool = False
    optimizer: str = "sgd"
    print_data: bool = False


def _four_to_five(v):
    return jnp.array([v[0], jnp.cos(v[1]), jnp.sin(v[1]), v[2], v[3]])


def _make_loss_fn(
    ts: jnp.ndarray,
    init_batch: jnp.ndarray,
    Q: jnp.ndarray,
    R: float,
    params: CartPoleParams,
    stability_weight: float = 0.0,
):
    dt = float(ts[1] - ts[0])

    def loss(K, batch_states):
        def ctrl(y, t):
            # Match deployed controller semantics: u = -K · state
            return jnp.clip(-(y @ K), -100.0, 100.0)

        sol = simulate_batch(ctrl, params, (ts[0], ts[-1]), ts, batch_states)
        bad = ~jnp.all(jnp.isfinite(sol.ys))
        cost = jnp.mean(
            jax.vmap(lambda tr: _trajectory_cost_impl(tr, K, Q, R, dt))(sol.ys)
        )

        if stability_weight > 0.0:
            A, B = _linearise(params)
            K4 = jnp.array([K[0], K[2], K[3], K[4]])
            eigvals = jnp.linalg.eigvals(A - B @ K4[None, :])
            penalty = jnp.sum(jnp.square(jnp.maximum(jnp.real(eigvals), 0.0)))
            cost = cost + stability_weight * penalty

        return jnp.where(bad, jnp.inf, cost)

    return loss


def train_linear_controller(
    initial_K: jnp.ndarray | None,
    initial_state: jnp.ndarray,
    config: LinearTrainingConfig = LinearTrainingConfig(),
    *,
    Q: jnp.ndarray | None = None,
    params: CartPoleParams = CartPoleParams(),
    print_data: bool = True,
):
    """Train linear controller with optional advanced features."""

    # Override config flag with explicit print_data parameter
    config.print_data = bool(print_data)

    if Q is None:
        Q = create_cost_matrices()

    if initial_K is None and config.lqr_warm_start:
        K4 = LQRController.from_linearisation(params).K.squeeze()
        initial_K = jnp.array([K4[0], 0.0, K4[1], K4[2], K4[3]])
    elif initial_K is None:
        raise ValueError("initial_K is required when lqr_warm_start=False")

    key = random.PRNGKey(config.seed)
    noise4 = config.perturb_std * random.normal(key, (config.batch_size, 4))
    if initial_state.shape[0] == 4:
        base4 = initial_state
    else:
        theta = jnp.arctan2(initial_state[2], initial_state[1])
        base4 = jnp.array([initial_state[0], theta, initial_state[3], initial_state[4]])
    batch4 = base4 + noise4
    init_batch = jax.vmap(_four_to_five)(batch4)

    schedules = {
        "none": config.learning_rate,
        "cosine": optax.cosine_decay_schedule(
            config.learning_rate, config.num_iterations
        ),
        "step": optax.piecewise_constant_schedule(
            config.learning_rate, {config.num_iterations // 2: 0.1}
        ),
    }
    lr_schedule = schedules[config.lr_schedule]
    optimizers = {"adam": optax.adam, "sgd": optax.sgd, "rmsprop": optax.rmsprop}
    optimizer = optimizers[config.optimizer](lr_schedule)
    opt_state = optimizer.init(initial_K)

    dt = 0.02
    ts = jnp.arange(0.0, config.trajectory_length + dt / 2, dt)

    loss_fn = _make_loss_fn(ts, init_batch, Q, 0.1, params, config.stability_weight)
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    history = TrainingHistory()
    K = initial_K

    initial_cost = loss_fn(K, init_batch)
    if not jnp.isfinite(initial_cost):
        if config.print_data:
            print("[TRAIN] LinearController started")
            print("[TRAIN] iter=0 time=0.000000s loss=inf")
            print("[TRAIN] LinearController finished in 0.000000s")
        return LinearController(K=initial_K), history

    # Training loop with logging
    start_total = time.perf_counter()
    if config.print_data:
        print("[TRAIN] LinearController started")
        print(f"[TRAIN] total_iters={config.num_iterations}")
        print("[TRAIN] Learning Parameters:")
        print(f"  learning_rate: {config.learning_rate}")
        print(f"  num_iterations: {config.num_iterations}")
        print(f"  trajectory_length: {config.trajectory_length}")
        print(f"  batch_size: {config.batch_size}")
        print(f"  perturb_std: {config.perturb_std}")
        print(f"  lr_schedule: {config.lr_schedule}")
        print(f"  stability_weight: {config.stability_weight}")
        print(f"  seed: {config.seed}")
        print(f"  lqr_warm_start: {config.lqr_warm_start}")
        print(f"  optimizer: {config.optimizer}")

    rng = key
    losses = []
    for i in range(config.num_iterations):
        iter_start = time.perf_counter()
        rng, subkey = random.split(rng)
        noise4 = config.perturb_std * random.normal(subkey, (config.batch_size, 4))
        batch4 = base4 + noise4
        batch_states = jax.vmap(_four_to_five)(batch4)

        cost, grads = value_and_grad(K, batch_states)
        updates, opt_state = optimizer.update(grads, opt_state)
        K = optax.apply_updates(K, updates)
        history.update(cost, K)
        iter_time = time.perf_counter() - iter_start

        losses.append(float(cost))
        # Print only every 50 iterations, and always print first and last
        if config.print_data and (i % 50 == 0 or i == config.num_iterations - 1 or i == 0):
            print(f"[TRAIN] iter={i} time={iter_time:.6f}s loss={float(cost):.6f}")

        if not jnp.isfinite(cost):
            break

    total_time = time.perf_counter() - start_total
    if config.print_data:
        print(f"[TRAIN] LinearController finished in {total_time:.6f}s")
        print("[TRAIN] Losses (every 50 iters):")
        for idx in range(0, len(losses), 50):
            print(f"  iter={idx} loss={losses[idx]:.6f}")
        if (len(losses) - 1) % 50 != 0:
            print(f"  iter={len(losses)-1} loss={losses[-1]:.6f}")

    controller = LinearController(K=K)
    return controller, history


def grid_search_linear_gains(
    initial_state: jnp.ndarray,
    gain_ranges: tuple[tuple[float, float], ...] | None = None,
    n_points: int = 5,
    params: CartPoleParams = CartPoleParams(),
):
    """Grid search for good initial gains."""

    if gain_ranges is None:
        gain_ranges = (
            (0.5, 2.0),
            (-25.0, -5.0),
            (5.0, 25.0),
            (0.5, 3.0),
            (0.5, 3.0),
        )

    Q = create_cost_matrices()
    R = 0.1
    dt = 0.02
    ts = jnp.arange(0.0, 3.0 + dt / 2, dt)
    loss_fn = _make_loss_fn(ts, initial_state, Q, R, params, 0.0)

    ranges = [jnp.linspace(start, end, n_points) for start, end in gain_ranges]
    grids = jnp.meshgrid(*ranges, indexing="ij")
    K_candidates = jnp.stack([grid.ravel() for grid in grids], axis=1)

    best_cost = jnp.inf
    best_K = None
    for K in K_candidates:
        try:
            cost = loss_fn(K, initial_state[None, :])
            if cost < best_cost:
                best_cost = cost
                best_K = jnp.array(K)
        except Exception:
            continue

    if best_K is None:
        best_K = jnp.array([1.0, -10.0, 10.0, 1.0, 1.0])

    return LinearController(K=best_K)


__all__ = [
    "LinearTrainingConfig",
    "train_linear_controller",
    "grid_search_linear_gains",
]
