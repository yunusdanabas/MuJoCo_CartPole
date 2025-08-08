"""
training/linear_training.py

Linear controller training using gradient-based optimization.
"""

from __future__ import annotations
from dataclasses import dataclass
import time

import jax
import jax.numpy as jnp
import optax
import jax.random as random

from controller.linear_controller import LinearController
from controller.lqr_controller import LQRController, _linearise  # for optional warm-start
from env.closedloop import simulate_batch, simulate
from env.cartpole import CartPoleParams
from lib.cost_functions import create_cost_matrices
from lib.training_utils import BaseTrainingConfig, TrainingHistory


@dataclass
class LinearTrainingConfig(BaseTrainingConfig):
    """Training hyper-parameters for linear controller."""
    state_weight: float = 1.0
    control_weight: float = 0.1
    optimizer: str = "adam"
    # ------- new flags -------
    batch_size: int = 16
    perturb_std: float = 0.05
    lr_schedule: str = "cosine"                # 'none' | 'cosine' | 'step'
    stability_weight: float = 0.0
    seed: int = 0
    lqr_warm_start: bool = False               # <-- OPTIONAL
    # ------------------------


TARGET = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0])


@jax.jit
def _trajectory_cost_impl(traj, K, Q, R, dt):
    err    = traj - TARGET
    forces = -(err @ K)
    state_cost = jnp.einsum('ij,jk,ik->i', err, Q, err)
    ctrl_cost  = R * forces**2
    return dt * jnp.sum(state_cost + ctrl_cost)


def _make_loss_fn(
    ts: jnp.ndarray,
    init_batch: jnp.ndarray,
    Q: jnp.ndarray,
    R: float,
    params: CartPoleParams,
    stability_weight: float = 0.0,
):
    """Return a closure suitable for jax.grad."""
    dt = float(ts[1] - ts[0])

    def loss(K, batch_states):
        def ctrl(y, t):
            err = y - TARGET
            return jnp.clip(-(err @ K), -100.0, 100.0)
        sol = simulate_batch(ctrl, params, (ts[0], ts[-1]), ts, batch_states)
        bad = ~jnp.all(jnp.isfinite(sol.ys))
        cost = jnp.mean(
            jax.vmap(lambda tr: _trajectory_cost_impl(tr, K, Q, R, dt))(sol.ys)
        )
        # Stability penalty (closed-loop eigenvalues)
        if stability_weight > 0.0:
            A, B = _linearise(params)
            eigvals = jnp.linalg.eigvals(A - B @ K[None, :])
            penalty = jnp.sum(jnp.square(jnp.maximum(jnp.real(eigvals), 0.0)))
            cost = cost + stability_weight * penalty
        return jnp.where(bad, jnp.inf, cost)
    return loss


def train_linear_controller(
    initial_K: jnp.ndarray | None,
    initial_state: jnp.ndarray,
    config: LinearTrainingConfig = LinearTrainingConfig(),
    Q: jnp.ndarray = None,
    params: CartPoleParams = CartPoleParams()
) -> tuple[LinearController, TrainingHistory]:
    """Train linear controller using batch robust gradient descent."""
    try:
        if Q is None:
            Q = create_cost_matrices()

        # ---------------------------------------------------------------
        # 0. Optional LQR warm-start
        # ---------------------------------------------------------------
        if initial_K is None and config.lqr_warm_start:
            K4 = LQRController.from_linearisation(params).K.squeeze()
            initial_K = jnp.array([K4[0], 0.0, K4[1], K4[2], K4[3]])
        elif initial_K is None:
            raise ValueError("initial_K is None and lqr_warm_start=False")

        # ---------------------------------------------------------------
        # 1. Build initial batch of perturbed states
        # ---------------------------------------------------------------
        key = random.PRNGKey(config.seed)
        noise4 = config.perturb_std * random.normal(key, (config.batch_size, 4))
        base4  = jnp.zeros(4) if initial_state.shape[0] == 4 else initial_state[:4]
        batch4 = base4 + noise4
        def _four_to_five(v):
            return jnp.array([v[0], jnp.cos(v[1]), jnp.sin(v[1]), v[2], v[3]])
        init_batch = jax.vmap(_four_to_five)(batch4)

        # ---------------------------------------------------------------
        # 2. Optimiser & LR schedule
        # ---------------------------------------------------------------
        schedules = {
            "none":   config.learning_rate,
            "cosine": optax.cosine_decay_schedule(config.learning_rate,
                                                  config.num_iterations),
            "step":   optax.piecewise_constant_schedule(
                          config.learning_rate,
                          {config.num_iterations // 2: 0.1}),
        }
        lr_schedule = schedules[config.lr_schedule]
        optimizers  = {"adam": optax.adam,
                       "sgd":  optax.sgd,
                       "rmsprop": optax.rmsprop}
        optimizer   = optimizers[config.optimizer](lr_schedule)
        opt_state = optimizer.init(initial_K)
        K = initial_K
        history = TrainingHistory()

        dt = 0.02
        ts = jnp.arange(0.0, config.trajectory_length + dt/2, dt)

        if config.verbose:
            print("Starting training...")

        # 3. JIT-compiled cost+grad (batch & stability penalty)
        loss_fn = _make_loss_fn(
            ts, init_batch, Q, config.control_weight,
            params, config.stability_weight
        )
        value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

        # Initial cost
        initial_cost = loss_fn(K, init_batch)
        if config.verbose:
            print(f"Initial cost: {initial_cost}")

        if not jnp.isfinite(initial_cost):
            print("Invalid initial cost")
            return LinearController(K=initial_K), TrainingHistory()

        log_interval = max(10, config.num_iterations // 10)
        start_time = time.time()
        rng = key

        for i in range(config.num_iterations):
            rng, subkey = random.split(rng)
            # refresh perturbations each step (same mapping helper)
            noise4 = config.perturb_std * random.normal(subkey, (config.batch_size, 4))
            batch4 = base4 + noise4
            batch_states = jax.vmap(_four_to_five)(batch4)

            cost, grads = value_and_grad(K, batch_states)
            updates, opt_state = optimizer.update(grads, opt_state)
            K = optax.apply_updates(K, updates)

            history.update(cost, K)

            if config.verbose and i % log_interval == 0:
                elapsed = time.time() - start_time
                print(f"Iteration {i:3d}: Cost = {cost:.6f} | Elapsed: {elapsed:.2f}s")

            if not jnp.isfinite(cost):
                if config.verbose:
                    print(f"Training diverged at iteration {i}")
                break

        final_controller = LinearController(K=K)

        if config.verbose and len(history.costs) > 0:
            print("Training completed!")

        return final_controller, history

    except Exception as e:
        print(f"Training failed: {e}")
        return LinearController(K=initial_K), TrainingHistory()


def grid_search_linear_gains(
    initial_state: jnp.ndarray,
    gain_ranges: tuple[tuple[float, float], ...] = None,
    n_points: int = 5,
    params: CartPoleParams = CartPoleParams()
) -> LinearController:
    """Grid search for optimal initial gains."""
    if gain_ranges is None:
        # Default ranges for [x, cos(θ), sin(θ), ẋ, θ̇]
        gain_ranges = (
            (0.5, 2.0),      # x position
            (-25.0, -5.0),   # cos(θ) - negative
            (5.0, 25.0),     # sin(θ) - positive
            (0.5, 3.0),      # x velocity
            (0.5, 3.0)       # angular velocity
        )
    
    best_cost = float('inf')
    best_K = None
    Q = create_cost_matrices()
    R = 0.1
    
    # Pre-compute time grid for cost evaluation
    dt = 0.02
    ts = jnp.arange(0.0, 3.0 + dt/2, dt)
    cost_fn = _make_loss_fn(ts, initial_state, Q, R, params)
    
    total_points = n_points ** len(gain_ranges)
    print(f"Grid search over {total_points:,} combinations...")
    
    # Create parameter grid
    ranges = [jnp.linspace(start, end, n_points) for start, end in gain_ranges]
    grids = jnp.meshgrid(*ranges, indexing='ij')
    K_candidates = jnp.stack([grid.ravel() for grid in grids], axis=1)
    
    stable_count = 0
    
    for i, K in enumerate(K_candidates):
        # Quick stability test
        try:
            controller_test = LinearController(K=jnp.array(K))
            dt_test = 0.05
            ts_test = jnp.arange(0.0, 2.0 + dt_test, dt_test)
            sol = simulate(controller_test, params, (0.0, 2.0), ts_test, initial_state)
            
            from lib.stability import check_trajectory_bounds
            if not check_trajectory_bounds(sol.ys):
                continue
                
        except Exception:
            continue
        
        stable_count += 1
        
        # Compute cost for stable candidates
        try:
            cost = cost_fn(jnp.array(K))
            
            if cost < best_cost:
                best_cost = float(cost)
                best_K = jnp.array(K)
                
        except Exception:
            continue
        
        # Progress update
        if (i + 1) % max(1, total_points // 10) == 0:
            progress = (i + 1) / total_points * 100
            print(f"Progress: {progress:.0f}% ({stable_count} stable)")
    
    if best_K is None:
        print("No stable gains found, using defaults")
        best_K = jnp.array([1.0, -10.0, 10.0, 1.0, 1.0])
    else:
        print(f"Best gains: {best_K}, Cost: {best_cost:.4f}")
        print(f"Stable: {stable_count}/{total_points}")
    
    return LinearController(K=best_K)


def validate_linear_training_setup(
    initial_K: jnp.ndarray,
    initial_state: jnp.ndarray,
    config: LinearTrainingConfig
) -> None:
    """Validate training setup."""
    if initial_K.shape != (5,):
        raise ValueError(f"K must have shape (5,), got {initial_K.shape}")
    
    if initial_state.shape != (5,):
        raise ValueError(f"State must have shape (5,), got {initial_state.shape}")
