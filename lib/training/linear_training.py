"""
training/linear_training.py

Linear controller training using gradient-based optimization.
"""

from __future__ import annotations
from dataclasses import dataclass
import time  # Add this at the top with other imports

import jax
import jax.numpy as jnp
import optax
from functools import partial

from controller.linear_controller import LinearController
from env.closedloop import simulate
from env.cartpole import CartPoleParams
from lib.cost_functions import create_cost_matrices, compute_trajectory_cost
from lib.stability import quick_stability_check
from lib.training_utils import BaseTrainingConfig, TrainingHistory


@dataclass
class LinearTrainingConfig(BaseTrainingConfig):
    """Linear controller training configuration."""
    state_weight: float = 1.0
    control_weight: float = 0.1
    optimizer: str = 'adam'


@jax.jit
def _trajectory_cost_impl(trajectory, K, Q, R, dt):
    """Helper function to compute trajectory cost."""
    forces = -trajectory @ K
    target = jnp.array([0., 1., 0., 0., 0.])
    errs   = trajectory - target
    state_cost = jnp.einsum('ij,jk,ik->i', errs, Q, errs)
    ctrl_cost  = R * forces**2
    return dt * jnp.sum(state_cost + ctrl_cost)



def _make_loss_fn(
    ts: jnp.ndarray,
    initial_state: jnp.ndarray,
    Q: jnp.ndarray,
    R: float,
    params: CartPoleParams,
):
    """Return a *non-jitted* closure suitable for jax.grad."""
    dt = float(ts[1] - ts[0])

    def loss(K):
        # build controller everytime so it's differentiable in K
        ctrl = LinearController(K=K).jit()
        sol  = simulate(ctrl, params, (ts[0], ts[-1]), ts, initial_state)

        bad = (
            (sol.ys is None) |
            (~jnp.all(jnp.isfinite(sol.ys))) |
            (sol.ys.shape[0] == 0)
        )
        return jnp.where(
            bad,
            jnp.inf,
            _trajectory_cost_impl(sol.ys, K, Q, R, dt)
        )

    return loss  # no jax.jit here; grad() will trace it just fine


def train_linear_controller(
    initial_K: jnp.ndarray,
    initial_state: jnp.ndarray,
    config: LinearTrainingConfig = LinearTrainingConfig(),
    Q: jnp.ndarray = None,
    params: CartPoleParams = CartPoleParams()
) -> tuple[LinearController, TrainingHistory]:
    """Train linear controller using gradient descent."""
    try:
        if Q is None:
            Q = create_cost_matrices()
        
        # Setup optimizer
        optimizers = {'adam': optax.adam, 'sgd': optax.sgd, 'rmsprop': optax.rmsprop}
        optimizer = optimizers[config.optimizer](config.learning_rate)
        opt_state = optimizer.init(initial_K)
        
        K = initial_K
        history = TrainingHistory()
        
        # Pre-compute time grid outside of JIT
        dt = 0.02
        ts = jnp.arange(0.0, config.trajectory_length + dt/2, dt)
        
        if config.verbose:
            print("Starting training...")
        
        # Create loss function with pre-computed time grid
        loss_fn = _make_loss_fn(ts, initial_state, Q, config.control_weight, params)
        grad_fn = jax.grad(loss_fn)  # no jit: shapes are fixed already
        
        # Initial cost
        initial_cost = loss_fn(K)
        if config.verbose:
            print(f"Initial cost: {initial_cost}")
        
        if not jnp.isfinite(initial_cost):
            print("Invalid initial cost")
            return LinearController(K=initial_K), TrainingHistory()
        
        # Training loop
        log_interval = max(10, config.num_iterations // 10)
        start_time = time.time()  # Start timing
        
        for i in range(config.num_iterations):
            grads = grad_fn(K)
            updates, opt_state = optimizer.update(grads, opt_state)
            K = optax.apply_updates(K, updates)
            
            cost = loss_fn(K)
            history.update(cost, K)
            
            if config.verbose and i % log_interval == 0:
                elapsed = time.time() - start_time
                print(f"Iteration {i:3d}: Cost = {cost:.6f} | Elapsed: {elapsed:.2f}s")
            
            if not jnp.isfinite(cost):
                if config.verbose:
                    print(f"Training diverged at iteration {i}")
                break
        
        # Final controller
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
