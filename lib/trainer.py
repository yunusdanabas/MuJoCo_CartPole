"""
lib/trainer.py
Generic batched trainer for cart-pole controllers.

Uses Diffrax roll-outs with our fast env.dynamics.
Handles three training modes:
1. Gradient-based (neural net + optax optimiser)
2. Gradient-free (callable update_fn for evolutionary strategies)  
3. Pure evaluation loop (controllers with no learnable params)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import time

from env.closedloop import simulate_batch  # Changed from simulate_batch
from env.cartpole import CartPoleParams
from env.helpers import total_energy

# --------------------------------------------------------------------------- #
# Config dataclass                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class TrainConfig:
    batch_size: int = 128
    t_span: tuple[float, float] = (0.0, 2.0)
    ts: jnp.ndarray | None = None               # if None → automatically linspace
    learning_rate: float = 1e-3
    num_epochs: int = 1_000
    seed: int = 0
    print_data: bool = False

# --------------------------------------------------------------------------- #
# Loss functions                                                               #
# --------------------------------------------------------------------------- #

def default_loss(ys, params: CartPoleParams):
    """Quadratic cost to keep cart near 0 and pole near upright."""
    x = ys[..., 0]
    cos_th, sin_th = ys[..., 1], ys[..., 2]
    th = jnp.arctan2(sin_th, cos_th)
    return jnp.mean(x**2 + 10.0 * th**2)

def energy_loss(ys, params: CartPoleParams):
    """Energy-based loss for swing-up tasks."""
    mc, mp, l, g = params.mc, params.mp, params.l, params.g
    target_energy = mp * g * l  # Energy at upright position
    
    # Compute energy for each state
    energies = jax.vmap(lambda state: total_energy(state, params))(ys.reshape(-1, ys.shape[-1]))
    energies = energies.reshape(ys.shape[:-1])
    
    # Penalize deviation from target energy
    energy_error = jnp.mean((energies - target_energy)**2)
    
    # Also penalize position deviation
    position_error = jnp.mean(ys[..., 0]**2)
    
    return energy_error + 0.1 * position_error

def combined_loss(ys, params: CartPoleParams):
    """Combined stabilization and energy loss (works with 5-state)."""
    x = ys[..., 0]
    cos_th, sin_th = ys[..., 1], ys[..., 2]
    th = jnp.arctan2(sin_th, cos_th)
    
    # Stabilization loss (stronger near upright)
    upright_weight = jnp.exp(-th**2)  # Higher weight when θ ≈ 0
    stab_loss = jnp.mean(upright_weight * (x**2 + 100.0 * th**2))
    
    # Energy loss (for swing-up)
    eng_loss = energy_loss(ys, params)
    
    return 0.7 * stab_loss + 0.3 * eng_loss

# --------------------------------------------------------------------------- #
# Training state                                                               #
# --------------------------------------------------------------------------- #

class TrainState(NamedTuple):
    controller: Any                # full controller object (not JIT-able)
    ctrl_fn: Any                   # jit-compiled callable for batching
    opt_state: optax.OptState | None
    key: jax.random.KeyArray

# --------------------------------------------------------------------------- #
# Initial condition samplers                                                   #
# --------------------------------------------------------------------------- #

def random_initial_conditions(key, batch_size):
    """Sample random initial conditions."""
    return jax.random.uniform(
        key, (batch_size, 4),
        minval=jnp.array([-2.0, -jnp.pi, -1.0, -1.0]),
        maxval=jnp.array([2.0, jnp.pi, 1.0, 1.0])
    )

def downward_initial_conditions(key, batch_size):
    """Sample initial conditions near downward position."""
    base = jnp.array([0., jnp.pi, 0., 0.])
    noise = 0.1 * jax.random.normal(key, (batch_size, 4))
    return base + noise

def upright_initial_conditions(key, batch_size):
    """Sample initial conditions near upright position."""
    base = jnp.array([0., 0., 0., 0.])
    noise = 0.1 * jax.random.normal(key, (batch_size, 4))
    return base + noise

# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def train(controller,
          params: CartPoleParams = CartPoleParams(),
          cfg: TrainConfig = TrainConfig(),
          loss_fn: Callable = default_loss,
          optimiser: optax.GradientTransformation | None = None,
          init_state_fn: Callable[[jax.random.KeyArray, int], jnp.ndarray] | None = None,
          *,
          print_data: bool = True):
    """
    Main training entry point.

    Arguments
    ---------
    controller
        Any object with __call__(state, t) and either:
        • eqx.is_array_like leaves (for gradient-based), or  
        • an external optimiser/update_fn handles updates
    optimiser
        optax optimiser; if None, runs in evaluation-only mode
    init_state_fn(key, batch) -> (batch, state_dim)
        Custom initial-condition sampler
    """
    # Setup configuration
    if not isinstance(cfg, TrainConfig):
        cfg = TrainConfig(**cfg.__dict__)

    # Sync logging flag
    cfg.print_data = bool(print_data)
    
    ts = cfg.ts if cfg.ts is not None else jnp.linspace(cfg.t_span[0], cfg.t_span[1], 201)
    
    # Setup optimiser
    if optimiser is None:
        optimiser = optax.identity()  # No-op optimiser for evaluation mode
    
    # Check if controller is trainable
    trainable_params = eqx.filter(controller, eqx.is_array_like)
    is_trainable = jax.tree_util.tree_leaves(trainable_params) != []
    
    if not is_trainable and optimiser is not optax.identity():
        if cfg.print_data:
            pass  # avoid extra prints beyond required format
        optimiser = optax.identity()
    
    opt_state = optimiser.init(trainable_params)
    key = jax.random.PRNGKey(cfg.seed)
    
    # Default initial condition sampler
    if init_state_fn is None:
        init_state_fn = downward_initial_conditions
    
    # ---------- JIT-compiled training step -------------------------------- #
    def _train_step(state: TrainState):
        ctrl, ctrl_fn, opt_state, key = state
        key, sub = jax.random.split(key)
        
        # Sample initial conditions
        init_states = init_state_fn(sub, cfg.batch_size)
        
        if is_trainable:
            # Gradient-based update
            def loss_for_grad(c):
                sol = simulate_batch(c.batched(), params, cfg.t_span, ts, init_states)
                return loss_fn(sol.ys, params)
            
            loss, grads = eqx.filter_value_and_grad(loss_for_grad)(ctrl)
            
            # Apply updates only to trainable parameters
            trainable_grads = eqx.filter(grads, eqx.is_array_like)
            updates, opt_state = optimiser.update(trainable_grads, opt_state, 
                                                 eqx.filter(ctrl, eqx.is_array_like))
            ctrl = eqx.apply_updates(ctrl, updates)
            ctrl_fn = ctrl.batched()
        else:
            # Evaluation only
            sol = simulate_batch(ctrl_fn, params, cfg.t_span, ts, init_states)
            loss = loss_fn(sol.ys, params)
        
        return TrainState(ctrl, ctrl_fn, opt_state, key), loss
    
    # ---------- Main training loop ---------------------------------------- #
    ctrl_fn = controller.jit().batched()
    state = TrainState(controller, ctrl_fn, opt_state, key)
    loss_history = []
    
    ctrl_name = type(controller).__name__
    start_total = time.perf_counter()
    if cfg.print_data:
        print(f"[TRAIN] {ctrl_name} started")
    
    for epoch in range(cfg.num_epochs):
        iter_start = time.perf_counter()
        state, loss = _train_step(state)
        iter_time = time.perf_counter() - iter_start
        loss_val = float(loss)
        loss_history.append(loss_val)
        
        if cfg.print_data:
            print(f"[TRAIN] iter={epoch} time={iter_time:.6f}s loss={loss_val:.6f}")
    
    total_time = time.perf_counter() - start_total
    if cfg.print_data:
        print(f"[TRAIN] {ctrl_name} finished in {total_time:.6f}s")
    
    return state.controller, jnp.array(loss_history)


# --------------------------------------------------------------------------- #
# Evaluation utilities                                                         #
# --------------------------------------------------------------------------- #

def evaluate(controller, 
             params: CartPoleParams = CartPoleParams(),
             initial_states: jnp.ndarray | None = None,
             t_span: tuple[float, float] = (0.0, 3.0),
             loss_fn: Callable = default_loss):
    """Evaluate controller performance on given initial conditions."""
    if initial_states is None:
        key = jax.random.PRNGKey(0)
        initial_states = random_initial_conditions(key, 32)
    
    ts = jnp.linspace(t_span[0], t_span[1], 301)
    sol = simulate_batch(controller.jit().batched(), params, t_span, ts, initial_states)
    
    loss = loss_fn(sol.ys, params)
    
    # Additional metrics
    cos_th_f, sin_th_f = sol.ys[:, -1, 1], sol.ys[:, -1, 2]
    th_f = jnp.arctan2(sin_th_f, cos_th_f)
    final_angle_error = jnp.mean(jnp.abs(th_f))     # |θ| at final time
    max_cart_position = jnp.max(jnp.abs(sol.ys[:, :, 0]))    # Max cart displacement
    
    return {
        'loss': float(loss),
        'final_angle_error': float(final_angle_error),
        'max_cart_position': float(max_cart_position),
        'solution': sol
    }


if __name__ == "__main__":
    # Quick test
    from controller.linear_controller import LinearController
    
    ctrl = LinearController(K=jnp.array([10., 0., 5., 0.]))
    config = TrainConfig(num_epochs=100, batch_size=32, print_data=True)
    
    trained_ctrl, history = train(ctrl, cfg=config)