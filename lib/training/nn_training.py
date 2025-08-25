"""
lib/training/nn_training.py
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
    grad_clip: float = 1.0  # gradient clipping norm

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
    # Compute target energy at upright position using helper function
    upright_state = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0])  # [x, cos0, sin0, ẋ, θ̇]
    target_energy = total_energy(upright_state, params)  # Use helper for consistency
    
    # Compute energy for each state
    energies = jax.vmap(lambda state: total_energy(state, params))(ys.reshape(-1, ys.shape[-1]))
    energies = energies.reshape(ys.shape[:-1])
    
    # Strong energy penalty for swing-up
    energy_error = jnp.mean((energies - target_energy)**2)
    
    # Weaker position penalty - let cart move during swing-up
    position_error = 0.01 * jnp.mean(ys[..., 0]**2)
    
    return energy_error + position_error

def swingup_loss(ys, params: CartPoleParams):
    """Improved swing-up focused loss with better energy and momentum guidance."""
    # Target energy at upright
    upright_state = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0])
    target_energy = total_energy(upright_state, params)
    
    # Current energies
    energies = jax.vmap(lambda state: total_energy(state, params))(ys.reshape(-1, ys.shape[-1]))
    energies = energies.reshape(ys.shape[:-1])
    
    # Energy loss - main objective for swing-up
    energy_error = jnp.mean((energies - target_energy)**2)
    
    # Extract state components
    cos_th, sin_th = ys[..., 1], ys[..., 2]
    th = jnp.arctan2(sin_th, cos_th)
    thdot = ys[..., 4]
    xdot = ys[..., 3]
    
    # Phase-aware momentum guidance:
    # When pole is in lower half (cos_th < 0), encourage positive angular velocity
    # When pole is in upper half (cos_th > 0), encourage small angular velocity for stabilization
    lower_half_mask = cos_th < 0
    upper_half_mask = cos_th > 0
    
    # Momentum guidance: encourage θ̇ > 0 in lower half, small θ̇ in upper half
    momentum_loss = jnp.mean(
        jnp.where(lower_half_mask, 
                  jnp.maximum(0.0, -thdot),  # Penalize negative θ̇ in lower half
                  jnp.abs(thdot))             # Penalize large θ̇ in upper half
    )
    
    # Cart movement guidance: encourage cart movement during swing-up, minimize during stabilization
    cart_movement = jnp.mean(
        jnp.where(lower_half_mask,
                 0.0,                        # No penalty for cart movement during swing-up
                 jnp.abs(xdot))              # Penalize cart movement during stabilization
    )
    
    # Angle-based position penalty: stronger penalty when near upright
    angle_error = jnp.mean(jnp.where(upper_half_mask, th**2, 0.1 * th**2))
    
    # Cart position penalty: minimal during swing-up, stronger during stabilization
    cart_position = jnp.mean(
        jnp.where(upper_half_mask,
                 ys[..., 0]**2,              # Strong cart position penalty when stabilizing
                 0.01 * ys[..., 0]**2)       # Weak penalty during swing-up
    )
    
    # Combine losses with appropriate weights
    total_loss = (
        1.0 * energy_error +      # Primary objective
        0.5 * momentum_loss +     # Momentum guidance
        0.1 * cart_movement +     # Cart movement guidance
        0.3 * angle_error +       # Angle stabilization
        0.2 * cart_position       # Cart position control
    )
    
    return total_loss

def combined_loss(ys, params: CartPoleParams):
    """Improved combined loss with better balance between swing-up and stabilization."""
    x = ys[..., 0]
    cos_th, sin_th = ys[..., 1], ys[..., 2]
    th = jnp.arctan2(sin_th, cos_th)
    
    # Phase-aware weighting: use angle to determine if we're in swing-up or stabilization phase
    # When |θ| > π/4, we're in swing-up phase; when |θ| < π/4, we're stabilizing
    swingup_phase = jnp.abs(th) > jnp.pi/4
    stab_phase = ~swingup_phase
    
    # Stabilization loss (stronger near upright)
    stab_loss = jnp.mean(
        jnp.where(stab_phase,
                 x**2 + 50.0 * th**2,        # Strong stabilization when near upright
                 0.1 * (x**2 + th**2))       # Weak penalty during swing-up
    )
    
    # Energy loss for swing-up (only active during swing-up phase)
    eng_loss = energy_loss(ys, params)
    eng_loss = jnp.mean(jnp.where(swingup_phase, eng_loss, 0.0))
    
    # Combine with phase-aware weighting
    return 0.6 * stab_loss + 0.4 * eng_loss

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
    """Sample random initial conditions in 5D format."""
    # Sample 4D [x, θ, ẋ, θ̇] then convert to 5D [x, cosθ, sinθ, ẋ, θ̇]
    states_4d = jax.random.uniform(
        key, (batch_size, 4),
        minval=jnp.array([-2.0, -jnp.pi, -1.0, -1.0]),
        maxval=jnp.array([2.0, jnp.pi, 1.0, 1.0])
    )
    # Convert to 5D canonical state and ensure float32
    x, th, xdot, thdot = jnp.split(states_4d, 4, axis=-1)
    return jnp.concatenate([x, jnp.cos(th), jnp.sin(th), xdot, thdot], axis=-1).astype(jnp.float32)

def downward_initial_conditions(key, batch_size):
    """Improved initial conditions for swing-up training with better variety."""
    # Base: [x, θ, ẋ, θ̇] = [0, π, 0, 0] → [x, cosπ, sinπ, ẋ, θ̇] = [0, -1, 0, 0, 0]
    base_4d = jnp.array([0., jnp.pi, 0., 0.])
    
    # Split key for different noise components
    key1, key2, key3 = jax.random.split(key, 3)
    
    # Position noise: allow some cart position variation
    x_noise = jax.random.normal(key1, (batch_size,)) * 0.5  # ±0.5 cart position
    
    # Angle noise: focus on lower half but allow some variation
    # Most samples near π (downward), some near π/2 and 3π/2
    angle_noise = jax.random.normal(key2, (batch_size,)) * 0.3  # ±0.3 radian
    angles = jnp.pi + angle_noise
    # Ensure angles stay in lower half [π/2, 3π/2]
    angles = jnp.clip(angles, jnp.pi/2, 3*jnp.pi/2)
    
    # Velocity noise: more aggressive for better swing-up training
    # Allow both positive and negative initial velocities
    xdot_noise = jax.random.normal(key3, (batch_size,)) * 1.0  # ±1.0 cart velocity
    thdot_noise = jax.random.normal(key3, (batch_size,)) * 0.8  # ±0.8 angular velocity
    
    # Combine all components
    states_4d = jnp.stack([
        x_noise,           # x position
        angles,            # θ angle
        xdot_noise,        # ẋ velocity
        thdot_noise        # θ̇ velocity
    ], axis=1)
    
    # Convert to 5D canonical state and ensure float32
    x, th, xdot, thdot = jnp.split(states_4d, 4, axis=-1)
    return jnp.concatenate([x, jnp.cos(th), jnp.sin(th), xdot, thdot], axis=-1).astype(jnp.float32)

def upright_initial_conditions(key, batch_size):
    """Sample initial conditions near upright position in 5D format."""
    # Base: [x, θ, ẋ, θ̇] = [0, 0, 0, 0] → [x, cos0, sin0, ẋ, θ̇] = [0, 1, 0, 0, 0]
    base_4d = jnp.array([0., 0., 0., 0.])
    noise_4d = 0.1 * jax.random.normal(key, (batch_size, 4))
    states_4d = base_4d + noise_4d
    # Convert to 5D canonical state and ensure float32
    x, th, xdot, thdot = jnp.split(states_4d, 4, axis=-1)
    return jnp.concatenate([x, jnp.cos(th), jnp.sin(th), xdot, thdot], axis=-1).astype(jnp.float32)

# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def train(controller,
          params: CartPoleParams = CartPoleParams(),
          cfg: TrainConfig = TrainConfig(),
          loss_fn: Callable | str = "default_loss",
          optimiser: optax.GradientTransformation | None = None,
          init_state_fn: Callable[[jax.random.KeyArray, int], jnp.ndarray] | str | None = None,
          *,
          print_data: bool = True):
    """
    Main training entry point.
    
    JIT optimization:
    - The per-epoch training step is JIT-compiled for speed.
    - The loss function for gradients is also JITted.
    - All controller calls are JITted for fast simulation.
    """
    # Setup configuration
    if not isinstance(cfg, TrainConfig):
        cfg = TrainConfig(**cfg.__dict__)

    # Sync logging flag
    cfg.print_data = bool(print_data)
    
    ts = cfg.ts if cfg.ts is not None else jnp.linspace(cfg.t_span[0], cfg.t_span[1], 201)
    
    # Handle string-based loss function
    if isinstance(loss_fn, str):
        loss_fn_map = {
            "default_loss": default_loss,
            "energy_loss": energy_loss,
            "swingup_loss": swingup_loss,  # Add the improved swing-up loss
            "combined_loss": combined_loss
        }
        if loss_fn not in loss_fn_map:
            raise ValueError(f"Unknown loss function: {loss_fn}. Available: {list(loss_fn_map.keys())}")
        loss_fn = loss_fn_map[loss_fn]
    
    # Setup optimiser
    if optimiser is None:
        # Default optimizer with better parameters for stability
        optimiser = optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip),  # Gradient clipping for stability
            optax.adam(learning_rate=cfg.learning_rate, b1=0.9, b2=0.999),
            optax.add_decayed_weights(1e-4)            # Weight decay for regularization
        )
    
    # Check if controller is trainable
    trainable_params = eqx.filter(controller, eqx.is_array_like)
    is_trainable = jax.tree_util.tree_leaves(trainable_params) != []
    
    if not is_trainable and optimiser is not optax.identity():
        if cfg.print_data:
            pass  # avoid extra prints beyond required format
        optimiser = optax.identity()
    
    opt_state = optimiser.init(trainable_params)
    key = jax.random.PRNGKey(cfg.seed)
    
    # Handle string-based initial condition sampler
    if isinstance(init_state_fn, str):
        init_state_map = {
            "random_initial_conditions": random_initial_conditions,
            "downward_initial_conditions": downward_initial_conditions,
            "upright_initial_conditions": upright_initial_conditions
        }
        if init_state_fn not in init_state_map:
            raise ValueError(f"Unknown initial condition function: {init_state_fn}. Available: {list(init_state_map.keys())}")
        init_state_fn = init_state_map[init_state_fn]
    elif init_state_fn is None:
        init_state_fn = downward_initial_conditions
    
    # ---------- JIT-compiled training step -------------------------------- #
    # Do NOT JIT _train_step itself, because TrainState contains non-array objects (controller, ctrl_fn)
    def _train_step(state: TrainState):
        ctrl, ctrl_fn, opt_state, key = state
        key, sub = jax.random.split(key)
        
        # Sample initial conditions
        init_states = init_state_fn(sub, cfg.batch_size)
        
        if is_trainable:
            # Gradient-based update
            @jax.jit
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
        
        if cfg.print_data and (epoch % 50 == 0 or epoch == cfg.num_epochs - 1):
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