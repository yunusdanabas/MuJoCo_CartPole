# lib/trainer.py
# # Description: This module contains the training loop for the neural network controller.
# It includes functions for training the controller, computing the loss, and evaluating the controller's performance.

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from functools import partial
from typing import Tuple, Callable
from lib.utils import *
from env.closedloop import simulate_closed_loop_nn

def train_nn_controller(
    controller: eqx.Module,
    params_system: Tuple[float, float, float, float],
    Q: jnp.ndarray,
    num_epochs: int,
    batch_size: int,
    t_span: Tuple[float, float],
    t_eval: jnp.ndarray,
    key: jax.random.PRNGKey,
    learning_rate: float = 3e-4,
    grad_clip: float = 1.0,
    lr_schedule: Optional[Callable] = None  # Add default value
) -> Tuple[eqx.Module, jnp.ndarray]:
    """
    Train neural network controller using energy-based loss and trajectory optimization.
    
    Args:
        controller: Initial neural network controller
        params_system: Tuple (mc, mp, l, g)
        Q: State cost matrix (5x5 for 5D state)
        optimizer: Optax optimizer
        num_epochs: Number of training epochs
        batch_size: Number of trajectories per batch
        t_span: Simulation time range (t0, t1)
        t_eval: Array of evaluation time points
        key: JAX random key
        lr_schedule: Optional learning rate schedule
        grad_clip: Gradient clipping value
        
    Returns:
        trained_controller: Optimized neural network controller
        loss_history: Array of loss values during training
    """
    # Initialize optimizer and learning rate
    if lr_schedule is None:
        lr_schedule = optax.constant_schedule(3e-4)
    # Initialize optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(learning_rate)
    )
    opt_state = optimizer.init(eqx.filter(controller, eqx.is_array))
    
    # Pre-compile critical functions
    simulate_fn = partial(
        simulate_closed_loop_nn,
        params=params_system,
        t_span=t_span,
        t=t_eval
    )
    loss_fn = partial(_compute_loss, Q=Q, params=params_system, simulate_fn=simulate_fn)
    
    @partial(jax.jit, static_argnames=["simulate_fn"])
    def update_step(controller, opt_state, batch_states, simulate_fn):
        loss, grads = jax.value_and_grad(loss_fn)(controller, batch_states)
        updates, opt_state = optimizer.update(grads, opt_state)
        controller = eqx.apply_updates(controller, updates)
        return controller, opt_state, loss
        
    loss_history = []
    for epoch in range(num_epochs):
        key, subkey = jax.random.split(key)
        
        # Sample and convert initial conditions
        batch_4d = sample_initial_conditions(
            batch_size, 
            x_range=(-2.0, 2.0),
            theta_range=(-jnp.pi, jnp.pi),
            key=subkey
        )
        batch_5d = jax.vmap(convert_4d_to_5d)(batch_4d)
        
        # Update model parameters
        controller, opt_state, loss = update_step(controller, opt_state, batch_5d, simulate_fn)
        loss_history.append(loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")
            
    return controller, jnp.array(loss_history)

def _compute_loss(
    controller: eqx.Module,
    batch_states: jnp.ndarray,
    Q: jnp.ndarray,
    params: Tuple[float, float, float, float],
    simulate_fn: Callable
) -> float:
    """
    Combined loss function for controller training.
    
    Components:
    1. Energy deviation from desired upright energy
    2. State regulation cost (x^T Q x)
    3. Control effort penalty (f^2)
    """
    mc, mp, l, g = params
    
    def per_trajectory_loss(state_5d):
        # Simulate trajectory
        sol = simulate_fn(controller=controller, initial_state=state_5d)
        states = sol.ys
        ts = sol.ts
        
        # Energy calculation
        current_energy = _compute_energy_nn(states, params)
        desired_energy = 2 * mp * g * l  # Energy for upright position
        energy_loss = jnp.mean((current_energy - desired_energy)**2)
        
        # State regulation cost
        state_cost = jnp.mean(jnp.einsum('ni,ij,nj->n', states, Q, states))
        
        # Control effort
        forces = jax.vmap(controller)(states, ts)
        control_cost = jnp.mean(forces**2)
        
        return energy_loss + state_cost + 0.1 * control_cost
    
    # Vectorize over batch dimension
    batch_loss = jnp.mean(jax.vmap(per_trajectory_loss)(batch_states))
    return batch_loss

def _compute_energy_nn(
    states: jnp.ndarray,
    params: Tuple[float, float, float, float]
) -> jnp.ndarray:
    """
    Compute total energy for NN states (5D format).
    
    Args:
        states: Array of shape (T, 5) [x, cosθ, sinθ, ẋ, θ̇]
        params: Tuple (mc, mp, l, g)
        
    Returns:
        energy: Array of shape (T,) containing total energy at each timestep
    """
    mc, mp, l, g = params
    x, cosθ, sinθ, ẋ, θ̇ = states.T
    
    # Kinetic energy components
    trans_energy = 0.5 * (mc + mp) * ẋ**2
    rot_energy = 0.5 * mp * l**2 * θ̇**2
    coriolis = -mp * l * cosθ * ẋ * θ̇
    
    # Potential energy
    potential = mp * g * l * (1 - cosθ)
    
    return trans_energy + rot_energy + coriolis + potential

def evaluate_controller(
    controller: eqx.Module,
    params_system: Tuple[float, float, float, float],
    initial_state: jnp.ndarray,
    t_span: Tuple[float, float],
    t_eval: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Evaluate controller performance from specific initial condition.
    
    Returns:
        ts: Time points (N,)
        states: State trajectory (N, 4) [x, θ, ẋ, θ̇]
    """
    # Convert to NN state format
    initial_state_5d = convert_4d_to_5d(initial_state)
    
    # Run simulation
    sol = simulate_closed_loop_nn(
        controller=controller,
        params=params_system,
        t_span=t_span,
        t=t_eval,
        initial_state=initial_state_5d
    )
    
    # Convert back to 4D states for analysis
    states_4d = jax.vmap(convert_5d_to_4d)(sol.ys)
    return sol.ts, states_4d