"""
lib/cost_functions.py
Cost computation utilities for cart-pole controllers.
"""

from __future__ import annotations
import jax.numpy as jnp


def create_cost_matrices(
    pos_weight: float = 1.0,
    cos_theta_weight: float = 10.0,
    sin_theta_weight: float = 10.0, 
    vel_weight: float = 0.1,
    angvel_weight: float = 0.1
) -> jnp.ndarray:
    """Create state cost matrix Q for [x, cos(θ), sin(θ), ẋ, θ̇]."""
    return jnp.diag(jnp.array([
        pos_weight,        # x position
        cos_theta_weight,  # cos(θ) - want close to 1
        sin_theta_weight,  # sin(θ) - want close to 0
        vel_weight,        # ẋ velocity
        angvel_weight      # θ̇ angular velocity
    ]))


def compute_trajectory_cost(
    trajectory: jnp.ndarray,
    forces: jnp.ndarray,
    Q: jnp.ndarray,
    R: float,
    dt: float
) -> float:
    """Compute LQR-style trajectory cost."""
    if trajectory.shape[0] == 0:
        return jnp.inf
        
    if trajectory.shape[1] != 5:
        raise ValueError(f"Trajectory must have shape (N, 5), got {trajectory.shape}")
        
    if forces.shape[0] != trajectory.shape[0]:
        raise ValueError("Forces and trajectory length mismatch")
    
    # Target state: upright and stationary
    target_state = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0])
    
    # State errors
    state_errors = trajectory - target_state
    
    # Costs
    state_costs = jnp.sum(state_errors * (Q @ state_errors.T).T, axis=1)
    control_costs = R * forces**2
    
    # Integrate over time
    return dt * jnp.sum(state_costs + control_costs)


def create_lqr_matrices(
    pos_weight: float = 1.0,
    angle_weight: float = 10.0, 
    vel_weight: float = 0.1,
    control_weight: float = 0.1
) -> tuple[jnp.ndarray, float]:
    """Create LQR cost matrices."""
    Q = create_cost_matrices(
        pos_weight=pos_weight,
        cos_theta_weight=angle_weight,
        sin_theta_weight=angle_weight,
        vel_weight=vel_weight,
        angvel_weight=vel_weight
    )
    
    return Q, control_weight