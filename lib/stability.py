"""
lib/stability.py

Stability checking utilities for controllers.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from env.cartpole import CartPoleParams


@jax.jit
def check_trajectory_bounds(
    trajectory: jnp.ndarray,
    position_limit: float = 2.0,
    angle_limit: float = 0.5,
    velocity_limit: float = 5.0,
    angular_velocity_limit: float = 10.0
) -> bool:
    """Check if trajectory stays within bounds."""
    if trajectory.shape[0] == 0:
        return False
        
    # Check bounds
    positions = trajectory[:, 0]
    cos_theta = trajectory[:, 1]
    sin_theta = trajectory[:, 2]
    angles = jnp.arctan2(sin_theta, cos_theta)
    velocities = trajectory[:, 3]
    angular_velocities = trajectory[:, 4]
    
    position_ok = jnp.all(jnp.abs(positions) < position_limit)
    angle_ok = jnp.all(jnp.abs(angles) < angle_limit)
    velocity_ok = jnp.all(jnp.abs(velocities) < velocity_limit)
    angular_velocity_ok = jnp.all(jnp.abs(angular_velocities) < angular_velocity_limit)
    
    return position_ok & angle_ok & velocity_ok & angular_velocity_ok


def quick_stability_check(
    controller,
    initial_state: jnp.ndarray,
    t_span: tuple[float, float] = (0.0, 2.0),
    params: CartPoleParams = CartPoleParams()
) -> bool:
    """Quick stability test for any controller."""
    try:
        from env.closedloop import simulate
        
        dt = 0.05
        ts = jnp.arange(t_span[0], t_span[1] + dt, dt)
        sol = simulate(controller, params, t_span, ts, initial_state)
        
        if not hasattr(sol, 'ys') or sol.ys is None or sol.ys.shape[0] == 0:
            return False
        
        return check_trajectory_bounds(sol.ys)
        
    except Exception:
        return False


def create_standard_test_states() -> jnp.ndarray:
    """Standard test states for stability testing."""
    return jnp.array([
        [0.0, 1.0, 0.0, 0.0, 0.0],      # Upright
        [0.1, 0.95, 0.31, 0.0, 0.0],    # Small perturbation
        [0.0, 0.0, 1.0, 0.0, 0.0],      # Horizontal
        [-0.2, 0.8, 0.6, 0.1, -0.1]     # Large perturbation with motion
    ])