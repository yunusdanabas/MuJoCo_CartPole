"""
env/cartpole.py - JAX cart-pole dynamics
State format: [x, cos(θ), sin(θ), ẋ, θ̇]
"""

from __future__ import annotations
from functools import partial
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from env.helpers import inverse_mass_matrix, total_energy


@dataclass(frozen=True)
class CartPoleParams:
    """Cart-pole physical parameters"""
    mc: float = 1.0  # Cart mass
    mp: float = 1.0  # Pole mass
    l : float = 1.0  # Pole length
    g : float = 9.81 # Gravity


# Dynamics function cache for performance
_dyn_cache: dict[tuple[int, int], callable] = {}

def _get_cached_dynamics(params: CartPoleParams, controller):
    """Get cached JIT-compiled dynamics function"""
    key = (id(params), id(controller))          # params AND controller
    
    if key not in _dyn_cache:
        def _cached_fn(state, t):
            return _dynamics_single(state, t, params=params, controller=controller)
        _dyn_cache[key] = _cached_fn
    
    return _dyn_cache[key]


@partial(jax.jit, static_argnames=("params", "controller"))
def _dynamics_single(
    state: Float[Array, "5"],
    t: float,
    *,
    params: CartPoleParams,
    controller,
) -> Float[Array, "5"]:
    """Core dynamics: compute state derivatives for single state vector"""
    
    x, cos_theta, sin_theta, xdot, thdot = state
    mp, l, g = params.mp, params.l, params.g

    # Get control force
    force = controller(state, t)

    # Solve for accelerations using mass matrix inverse
    m11, m12, m22, _ = inverse_mass_matrix(cos_theta, params)
    rhs1 = force
    rhs2 = mp * g * l * sin_theta - mp * l * sin_theta * thdot * thdot
    xddot = m11 * rhs1 + m12 * rhs2
    thddot = m12 * rhs1 + m22 * rhs2

    # Trigonometric derivatives: d/dt[cos(θ)] = -sin(θ)θ̇, d/dt[sin(θ)] = cos(θ)θ̇
    cos_theta_dot = -sin_theta * thdot
    sin_theta_dot = cos_theta * thdot
    
    return jnp.array([xdot, cos_theta_dot, sin_theta_dot, xddot, thddot])


def dynamics(
    state: Float[Array, "... 5"],
    t: float | Array,
    params: CartPoleParams = CartPoleParams(),
    controller=lambda s, t: 0.0,
) -> Float[Array, "... 5"]:
    """
    Compute cart-pole dynamics derivatives.
    
    Args:
        state: State vector(s) [x, cos(θ), sin(θ), ẋ, θ̇]
        t: Time
        params: Physical parameters
        controller: Control function (state, time) -> force
        
    Returns:
        State derivatives [ẋ, -sin(θ)θ̇, cos(θ)θ̇, ẍ, θ̈]
    """
    # Validate input format
    if state.shape[-1] != 5:
        raise ValueError(f"Expected state format [x, cos(θ), sin(θ), ẋ, θ̇], got shape {state.shape}")
    
    # Handle batched vs single state
    dynamics_fn = _get_cached_dynamics(params, controller)
    
    if state.ndim > 1:
        return jax.vmap(dynamics_fn, in_axes=(0, None))(state, t)
    else:
        return dynamics_fn(state, t)


@partial(jax.jit, static_argnames=("params", "controller"))
def batch_dynamics(
    states: Float[Array, "batch 5"],
    t: float,
    params: CartPoleParams = CartPoleParams(),
    controller=lambda s, t: 0.0,
) -> Float[Array, "batch 5"]:
    """Vectorized dynamics for batch of state vectors"""
    if states.shape[-1] != 5:
        raise ValueError(f"Expected state format [x, cos(θ), sin(θ), ẋ, θ̇], got shape {states.shape}")
    
    dynamics_fn = _get_cached_dynamics(params, controller)
    return jax.vmap(dynamics_fn, in_axes=(0, None))(states, t)


def compute_energy(
    state: Float[Array, "... 5"],
    params: CartPoleParams = CartPoleParams()
) -> Float[Array, "..."]:
    """Compute total energy for state vector(s)"""
    if state.shape[-1] != 5:
        raise ValueError(f"Expected state format [x, cos(θ), sin(θ), ẋ, θ̇], got shape {state.shape}")
    
    if state.ndim > 1:
        return jax.vmap(total_energy, in_axes=(0, None))(state, params)
    else:
        return total_energy(state, params)

