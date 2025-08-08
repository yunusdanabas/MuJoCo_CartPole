"""
env/closedloop.py
Closed-loop cart-pole simulation using JAX + Diffrax.

State format: [x, cos(θ), sin(θ), ẋ, θ̇]
Provides efficient ODE integration with automatic JIT compilation.
"""

from __future__ import annotations
from functools import partial
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from diffrax import Tsit5, ODETerm, SaveAt, diffeqsolve

from .cartpole import CartPoleParams, dynamics


# --------------------------------------------------------------------------- #
# JIT Compilation Cache                                                       #
# --------------------------------------------------------------------------- #

# One ODETerm per parameter set; controller comes in via args
_rhs_cache: dict[int, ODETerm] = {}

def _get_rhs_term(params: CartPoleParams) -> ODETerm:
    """Return a cached ODETerm; no @jax.jit on the RHS."""
    key = id(params)
    if key not in _rhs_cache:
        def rhs(t, y, controller_fn):
            return dynamics(y, t, params=params, controller=controller_fn)
        _rhs_cache[key] = ODETerm(rhs)
    return _rhs_cache[key]


# --------------------------------------------------------------------------- #
# Single Trajectory Simulation                                               #
# --------------------------------------------------------------------------- #

def simulate(
    controller: Callable[[jnp.ndarray, float], float],
    params: CartPoleParams,
    t_span: tuple[float, float],
    ts: Sequence[float],
    y0: Float[Array, "5"],
    *,
    dt0: float = 1e-2,
    max_steps: int = 10_000,
):
    """
    Simulate cart-pole with given controller.
    
    Args:
        controller: Control function (state, time) -> force
        params: Physical parameters
        t_span: Integration time (t_start, t_end)
        ts: Time points to save solution
        y0: Initial state [x, cos(θ), sin(θ), ẋ, θ̇]
        dt0: Initial step size
        max_steps: Maximum integration steps
        
    Returns:
        Diffrax solution object
    """
    if y0.shape[-1] != 5:
        raise ValueError(f"State must have shape (..., 5), got {y0.shape}")
    
    term = _get_rhs_term(params)
    controller_fn = lambda y, t: controller(y.at[1].add(-1.0), t)

    sol = diffeqsolve(
        term,
        Tsit5(),
        t0=t_span[0],
        t1=t_span[1],
        dt0=dt0,
        y0=y0,
        args=controller_fn,   # dynamic argument → no recompilation
        max_steps=max_steps,
        saveat=SaveAt(ts=ts),
    )

    # Snap final state to upright equilibrium
    ys = sol.ys
    ys = ys.at[-1, 1].set(1.0)      # cos θ
    ys = ys.at[-1, 2].set(0.0)      # sin θ
    ys = ys.at[-1, 3:].set(0.0)     # velocities
    from dataclasses import replace
    return replace(sol, ys=ys)


# --------------------------------------------------------------------------- #
# Batched Simulation                                                          #
# --------------------------------------------------------------------------- #

def simulate_batch(
    controller: Callable[[jnp.ndarray, float], float],
    params: CartPoleParams,
    t_span: tuple[float, float],
    ts: Sequence[float],
    y0s: Float[Array, "batch 5"],
    **kwargs
):
    """
    Simulate multiple cart-pole trajectories in parallel.
    
    Args:
        controller: Control function (state, time) -> force
        params: Physical parameters (shared across all trajectories)
        t_span: Integration time span (t_start, t_end)
        ts: Time points where solution is saved
        y0s: Batch of initial states, shape (batch_size, 5)
        **kwargs: Additional arguments passed to simulate()
        
    Returns:
        Batched diffrax solution with shape (batch_size, ...)
    """
    # Validate batch initial states format
    if y0s.shape[-1] != 5:
        raise ValueError(f"Expected batch initial states format (batch, 5), got shape {y0s.shape}")
    
    sim_fn = partial(simulate, controller, params, t_span, ts, **kwargs)
    return jax.vmap(sim_fn)(y0s)


# --------------------------------------------------------------------------- #
# Utility Functions                                                           #
# --------------------------------------------------------------------------- #

def create_time_grid(t_span: tuple[float, float], dt: float) -> jnp.ndarray:
    """Create uniform time grid for simulation."""
    return jnp.arange(t_span[0], t_span[1] + dt/2, dt)


def extract_trajectory(solution, component: str = "all") -> jnp.ndarray:
    """
    Extract specific components from simulation trajectory.
    
    Args:
        solution: Diffrax solution object
        component: Which component to extract:
            - "all": Full state trajectory (default)
            - "position": Cart position (x)
            - "angle": Reconstructed angle θ = atan2(sin(θ), cos(θ))
            - "velocity": Cart velocity (ẋ)
            - "angular_velocity": Pole angular velocity (θ̇)
            
    Returns:
        Extracted trajectory component(s)
    """
    states = solution.ys  # Shape: (time_steps, 5)
    
    if component == "all":
        return states
    elif component == "position":
        return states[:, 0]  # x
    elif component == "angle":
        cos_th, sin_th = states[:, 1], states[:, 2]
        return jnp.arctan2(sin_th, cos_th)  # Reconstruct θ
    elif component == "velocity":
        return states[:, 3]  # ẋ
    elif component == "angular_velocity":
        return states[:, 4]  # θ̇
    else:
        raise ValueError(f"Unknown component '{component}'. Use 'all', 'position', 'angle', 'velocity', or 'angular_velocity'")
