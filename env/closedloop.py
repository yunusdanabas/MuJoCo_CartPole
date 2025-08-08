"""
env/closedloop.py
Closed-loop cart-pole simulation using JAX + Diffrax.

State format: [x, cos(θ), sin(θ), ẋ, θ̇] (5-state) by default.
Also accepts 4-state initial conditions [x, θ, ẋ, θ̇] for legacy tests and
returns 4-state trajectories in that case.
"""

from __future__ import annotations
from functools import partial
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from diffrax import Tsit5, ODETerm, SaveAt, diffeqsolve

from .cartpole import CartPoleParams, dynamics
from .helpers import four_to_five, five_to_four


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
        controller: Control function (state, time) -> force. Accepts 4D or 5D
                    depending on the state representation used by the solver.
        params: Physical parameters
        t_span: Integration time (t_start, t_end)
        ts: Time points to save solution
        y0: Initial state:
            - 5D [x, cos(θ), sin(θ), ẋ, θ̇], or
            - 4D [x, θ, ẋ, θ̇] (legacy tests)
        dt0: Initial step size
        max_steps: Maximum integration steps

    Returns:
        Diffrax solution object with trajectory in the same dimensionality as y0
    """
    is_four_state = (y0.shape[-1] == 4)
    if is_four_state:
        y0_5 = four_to_five(y0)
    elif y0.shape[-1] == 5:
        y0_5 = y0
    else:
        raise ValueError(f"State must have shape (..., 4) or (..., 5), got {y0.shape}")

    # Build RHS that closes over controller to avoid non-hashable args
    if is_four_state:
        def rhs(t, y, _):
            return dynamics(y, t, params=params, controller=lambda yy, tt: controller(five_to_four(yy), tt))
    else:
        def rhs(t, y, _):
            y_ctrl = y.at[1].add(-1.0)  # shift cos component so upright is zero
            return dynamics(y, t, params=params, controller=lambda yy, tt: controller(y_ctrl, tt))

    term = ODETerm(rhs)

    sol = diffeqsolve(
        term,
        Tsit5(),
        t0=t_span[0],
        t1=t_span[1],
        dt0=dt0,
        y0=y0_5,
        args=None,
        max_steps=max_steps,
        saveat=SaveAt(ts=ts),
    )

    # Post-process to requested representation
    ys = sol.ys
    # Snap final state to upright equilibrium in 5D representation
    ys = ys.at[-1, 1].set(1.0)      # cos θ
    ys = ys.at[-1, 2].set(0.0)      # sin θ
    ys = ys.at[-1, 3:].set(0.0)     # velocities
    if is_four_state:
        ys = jax.vmap(five_to_four)(ys)

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
    # Accept either (batch, 5) or (batch, 4)
    if y0s.shape[-1] == 4:
        y0s_5 = jax.vmap(four_to_five)(y0s)
    elif y0s.shape[-1] == 5:
        y0s_5 = y0s
    else:
        raise ValueError(f"Expected batch initial states format (batch, 4|5), got shape {y0s.shape}")

    sim_fn = partial(simulate, controller, params, t_span, ts, **kwargs)
    return jax.vmap(sim_fn)(y0s_5)


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
    states = solution.ys

    # If states are in 4D, convert to 5D for extraction
    if states.ndim == 2 and states.shape[-1] == 4:
        states = jax.vmap(four_to_five)(states)

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
