"""
env/cartpole.py
-------------------------------------------------------------------------------
Cart-Pole continuous-time dynamics (JAX).

Supports both state encodings:
4-state: [x, theta, x_dot, theta_dot]
5-state: [x, cos theta, sin theta, x_dot, theta_dot]

"""

from __future__ import annotations
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .helpers import (
    four_to_five,
    five_to_four,
    inverse_mass_matrix,
    total_energy,
)

# -----------------------------------------------------------------------------#
# Parameters                                                                    #
# -----------------------------------------------------------------------------#

@dataclass(frozen=True)
class CartPoleParams:
    mc: float = 1.0     # Cart mass  [kg]
    mp: float = 1.0     # Pole mass  [kg]
    l:  float = 1.0     # Pole length [m]
    g:  float = 9.81    # Gravity     [m s^-2]

# -----------------------------------------------------------------------------#
# Core dynamics                                                                 #
# -----------------------------------------------------------------------------#

@jax.jit
def _dynamics_single(
    state: Float[Array, "state_dim"],
    t: float,
    params: CartPoleParams,
    controller,
) -> Float[Array, "state_dim"]:
    """JIT-compiled dynamics for one cart-pole."""

    # Ensure 5-state representation internally
    state5 = jax.lax.cond(
        state.shape[0] == 4,
        lambda s: four_to_five(s),
        lambda s: s,
        state,
    )

    # Unpack for readability
    x, cos_theta, sin_theta, x_dot, theta_dot = state5
    mp, l, g = params.mp, params.l, params.g

    # Control input (scalar)
    force = controller(state5, t)

    # Pre-compute inverse mass matrix
    m11, m12, m22, _ = inverse_mass_matrix(cos_theta, params)

    # Right-hand side: τ_g + B u − C q̇
    rhs1 = force
    rhs2 = mp * g * l * sin_theta - mp * l * sin_theta * theta_dot * theta_dot

    # Accelerations using analytic M^-1
    x_ddot     = m11 * rhs1 + m12 * rhs2
    theta_ddot = m12 * rhs1 + m22 * rhs2

    # Time-derivatives of cos(theta) and sin(theta)
    cos_dot = -sin_theta * theta_dot
    sin_dot =  cos_theta * theta_dot

    state5_dot = jnp.array([x_dot, cos_dot, sin_dot, x_ddot, theta_ddot])

    # If caller gave 4-state, convert derivative back to 4-state layout
    return jax.lax.cond(
        state.shape[0] == 4,
        lambda s: five_to_four(s),
        lambda s: s,
        state5_dot,
    )

# -----------------------------------------------------------------------------#
# Public vectorised interface                                                   #
# -----------------------------------------------------------------------------#

def dynamics(
    state: Float[Array, "... state_dim"],
    t: float | Array,
    params: CartPoleParams = CartPoleParams(),
    controller=lambda s, t: 0.0,
) -> Float[Array, "... state_dim"]:
    """Vectorised cart-pole dynamics: works on scalars **or** batches."""
    # Handle scalar time input
    t_scalar = float(t) if jnp.isscalar(t) else t
    
    _dyn = jax.jit(
        _dynamics_single,
        static_argnames=("params", "controller"),
    )
    
    if state.ndim > 1:
        return jax.vmap(_dyn, in_axes=(0, None, None, None))(
            state, t_scalar, params, controller
        )
    else:
        return _dyn(state, t_scalar, params, controller)

# -----------------------------------------------------------------------------#
# Legacy compatibility functions                                                #
# -----------------------------------------------------------------------------#

def cartpole_dynamics(t, state, args):
    """Legacy function for backward compatibility."""
    params, controller = args
    if isinstance(params, (tuple, list)):
        mc, mp, l, g = params
        params = CartPoleParams(mc=mc, mp=mp, l=l, g=g)
    
    return dynamics(state, t, params, controller)


def cartpole_dynamics_nn(t, state, args):
    """Legacy function for 5-state representation."""
    params, controller = args
    if isinstance(params, (tuple, list)):
        mc, mp, l, g = params
        params = CartPoleParams(mc=mc, mp=mp, l=l, g=g)
    
    return dynamics(state, t, params, controller)
