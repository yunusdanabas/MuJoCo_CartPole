"""
Helper functions for cart-pole dynamics.
"""

from __future__ import annotations

import jax                         
import jax.numpy as jnp          
from jaxtyping import Array, Float

# -----------------------------------------------------------------------------#
# State conversion helpers                                                      #
# -----------------------------------------------------------------------------#

def four_to_five(state4: Float[Array, "...4"]) -> Float[Array, "...5"]:
    """Convert 4-state [x, theta, x_dot, theta_dot] 
        to 5-state [x, cos(theta), sin(theta), x_dot, theta_dot]."""
    
    x, th, x_dot, th_dot = jnp.split(state4, 4, axis=-1)
    return jnp.concatenate(
        [x, jnp.cos(th), jnp.sin(th), x_dot, th_dot], axis=-1
    )


def five_to_four(state5: Float[Array, "...5"]) -> Float[Array, "...4"]:
    """Convert 5-state [x, cos(theta), sin(theta), x_dot, theta_dot] 
        to 4-state [x, theta, x_dot, theta_dot]."""
    
    x, cos_th, sin_th, x_dot, th_dot = jnp.split(state5, 5, axis=-1)
    th = jnp.arctan2(sin_th, cos_th)
    return jnp.concatenate([x, th, x_dot, th_dot], axis=-1)

# -----------------------------------------------------------------------------#
# Mass matrix operations                                                        #
# -----------------------------------------------------------------------------#

def inverse_mass_matrix(
    cos_th: Float[Array, "..."], params
) -> tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
    """Analytic inverse of the 2x2 mass matrix."""

    mc, mp, l = params.mc, params.mp, params.l

    a   = mc + mp                    # M[0,0]: total mass
    b   = -mp * l * cos_th          # M[0,1] = M[1,0]: coupling term
    c   = mp * l**2                 # M[1,1]: pole rotational inertia

    det = a * c - b * b + 1e-9      # determinant with regularization
    
    m11 =  c / det                  # (M^-1)[0,0]
    m12 = -b / det                  # (M^-1)[0,1] = (M^-1)[1,0]
    m22 =  a / det                  # (M^-1 )[1,1]
    return m11, m12, m22, det

# -----------------------------------------------------------------------------#
# Energy calculations                                                           #
# -----------------------------------------------------------------------------#

@jax.jit
def total_energy(state, params):
    """Calculate total mechanical energy (kinetic + potential)."""
    s5 = jax.lax.cond(
        state.shape[-1] == 4, 
        four_to_five, 
        lambda x: x, 
        state
    )
    
    _, cos_th, sin_th, x_dot, th_dot = jnp.split(s5, 5, axis=-1)
    mc, mp, l, g = params.mc, params.mp, params.l, params.g
    
    ke_cart = 0.5 * mc * x_dot**2                               # cart kinetic energy
    pole_vx = x_dot + l * th_dot * cos_th                       # pole x-velocity 
    pole_vy = -l * th_dot * sin_th                              # pole y-velocity
    ke_pole = 0.5 * mp * (pole_vx**2 + pole_vy**2)             # pole kinetic energy
    pe = mp * g * l * cos_th                                    # gravitational potential
    
    return (ke_cart + ke_pole + pe).squeeze()


def kinetic_energy(state, params):
    """Calculate kinetic energy only."""
    s5 = jax.lax.cond(state.shape[-1] == 4, four_to_five, lambda x: x, state)
    _, cos_th, sin_th, x_dot, th_dot = jnp.split(s5, 5, axis=-1)
    mc, mp, l = params.mc, params.mp, params.l
    
    ke_cart = 0.5 * mc * x_dot**2                               # cart kinetic energy
    pole_vx = x_dot + l * th_dot * cos_th                       # pole velocity components 
    pole_vy = -l * th_dot * sin_th                              # pole velocity components
    ke_pole = 0.5 * mp * (pole_vx**2 + pole_vy**2)             # pole kinetic energy
    
    return (ke_cart + ke_pole).squeeze()


def potential_energy(state, params):
    """Calculate potential energy only."""
    s5 = jax.lax.cond(state.shape[-1] == 4, four_to_five, lambda x: x, state)
    _, cos_th, _, _, _ = jnp.split(s5, 5, axis=-1)
    mp, l, g = params.mp, params.l, params.g
    
    return (mp * g * l * cos_th).squeeze()