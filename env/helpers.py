"""
env/helpers.py
Helper functions for cart-pole dynamics.
State format: [x, cos(θ), sin(θ), ẋ, θ̇]
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from functools import partial


# --------------------------------------------------------------------------- #
# State Format Conversion (Visualization Only)
# --------------------------------------------------------------------------- #

@jax.jit
def four_to_five(state4: Float[Array, "...4"]) -> Float[Array, "...5"]:
    """Convert [x, θ, ẋ, θ̇] to [x, cos(θ), sin(θ), ẋ, θ̇] format"""
    x, th, x_dot, th_dot = jnp.split(state4, 4, axis=-1)
    return jnp.concatenate(
        [x, jnp.cos(th), jnp.sin(th), x_dot, th_dot], axis=-1
    )


@jax.jit
def five_to_four(state5: Float[Array, "...5"]) -> Float[Array, "...4"]:
    """Convert [x, cos(θ), sin(θ), ẋ, θ̇] to [x, θ, ẋ, θ̇] format"""
    x, cos_th, sin_th, x_dot, th_dot = jnp.split(state5, 5, axis=-1)
    th = jnp.arctan2(sin_th, cos_th)
    return jnp.concatenate([x, th, x_dot, th_dot], axis=-1)


# --------------------------------------------------------------------------- #
# Mass Matrix Operations
# --------------------------------------------------------------------------- #

def inverse_mass_matrix(
    cos_th: Float[Array, "..."], 
    params
) -> tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
    """
    Compute inverse of 2x2 cart-pole mass matrix analytically.
    
    Returns: m11, m12, m22, det where M^-1 = [[m11, m12], [m12, m22]]
    """
    mc, mp, l = params.mc, params.mp, params.l

    # Mass matrix: M = [[a, b], [b, c]]
    a = mc + mp              # Total mass
    b = -mp * l * cos_th     # Coupling term
    c = mp * l**2            # Pole inertia

    # Compute determinant with regularization to avoid singularity
    det = a * c - b * b
    det = jnp.where(det < 1e-12, det + 1e-9, det)
    
    # Inverse matrix elements
    m11 = c / det            # M^-1[0,0]
    m12 = -b / det           # M^-1[0,1] = M^-1[1,0]
    m22 = a / det            # M^-1[1,1]
    
    return m11, m12, m22, det


# --------------------------------------------------------------------------- #
# Energy Calculations
# --------------------------------------------------------------------------- #

@partial(jax.jit, static_argnames=("params",))
def potential_energy(state: Float[Array, "...5"], params) -> Float[Array, "..."]:
    """Calculate gravitational potential energy"""
    if state.shape[-1] != 5:
        raise ValueError(f"Expected shape (..., 5), got {state.shape}")
    
    _, cos_th, _, _, _ = jnp.split(state, 5, axis=-1)
    mp, l, g = params.mp, params.l, params.g
    
    # PE = m*g*h where h = l*(1 + cos(θ)) (offset so PE ≥ 0 at bottom)
    return (mp * g * l * (cos_th + 1.0)).squeeze()


@partial(jax.jit, static_argnames=("params",))
def kinetic_energy(state: Float[Array, "...5"], params) -> Float[Array, "..."]:
    """Calculate total kinetic energy (cart + pole)"""
    if state.shape[-1] != 5:
        raise ValueError(f"Expected shape (..., 5), got {state.shape}")
    
    _, cos_th, sin_th, x_dot, th_dot = jnp.split(state, 5, axis=-1)
    mc, mp, l = params.mc, params.mp, params.l
    
    # Cart kinetic energy
    ke_cart = 0.5 * mc * x_dot**2
    
    # Pole kinetic energy (need pole velocity in x,y coordinates)
    pole_vx = x_dot + l * th_dot * cos_th    # d/dt(x + l*sin(θ))
    pole_vy = -l * th_dot * sin_th           # d/dt(l*cos(θ))
    ke_pole = 0.5 * mp * (pole_vx**2 + pole_vy**2)
    
    return (ke_cart + ke_pole).squeeze()


@partial(jax.jit, static_argnames=("params",))
def total_energy(state: Float[Array, "...5"], params) -> Float[Array, "..."]:
    """Calculate total mechanical energy (kinetic + potential)"""
    if state.shape[-1] != 5:
        raise ValueError(f"Expected shape (..., 5), got {state.shape}")
    
    return kinetic_energy(state, params) + potential_energy(state, params)