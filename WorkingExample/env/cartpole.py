# env/cartpole.py

import jax
import jax.numpy as jnp
from functools import partial
from diffrax import ODETerm, Tsit5, diffeqsolve, SaveAt
import equinox as eqx


def cartpole_dynamics(t, state, args):
    """
    A closed-loop variant of cartpole_dynamics.
    Instead of a time-based force_func, we call a controller(state, t) that returns f.
    """
    params, controller = args  # Unpack system params and controller
    q = state[:2]    # [x, theta]
    q_dot = state[2:]  # [x_dot, theta_dot]

    # Controller determines the force based on current state and time
    f = controller(state, t)  # controller is a callable: controller(state, t) -> scalar force

    # Same dynamics computation as the original cartpole_dynamics
    mc, mp, l, g = params
    cos_theta = jnp.cos(q[1])
    sin_theta = jnp.sin(q[1])

    # Mass matrix
    M = jnp.array([
        [mc + mp, -mp * l * cos_theta],
        [-mp * l * cos_theta, mp * l**2]
    ])

    # Coriolis matrix
    C = jnp.array([
        [0, mp * l * sin_theta * q_dot[1]],
        [0, 0]
    ])

    # Gravitational forces
    tau_g = jnp.array([0, mp * g * l * sin_theta])

    # Input force distribution
    B = jnp.array([1, 0])

    # Compute accelerations
    rhs = tau_g + B * f - C @ q_dot
    q_ddot = jnp.linalg.solve(M, rhs)

    return jnp.concatenate([q_dot, q_ddot])


def cartpole_dynamics_nn(t, state, args):
    x_pos = state[0]
    cos_theta = state[1]
    sin_theta = state[2]
    x_dot = state[3]
    theta_dot = state[4]

    params, f_func = args
    mc, mp, l, g = params

    # External force
    f = f_func(t, state)

    # Mass matrix
    M = jnp.array([
        [mc + mp, -mp * l * cos_theta],
        [-mp * l * cos_theta, mp * l**2]
    ])

    # Coriolis matrix
    C = jnp.array([
        [0, mp * l * sin_theta * theta_dot],
        [0, 0]
    ])

    M += jnp.eye(2) * 1e-6  # Regularize the matrix

    # tau_g(q)
    tau_g = jnp.array([0.0, mp * g * l * sin_theta])

    # B
    B = jnp.array([1.0, 0.0])

    
    # tau = tau_g + B*f - C*q_dot
    q_dot = jnp.array([x_dot, theta_dot])
    rhs = tau_g + B * f - (C @ q_dot)

    # q_ddot
    q_ddot = jnp.linalg.solve(M, rhs)
    x_ddot = q_ddot[0]
    theta_ddot = q_ddot[1]

    # Time derivatives of cos(theta) and sin(theta)
    cos_theta_dot = -sin_theta * theta_dot
    sin_theta_dot = cos_theta * theta_dot

    return jnp.array([
        x_dot,
        cos_theta_dot,
        sin_theta_dot,
        x_ddot,
        theta_ddot
    ])
