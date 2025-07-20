# env/cartpole.py
# Description: Cartpole dynamics for a cart-pole system.

import jax
import jax.numpy as jnp

def cartpole_dynamics(t, state, args):
    """
    Basic cart-pole dynamics in continuous time.
    state = [x, theta, x_dot, theta_dot]
    """
    params, controller = args
    x, theta, x_dot, theta_dot = state

    # Control force from user-defined controller
    f = controller(state, t)

    mc, mp, l, g = params
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    # Mass matrix
    M = jnp.array([
        [mc + mp, -mp * l * cos_theta],
        [-mp * l * cos_theta, mp * l**2]
    ])

    # Coriolis-like term
    C = jnp.array([
        [0, mp * l * sin_theta * theta_dot],
        [0, 0]
    ])

    # Gravity
    tau_g = jnp.array([0, mp * g * l * sin_theta])

    # B matrix
    B = jnp.array([1, 0])

    rhs = tau_g + B * f - C @ jnp.array([x_dot, theta_dot])
    q_ddot = jnp.linalg.solve(M, rhs)

    return jnp.array([x_dot, theta_dot, q_ddot[0], q_ddot[1]])



def cartpole_dynamics_nn(t, state, args):
    """
    5-element version. 
    state = [x, cosθ, sinθ, x_dot, θ_dot]

    returns d/dt [x, cosθ, sinθ, x_dot, θ_dot].
    """
    params, controller = args
    mc, mp, l, g = params

    # Unpack the state
    x = state[0]
    cos_theta = state[1]
    sin_theta = state[2]
    x_dot = state[3]
    theta_dot = state[4]

    # Evaluate control force from the controller
    f = controller(state, t)  # still a function of the 5-element state

    # Mass matrix
    M = jnp.array([
        [mc + mp, -mp * l * cos_theta],
        [-mp * l * cos_theta, mp * l**2]
    ])

    # Coriolis matrix
    C = jnp.array([
        [0.0, mp * l * sin_theta * theta_dot],
        [0.0, 0.0]
    ])

    # Gravity
    tau_g = jnp.array([0.0, mp * g * l * sin_theta])

    # B
    B = jnp.array([1.0, 0.0])

    # Evaluate the acceleration
    q_dot = jnp.array([x_dot, theta_dot])
    rhs = tau_g + B * f - (C @ q_dot)
    q_ddot = jnp.linalg.solve(M + 1e-9*jnp.eye(2), rhs)  # optional regularization

    x_ddot = q_ddot[0]
    theta_ddot = q_ddot[1]

    # Also compute time derivatives of cosθ and sinθ
    cos_theta_dot = -sin_theta * theta_dot
    sin_theta_dot =  cos_theta * theta_dot

    # Return the full derivative
    return jnp.array([
        x_dot,
        cos_theta_dot,
        sin_theta_dot,
        x_ddot,
        theta_ddot
    ])
