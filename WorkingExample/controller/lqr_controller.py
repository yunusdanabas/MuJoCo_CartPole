#==================== controller/lqr_controller.py ====================

import jax.numpy as jnp
from scipy.linalg import solve_continuous_are

def linearize_cartpole(params):
    mc, mp, l, g = params
    # Linearization about the upright state (x=0, θ=0)
    A = jnp.array([
        [0,     0,                1,                 0],
        [0,     0,                0,                 1],
        [0, (mp * g) / mc,        0,                 0],
        [0, (g * (mc + mp)) / (l * mc),  0,         0]
    ])
    B = jnp.array([[0],
                   [0],
                   [1 / mc],
                   [1 / (l * mc)]])
    return A, B

def compute_lqr_gain(A, B, Q, R):
    P = solve_continuous_are(A, B, Q, R)
    K = jnp.linalg.inv(jnp.array(R)) @ jnp.array(B).T @ P
    return jnp.array(K)


  