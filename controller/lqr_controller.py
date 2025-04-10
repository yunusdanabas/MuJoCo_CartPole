# controller/lqr_controller.py
# Description: This module implements a Linear Quadratic Regulator (LQR) controller for a cart-pole system.
# It computes the control force using the LQR gain matrix and linearizes the system dynamics.
# It also includes functions to compute the LQR gain and simulate the closed-loop system.

import jax.numpy as jnp
from jax.numpy.linalg import inv

from env.cartpole import cartpole_dynamics

def linearize_cartpole(params):
    """
    Linearize around upright equilibrium (x=0, theta=0, x_dot=0, theta_dot=0).
    Returns A, B (4x4, 4x1).
    """
    mc, mp, l, g = params

    # For small theta around 0, the linearized system:
    # A = [[0,    0,    1,        0],
    #      [0,    0,    0,        1],
    #      [0,  mp*g/mc, 0,       0],
    #      [0,  (mc+mp)*g/(mc*l), 0, 0]]
    # B = [[0],
    #      [0],
    #      [1/mc],
    #      [1/(mc*l)]]
    A = jnp.array([
        [0.,              0.,              1.,         0.],
        [0.,              0.,              0.,         1.],
        [0.,    (mp*g)/mc,                 0.,         0.],
        [0., (mc+mp)*g/(mc*l),             0.,         0.]
    ])
    B = jnp.array([
        [0.],
        [0.],
        [1./mc],
        [1./(mc*l)]
    ])
    return A, B

def compute_lqr_gain(A, B, Q, R):
    """
    Solve the continuous-time algebraic Riccati equation for K.
    K = R^{-1} B^T P
    """
    # CARE: A'P + P A - P B R^-1 B' P + Q = 0
    # We'll do a direct numeric approach or use a known solver
    # For brevity, let's do a manual iteration (not the most robust, but simple).
    # In practice, you'd do something like slycot or a robust solver.
    # This is a naive iterative approach:

    P = jnp.eye(A.shape[0])
    for _ in range(200):
        dP = A.T @ P + P @ A - P @ B @ inv(R) @ B.T @ P + Q
        P = P + 0.01 * dP  # gradient step
    K = inv(R) @ B.T @ P
    return K
