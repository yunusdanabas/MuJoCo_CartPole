# controller/lqr_controller.py
# Description: This module implements a Linear Quadratic Regulator (LQR) controller for a cart-pole system.
# It computes the control force using the LQR gain matrix and linearizes the system dynamics.
# It also includes functions to compute the LQR gain and simulate the closed-loop system.

import jax.numpy as jnp
from jax.numpy.linalg import inv

from pathlib import Path
import yaml

from env.cartpole import cartpole_dynamics
from lib.utils import convert_5d_to_4d

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


def create_lqr_controller(
    params: tuple[float, float, float, float],
    Q: jnp.ndarray,
    R: jnp.ndarray | float,
) -> callable:
    """Return an LQR controller function for the given parameters."""
    if isinstance(R, float):
        R = jnp.array([[R]])
    A, B = linearize_cartpole(params)
    K = compute_lqr_gain(A, B, Q, R)

    def policy(state: jnp.ndarray, t: float = 0.0) -> float:
        if state.shape[0] == 5:
            state = convert_5d_to_4d(state)
        return float((-(K @ state))[0])

    return policy


# Default policy using configuration values
_CFG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
try:
    with _CFG_PATH.open("r") as f:
        _CFG = yaml.safe_load(f) or {}
except FileNotFoundError:
    _CFG = {}

_DEFAULT_PARAMS = tuple(
    _CFG.get("nn_training", {}).get("params_system", [1.0, 0.1, 0.5, 9.81])
)
_LQR = _CFG.get("lqr", {})
_DEFAULT_Q = jnp.diag(jnp.array(_LQR.get("Q", [50.0, 100.0, 5.0, 20.0])))
_DEFAULT_R = jnp.array([[float(_LQR.get("R", 0.1))]])

# Instantiate the default policy so scripts can import it directly
lqr_policy = create_lqr_controller(_DEFAULT_PARAMS, _DEFAULT_Q, _DEFAULT_R)
