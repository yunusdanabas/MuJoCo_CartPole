"""
Time-invariant LQR controller.

Either pass in a pre-computed gain K, or call the
from_linearisation() factory to derive K at the upright
equilibrium (θ≈0) using continuous-time ARE.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp

try:
    import yaml
except ImportError:
    yaml = None

from controller.base import Controller
from env.cartpole import CartPoleParams
from env.helpers import five_to_four

# --------------------------------------------------------------------- helpers
def _linearise(params: CartPoleParams):
    """Linearize around upright equilibrium (x=0, θ=0, ẋ=0, θ̇=0)."""
    mc, mp, l, g = params.mc, params.mp, params.l, params.g
    A = jnp.array([
        [0, 0,        1, 0],
        [0, 0,        0, 1],
        [0, mp*g/mc,  0, 0],
        [0, g*(mc+mp)/(l*mc), 0, 0]
    ], dtype=jnp.float32)
    
    B = jnp.array([
        [0],
        [0],
        [1/mc],
        [1/(l*mc)]
    ], dtype=jnp.float32)
    
    return A, B


@jax.jit
def _solve_care_iterative(A, B, Q, R, *, iters: int = 500, alpha: float = 1e-3):
    """
    Continuous-time Algebraic Riccati Equation (ARE) solver
    using a simple (but robust) projected-gradient iteration:

        P₀ = Q
        for k = 0 … iters-1:
            Res  = AᵀP + P A − P B R⁻¹BᵀP + Q
            P   ← P − α · Res

    Converges for small α when (A,B) is stabilisable and (Q,R) ≻ 0.
    The final symmetrisation guarantees P = Pᵀ.
    """
    R_inv = jnp.linalg.inv(R)

    def body(P, _):
        resid = (A.T @ P + P @ A
                 - P @ B @ R_inv @ B.T @ P
                 + Q)
        P_new = P - alpha * resid
        return 0.5 * (P_new + P_new.T), None           # force symmetry

    P0, _ = jax.lax.scan(body, Q, None, length=iters)
    return P0


def _lqr_gain(A, B, Q, R):
    """Pure-JAX continuous-time LQR gain via iterative CARE."""
    P = _solve_care_iterative(A, B, Q, R)
    return jnp.squeeze(jnp.linalg.solve(R, B.T @ P))   # shape (1, 4)


# --------------------------------------------------------------------- class
@dataclass(frozen=True)
class LQRController(Controller):
    K: jnp.ndarray                # shape (1,4) or (4,)

    @classmethod
    def from_linearisation(cls,
                           params: CartPoleParams,
                           Q=jnp.diag(jnp.array([10., 10., 1., 1.])),
                           R=jnp.array([[0.1]])):
        """Create LQR controller from system linearisation."""
        A, B = _linearise(params)
        K = _lqr_gain(A, B, Q, R)
        return cls(K=K)

    def _force(self, state, _t):
        s4 = jax.lax.cond(
            state.shape[-1] == 5,
            five_to_four,
            lambda x: x,
            state
        )
        return -jnp.dot(self.K, s4)


# Legacy compatibility functions
def linearize_cartpole(params):
    """Legacy function for backward compatibility."""
    if isinstance(params, (tuple, list)):
        mc, mp, l, g = params
        params = CartPoleParams(mc=mc, mp=mp, l=l, g=g)
    return _linearise(params)


def compute_lqr_gain(A, B, Q, R):
    """Legacy function for backward compatibility."""
    return _lqr_gain(A, B, Q, R)


def create_lqr_controller(params, Q, R):
    """Legacy function for backward compatibility."""
    if isinstance(params, (tuple, list)):
        mc, mp, l, g = params
        params = CartPoleParams(mc=mc, mp=mp, l=l, g=g)
    
    if isinstance(R, float):
        R = jnp.array([[R]])
    
    controller = LQRController.from_linearisation(params, Q, R)
    
    def policy(state, t=0.0):
        return float(controller._force(state, t))
    
    return policy


# Default policy using configuration values
_CFG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
try:
    if yaml is not None:
        with _CFG_PATH.open("r") as f:
            _CFG = yaml.safe_load(f) or {}
    else:
        _CFG = {}
except FileNotFoundError:
    _CFG = {}

_DEFAULT_PARAMS = CartPoleParams(
    *_CFG.get("nn_training", {}).get("params_system", [1.0, 0.1, 0.5, 9.81])
)
_LQR = _CFG.get("lqr", {})
_DEFAULT_Q = jnp.diag(jnp.array(_LQR.get("Q", [50.0, 100.0, 5.0, 20.0])))
_DEFAULT_R = jnp.array([[float(_LQR.get("R", 0.1))]])

# Instantiate default policy
lqr_policy = create_lqr_controller(_DEFAULT_PARAMS, _DEFAULT_Q, _DEFAULT_R)
