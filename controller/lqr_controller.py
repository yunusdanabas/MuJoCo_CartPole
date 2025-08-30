"""
LQR Controller for Cart-Pole Systems

Implements optimal linear-quadratic regulator control using SciPy's solve_continuous_are.
Supports both 4D [x, θ, ẋ, θ̇] and 5D [x, cos(θ), sin(θ), ẋ, θ̇] state formats.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_continuous_are

from controller.base import Controller
from env.cartpole import CartPoleParams
from env.helpers import five_to_four

__all__ = [
    "LQRController",
    "linearize_cartpole",
    "compute_lqr_gain",
    "create_lqr_controller",
]


def _linearise(params: CartPoleParams) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Linearize cart-pole system about upright equilibrium (θ=0)."""
    mc, mp, l, g = params.mc, params.mp, params.l, params.g
    
    # State matrix A (4x4) - linearized dynamics
    A = jnp.array([
        [0.0, 0.0, 1.0, 0.0],                    # ẋ = ẋ
        [0.0, 0.0, 0.0, 1.0],                    # θ̇ = θ̇  
        [0.0, mp * g / mc, 0.0, 0.0],            # ẍ = (mp*g/mc) * θ
        [0.0, g * (mc + mp) / (l * mc), 0.0, 0.0], # θ̈ = (g*(mc+mp)/(l*mc)) * θ
    ], dtype=jnp.float32)

    # Input matrix B (4x1) - control influence
    B = jnp.array([
        [0.0],           # Force doesn't affect position directly
        [0.0],           # Force doesn't affect angle directly
        [1.0 / mc],      # Force affects cart acceleration
        [1.0 / (l * mc)], # Force affects pole angular acceleration
    ], dtype=jnp.float32)
    
    return A, B


def _lqr_gain(A: jnp.ndarray, B: jnp.ndarray, Q: jnp.ndarray, R: jnp.ndarray) -> jnp.ndarray:
    """Compute optimal LQR gain K using SciPy's solve_continuous_are."""
    # Convert to numpy for SciPy compatibility
    A_np, B_np, Q_np, R_np = map(np.array, [A, B, Q, R])
    
    # Solve Algebraic Riccati Equation: A^T P + P A - P B R^(-1) B^T P + Q = 0
    P = solve_continuous_are(A_np, B_np, Q_np, R_np)
    
    # Optimal gain: K = R^(-1) * B^T * P
    K = np.linalg.solve(R_np, B_np.T @ P)
    
    return jnp.array(K, dtype=A.dtype)


@dataclass(frozen=True)
class LQRController(Controller):
    """Time-invariant LQR controller with optimal gains."""

    K: jnp.ndarray  # Shape (4,) - optimal gains for 4-state system

    @classmethod
    def from_linearisation(
        cls,
        params: CartPoleParams,
        Q: jnp.ndarray = None,
        R: jnp.ndarray = None,
    ) -> "LQRController":
        """Create LQR controller from system parameters."""
        # Default cost matrices for balanced performance
        if Q is None:
            Q = jnp.diag(jnp.array([20.0, 50.0, 5.0, 5.0]))
        if R is None:
            R = jnp.array([[5.0]])
        
        A, B = _linearise(params)
        K = _lqr_gain(A, B, Q, R)
        return cls(K=K)

    def _force(self, state: jnp.ndarray, _t: float) -> jnp.ndarray:
        """Compute control force: u = -K * state."""
        # Convert 5D state to 4D if needed
        state_4d = five_to_four(state) if state.shape[-1] == 5 else state
        
        # Control law: u = -K * x
        force = -jnp.dot(self.K, state_4d)
        return force.reshape(())  # Ensure scalar output


# ============================================================================
# Backward Compatibility Functions
# ============================================================================

def linearize_cartpole(params) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Legacy: Return (A, B) matrices for given parameters."""
    if isinstance(params, (tuple, list)):
        mc, mp, l, g = params
        params = CartPoleParams(mc=mc, mp=mp, l=l, g=g)
    return _linearise(params)


def compute_lqr_gain(A, B, Q, R) -> jnp.ndarray:
    """Legacy: Compute LQR gain K."""
    return _lqr_gain(A, B, Q, R)


def create_lqr_controller(params, Q, R) -> Callable[[jnp.ndarray, float], float]:
    """Legacy: Create callable policy function."""
    if isinstance(params, (tuple, list)):
        mc, mp, l, g = params
        params = CartPoleParams(mc=mc, mp=mp, l=l, g=g)

    if isinstance(R, float):
        R = jnp.array([[R]])

    controller = LQRController.from_linearisation(params, Q, R)

    def policy(state, t=0.0):
        return float(controller._force(state, t))

    return policy


# ============================================================================
# Default Configuration
# ============================================================================

_DEFAULT_PARAMS = CartPoleParams(mc=1.0, mp=0.1, l=0.5, g=9.81)
_DEFAULT_Q = jnp.diag(jnp.array([20.0, 50.0, 5.0, 5.0]))
_DEFAULT_R = jnp.array([[5.0]])

# Default policy for interactive demos
lqr_policy = create_lqr_controller(_DEFAULT_PARAMS, _DEFAULT_Q, _DEFAULT_R)
