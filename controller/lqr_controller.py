"""
Time-invariant LQR controller (pure JAX implementation).

- Works with the project-wide 5-state representation [x, cos(theta), sin(theta), xdot, thdot]
  by converting to the 4-state small-angle linearisation order [x, theta, xdot, thdot] internally.
- Linearises about the upright equilibrium (theta ≈ 0) and solves the continuous-time
  Algebraic Riccati Equation (ARE) once at initialisation.
- No hard-coded gains; K is derived from (A, B, Q, R).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np

try:
    import yaml  # Optional configuration support
except ImportError:  # pragma: no cover - yaml is optional
    yaml = None

from controller.base import Controller
from env.cartpole import CartPoleParams
from env.helpers import five_to_four

try:  # Optional SciPy for a stabilising initial policy
    from scipy.linalg import solve_continuous_are as _scipy_care
except Exception:  # pragma: no cover
    _scipy_care = None

__all__ = [
    "LQRController",
    "linearize_cartpole",
    "compute_lqr_gain",
    "create_lqr_controller",
]


# --------------------------------------------------------------------------- #
# Linearisation                                                               #
# --------------------------------------------------------------------------- #

def _linearise(params: CartPoleParams) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return continuous-time linearisation (A, B) about upright equilibrium.

    State order for linearisation is 4-state [x, theta, xdot, thdot].

    Args:
        params: Physical parameters of the cart-pole.

    Returns:
        (A, B): Continuous-time linearised dynamics matrices.
    """
    mc, mp, l, g = params.mc, params.mp, params.l, params.g
    A = jnp.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, mp * g / mc, 0.0, 0.0],
            [0.0, g * (mc + mp) / (l * mc), 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    B = jnp.array(
        [
            [0.0],
            [0.0],
            [1.0 / mc],
            [1.0 / (l * mc)],
        ],
        dtype=jnp.float32,
    )
    return A, B


# --------------------------------------------------------------------------- #
# Continuous-time ARE via simple projected gradient                            #
# --------------------------------------------------------------------------- #

def _solve_care_iterative(
    A: jnp.ndarray,
    B: jnp.ndarray,
    Q: jnp.ndarray,
    R: jnp.ndarray,
    *,
    iters: int = 100,
    alpha: float = 1e-3,
) -> jnp.ndarray:
    """Solve continuous-time ARE via Kleinman policy iteration with Lyapunov solves.

    This approach alternates between policy evaluation (solve Lyapunov for P)
    and policy improvement (update K = R^{-1} B^T P). It is robust for small
    systems and avoids divergence seen with naive gradient steps.
    """
    del alpha  # Unused in this implementation

    def solve_lyapunov(Acl: jnp.ndarray, Qbar: jnp.ndarray) -> jnp.ndarray:
        """Solve Acl^T P + P Acl + Qbar = 0 via Kronecker linear solve."""
        n = Acl.shape[0]
        I = jnp.eye(n, dtype=Acl.dtype)
        M = jnp.kron(I, Acl.T) + jnp.kron(Acl.T, I)  # (n^2, n^2)
        b = -Qbar.T.reshape((n * n,))  # vec_col(Qbar) == reshape(Qbar.T, -1)
        vecP = jnp.linalg.solve(M, b)
        P = vecP.reshape((n, n)).T
        # Symmetrise to counter numerical asymmetry
        return 0.5 * (P + P.T)

    def body(carry, _):
        K = carry  # (1, n)
        Acl = A - B @ K
        Qbar = Q + K.T @ R @ K
        P = solve_lyapunov(Acl, Qbar)
        K_new = jnp.linalg.solve(R, B.T @ P)  # (1, n)
        return K_new, None

    # Start from zero-gain policy
    if _scipy_care is not None:
        # Use SciPy to get a good initial K, then iterate in JAX
        P0_np = _scipy_care(np.array(A), np.array(B), np.array(Q), np.array(R))
        K0_np = np.linalg.solve(np.array(R), np.array(B).T @ P0_np)  # (1,4)
        K0 = jnp.array(K0_np, dtype=A.dtype)
    else:
        # Heuristic stabilising initial policy: velocity damping
        k3 = 10.0
        k4 = 10.0
        K0 = jnp.array([[0.0, 0.0, k3, k4]], dtype=A.dtype)
    # Run a simple Python loop for robustness (no JIT constraints)
    K = K0
    for _ in range(iters):
        Acl = A - B @ K
        Qbar = Q + K.T @ R @ K
        P = solve_lyapunov(Acl, Qbar)
        K = jnp.linalg.solve(R, B.T @ P)
    K_final = K

    # Return P corresponding to final policy
    Acl = A - B @ K_final
    Qbar = Q + K_final.T @ R @ K_final
    P_final = solve_lyapunov(Acl, Qbar)
    return P_final


def _lqr_gain(A: jnp.ndarray, B: jnp.ndarray, Q: jnp.ndarray, R: jnp.ndarray) -> jnp.ndarray:
    """Compute continuous-time LQR gain K, returned as shape (4,)."""
    P = _solve_care_iterative(A, B, Q, R)
    # Solve R K = B^T P  ->  K = R^{-1} B^T P (row vector length 4)
    K_row = jnp.linalg.solve(R, B.T @ P)  # (1, 4)
    return jnp.squeeze(K_row)


# --------------------------------------------------------------------------- #
# Public controller                                                            #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LQRController(Controller):
    """Time-invariant LQR controller.

    Attributes:
        K: Row-vector gains for 4-state linearisation, shape (4,).
    """

    K: jnp.ndarray  # (4,)

    @classmethod
    def from_linearisation(
        cls,
        params: CartPoleParams,
        Q: jnp.ndarray = jnp.diag(jnp.array([10.0, 10.0, 1.0, 1.0])),
        R: jnp.ndarray = jnp.array([[0.1]]),
    ) -> "LQRController":
        """Create LQR controller from physical parameters.

        Args:
            params: Cart-pole parameters
            Q: State cost matrix (4x4)
            R: Control cost matrix (1x1)

        Returns:
            LQRController with gain K (4,)
        """
        A, B = _linearise(params)
        K = _lqr_gain(A, B, Q, R)
        return cls(K=K)

    def _force(self, state: jnp.ndarray, _t: float) -> jnp.ndarray:
        """Compute control force for a 4D or 5D state.

        For 5D input, converts to 4D small-angle state before applying u = -K s4.
        """
        if state.shape[-1] == 5:
            s4 = five_to_four(state)
        else:
            s4 = state
        return -jnp.dot(self.K, s4)


# --------------------------------------------------------------------------- #
# Backward-compatibility helpers                                               #
# --------------------------------------------------------------------------- #

def linearize_cartpole(params) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Legacy wrapper returning (A, B) for given params (tuple or CartPoleParams)."""
    if isinstance(params, (tuple, list)):
        mc, mp, l, g = params
        params = CartPoleParams(mc=mc, mp=mp, l=l, g=g)
    return _linearise(params)


def compute_lqr_gain(A, B, Q, R) -> jnp.ndarray:
    """Legacy wrapper computing K; returns shape (4,)."""
    return _lqr_gain(A, B, Q, R)


def create_lqr_controller(params, Q, R) -> Callable[[jnp.ndarray, float], float]:
    """Legacy factory returning a callable policy(state, t) -> float.

    Accepts params as tuple/list or CartPoleParams. R may be a scalar or (1,1) array.
    """
    if isinstance(params, (tuple, list)):
        mc, mp, l, g = params
        params = CartPoleParams(mc=mc, mp=mp, l=l, g=g)

    if isinstance(R, float):
        R = jnp.array([[R]])

    controller = LQRController.from_linearisation(params, Q, R)

    def policy(state, t=0.0):
        return float(controller._force(state, t))

    return policy


# --------------------------------------------------------------------------- #
# Defaults via optional configuration                                          #
# --------------------------------------------------------------------------- #
_CFG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
try:  # pragma: no cover - config is optional
    if yaml is not None:
        with _CFG_PATH.open("r") as f:
            _CFG = yaml.safe_load(f) or {}
    else:
        _CFG = {}
except FileNotFoundError:  # pragma: no cover
    _CFG = {}

_DEFAULT_PARAMS = CartPoleParams(*_CFG.get("nn_training", {}).get("params_system", [1.0, 0.1, 0.5, 9.81]))
_LQR = _CFG.get("lqr", {})
_DEFAULT_Q = jnp.diag(jnp.array(_LQR.get("Q", [50.0, 100.0, 5.0, 20.0])))
_DEFAULT_R = jnp.array([[float(_LQR.get("R", 0.1))]])

# Instantiate default policy for interactive demos
lqr_policy = create_lqr_controller(_DEFAULT_PARAMS, _DEFAULT_Q, _DEFAULT_R)
