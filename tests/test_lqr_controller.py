import jax.numpy as jnp
import numpy as np
import pytest
from controller.lqr_controller import _solve_care_iterative, _linearise, LQRController
from env.cartpole import CartPoleParams


def test_pure_jax_care():
    """Test that the pure JAX CARE solver works correctly"""
    params = CartPoleParams()
    A, B = _linearise(params)
    Q = jnp.diag(jnp.array([10., 10., 1., 1.]))
    R = jnp.array([[0.1]])
    
    P = _solve_care_iterative(A, B, Q, R)
    
    # Check that residual norm is tiny
    R_inv = 10.0  # 1 / 0.1
    res = A.T @ P + P @ A - P @ B * R_inv @ B.T @ P + Q
    assert jnp.linalg.norm(res) < 1e-2


def test_lqr_controller_creation():
    """Test that LQRController can be created without scipy"""
    params = CartPoleParams()
    controller = LQRController.from_linearisation(params)
    assert controller.K.shape in ((1, 4), (4,))


def test_closed_loop_stability():
    """Test that LQR controller produces stable closed-loop system"""
    params = CartPoleParams()
    A, B = _linearise(params)
    K = LQRController.from_linearisation(params).K
    
    # Check closed-loop eigenvalues
    eigs = np.linalg.eigvals(A - B @ K.reshape(1, -1))
    assert np.all(np.real(eigs) < 0), "closed-loop poles not in LHP"


def test_lqr_zero_state_response():
    """Test LQR controller response to zero state"""
    params = CartPoleParams()
    lqr = LQRController.from_linearisation(params).jit()
    
    zero_state = jnp.zeros(4)
    force = lqr(zero_state, 0.0)
    assert jnp.isclose(force, 0.0, atol=1e-6)


def test_lqr_gain_properties():
    """Test properties of LQR gain matrix"""
    params = CartPoleParams()
    lqr = LQRController.from_linearisation(params)
    
    # Gain should be finite
    assert jnp.all(jnp.isfinite(lqr.K))
    
    # Gain should have correct shape
    assert lqr.K.shape == (4,)  # 1x4 flattened to (4,)