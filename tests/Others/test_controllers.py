"""
pytest -q tests/test_controllers.py
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

from controller.linear_controller import LinearController
from controller.lqr_controller import LQRController, _linearise
from controller.nn_controller import NNController
from env.cartpole import CartPoleParams
from env.helpers import four_to_five


# ------------------------------------------------------------------- Linear
def test_linear_controller_scalar_and_batch():
    """Test linear controller with scalar and batch inputs"""
    K = jnp.array([1.0, 0.0, 2.0, 0.0])
    lin = LinearController(K=K).jit()

    # Test scalar (4-state)
    s4 = jnp.array([0.5, 0.1, -0.2, 0.0])
    expected = -(K[0] * s4[0] + K[2] * s4[2])
    assert np.isclose(lin(s4, 0.0), expected, atol=1e-6)

    # Test scalar (5-state)
    assert np.isclose(lin(four_to_five(s4), 0.0), expected, atol=1e-6)

    # Test batched
    states = jnp.stack([s4, 2 * s4])
    forces = lin.batched()(states, 0.0)
    assert forces.shape == (2,)
    np.testing.assert_allclose(
        forces,
        jnp.array([expected, 2 * expected]),
        rtol=1e-6,
    )


# --------------------------------------------------------------------- LQR
def test_lqr_gain_and_stability():
    """Test LQR controller gain and stability properties"""
    params = CartPoleParams()
    lqr = LQRController.from_linearisation(params).jit()

    # zero state → zero force
    assert np.isclose(lqr(jnp.zeros(4), 0.0), 0.0, atol=1e-6)

    # closed-loop eigenvalues < 0 (continuous-time stability check)
    A, B = _linearise(params)
    K = np.array(lqr.K).reshape(1, -1)
    eigs = np.linalg.eigvals(np.array(A) - np.array(B) @ K)
    assert np.all(np.real(eigs) < 0), "closed-loop poles not in LHP"


# ----------------------------------------------------------------------- NN
def test_nn_controller_shapes_and_grad():
    """Test NN controller shapes and gradient computation"""
    import jax
    
    ctrl = NNController.init().jit()
    s = jnp.zeros(5)
    
    # Test forward pass
    force = ctrl(s, 0.0)
    assert force.shape == ()
    
    # Test gradient computation
    def loss_fn(net):
        return NNController(net).jit()(s, 0.0)
    
    g = jax.grad(loss_fn)(ctrl.net)
    leaves = jax.tree_util.tree_leaves(g)
    param_leaves = jax.tree_util.tree_leaves(ctrl.net)
    
    assert all(l.shape == p.shape for l, p in zip(leaves, param_leaves))


# ------------------------------------------------------------------ Base class
def test_controller_base_interface():
    """Test that all controllers follow the base interface"""
    params = CartPoleParams()
    state = jnp.zeros(4)
    t = 0.0

    lin = LinearController(K=jnp.array([1., 0., 0., 0.])).jit()
    lqr = LQRController.from_linearisation(params).jit()
    nn = NNController.init().jit()

    for ctrl in (lin, lqr, nn):
        force = ctrl(state, t)
        assert force.shape == ()
        assert jnp.isfinite(force)


# ------------------------------------------------------------------ Integration
def test_controller_environment_integration():
    """Test controllers work with environment integration"""
    from env.closedloop import simulate
    
    params = CartPoleParams()
    tgrid = jnp.linspace(0.0, 0.1, 11)
    
    controllers = [
        LinearController(K=jnp.array([10., 0., 5., 0.])).jit(),
        LQRController.from_linearisation(params).jit(),
        NNController.init().jit()
    ]
    
    init_state = jnp.array([0.01, 0.01, 0.0, 0.0])
    
    for ctrl in controllers:
        sol = simulate(ctrl, params, (0., 0.1), tgrid, init_state)
        assert sol.ys.shape == (len(tgrid), 4)
        assert jnp.all(jnp.isfinite(sol.ys))