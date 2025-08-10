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
    """Test linear controller with scalar and batch 5-state inputs"""
    K = jnp.array([1.0, 0.0, 0.0, 2.0, 0.0])
    lin = LinearController(K=K).jit()

    # Test scalar (5-state)
    theta = 0.1
    s5 = jnp.array([0.5, jnp.cos(theta), jnp.sin(theta), -0.2, 0.0])
    expected = -(K @ s5)
    assert np.isclose(lin(s5, 0.0), expected, atol=1e-6)

    # Test batched
    states = jnp.stack([s5, 2 * s5])
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

    # zero state → zero force (4-state interface retained)
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
    """Test that controllers follow the base interface with 5-state (Linear, NN) and 4-state (LQR)."""
    params = CartPoleParams()
    t = 0.0

    lin = LinearController(K=jnp.array([1., 0., 0., 0., 0.])).jit()
    nn = NNController.init().jit()
    lqr = LQRController.from_linearisation(params).jit()

    state5 = jnp.zeros(5)
    for ctrl in (lin, nn):
        force = ctrl(state5, t)
        assert force.shape == ()
        assert jnp.isfinite(force)

    state4 = jnp.zeros(4)
    force_lqr = lqr(state4, t)
    assert force_lqr.shape == ()
    assert jnp.isfinite(force_lqr)


# ------------------------------------------------------------------ Integration
def test_controller_environment_integration():
    """Test controllers work with environment integration using 5-state initial conditions"""
    from env.closedloop import simulate
    
    params = CartPoleParams()
    tgrid = jnp.linspace(0.0, 0.1, 11)
    
    controllers = [
        LinearController(K=jnp.array([10., 0., 5., 0., 0.])).jit(),
        LQRController.from_linearisation(params).jit(),
        NNController.init().jit()
    ]
    
    init_state4 = jnp.array([0.01, 0.01, 0.0, 0.0])
    init_state = four_to_five(init_state4)
    
    for ctrl in controllers:
        sol = simulate(ctrl, params, (0., 0.1), tgrid, init_state)
        assert sol.ys.shape == (len(tgrid), 5)
        assert jnp.all(jnp.isfinite(sol.ys))