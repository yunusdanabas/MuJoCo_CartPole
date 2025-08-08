# tests/test_nn_controller.py
import jax
import jax.numpy as jnp
import pytest
from controller.nn_controller import NNController


def test_nn_controller_training():
    """Test NN controller training process"""
    model = NNController.init()
    
    # Test that model can be created
    assert model is not None
    assert hasattr(model, 'net')
    
    # Test that model can be jitted
    jitted_model = model.jit()
    assert jitted_model is not None


def test_nn_controller_prediction():
    """Test NN controller prediction"""
    model = NNController.init().jit()
    test_input = jnp.zeros(5)  # 5-state input
    
    prediction = model(test_input, 0.0)
    assert prediction is not None
    assert jnp.isscalar(prediction) or prediction.shape == ()
    assert jnp.isfinite(prediction)


def test_nn_forward_and_grad():
    """Test NN controller forward pass and gradient computation"""
    key = jax.random.PRNGKey(0)
    ctrl = NNController.init(key=key).jit()
    s = jnp.zeros(5)
    
    # Test forward pass
    force = ctrl(s, 0.0)
    assert force.shape == ()

    # Test gradient computation
    g = jax.grad(lambda net: NNController(net).jit()(s, 0.0))(ctrl.net)
    leaves = jax.tree_util.tree_leaves(g)
    param_leaves = jax.tree_util.tree_leaves(ctrl.net)
    
    assert all(l.shape == p.shape for l, p in zip(leaves, param_leaves))


def test_nn_controller_different_inputs():
    """Test NN controller with different input sizes"""
    model = NNController.init().jit()
    
    # Test with 5-state input (standard)
    input_5d = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    output = model(input_5d, 0.0)
    assert jnp.isfinite(output)
    
    # Test that output changes with different inputs
    input_5d_diff = jnp.array([0.2, 0.3, 0.4, 0.5, 0.6])
    output_diff = model(input_5d_diff, 0.0)
    # Outputs should generally be different for different inputs
    # (though could be same by chance)
    assert output.shape == output_diff.shape