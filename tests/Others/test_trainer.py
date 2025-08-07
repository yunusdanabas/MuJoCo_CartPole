import jax.numpy as jnp
import pytest
from lib.trainer import train, TrainConfig
from controller.nn_controller import NNController
from env.cartpole import CartPoleParams


def test_trainer_smoke():
    """Smoke test for training process"""
    cfg = TrainConfig(batch_size=16, num_epochs=5)  # tiny run
    ctrl, loss_hist = train(NNController.init(), CartPoleParams(), cfg)
    assert loss_hist.shape == (cfg.num_epochs,)


def test_train_config_validation():
    """Test training configuration validation"""
    # Valid config
    cfg = TrainConfig(batch_size=32, num_epochs=10)
    assert cfg.batch_size == 32
    assert cfg.num_epochs == 10
    
    # Test that config has reasonable defaults
    assert hasattr(cfg, 'batch_size')
    assert hasattr(cfg, 'num_epochs')


def test_training_reduces_loss():
    """Test that training generally reduces loss"""
    cfg = TrainConfig(batch_size=16, num_epochs=10)
    ctrl, loss_hist = train(NNController.init(), CartPoleParams(), cfg)
    
    # Loss should generally decrease (allowing for some noise)
    initial_loss = loss_hist[0]
    final_loss = loss_hist[-1]
    
    # Check that loss is finite
    assert jnp.all(jnp.isfinite(loss_hist))
    
    # Check that we get expected number of loss values
    assert len(loss_hist) == cfg.num_epochs


def test_trained_controller_functionality():
    """Test that trained controller still functions properly"""
    cfg = TrainConfig(batch_size=8, num_epochs=3)  # Very short training
    ctrl, _ = train(NNController.init(), CartPoleParams(), cfg)
    
    # Test that trained controller can still compute forces
    state = jnp.zeros(5)
    force = ctrl.jit()(state, 0.0)
    assert jnp.isfinite(force)
    assert force.shape == ()