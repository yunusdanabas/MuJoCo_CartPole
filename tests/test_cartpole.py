"""Tests for cart-pole dynamics with 5-state format"""
import jax.numpy as jnp
import pytest
from env.cartpole import dynamics, CartPoleParams, compute_energy
from env.helpers import total_energy, four_to_five


def test_cartpole_functionality():
    """Test basic cartpole dynamics functionality with 5-state"""
    params = CartPoleParams()
    
    # Create 5-state representation: [x, cos(θ), sin(θ), ẋ, θ̇]
    theta = jnp.pi - 0.1  # Near inverted
    state = jnp.array([0., jnp.cos(theta), jnp.sin(theta), 0., 0.])
    
    # Test dynamics function
    dot = dynamics(state, 0.0, params, lambda *_: 0.)
    assert dot.shape == state.shape
    assert jnp.all(jnp.isfinite(dot))
    
    # Check derivative structure
    assert jnp.isclose(dot[0], state[3])  # ẋ = x_dot
    assert jnp.isclose(dot[1], -jnp.sin(theta) * state[4])  # d/dt[cos(θ)] = -sin(θ)θ̇
    assert jnp.isclose(dot[2], jnp.cos(theta) * state[4])   # d/dt[sin(θ)] = cos(θ)θ̇


def test_cartpole_performance():
    """Test that dynamics computation is reasonably fast"""
    import time
    params = CartPoleParams()
    
    # 5-state representation
    theta = jnp.pi - 0.1
    state = jnp.array([0., jnp.cos(theta), jnp.sin(theta), 0., 0.])
    
    start_time = time.time()
    for _ in range(1000):
        dynamics(state, 0.0, params, lambda *_: 0.)
    elapsed = time.time() - start_time

    #print(f"Elapsed time for 1000 dynamics evaluations: {elapsed:.4f} seconds")
    
    # Should complete 1000 dynamics evaluations in less than 1 second
    assert elapsed < 1.0


def test_cartpole_edge_case():
    """Test cartpole behavior at edge cases with 5-state"""
    params = CartPoleParams()
    
    # Test at exactly upright position [x, cos(0), sin(0), ẋ, θ̇]
    upright_state = jnp.array([0., 1., 0., 0., 0.])
    dot = dynamics(upright_state, 0.0, params, lambda *_: 0.)
    assert jnp.all(jnp.isfinite(dot))
    
    # Test energy conservation property
    theta = jnp.pi - 0.1
    state = jnp.array([0., jnp.cos(theta), jnp.sin(theta), 0., 0.])
    energy = total_energy(state, params)
    assert jnp.isfinite(energy)
    assert energy > 0  # Should have positive energy in this configuration


def test_energy_consistency():
    """Test energy calculations are consistent"""
    params = CartPoleParams()
    
    # Test various states
    states = jnp.array([
        [0., 1., 0., 0., 0.],           # Upright, stationary
        [0., -1., 0., 0., 0.],          # Hanging down, stationary  
        [1., 1., 0., 1., 0.],           # Moving cart, upright pole
        [0., 0., 1., 0., 1.],           # Horizontal pole, spinning
    ])
    
    for state in states:
        energy = compute_energy(state, params)
        assert jnp.isfinite(energy)
        assert energy >= 0  # Energy should be non-negative with our offset


def test_batch_dynamics():
    """Test batch dynamics computation"""
    from env.cartpole import batch_dynamics
    
    params = CartPoleParams()
    
    # Create batch of 5-state vectors
    batch_states = jnp.array([
        [0., 1., 0., 0., 0.],           # Upright
        [0., -1., 0., 0., 0.],          # Hanging  
        [1., 1., 0., 0., 0.],           # Displaced cart
    ])
    
    dot_batch = batch_dynamics(batch_states, 0.0, params)
    
    assert dot_batch.shape == (3, 5)
    assert jnp.all(jnp.isfinite(dot_batch))
    
    # Compare with individual computations
    for i, state in enumerate(batch_states):
        dot_single = dynamics(state, 0.0, params)
        assert jnp.allclose(dot_batch[i], dot_single, atol=1e-10)


def test_input_validation():
    """Test that functions properly validate 5-state input"""
    params = CartPoleParams()
    
    # Test with wrong input size
    wrong_state = jnp.array([0., 0., 0., 0.])  # 4-state
    
    # Match actual error message from dynamics()
    with pytest.raises(ValueError, match=r"Expected state format \[x, cos\(θ\), sin\(θ\), ẋ, θ̇\], got shape"):
        dynamics(wrong_state, 0.0, params)
    
    # Match actual error message from helpers.py energy functions  
    with pytest.raises(ValueError, match=r"Expected shape \(..., 5\), got"):
        total_energy(wrong_state, params)