"""Tests for closed-loop cart-pole simulation with 5-state format"""
import jax.numpy as jnp
from env.closedloop import simulate, extract_trajectory
from controller.linear_controller import LinearController
from env.cartpole import CartPoleParams
from env.helpers import four_to_five


def test_response_to_inputs():
    """Test closed loop system response to different inputs"""
    params = CartPoleParams()
    tgrid = jnp.linspace(0.0, 0.1, 11)
    lin = LinearController(K=jnp.array([10., -5., 5., 5., 2.])).jit()
    
    # Test response to small perturbation (convert 4-state to 5-state)
    small_perturbation_4 = jnp.array([0.01, 0.01, 0.0, 0.0])
    small_perturbation = four_to_five(small_perturbation_4)
    
    sol = simulate(lin, params, (0., 0.1), tgrid, small_perturbation)
    assert sol.ys.shape == (len(tgrid), 5)  # 5-state format
    
    # Final state should be closer to origin than initial state
    initial_norm = jnp.linalg.norm(small_perturbation)
    final_norm = jnp.linalg.norm(sol.ys[-1])
    assert final_norm < initial_norm


def test_performance_under_conditions():
    """Test closed loop system performance under various conditions"""
    params = CartPoleParams()
    tgrid = jnp.linspace(0.0, 1.0, 101)
    lin = LinearController(K=jnp.array([50., -20., 20., 20., 10.])).jit()
    
    # Test stabilization from hanging down position
    hanging_state_4 = jnp.array([0., jnp.pi, 0., 0.])
    hanging_state = four_to_five(hanging_state_4)
    
    sol = simulate(lin, params, (0., 1.0), tgrid, hanging_state)
    
    # System should eventually stabilize - extract angle from 5-state
    final_angle = abs(extract_trajectory(sol, "angle")[-1])
    assert final_angle < 0.5  # Within 0.5 radians of upright


def test_closed_loop_stability():
    """Test that the closed loop system is stable"""
    params = CartPoleParams()
    tgrid = jnp.linspace(0.0, 2.0, 201)
    lin = LinearController(K=jnp.array([20., -10., 10., 10., 5.])).jit()
    
    # Start from small perturbation
    init_state_4 = jnp.array([0.1, 0.1, 0.0, 0.0])
    init_state = four_to_five(init_state_4)
    
    sol = simulate(lin, params, (0., 2.0), tgrid, init_state)
    
    # Check that the system doesn't blow up
    assert jnp.all(jnp.isfinite(sol.ys))
    assert jnp.max(jnp.abs(sol.ys)) < 100  # Reasonable bounds


def test_extract_trajectory_components():
    """Test trajectory extraction utility functions"""
    params = CartPoleParams()
    tgrid = jnp.linspace(0.0, 0.5, 51)
    lin = LinearController(K=jnp.array([10., -5., 5., 5., 2.])).jit()
    
    # Small perturbation
    init_state = jnp.array([0.1, jnp.cos(0.1), jnp.sin(0.1), 0.0, 0.0])
    sol = simulate(lin, params, (0., 0.5), tgrid, init_state)
    
    # Test different extraction modes
    full_traj = extract_trajectory(sol, "all")
    position = extract_trajectory(sol, "position")
    angle = extract_trajectory(sol, "angle")
    velocity = extract_trajectory(sol, "velocity")
    angular_velocity = extract_trajectory(sol, "angular_velocity")
    
    assert full_traj.shape == (len(tgrid), 5)
    assert position.shape == (len(tgrid),)
    assert angle.shape == (len(tgrid),)
    assert velocity.shape == (len(tgrid),)
    assert angular_velocity.shape == (len(tgrid),)
    
    # Check angle reconstruction is reasonable
    assert jnp.abs(angle[0] - 0.1) < 1e-6  # Initial angle should match