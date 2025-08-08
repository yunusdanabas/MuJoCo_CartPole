"""Tests for linear controller functionality with 5-state format."""
import jax.numpy as jnp
import pytest
from controller.linear_controller import LinearController, create_pd_controller, create_zero_controller
from lib.training.linear_training import LinearTrainingConfig, train_linear_controller


def test_linear_controller_response():
    """Test linear controller response to different 5-state inputs"""
    K = jnp.array([1., 0., 0., 0., 0.])
    lin = LinearController(K=K).jit()
    
    # Test response to zero state
    assert jnp.isclose(lin(jnp.zeros(5), 0.0), 0.0, atol=1e-6)
    
    # Test response to unit position
    state_unit = jnp.array([1., 1., 0., 0., 0.])  # x=1, cos(θ)=1, sin(θ)=0, etc.
    assert jnp.isclose(lin(state_unit, 0.0), -1.0, atol=1e-6)
    
    # Test response to negative position
    state_neg = jnp.array([-1., 1., 0., 0., 0.])
    assert jnp.isclose(lin(state_neg, 0.0), 1.0, atol=1e-6)


def test_linear_controller_configuration():
    """Test linear controller configuration for 5-state"""
    K = jnp.array([2., 1., 3., 0.5, 1.])
    controller = LinearController(K=K)
    
    # Test that gain matrix is stored correctly
    assert jnp.allclose(controller.K, K)
    
    # Test controller force calculation
    state = jnp.array([1., 1., 0., 1., 1.])
    expected_force = -(K @ state)
    actual_force = controller.jit()(state, 0.0)
    assert jnp.isclose(actual_force, expected_force, atol=1e-6)


def test_linear_controller_batched():
    """Test linear controller with batched 5-state inputs"""
    K = jnp.array([1., 0., 0., 2., 0.])
    lin = LinearController(K=K).jit()
    
    # Create batch of 5-state vectors
    states = jnp.array([
        [1., 1., 0., 0., 0.],      # x=1
        [0., 1., 0., 1., 0.],      # x_dot=1  
        [0.5, 1., 0., 0.5, 0.]     # x=0.5, x_dot=0.5
    ])
    
    forces = lin(states, 0.0)
    expected_forces = jnp.array([-1., -2., -1.5])
    
    assert forces.shape == (3,)
    assert jnp.allclose(forces, expected_forces, atol=1e-6)


def test_linear_controller_five_state_format():
    """Test controller handles 5-state format correctly"""
    K = jnp.array([1., 2., 0.5, 1., 0.])
    controller = LinearController(K=K).jit()
    
    # Test with 5-state format [x, cos(θ), sin(θ), ẋ, θ̇] 
    theta = 0.1
    state_5 = jnp.array([1., jnp.cos(theta), jnp.sin(theta), 0.5, 0.2])
    
    force = controller(state_5, 0.0)
    expected_force = -(1.*1. + 2.*jnp.cos(theta) + 0.5*jnp.sin(theta) + 1.*0.5 + 0.*0.2)
    
    assert jnp.isclose(force, expected_force, atol=1e-6)


def test_linear_controller_angle_handling():
    """Test controller handles angle representation correctly"""
    K = jnp.array([0., 2., 0.5, 0., 0.])
    controller = LinearController(K=K).jit()
    
    # Test various angles
    angles = jnp.array([0., jnp.pi/6, jnp.pi/4, jnp.pi/2])
    
    for theta in angles:
        state = jnp.array([0., jnp.cos(theta), jnp.sin(theta), 0., 0.])
        force = controller(state, 0.0)
        expected_force = -(2.*jnp.cos(theta) + 0.5*jnp.sin(theta))
        assert jnp.isclose(force, expected_force, atol=1e-6)


def test_linear_controller_validation():
    """Test input validation for LinearController"""
    # Test wrong shape should raise error
    with pytest.raises(ValueError, match="K must have shape"):
        LinearController(K=jnp.array([1., 2.]))  # Wrong shape
    
    with pytest.raises(ValueError, match="K must have shape"):
        LinearController(K=jnp.array([1., 2., 3., 4., 5., 6.]))  # Wrong shape
    
    # Test correct shape should work
    controller = LinearController(K=jnp.array([1., 2., 3., 4., 5.]))
    assert controller.K.shape == (5,)


def test_factory_functions():
    """Test factory functions for creating controllers"""
    # Test PD controller creation
    pd_controller = create_pd_controller(kp_pos=2.0, kd_pos=1.5, kp_angle=15.0, kd_angle=1.8)
    expected_K = jnp.array([2.0, -15.0, 15.0, 1.5, 1.8])  # Updated for 5-state format
    assert jnp.allclose(pd_controller.K, expected_K)
    
    # Test zero controller
    zero_controller = create_zero_controller()
    assert jnp.allclose(zero_controller.K, jnp.zeros(5))
    
    # Test zero controller produces no force
    state = jnp.array([1., 1., 0., 1., 1.])
    force = zero_controller.jit()(state, 0.0)
    assert jnp.isclose(force, 0.0, atol=1e-6)


def test_controller_consistency():
    """Test that controller behavior is consistent across different call patterns"""
    K = jnp.array([1.5, 8.0, 0.8, 1.2, 2.0])
    controller = LinearController(K=K)
    jit_controller = controller.jit()
    
    test_state = jnp.array([0.2, 0.95, 0.31, 0.3, -0.1])
    
    # Test that jit and non-jit versions give same result
    force_eager = controller(test_state, 0.0)
    force_jit = jit_controller(test_state, 0.0)
    
    assert jnp.isclose(force_eager, force_jit, atol=1e-6)
    
    # Test that manual computation matches
    expected_force = -jnp.dot(K, test_state)
    assert jnp.isclose(force_jit, expected_force, atol=1e-6)


def test_controller_profiling():
    """Test controller profiling functionality"""
    K = jnp.array([1., 2., 3., 4., 5.])
    controller = LinearController(K=K).jit()
    
    state = jnp.array([0.1, 0.95, 0.31, 0.02, 0.01])
    
    # Test profiling mode
    force, latency = controller(state, 0.0, profile=True)
    
    # Check that force is computed correctly
    expected_force = -jnp.dot(K, state)
    assert jnp.isclose(force, expected_force, atol=1e-6)
    
    # Check that latency is reasonable (should be very small for compiled function)
    assert isinstance(latency, float)
    assert latency >= 0.0
    assert latency < 0.1  # Should be much less than 100ms


def test_batched_operation():
    """Test that batched operations work correctly"""
    K = jnp.array([1., 2., 3., 4., 5.])
    controller = LinearController(K=K).jit()
    
    # Single state vector
    state_1d = jnp.array([1., 0.8, 0.6, 1., 0.])
    force_1d = controller(state_1d, 0.0)
    expected_1d = -jnp.dot(K, state_1d)
    assert jnp.isclose(force_1d, expected_1d, atol=1e-6)
    assert jnp.isscalar(force_1d) or force_1d.shape == ()
    
    # Batch of states
    states_2d = jnp.array([
        [1., 0.8, 0.6, 1., 0.],
        [0., 1., 0., 0., 1.]
    ])
    forces_2d = controller(states_2d, 0.0)
    expected_2d = jnp.array([
        -jnp.dot(K, states_2d[0]),
        -jnp.dot(K, states_2d[1])
    ])
    assert jnp.allclose(forces_2d, expected_2d, atol=1e-6)
    assert forces_2d.shape == (2,)


def test_trigonometric_consistency():
    """Test that controller works with trigonometric state representation"""
    K = jnp.array([0., 1., 1., 0., 0.])
    controller = LinearController(K=K).jit()
    
    # Test that cos²θ + sin²θ = 1 constraint is maintained
    angles = jnp.linspace(0, 2*jnp.pi, 10)
    
    for theta in angles:
        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)
        
        # Verify trigonometric identity
        assert jnp.isclose(cos_theta**2 + sin_theta**2, 1.0, atol=1e-10)
        
        state = jnp.array([0., cos_theta, sin_theta, 0., 0.])
        force = controller(state, 0.0)
        expected_force = -(cos_theta + sin_theta)
        
        assert jnp.isclose(force, expected_force, atol=1e-6)


def test_physical_bounds():
    """Test controller with physically reasonable states"""
    K = jnp.array([1., 10., 10., 1., 1.])
    controller = LinearController(K=K).jit()
    
    # Test with various physical scenarios
    scenarios = [
        # [x, cos(θ), sin(θ), ẋ, θ̇]
        [0., 1., 0., 0., 0.],           # Upright equilibrium
        [0.1, 0.95, 0.31, 0., 0.],     # Small angle (~18°)
        [0., 0.7, 0.7, 0., 0.],        # 45° angle
        [0.5, 0.8, 0.6, 0.1, 0.1],     # Moving with angle
    ]
    
    for state in scenarios:
        state_array = jnp.array(state)
        force = controller(state_array, 0.0)
        
        # Force should be finite
        assert jnp.isfinite(force)
        
        # Force should be reasonable magnitude (not too large)
        assert jnp.abs(force) < 100.0  # Reasonable force limit


def test_linear_training_path():
    """Ensure basic training pipeline produces finite cost."""
    initial_K = jnp.array([1.0, -10.0, 10.0, 1.0, 1.0])
    initial_state = jnp.array([0.1, 0.95, 0.31, 0.0, 0.0])
    config = LinearTrainingConfig(num_iterations=5, trajectory_length=1.0, learning_rate=0.02)
    _, history = train_linear_controller(initial_K, initial_state, config)
    assert len(history.costs) > 0
    assert jnp.isfinite(jnp.array(history.costs)).all()


if __name__ == "__main__":
    # Run tests
    test_functions = [
        test_linear_controller_response,
        test_linear_controller_configuration,
        test_linear_controller_batched,
        test_linear_controller_five_state_format,
        test_linear_controller_angle_handling,
        test_factory_functions,
        test_controller_consistency,
        test_batched_operation,
        test_trigonometric_consistency,
        test_physical_bounds
    ]
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")
    
    print("All basic tests completed!")