import jax.numpy as jnp
import pytest
from lib.utils import clamp, clip_by_norm, safe_divide, normalize_angle


# Test data
edge_case_input = jnp.array([0.0, 0.0, 0.0, 0.0])
typical_input = jnp.array([0.1, 0.0, 0.1, 0.0])


def test_basic_imports():
    """Test that utils module imports successfully"""
    assert True


def test_clamp_function():
    """Test clamp utility function"""
    x = jnp.array([-2., 0., 2.])
    y = clamp(x, -1., 1.)
    assert y.min() >= -1 and y.max() <= 1
    
    # Test edge cases
    assert jnp.allclose(clamp(jnp.array([0.5]), -1., 1.), jnp.array([0.5]))
    assert jnp.allclose(clamp(jnp.array([2.0]), -1., 1.), jnp.array([1.0]))
    assert jnp.allclose(clamp(jnp.array([-2.0]), -1., 1.), jnp.array([-1.0]))


def test_clip_by_norm():
    """Test clip_by_norm utility function"""
    v = jnp.array([3.0, 4.0])  # norm 5
    clipped = clip_by_norm(v, 2.0)
    assert abs(jnp.linalg.norm(clipped) - 2.0) < 1e-6
    
    # Test that small vectors are unchanged
    small_v = jnp.array([0.5, 0.5])
    clipped_small = clip_by_norm(small_v, 2.0)
    assert jnp.allclose(small_v, clipped_small)


def test_utility_functions_with_edge_cases():
    """Test utility functions with edge case inputs"""
    # Test with zero input
    zero_result = clamp(edge_case_input, -1., 1.)
    assert jnp.allclose(zero_result, edge_case_input)
    
    # Test with typical input
    typical_result = clamp(typical_input, -1., 1.)
    assert jnp.all(typical_result >= -1.) and jnp.all(typical_result <= 1.)


def test_utility_functions_shapes():
    """Test that utility functions preserve shapes correctly"""
    test_array = jnp.array([[1., 2.], [3., 4.]])
    clamped = clamp(test_array, 0., 3.)
    assert clamped.shape == test_array.shape
    
    # Test clip_by_norm with 2D input
    clipped = clip_by_norm(test_array.flatten(), 2.0)
    assert clipped.shape == test_array.flatten().shape


def test_safe_divide():
    """Test safe division utility function"""
    numerator = jnp.array([1.0, 2.0, 3.0])
    denominator = jnp.array([1.0, 0.0, 2.0])
    
    result = safe_divide(numerator, denominator)
    assert jnp.all(jnp.isfinite(result))
    
    # Test with all zeros
    zero_denom = jnp.zeros(3)
    result_zero = safe_divide(numerator, zero_denom)
    assert jnp.all(jnp.isfinite(result_zero))


def test_normalize_angle():
    """Test angle normalization utility function"""
    # Test angles outside [-π, π]
    angles = jnp.array([3*jnp.pi, -3*jnp.pi, jnp.pi/2, 0.0])
    normalized = normalize_angle(angles)
    
    # All should be in [-π, π]
    assert jnp.all(normalized >= -jnp.pi)
    assert jnp.all(normalized <= jnp.pi)
    
    # Test specific values
    assert jnp.isclose(normalize_angle(3*jnp.pi), -jnp.pi, atol=1e-6)
    assert jnp.isclose(normalize_angle(jnp.pi/2), jnp.pi/2, atol=1e-6)