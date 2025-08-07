# tests/test_helpers.py
import jax.numpy as jnp
from lib.utils import clamp, clip_by_norm, safe_divide, normalize_angle


def test_helper_function_clamp():
    """Test clamp helper function"""
    # Test basic clamping
    x = jnp.array([-2., 0., 2.])
    y = clamp(x, -1., 1.)
    assert y.min() >= -1 and y.max() <= 1
    
    # Test that values within bounds are unchanged
    x_within = jnp.array([-0.5, 0., 0.5])
    y_within = clamp(x_within, -1., 1.)
    assert jnp.allclose(x_within, y_within)


def test_helper_function_clip_by_norm():
    """Test clip_by_norm helper function"""
    # Test vector with norm > max_norm
    v = jnp.array([3.0, 4.0])  # norm = 5
    clipped = clip_by_norm(v, 2.0)
    assert abs(jnp.linalg.norm(clipped) - 2.0) < 1e-6
    
    # Test vector with norm < max_norm (should be unchanged)
    v_small = jnp.array([0.5, 0.5])  # norm < 1
    clipped_small = clip_by_norm(v_small, 2.0)
    assert jnp.allclose(v_small, clipped_small)


def test_helper_function_edge_cases():
    """Test helper functions with edge cases"""
    # Test clamp with equal bounds
    x = jnp.array([-1., 0., 1.])
    y = clamp(x, 0., 0.)
    assert jnp.all(y == 0.)
    
    # Test clip_by_norm with zero vector
    zero_vec = jnp.zeros(3)
    clipped_zero = clip_by_norm(zero_vec, 1.0)
    assert jnp.allclose(zero_vec, clipped_zero)


def test_helper_safe_divide():
    """Test safe_divide helper function"""
    # Test normal division
    result = safe_divide(jnp.array([4.0, 6.0]), jnp.array([2.0, 3.0]))
    expected = jnp.array([2.0, 2.0])
    assert jnp.allclose(result, expected)
    
    # Test division by zero
    result_zero = safe_divide(jnp.array([1.0, 2.0]), jnp.array([0.0, 0.0]))
    assert jnp.all(jnp.isfinite(result_zero))


def test_helper_normalize_angle():
    """Test normalize_angle helper function"""
    # Test wrapping of large positive angle
    large_positive = normalize_angle(5*jnp.pi/2)
    assert jnp.isclose(large_positive, jnp.pi/2, atol=1e-6)
    
    # Test wrapping of large negative angle
    large_negative = normalize_angle(-5*jnp.pi/2)
    assert jnp.isclose(large_negative, -jnp.pi/2, atol=1e-6)
    
    # Test angle already in range
    normal_angle = normalize_angle(jnp.pi/4)
    assert jnp.isclose(normal_angle, jnp.pi/4, atol=1e-6)