import pytest
from dataclasses import dataclass
import time
import jax
import jax.numpy as jnp
from controller.base import Controller, _jit_cache


@dataclass(frozen=True)
class DummyCtrl(Controller):
    gain: float = 1.0

    def _force(self, state, _t):
        return -self.gain * jnp.sum(state)  # More realistic: use all state components


@dataclass(frozen=True)  
class MultiDimCtrl(Controller):
    """Controller that returns multi-dimensional output for testing edge cases"""
    
    def _force(self, state, _t):
        return jnp.array([state[0], -state[1]])  # 2D output


# --------------------------------------------------------------------------- #
# Core functionality tests                                                    #
# --------------------------------------------------------------------------- #

def test_batched_and_scalar():
    """Test that controller works for both scalar and batched inputs"""
    ctrl = DummyCtrl(gain=2.0).jit()

    # Test scalar input
    s = jnp.array([1., 0.5, 0.2, 0.1])
    expected_scalar = -2.0 * jnp.sum(s)
    assert jnp.allclose(ctrl(s, 0.0), expected_scalar)

    # Test batched input
    states = jnp.stack([s, 0.5 * s, 2.0 * s])
    forces = ctrl.batched()(states, 0.0)
    assert forces.shape == (3,)
    expected_batch = jnp.array([expected_scalar, 0.5 * expected_scalar, 2.0 * expected_scalar])
    assert jnp.allclose(forces, expected_batch)


def test_controller_interface():
    """Test that controller follows the base interface"""
    ctrl = DummyCtrl(gain=1.0)
    
    # Test that it has required methods
    assert hasattr(ctrl, '_force')
    assert hasattr(ctrl, 'jit')
    assert hasattr(ctrl, 'batched')
    assert hasattr(ctrl, '__call__')
    
    # Test that _force returns appropriate shape
    state = jnp.array([1., 0., 0., 0.])
    force = ctrl._force(state, 0.0)
    assert jnp.isscalar(force) or force.shape == ()


# --------------------------------------------------------------------------- #
# JIT compilation and caching tests                                          #
# --------------------------------------------------------------------------- #

def test_jit_compilation():
    """Test JIT compilation behavior"""
    ctrl = DummyCtrl(gain=3.0)
    
    # Before JIT
    assert ctrl._compiled is None
    
    # After JIT  
    jit_ctrl = ctrl.jit()
    assert jit_ctrl._compiled is not None
    assert callable(jit_ctrl._compiled)
    
    # Test that JIT controller produces same results
    state = jnp.array([1., 2., 3., 4.])
    eager_result = ctrl(state, 0.0)
    jit_result = jit_ctrl(state, 0.0)
    assert jnp.allclose(eager_result, jit_result)


def test_cache_reuse():
    """Test that identical controllers share compiled functions"""
    # Clear cache for clean test
    _jit_cache.clear()
    
    ctrl1 = DummyCtrl(gain=2.0)
    ctrl2 = DummyCtrl(gain=2.0)  # Identical parameters
    ctrl3 = DummyCtrl(gain=3.0)  # Different parameters
    
    # JIT compile all
    jit_ctrl1 = ctrl1.jit()
    jit_ctrl2 = ctrl2.jit()
    jit_ctrl3 = ctrl3.jit()
    
    # Check cache usage
    assert len(_jit_cache) == 2  # Only 2 unique compiled functions
    assert jit_ctrl1._compiled is jit_ctrl2._compiled  # Same compiled function
    assert jit_ctrl1._compiled is not jit_ctrl3._compiled  # Different compiled function


def test_in_place_modification():
    """Test that jit() modifies controller in-place correctly"""
    ctrl = DummyCtrl(gain=5.0)
    original_id = id(ctrl)
    
    # JIT should return the same object
    jit_ctrl = ctrl.jit()
    assert id(jit_ctrl) == original_id
    assert ctrl._compiled is not None
    
    # Calling jit() again should return same object
    jit_ctrl2 = ctrl.jit()
    assert id(jit_ctrl2) == original_id


# --------------------------------------------------------------------------- #
# Profiling tests                                                             #
# --------------------------------------------------------------------------- #

def test_profiling_mode():
    """Test profiling functionality"""
    ctrl = DummyCtrl(gain=1.0).jit()
    state = jnp.array([1., 0., 0., 0.])
    
    # Normal mode
    force_normal = ctrl(state, 0.0, profile=False)
    assert jnp.isscalar(force_normal)
    
    # Profiling mode
    result = ctrl(state, 0.0, profile=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    
    force_profiled, latency = result
    assert jnp.allclose(force_normal, force_profiled)
    assert isinstance(latency, float)
    assert latency > 0  # Should have some positive latency


def test_profiling_accuracy():
    """Test that profiling gives reasonable timing results"""
    ctrl = DummyCtrl(gain=1.0).jit()
    state = jnp.array([1., 0., 0., 0.])
    
    # Warm-up JIT
    ctrl(state, 0.0)
    
    # Multiple profiling runs
    latencies = []
    for _ in range(5):
        _, latency = ctrl(state, 0.0, profile=True)
        latencies.append(latency)
    
    # Check reasonable bounds (should be microseconds, not seconds)
    avg_latency = sum(latencies) / len(latencies)
    assert 1e-6 < avg_latency < 1e-2  # Between 1μs and 10ms


# --------------------------------------------------------------------------- #
# Edge cases and error handling                                               #
# --------------------------------------------------------------------------- #

def test_multi_dimensional_output():
    """Test controller with multi-dimensional output"""
    ctrl = MultiDimCtrl().jit()
    
    # Single state
    state = jnp.array([1., 2., 0., 0.])
    force = ctrl(state, 0.0)
    assert force.shape == (2,)
    assert jnp.allclose(force, jnp.array([1., -2.]))
    
    # Batch states
    states = jnp.stack([state, 2 * state])
    forces = ctrl.batched()(states, 0.0)
    assert forces.shape == (2, 2)


def test_different_state_dimensions():
    """Test with different input state dimensions"""
    ctrl = DummyCtrl(gain=1.0).jit()
    
    # Test with different state sizes
    state_3d = jnp.array([1., 2., 3.])
    state_5d = jnp.array([1., 2., 3., 4., 5.])
    
    force_3d = ctrl(state_3d, 0.0)
    force_5d = ctrl(state_5d, 0.0)
    
    assert jnp.isscalar(force_3d)
    assert jnp.isscalar(force_5d)
    assert force_3d == -6.0  # sum([1,2,3])
    assert force_5d == -15.0  # sum([1,2,3,4,5])


def test_time_parameter():
    """Test that time parameter is properly passed"""
    @dataclass(frozen=True)
    class TimeAwareCtrl(Controller):
        def _force(self, state, t):
            return t * jnp.sum(state)  # Force depends on time
    
    ctrl = TimeAwareCtrl().jit()
    state = jnp.array([1., 1., 1., 1.])
    
    force_t0 = ctrl(state, 0.0)
    force_t1 = ctrl(state, 1.0)
    force_t2 = ctrl(state, 2.0)
    
    assert force_t0 == 0.0
    assert force_t1 == 4.0
    assert force_t2 == 8.0


def test_large_batch_performance():
    """Test performance with large batches"""
    ctrl = DummyCtrl(gain=1.0).jit()
    
    # Large batch
    batch_size = 1000
    states = jnp.ones((batch_size, 4))
    
    # Should handle large batches without errors
    forces = ctrl.batched()(states, 0.0)
    assert forces.shape == (batch_size,)
    assert jnp.all(forces == -4.0)  # Each state sums to 4


# --------------------------------------------------------------------------- #
# Performance regression tests                                                #
# --------------------------------------------------------------------------- #

def test_compilation_overhead():
    """Test that JIT compilation doesn't add excessive overhead"""
    ctrl = DummyCtrl(gain=1.0)
    state = jnp.array([1., 0., 0., 0.])
    
    # Time eager execution
    start = time.perf_counter()
    for _ in range(100):
        ctrl(state, 0.0)
    eager_time = time.perf_counter() - start
    
    # Time JIT execution (after warm-up)
    jit_ctrl = ctrl.jit()
    jit_ctrl(state, 0.0)  # Warm-up
    
    start = time.perf_counter()
    for _ in range(100):
        jit_ctrl(state, 0.0)
    jit_time = time.perf_counter() - start
    
    # JIT should be faster (or at least not much slower)
    print(f"Eager: {eager_time:.4f}s, JIT: {jit_time:.4f}s")
    # Don't assert performance here as it's hardware dependent
    # but print for manual inspection


# --------------------------------------------------------------------------- #
# Integration tests                                                           #
# --------------------------------------------------------------------------- #

def test_with_jax_transformations():
    """Test that controllers work with JAX transformations"""
    ctrl = DummyCtrl(gain=1.0).jit()
    
    # Test with vmap
    states = jnp.ones((5, 4))
    batch_fn = jax.vmap(lambda s: ctrl(s, 0.0))
    forces = batch_fn(states)
    assert forces.shape == (5,)
    
    # Test with grad (if controller is differentiable)
    def loss_fn(gain_val):
        test_ctrl = DummyCtrl(gain=gain_val).jit()
        return test_ctrl(jnp.ones(4), 0.0) ** 2
    
    grad_fn = jax.grad(loss_fn)
    gradient = grad_fn(2.0)
    assert jnp.isfinite(gradient)


if __name__ == "__main__":
    # Run a quick smoke test
    test_batched_and_scalar()
    test_controller_interface()
    test_jit_compilation()
    test_cache_reuse()
    print("✅ All tests passed!")