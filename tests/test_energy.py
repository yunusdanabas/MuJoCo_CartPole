import os
import psutil
import tracemalloc
import jax.numpy as jnp
from env.cartpole import CartPoleParams, dynamics
from env.helpers import total_energy

def _zero_ctrl(_, __): 
    return 0.0

def test_energy_conservation():
    """Test that energy is conserved during passive dynamics."""
    p = CartPoleParams()
    dt = 0.01
    s = jnp.array([0., jnp.pi - 0.1, 0., 0.])   # almost upright

    e0 = float(total_energy(s, p))
    for _ in range(1_000):
        s = s + dt * dynamics(s, 0., p, _zero_ctrl)
    e1 = float(total_energy(s, p))

    relative_error = abs(e1 - e0) / max(abs(e0), 1e-9)
    assert relative_error <= 1e-5, f"Energy drift too large: {relative_error:.2e}"

def test_memory_regression():
    """Test that memory usage doesn't grow excessively during long runs."""
    if os.getenv("CI") == "true":
        return  # Skip on CI
    
    p, dt, steps = CartPoleParams(), 0.001, 100_000
    s = jnp.zeros(4)
    proc = psutil.Process(os.getpid())

    tracemalloc.start()
    rss0 = proc.memory_info().rss
    
    for _ in range(steps):
        s = s + dt * dynamics(s, 0., p, _zero_ctrl)
    
    _ = float(s[0])  # sync
    rss1 = proc.memory_info().rss
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    rss_growth = (rss1 - rss0) / rss0
    peak_growth = (peak - rss0) / rss0
    
    assert rss_growth <= 0.05, f"RSS growth too large: {rss_growth:.1%}"
    assert peak_growth <= 0.05, f"Peak memory growth too large: {peak_growth:.1%}"

