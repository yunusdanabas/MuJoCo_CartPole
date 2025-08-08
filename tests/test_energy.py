"""
tests/test_energy.py
--------------------

Guards:
1. Energy drift ≤ 1 e-5 over 1 000 passive Euler steps.
2. RSS / peak-alloc growth ≤ 5 % over 100 k steps.

Skip the long memory test automatically on CI
(by setting CI=true in your workflow).
"""

from __future__ import annotations
import os
import psutil
import tracemalloc
import jax.numpy as jnp
from env.cartpole import CartPoleParams, dynamics
from env.helpers import total_energy


def _zero_ctrl(_, __):
    """Zero control input for passive dynamics"""
    return 0.0


def test_energy_conservation():
    """No-control dynamics should conserve mechanical energy."""
    params = CartPoleParams()
    dt = 0.01
    state = jnp.array([0.0, jnp.pi - 0.1, 0.0, 0.0, 0.0])  # near-upright 5-state

    e0 = float(total_energy(state, params))
    for _ in range(1_000):
        state = state + dt * dynamics(state, 0.0, params, _zero_ctrl)
        
    e1 = float(total_energy(state, params))

    rel_err = abs(e1 - e0) / max(abs(e0), 1e-9)
    assert rel_err <= 1e-5, f"energy drift {rel_err:.2e} > 1e-5"


def test_memory_regression():
    """Long passive run should not leak (> 5 % extra RSS)."""
    if os.getenv("CI") == "true":
        return  # Skip on CI
        
    params, dt, steps = CartPoleParams(), 0.001, 100_000
    state = jnp.zeros(5)
    proc = psutil.Process(os.getpid())

    tracemalloc.start()
    rss0 = proc.memory_info().rss

    for _ in range(steps):
        state = state + dt * dynamics(state, 0.0, params, _zero_ctrl)

    # host/device sync before measuring RSS
    _ = float(state[0])

    rss1 = proc.memory_info().rss
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    rss_growth = (rss1 - rss0) / rss0
    peak_growth = (peak - rss0) / rss0

    assert rss_growth <= 0.05, f"RSS grew {rss_growth:.1%} (> 5 %)"
    assert peak_growth <= 0.05, f"peak alloc grew {peak_growth:.1%} (> 5 %)"
