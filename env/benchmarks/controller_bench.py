#!/usr/bin/env python
"""
env/benchmarks/controller_bench.py

Controller latency + return-to-upright benchmark.

Measures:
    • mean / 99-th % call latency of controller.__call__
    • crude performance score (angle-error integral)
Outputs JSON to results/controller_bench_last.json
"""

from __future__ import annotations
import argparse
import statistics
import json
import pathlib
import time

import jax
import jax.numpy as jnp
from env import CartPoleParams, simulate_batch  

# Import controllers
from controller.linear_controller import LinearController
from controller.lqr_controller import LQRController
from controller.nn_controller import NNController

# -------------------------------------------------------------------------- #
# Helper functions                                                             #
# -------------------------------------------------------------------------- #

def _angle_error(ys):
    """Compute mean absolute angle error over trajectory."""
    th = ys[..., 1]
    return jnp.mean(jnp.abs(th))

def _latency_benchmark(ctrl, batch_states, n_trials=100):
    """Measure controller call latency statistics."""
    times_ns = []
    
    # Warm-up
    for _ in range(10):
        _ = ctrl.batched()(batch_states, 0.0)
        jax.block_until_ready(_)
    
    # Actual timing
    for _ in range(n_trials):
        t0 = time.perf_counter_ns()
        forces = ctrl.batched()(batch_states, 0.0)
        jax.block_until_ready(forces)
        times_ns.append(time.perf_counter_ns() - t0)
    
    # Convert to per-element times
    per_element_ns = [t / batch_states.shape[0] for t in times_ns]
    
    return {
        'mean_ns': statistics.mean(per_element_ns),
        'p99_ns': statistics.quantiles(per_element_ns, n=100)[-1],
        'std_ns': statistics.stdev(per_element_ns)
    }

def _performance_score(ctrl, params, initial_states, t_span=(0.0, 2.0)):
    """Compute performance score for swing-up task."""
    ts = jnp.linspace(t_span[0], t_span[1], 401)
    
    try:
        sol = simulate_batch(ctrl.batched(), params, t_span, ts, initial_states)  # Updated function name
        angle_error = float(_angle_error(sol.ys))
        
        # Check for stability (states shouldn't blow up)
        max_pos = float(jnp.max(jnp.abs(sol.ys[..., 0])))
        if max_pos > 10.0:  # Cart moved too far
            return float('inf')
            
        return angle_error
    except Exception:
        return float('inf')

def _make_controller(kind, key=None):
    """Factory function for creating controllers."""
    if kind == "linear":
        # Hand-tuned gains
        return LinearController(K=jnp.array([10., 0., 5., 0.])).jit()
    elif kind == "lqr":
        return LQRController.from_linearisation(CartPoleParams()).jit()
    elif kind == "nn":
        if key is None:
            key = jax.random.PRNGKey(0)
        return NNController.init(hidden_dims=(64, 64), key=key).jit()
    else:
        raise ValueError(f"Unknown controller type: {kind}")

# -------------------------------------------------------------------------- #
# Main benchmark                                                               #
# -------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Controller benchmark")
    parser.add_argument("--controller", choices=["linear", "lqr", "nn"], 
                       default="lqr", help="Controller type to benchmark")
    parser.add_argument("--batch", type=int, default=256, 
                       help="Batch size for latency test")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                       help="JAX device to use")
    parser.add_argument("--trials", type=int, default=100,
                       help="Number of timing trials")
    args = parser.parse_args()

    # Setup
    jax.config.update("jax_platform_name", args.device)
    params = CartPoleParams()
    key = jax.random.PRNGKey(42)
    
    # Create controller
    ctrl = _make_controller(args.controller, key)
    
    print(f"Benchmarking {args.controller.upper()} controller on {args.device.upper()}")
    print("-" * 60)
    
    # ---------- Latency benchmark ----------------------------------------- #
    print("Running latency benchmark...")
    states = jnp.zeros((args.batch, 4))
    latency_stats = _latency_benchmark(ctrl, states, args.trials)
    
    mean_us = latency_stats['mean_ns'] / 1e3
    p99_us = latency_stats['p99_ns'] / 1e3
    std_us = latency_stats['std_ns'] / 1e3
    
    print(f"Latency (per call): {mean_us:.2f} ± {std_us:.2f} µs")
    print(f"99th percentile: {p99_us:.2f} µs")
    
    # ---------- Performance score ----------------------------------------- #
    print("Running performance benchmark...")
    
    # Test scenarios
    scenarios = {
        'upright_small': jnp.array([[0.0, 0.1, 0.0, 0.0]] * args.batch),  # Small perturbation
        'hanging_down': jnp.array([[0.0, jnp.pi, 0.0, 0.0]] * args.batch),  # Swing-up
        'random_start': jax.random.uniform(key, (args.batch, 4), 
                                         minval=jnp.array([-1.0, -jnp.pi, -0.5, -0.5]),
                                         maxval=jnp.array([1.0, jnp.pi, 0.5, 0.5]))
    }
    
    scores = {}
    for scenario_name, initial_states in scenarios.items():
        score = _performance_score(ctrl, params, initial_states)
        scores[scenario_name] = score
        print(f"{scenario_name:>15}: {score:.3f}")
    
    # Overall score (weighted average)
    overall_score = (
        0.3 * scores['upright_small'] + 
        0.5 * scores['hanging_down'] + 
        0.2 * scores['random_start']
    )
    
    print(f"{'Overall score':>15}: {overall_score:.3f}")
    print("-" * 60)
    
    # ---------- Save results ---------------------------------------------- #
    results = {
        'controller': args.controller,
        'device': args.device,
        'batch_size': args.batch,
        'latency': {
            'mean_us': float(mean_us),
            'p99_us': float(p99_us),
            'std_us': float(std_us)
        },
        'performance': {
            'scenarios': {k: float(v) for k, v in scores.items()},
            'overall_score': float(overall_score)
        }
    }
    
    pathlib.Path("results").mkdir(exist_ok=True)
    with open("results/controller_bench_last.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to results/controller_bench_last.json")


if __name__ == "__main__":
    main()