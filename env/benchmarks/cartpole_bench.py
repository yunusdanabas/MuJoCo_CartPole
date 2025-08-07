#!/usr/bin/env python
"""
env/benchmarks/cartpole_bench.py

Cart-pole micro-benchmark.

Examples
--------
$ python -m env.benchmarks.cartpole_bench
$ python -m env.benchmarks.cartpole_bench --n 5e4 --batch 256 --profile
$ python -m env.benchmarks.cartpole_bench --device gpu
"""

from __future__ import annotations
import argparse
import time
import statistics
import json
import pathlib
import tracemalloc
import psutil

import jax
import jax.numpy as jnp
from env import CartPoleParams, dynamics  # Updated import

# -----------------------------------------------------------------------------#
# CLI                                                                           #
# -----------------------------------------------------------------------------#

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--n",      type=int,   default=10_000, help="# steps")
    p.add_argument("--batch",  type=int,   default=1,      help="batch size")
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--profile", action="store_true", help="dump cProfile stats")
    return p.parse_args()

# -----------------------------------------------------------------------------#
# Core helpers                                                                  #
# -----------------------------------------------------------------------------#

@jax.jit
def _step(s, p):
    """Single dynamics step with zero-force controller."""
    return dynamics(s, 0.0, p, lambda s, t: 0.), None

def _run(n, batch, params):
    """Run tight benchmark loop."""
    state = jnp.zeros((batch, 4))
    _step(state, params)  # warm-up JIT
    times_ns = []
    
    for _ in range(n):
        t0 = time.perf_counter_ns()
        state, _ = _step(state, params)
        times_ns.append(time.perf_counter_ns() - t0)
    
    return times_ns

# -----------------------------------------------------------------------------#
# Entry point                                                                   #
# -----------------------------------------------------------------------------#

def main() -> None:
    args = _cli()
    jax.config.update("jax_platform_name", args.device)

    # Optional profiling
    if args.profile:
        import cProfile
        prof = cProfile.Profile()
        prof.enable()

    # Memory tracking
    tracemalloc.start()
    times = _run(args.n, args.batch, CartPoleParams())
    peak = tracemalloc.get_traced_memory()[1] / 2**20  # MiB
    tracemalloc.stop()

    # Save profile if requested
    if args.profile:
        prof.disable()
        out = pathlib.Path("results/profile_env.pstats")
        out.parent.mkdir(exist_ok=True, parents=True)
        prof.dump_stats(out)
        print(f"[profile] saved → {out}")

    # Compute statistics
    mean = statistics.mean(times) / 1e3  # µs
    p99 = statistics.quantiles(times, n=100)[-1] / 1e3
    
    print(f"mean {mean:.2f} µs | 99-th % {p99:.2f} µs | peak RSS {peak:.1f} MiB")

    # Save results for CI/trending
    pathlib.Path("results").mkdir(exist_ok=True)
    with open("results/bench_last.json", "w") as fp:
        json.dump({
            "mean_us": mean, 
            "p99_us": p99, 
            "peak_mb": peak,
            "batch_size": args.batch,
            "n_steps": args.n,
            "device": args.device
        }, fp, indent=2)

if __name__ == "__main__":
    main()