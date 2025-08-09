"""
examples/basic_lqr.py

Run a basic LQR-controlled cart-pole rollout using the 5-state formulation
[x, cos(theta), sin(theta), xdot, thdot] and save a plot to results/trajectory_lqr.png.
"""
from __future__ import annotations

import jax.numpy as jnp

from env.cartpole import CartPoleParams
from env.closedloop import simulate, create_time_grid
from env.helpers import four_to_five
from controller.lqr_controller import LQRController
from lib.visualizer import plot_trajectory

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from time import perf_counter

def main():

    print("Starting LQR simulation...")

    params = CartPoleParams()

    # Time grid
    t_span = (0.0, 5.0)
    ts = create_time_grid(t_span, dt=0.01)

    # Initial state: small angle from upright, zero velocities (4D -> 5D)
    init_4d = jnp.array([0.0, 0.1, 0.0, 0.0])
    y0 = four_to_five(init_4d)

    # LQR controller
    t0 = perf_counter()
    lqr = LQRController.from_linearisation(params).jit()
    t1 = perf_counter()
    # Optional print to mimic training logs for consistency in examples
    print("[TRAIN] LQRController started")
    print(f"[TRAIN] iter=0 time={(t1 - t0):.6f}s loss=0.000000")
    print(f"[TRAIN] LQRController finished in {(t1 - t0):.6f}s")

    # Simulate and plot
    sol = simulate(lqr, params, t_span, ts, y0)
    plot_trajectory(sol.ys, sol.ts, title="LQR Trajectory", save_path="trajectory_lqr.png")
    print("LQR simulation completed.")

if __name__ == "__main__":
    main()