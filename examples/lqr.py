"""
Run an LQR-controlled cart-pole rollout using the 5-state formulation
[x, cos(theta), sin(theta), xdot, thdot] and save a plot to results/trajectory_lqr.png.
"""
from __future__ import annotations

import time
import jax.numpy as jnp

from env.cartpole import CartPoleParams
from env.closedloop import simulate, create_time_grid
from controller.lqr_controller import LQRController
from lib.visualizer import plot_trajectory
from lib.training.lqr_training import train_lqr_controller, LQRTrainingConfig


def main():
    print("[INFO] Training Started for LQRController")

    params = CartPoleParams()

    # Time grid
    t_span = (0.0, 5.0)
    ts = create_time_grid(t_span, dt=0.01)

    # Initial state: small angle from upright, zero velocities (5D)
    y0 = jnp.array([0.0, 0.99, 0.1, 0.0, 0.0])

    t0 = time.perf_counter()
    lqr = train_lqr_controller(params, LQRTrainingConfig(print_data=True), print_data=True)
    t1 = time.perf_counter()
    print(f"[INFO] Training Finished for LQRController in {(t1 - t0):.2f} seconds")

    # Simulate and plot
    sol = simulate(lqr.jit(), params, t_span, ts, y0)
    plot_trajectory(sol.ys, sol.ts, title="LQR Trajectory", save_path="trajectory_lqr.png")


if __name__ == "__main__":
    main()