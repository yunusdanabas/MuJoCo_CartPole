"""
LQR controller training and simulation example.
Demonstrates LQR controller training with improved cost matrices for smooth control.
"""

from __future__ import annotations

import time
import jax.numpy as jnp
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from env.cartpole import CartPoleParams
from env.closedloop import simulate, create_time_grid
from controller.lqr_controller import LQRController
from lib.visualizer import plot_trajectory
from lib.training.lqr_training import train_lqr_controller, LQRTrainingConfig


def main():
    """Train LQR controller and run simulation."""
    print("[INFO] Training Started for LQRController")

    # System parameters
    params = CartPoleParams()

    # Simulation setup
    t_span = (0.0, 10.0)
    ts = create_time_grid(t_span, dt=0.01)

    # Initial state: small angle from upright, zero velocities (5D)
    y0 = jnp.array([0.0, 0.99, 0.1, 0.0, 0.0])

    # Train LQR controller
    t0 = time.perf_counter()
    lqr = train_lqr_controller(params, LQRTrainingConfig(print_data=True), print_data=True)
    t1 = time.perf_counter()
    print(f"[INFO] Training Finished for LQRController in {(t1 - t0):.2f} seconds")
    
    # JIT compile for faster simulation
    lqr = lqr.jit()

    # Run simulation and plot results
    sol = simulate(lqr, params, t_span, ts, y0)
    plot_trajectory(
        sol.ys, 
        sol.ts, 
        title="LQR Trajectory", 
        save_path="trajectory_lqr.png", 
        show_plot=True
    )


if __name__ == "__main__":
    main()