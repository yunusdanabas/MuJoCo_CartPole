"""
examples/basic_nn.py

Run a basic NN-controlled cart-pole swing-up using the 5-state formulation
[x, cos(theta), sin(theta), xdot, thdot] and save a plot to results/trajectory_nn.png.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from env.cartpole import CartPoleParams
from env.closedloop import simulate, create_time_grid
from controller.nn_controller import NNController
from lib.visualizer import plot_trajectory
from lib.trainer import train, TrainConfig

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

def main():
    params = CartPoleParams()

    # Time grid
    t_span = (0.0, 8.0)
    ts = create_time_grid(t_span, dt=0.01)

    # Initial state: downwards, zero velocities (5D)
    y0 = jnp.array([0.0, -1.0, 0.0, 0.0, 0.0])

    # Deterministic NN controller (train briefly)
    nn = NNController.init(seed=0)
    cfg = TrainConfig(batch_size=32, num_epochs=10, print_data=True, t_span=t_span, ts=ts)
    nn_trained, _ = train(nn, params, cfg)

    # Simulate and plot
    sol = simulate(nn_trained.jit(), params, t_span, ts, y0)
    plot_trajectory(sol.ys, sol.ts, title="NN Trajectory", save_path="trajectory_nn.png")


if __name__ == "__main__":
    main()