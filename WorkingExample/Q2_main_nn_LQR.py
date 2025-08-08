# Q2_main_nn_LQR.py (refactored)
# Demonstrate NN controller and LQR controller using the 5-state environment.

from __future__ import annotations

import jax
import jax.numpy as jnp

from env.cartpole import CartPoleParams
from env.closedloop import simulate, create_time_grid
from controller.nn_controller import NNController
from controller.lqr_controller import LQRController
from lib.visualizer import compare_trajectories


def main():
    params = CartPoleParams()

    # Time grid
    t_span = (0.0, 8.0)
    ts = create_time_grid(t_span, dt=0.01)

    # Initial state (5D): downwards, zero velocities
    y0 = jnp.array([0.0, -1.0, 0.0, 0.0, 0.0])

    # 1) NN Controller (deterministic weights)
    nn = NNController.init(seed=0).jit()
    sol_nn = simulate(nn, params, t_span, ts, y0)

    # 2) LQR Controller
    lqr = LQRController.from_linearisation(params).jit()
    sol_lqr = simulate(lqr, params, t_span, ts, y0)

    # Plot comparison
    compare_trajectories(
        [sol_nn.ys, sol_lqr.ys],
        ["NN", "LQR"],
        title="NN vs LQR (5-state)",
        save_path="trajectory_nn_lqr.png",
    )


if __name__ == "__main__":
    main()
