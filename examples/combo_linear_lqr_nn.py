"""
examples/combo_linear_lqr_nn.py

Run linear (PD), LQR, and NN controllers sequentially for quick comparison.
Saves figures to results/trajectory_linear.png, results/trajectory_lqr.png, results/trajectory_nn.png.
"""
from __future__ import annotations

import jax.numpy as jnp

from env.cartpole import CartPoleParams
from env.closedloop import simulate, create_time_grid
from env.helpers import four_to_five
from controller.linear_controller import create_pd_controller
from controller.lqr_controller import LQRController
from controller.nn_controller import NNController
from lib.visualizer import plot_trajectory


def run_and_plot(ctrl, params, t_span, ts, y0, title, outfile):
    sol = simulate(ctrl, params, t_span, ts, y0)
    plot_trajectory(sol.ys, sol.ts, title=title, save_path=outfile)


def main():
    params = CartPoleParams()
    t_span = (0.0, 6.0)
    ts = create_time_grid(t_span, dt=0.01)

    # Initial conditions
    init_4d = jnp.array([0.2, 0.2, 0.0, 0.0])
    y0_5 = four_to_five(init_4d)

    # 1) Linear PD
    linear = create_pd_controller(kp_pos=2.0, kd_pos=1.0, kp_angle=20.0, kd_angle=2.0).jit()
    run_and_plot(linear, params, t_span, ts, y0_5, "Linear PD Trajectory", "trajectory_linear.png")

    # 2) LQR
    lqr = LQRController.from_linearisation(params).jit()
    run_and_plot(lqr, params, t_span, ts, y0_5, "LQR Trajectory", "trajectory_lqr.png")

    # 3) NN
    nn = NNController.init(seed=0).jit()
    run_and_plot(nn, params, t_span, ts, y0_5, "NN Trajectory", "trajectory_nn.png")


if __name__ == "__main__":
    main()