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
from lib.training.nn_training import train, TrainConfig
from lib.visualizer import compare_trajectories


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
    sol_linear = simulate(linear, params, t_span, ts, y0_5)

    # 2) LQR
    lqr = LQRController.from_linearisation(params).jit()
    sol_lqr = simulate(lqr, params, t_span, ts, y0_5)

    # 3) NN (brief training)
    nn = NNController.init(seed=0)
    cfg = TrainConfig(batch_size=32, num_epochs=10, print_data=True, t_span=t_span, ts=ts)
    nn_trained, _ = train(nn, params, cfg)
    sol_nn = simulate(nn_trained.jit(), params, t_span, ts, y0_5)

    # Overlay plot
    compare_trajectories(
        [sol_linear.ys, sol_lqr.ys, sol_nn.ys],
        ["Linear", "LQR", "NN"],
        title="Linear vs LQR vs NN",
        save_path="trajectory_combo.png",
    )


if __name__ == "__main__":
    main()