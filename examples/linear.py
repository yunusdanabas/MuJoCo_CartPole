"""Linear controller training example with progress logging and final plot."""

from __future__ import annotations

import time
import jax.numpy as jnp

from lib.training.linear_training import LinearTrainingConfig, train_linear_controller
from env.cartpole import CartPoleParams
from env.closedloop import simulate, create_time_grid
from lib.visualizer import plot_trajectory


def main():
    print("[INFO] Training Started for LinearController")

    initial_K = jnp.array([1.0, -10.0, 10.0, 1.0, 1.0])
    initial_state = jnp.array([0.1, 0.95, 0.31, 0.0, 0.0])

    total_iters = 300
    config = LinearTrainingConfig(learning_rate=0.02, num_iterations=total_iters, trajectory_length=2.0, batch_size=16, print_data=True)

    t0 = time.perf_counter()
    controller, history = train_linear_controller(initial_K, initial_state, config, print_data=True)
    t1 = time.perf_counter()
    print(f"[INFO] Training Finished for LinearController in {(t1 - t0):.2f} seconds")

    # Final rollout and plot
    params = CartPoleParams()
    t_span = (0.0, 5.0)
    ts = create_time_grid(t_span, dt=0.01)
    y0 = initial_state
    sol = simulate(controller.jit(), params, t_span, ts, y0)
    plot_trajectory(sol.ys, sol.ts, title="Linear Trajectory", save_path="trajectory_linear.png")

    return controller


if __name__ == "__main__":
    main()

