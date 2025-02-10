# file: train_nn_controller.py

import jax.random as rnd
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import equinox as eqx

from controller.neuralnetwork_controller import CartPolePolicy
from lib.trainer import train_nn_controller, rollout_once
from lib.utils import plot_trajectories3, plot_cost, sample_initial_conditions


def main():
    # 1) System parameters
    mc = 1.0
    mp = 1.0
    l = 1.0
    g = 9.81
    params = (mc, mp, l, g)

    # 2) Create random key & instantiate the NN policy
    key = rnd.PRNGKey(31)
    model_key, _ = rnd.split(key)
    # Let's specify hidden_dims if you want more layers
    nn_policy = CartPolePolicy(model_key, in_dim=5, hidden_dims=(64, 128, 64), out_dim=1)

    # 3) Train
    print("Training")
    trained_policy, cost_history = train_nn_controller(
        nn_policy,
        params=params,
        num_iterations=3_000,
        batch_size=32,
        T=7.0,
        dt=0.01,
        lr=1e-4,
        key=key
    )

    # 4) Plot training cost
    plot_cost(np.array(cost_history), title="NN Training Cost")

    # 5) Test with an initial condition near downward: (x=0, theta=pi, etc.)
    test_ic = jnp.array([0.0, jnp.pi, 0.0, 0.0])  # shape (4,) -> will convert to (5,) internally
    total_cost, states, times, forces = rollout_once(trained_policy, params, test_ic, T=7.0, dt=0.01)
    print(f"Test Rollout Cost: {float(total_cost):.4f}")

    # 6) Extract theta from cos/sin
    # states: shape (N, 5) = [x, cos_th, sin_th, x_dot, th_dot]
    x_vals = states[:, 0]
    cos_th = states[:, 1]
    sin_th = states[:, 2]
    x_dot_vals = states[:, 3]
    th_dot_vals = states[:, 4]
    theta = jnp.arctan2(sin_th, cos_th)

    print("\nSample of final states:")
    print("  x =", x_vals[-5:])
    print("  theta =", theta[-5:])

    # Quick plot of theta vs time
    plt.figure()
    plt.plot(times, theta, label="Theta (rad)")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.title("Pendulum Angle (Theta) Over Time")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 7) Save the trained model
    eqx.tree_serialise_leaves("trained_nn_model.eqx", trained_policy)
    print("Saved trained model to 'trained_nn_model.eqx'.")

if __name__ == "__main__":
    main()
