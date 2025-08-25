"""Demonstrate advanced linear controller training features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lib.training.linear_training import LinearTrainingConfig, train_linear_controller
from controller.lqr_controller import _linearise
from env.cartpole import CartPoleParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Run with fewer iterations")
    args = parser.parse_args()

    iters = 20 if args.fast else 100

    config = LinearTrainingConfig(
        learning_rate=0.02,
        num_iterations=iters,
        trajectory_length=2.0,
        batch_size=32,
        lr_schedule="cosine",
        stability_weight=0.1,
        lqr_warm_start=True,
    )

    initial_state = jnp.array([0.1, 0.95, 0.31, 0.0, 0.0])
    initial_K = jnp.array([1.0, -10.0, 10.0, 1.0, 1.0])
    controller, history = train_linear_controller(initial_K, initial_state, config)
    
    # JIT the controller for faster execution
    controller = controller.jit()

    print("Advanced flags enabled:")
    print(f"  batch_size={config.batch_size}")
    print(f"  lr_schedule={config.lr_schedule}")
    print(f"  stability_weight={config.stability_weight}")
    print(f"  lqr_warm_start={config.lqr_warm_start}")
    print("Final cost:", history.costs[-1])

    A, B = _linearise(CartPoleParams())
    K4 = jnp.array([controller.K[0], controller.K[2], controller.K[3], controller.K[4]])
    eigvals = jnp.linalg.eigvals(A - B @ K4[None, :])
    print("Closed-loop eigenvalues:", eigvals)

    return controller


if __name__ == "__main__":
    main()

