"""Basic linear controller training example."""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lib.training.linear_training import LinearTrainingConfig, train_linear_controller


def main():
    initial_K = jnp.array([1.0, -10.0, 10.0, 1.0, 1.0])
    initial_state = jnp.array([0.1, 0.95, 0.31, 0.0, 0.0])

    config = LinearTrainingConfig(learning_rate=0.02, num_iterations=50, trajectory_length=2.0)

  
    controller, history = train_linear_controller(initial_K, initial_state, config)

    print("Initial cost:", history.costs[0])
    print("Final cost:", history.costs[-1])
    print("Cost trend:")
    for i, c in enumerate(history.costs):
        if i % 10 == 0 or i == len(history.costs) - 1:
            print(f"  iter {i:02d}: {c:.6f}")

    return controller


if __name__ == "__main__":
    main()

