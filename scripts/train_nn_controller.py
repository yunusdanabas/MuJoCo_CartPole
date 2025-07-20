# train_nn_controller.py
# Description: This script trains a neural network controller for a cart-pole system using JAX and Equinox. 
# It includes training, evaluation, and plotting functionalities.

import os
import sys
from pathlib import Path
import yaml
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from controller.nn_controller import (
    CartPoleNN,
    train_nn_controller,
    evaluate_controller,
)
from lib.utils import convert_4d_to_5d, compute_trajectory_cost
from lib.visualizer import (
    plot_trajectory_comparison2,
    plot_energy,
    plot_cost_comparison,
    plot_control_forces_comparison,
)

# Load configuration
CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    _CONFIG = yaml.safe_load(f) or {}

MODEL_SAVE_PATH = _CONFIG.get("nn_model_path", "saved_models/nn_controller.eqx")
nn_cfg = _CONFIG.get("nn_training", {})
PARAMS_SYSTEM = tuple(nn_cfg.get("params_system", [1.0, 0.1, 0.5, 9.81]))
Q_MATRIX = jnp.diag(jnp.array(nn_cfg.get("q_weights", [0.1, 10.0, 10.0, 0.1, 0.1])))
_span = nn_cfg.get("t_span", [0.0, 10.0])
_eval_pts = nn_cfg.get("t_eval_points", 100)
TRAIN_CONFIG = {
    'num_epochs': nn_cfg.get('num_epochs', 500),
    'batch_size': nn_cfg.get('batch_size', 32),
    't_span': tuple(_span),
    't_eval': jnp.linspace(_span[0], _span[1], _eval_pts),
    'learning_rate': float(nn_cfg.get('learning_rate', 3e-4)),
    'grad_clip': float(nn_cfg.get('grad_clip', 1.0)),
    'cost_weights': tuple(nn_cfg.get('cost_weights', [10.0, 1.0, 0.01])),
    'hidden_dims': tuple(nn_cfg.get('hidden_dims', [64, 64])),
    'curriculum_learning': nn_cfg.get('curriculum_learning', False),
    'curriculum_stages': nn_cfg.get('curriculum_stages', 1),
}

def main():
    """Train the neural network swing-up controller and save the result."""

    key = jax.random.PRNGKey(42)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    controller = CartPoleNN(key=key, hidden_dims=TRAIN_CONFIG['hidden_dims'])
    print("Initialized new neural network controller")
    print(f"Architecture: {TRAIN_CONFIG['hidden_dims']}")
    print(f"Total parameters: {sum(p.size for p in jax.tree_util.tree_leaves(eqx.filter(controller, eqx.is_inexact_array)))}")

    print("\nStarting training...")
    trained_controller, loss_history = train_nn_controller(
        controller=controller,
        params_system=PARAMS_SYSTEM,
        Q=Q_MATRIX,
        num_epochs=TRAIN_CONFIG['num_epochs'],
        batch_size=TRAIN_CONFIG['batch_size'],
        t_span=TRAIN_CONFIG['t_span'],
        t_eval=TRAIN_CONFIG['t_eval'],
        key=key,
        learning_rate=TRAIN_CONFIG['learning_rate'],
        grad_clip=TRAIN_CONFIG['grad_clip'],
        cost_weights=TRAIN_CONFIG['cost_weights'],
    )

    eqx.tree_serialise_leaves(MODEL_SAVE_PATH, trained_controller)
    print(f"\nSaved trained controller to {MODEL_SAVE_PATH}")

    # Evaluate the final controller from the downward position
    initial_state = jnp.array([0.0, jnp.pi, 0.0, 0.0])
    ts, states = evaluate_controller(
        trained_controller,
        PARAMS_SYSTEM,
        initial_state,
        TRAIN_CONFIG['t_span'],
        TRAIN_CONFIG['t_eval'],
    )

    plt.figure(figsize=(10, 4))
    plt.semilogy(loss_history)
    plt.title("Training Loss Progress")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plot_trajectory_comparison2(ts, [states], labels=["NN Controller"],
                                title_prefix="Swing-Up Performance")
    plot_energy(ts, states, PARAMS_SYSTEM, title="Energy During Swing-Up")

if __name__ == "__main__":
    main()
