# train_nn_controller.py
# Description: This script trains a neural network controller for a cart-pole system using JAX and Equinox. 
# It includes training, evaluation, and plotting functionalities.

import os
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from controller.neuralnetwork_controller import CartPoleNN
from lib.trainer import train_nn_controller, evaluate_controller
from lib.utils import *

# Configuration
MODEL_SAVE_PATH = "saved_models/nn_controller.eqx"
PARAMS_SYSTEM = (1.0, 0.1, 0.5, 9.81)  # (mc, mp, l, g)
Q_MATRIX = jnp.diag(jnp.array([0.1, 10.0, 10.0, 0.1, 0.1]))  # 5D state weights
TRAIN_CONFIG = {
    'num_epochs': 500,
    'batch_size': 32,
    't_span': (0.0, 10.0),
    't_eval': jnp.linspace(0, 10, 100),
    'learning_rate': 3e-4,
    'grad_clip': 1.0
}

def main():
    # Initialize everything
    key = jax.random.PRNGKey(42)
    
    # Create model directory if needed
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    # Initialize controller
    controller = CartPoleNN(key=key)
    print("Initialized new neural network controller")

    # Train the controller
    print("\nStarting training...")
    trained_controller, loss_history = train_nn_controller(
        controller=controller,
        params_system=PARAMS_SYSTEM,
        Q=Q_MATRIX,
        num_epochs=500,
        batch_size=32,
        t_span=(0.0, 10.0),
        t_eval=jnp.linspace(0, 10, 100),
        key=key,
        learning_rate=1e-4
    )
    
    # Save trained model
    eqx.tree_serialise_leaves(MODEL_SAVE_PATH, trained_controller)
    print(f"\nSaved trained controller to {MODEL_SAVE_PATH}")

    # Evaluate from downward position
    initial_state = jnp.array([0.0, jnp.pi, 0.0, 0.0])  # Downward position
    ts, states = evaluate_controller(
        trained_controller,
        PARAMS_SYSTEM,
        initial_state,
        t_span=TRAIN_CONFIG['t_span'],
        t_eval=TRAIN_CONFIG['t_eval']
    )

    # Plot training progress
    plt.figure(figsize=(10, 4))
    plt.semilogy(loss_history)
    plt.title("Training Loss Progress")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot trajectory results
    plot_trajectory_comparison2(
        ts, [states], 
        labels=["NN Controller"],
        title_prefix="Swing-Up Performance"
    )

    # Plot energy history
    plot_energy(
        ts, states, PARAMS_SYSTEM,
        title="Energy During Swing-Up"
    )

    # Compare with LQR (if available)
    try:
        from controller.lqr_controller import lqr_policy
        # Run LQR simulation from near upright
        lqr_initial_state = jnp.array([0.0, 0.1, 0.0, 0.0])  # Near upright
        _, lqr_states = evaluate_controller(
            lqr_policy, PARAMS_SYSTEM,
            lqr_initial_state, TRAIN_CONFIG['t_span'], TRAIN_CONFIG['t_eval']
        )
        
        # Cost comparison
        nn_cost, nn_forces = compute_trajectory_cost(
            Q_MATRIX, 
            jax.vmap(convert_4d_to_5d)(states), 
            lambda s, t: trained_controller(s, t), 
            ts
        )
        lqr_cost, lqr_forces = compute_trajectory_cost(
            Q_MATRIX[:4,:4],  # LQR uses 4D state
            states, 
            lqr_policy, 
            ts
        )
        
        plot_cost_comparison(nn_cost, lqr_cost)
        plot_control_forces_comparison(ts, nn_forces, lqr_forces)
        
    except ImportError:
        print("\nLQR controller not found - skipping comparison plots")

if __name__ == "__main__":
    main()