"""
Train a linear controller for the cart-pole using LQR warm start and plot the result.
This example demonstrates robust training with proper validation and visualization.
"""

from __future__ import annotations

import time
import jax.numpy as jnp
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from lib.training.linear_training import LinearTrainingConfig, train_linear_controller
from env.cartpole import CartPoleParams
from env.closedloop import simulate, create_time_grid
from lib.visualizer import plot_trajectory


def main():
    """Main training and simulation function."""
    print("[INFO] Training LinearController with LQR warm start...")
    
    # ==================== Configuration ====================
    # Initial state: slightly perturbed from upright position
    initial_state = jnp.array([0.1, 0.95, 0.31, 0.0, 0.0])
    
    # Training configuration (robust, recommended values)
    config = LinearTrainingConfig(
        lqr_warm_start=True,         # Use LQR for initial gains
        learning_rate=0.005,        # Lower learning rate for stability
        num_iterations=2000,        # More iterations for better convergence
        batch_size=64,              # Larger batch for better gradient estimates
        trajectory_length=4.0,      # Longer rollout for more robust controller
        print_data=True
    )
    
    # ==================== Training ====================
    print("[INFO] Starting controller training...")
    t0 = time.perf_counter()
    
    controller, history = train_linear_controller(
        None,                      # Let trainer use LQR warm start
        initial_state,
        config,
        print_data=True
    )
    
    t1 = time.perf_counter()
    print(f"[INFO] Training finished in {(t1 - t0):.2f} seconds")
    
    # ==================== Optimization ====================
    # JIT compile the controller for faster simulation
    controller = controller.jit()
    
    # ==================== Validation ====================
    # Print final controller gains for inspection
    print(f"[INFO] Final controller gains: {controller.K}")
    
    # Validate training results
    if history.costs[-1] > history.costs[0]:
        print("[WARNING] Final cost is higher than initial cost - training may have failed")
    
    print(f"[INFO] Cost reduction: {history.costs[0]:.4f} -> {history.costs[-1]:.4f}")
    print(f"[INFO] Training converged: {'Yes' if len(history.costs) < config.num_iterations else 'No'}")
    
    # ==================== Simulation ====================
    print("[INFO] Running simulation with trained controller...")
    
    # Setup simulation parameters
    params = CartPoleParams()
    t_span = (0.0, 6.0)
    ts = create_time_grid(t_span, dt=0.01)
    y0 = initial_state
    
    # Run simulation
    sol = simulate(controller, params, t_span, ts, y0)
    
    # ==================== Visualization ====================
    # Plot and save the trajectory
    plot_trajectory(
        sol.ys, 
        sol.ts, 
        title="Linear Controller Trajectory", 
        save_path="trajectory_linear.png", 
        show_plot=True
    )
    
    print("[INFO] Simulation complete. Plot saved as 'trajectory_linear.png'")
    return controller


if __name__ == "__main__":
    main()

