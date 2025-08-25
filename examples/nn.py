"""
Run a NN-controlled cart-pole swing-up using the 5-state formulation
[x, cos(theta), sin(theta), xdot, thdot] and save a plot to results/trajectory_nn.png.
"""
from __future__ import annotations

import time
import jax
import jax.numpy as jnp
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from env.cartpole import CartPoleParams
from env.closedloop import simulate, create_time_grid
from controller.nn_controller import NNController
from lib.visualizer import plot_trajectory
from lib.training.nn_training import train, TrainConfig
import optax


def main():
    print("[INFO] Training Started for NNController")

    params = CartPoleParams()

    # Time grid - shorter for better training stability
    t_span = (0.0, 4.0)
    ts = create_time_grid(t_span, dt=0.02)

    # Initial state: downwards, zero velocities (5D)
    y0 = jnp.array([0.0, -1.0, 0.0, 0.0, 0.0])

    # Deterministic NN controller (train)
    nn = NNController.init(seed=0)
    
    # Learning rate schedule: slower decay for better convergence
    lr_schedule = optax.cosine_decay_schedule(
        init_value=1e-3,          # Higher initial LR for faster initial learning
        decay_steps=100,
        alpha=0.1                  # Final LR = 1e-3 * 0.1 = 1e-4 (less aggressive decay)
    )
    
    # Optimizer with learning rate schedule and better stability
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),      # Gradient clipping for stability
        optax.adam(learning_rate=lr_schedule, b1=0.9, b2=0.999),
        optax.add_decayed_weights(1e-4)      # Weight decay for regularization
    )
    
    # Better training configuration for 1000 epochs
    cfg = TrainConfig(
        batch_size=256,           # Larger batch for more stable gradients
        num_epochs=1000,          # More epochs for thorough learning
        t_span=t_span,
        ts=ts,
        print_data=True
    )

    t0 = time.perf_counter()
    nn_trained, loss_history = train(
        nn, 
        params, 
        cfg, 
        loss_fn="swingup_loss",   # Use improved swing-up loss
        optimiser=optimizer,
        init_state_fn="downward_initial_conditions",  # Focus on swing-up task
        print_data=True
    )
    t1 = time.perf_counter()
    print(f"[INFO] Training Finished for NNController in {(t1 - t0):.2f} seconds")
    print(f"[INFO] Final loss: {loss_history[-1]:.6f}")
    
    # JIT the controller for faster simulation
    nn_trained = nn_trained.jit()

    # Simulate and plot
    sol = simulate(nn_trained, params, t_span, ts, y0)
    plot_trajectory(sol.ys, sol.ts, title="NN Trajectory", save_path="trajectory_nn.png", show_plot=True)


if __name__ == "__main__":
    main()