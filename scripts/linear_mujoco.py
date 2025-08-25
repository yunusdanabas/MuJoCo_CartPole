"""
Interactive MuJoCo simulation with a trained linear controller using LQR warm start.
Supports debug mode with pre-determined gains for quick testing.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
import yaml
from lib.training.linear_training import LinearTrainingConfig, train_linear_controller
from env.cartpole import CartPoleParams
from scripts.mujoco_core import run_interactive


def _load_disturbance_schedule(path="disturbance.yaml"):
    """Load and validate disturbance schedule from YAML file."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    # Ensure all values are floats for type safety
    disturbances = config["disturbances"]
    for d in disturbances:
        d["start"] = float(d["start"])
        d["end"] = float(d["end"])
        d["value"] = float(d["value"])
    return disturbances


def _disturbance(t: float) -> float:
    """Apply disturbance force based on time schedule."""
    # Load once and cache
    if not hasattr(_disturbance, "_schedule"):
        _disturbance._schedule = _load_disturbance_schedule()
    
    for d in _disturbance._schedule:
        if d["start"] <= t < d["end"]:
            return float(d["value"])
    return 0.0


def _init_state(model, data):
    """Set initial conditions for the cart-pole system."""
    # Reset all positions and velocities
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    
    # Set specific initial conditions (cart at origin, pole slightly tilted)
    if model.nq >= 2:
        data.qpos[0] = 0.0     # cart x position
        data.qpos[1] = 0.5     # pole angle (radians)
    if model.nv >= 2:
        data.qvel[0] = -0.15   # cart x velocity
        data.qvel[1] = -0.35   # pole angular velocity
    
    mujoco.mj_forward(model, data)


def create_debug_controller():
    """Create controller with pre-determined gains for debug mode."""
    from controller.linear_controller import LinearController
    
    # Pre-determined gains from successful training
    debug_gains = np.array([-9.1261015, -0.0957633, 92.03127, -10.561087, 27.554216])
    
    print(f"[DEBUG] Using pre-determined gains: {debug_gains}")
    return LinearController(K=debug_gains).jit()


def train_controller(initial_state: np.ndarray, config: LinearTrainingConfig):
    """Train a linear controller with given configuration."""
    print("[INFO] Training LinearController for MuJoCo simulation...")
    
    # Train the controller
    controller, history = train_linear_controller(
        None,                      # Let trainer use LQR warm start
        initial_state,
        config,
        print_data=True
    )
    
    # JIT the controller for fast simulation
    controller = controller.jit()
    
    print(f"[INFO] Final controller gains: {controller.K}")
    print(f"[INFO] Cost reduction: {history.costs[0]:.4f} -> {history.costs[-1]:.4f}")
    
    return controller


def main():
    """Main function: train controller and run MuJoCo simulation."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Linear controller MuJoCo simulation")
    parser.add_argument("--debug", action="store_true", help="Use debug mode (skip training)")
    args = parser.parse_args()
    
    # Initial state: slightly perturbed from upright
    initial_state = np.array([0.1, 0.95, 0.31, 0.0, 0.0])
    
    if args.debug:
        # Debug mode: use pre-determined gains
        print("[DEBUG] Running in debug mode - skipping training")
        controller = create_debug_controller()
    else:
        # Training mode: train controller with robust configuration
        config = LinearTrainingConfig(
            lqr_warm_start=True,         # Use LQR for initial gains
            learning_rate=0.005,        # Lower learning rate for stability
            num_iterations=2000,        # More iterations for better convergence
            batch_size=64,              # Larger batch for better gradient estimates
            trajectory_length=4.0,      # Longer rollout for more robust controller
            print_data=True
        )
        
        controller = train_controller(initial_state, config)
    
    # Create controller function for MuJoCo
    controller_fn = lambda s, t: float(controller(np.array(s, dtype=np.float32), float(t)))
    
    # Run interactive MuJoCo simulation
    print("[INFO] Starting MuJoCo simulation...")
    run_interactive(
        "cart_pole_minimal.xml",
        controller_fn,
        out_plot="linear_mujoco.png",
        disturbance_fn=_disturbance,
        setup_callback=_init_state
    )


if __name__ == "__main__":
    main()