"""
scripts/nn_mujoco.py

Interactive MuJoCo simulation with NN controller.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml
import jax
import jax.numpy as jnp
import optax
from controller.nn_controller import NNController
from lib.training.nn_training import train, TrainConfig
from scripts.mujoco_core import run_interactive


def _load_disturbance_schedule(path="disturbance.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    # Ensure all values are floats
    disturbances = config["disturbances"]
    for d in disturbances:
        d["start"] = float(d["start"])
        d["end"] = float(d["end"])
        d["value"] = float(d["value"])
    return disturbances


def _disturbance(t: float) -> float:
    if not hasattr(_disturbance, "_schedule"):
        _disturbance._schedule = _load_disturbance_schedule()
    for d in _disturbance._schedule:
        if d["start"] <= t < d["end"]:
            return d["value"]
    return 0.0


def _init_state(model, data):
    """Set initial state: pole downwards, zero velocities."""
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    if model.nq >= 2:
        data.qpos[0] = 0.0     # cart x
        data.qpos[1] = np.pi   # pole angle (downwards)
    if model.nv >= 2:
        data.qvel[0] = 0.0     # cart xdot
        data.qvel[1] = 0.0     # pole thdot
    import mujoco
    mujoco.mj_forward(model, data)


def main():  # pragma: no cover - interactive
    try:
        import mujoco
    except ImportError:
        print("MuJoCo not available. Install with: conda install -c conda-forge mujoco")
        return
    
    print("[INFO] Training NN Controller for MuJoCo simulation...")
    
    # Create and train the controller
    nn = NNController.init(seed=0)
    
    # Learning rate schedule: cosine decay from 5e-4 to 1e-5
    lr_schedule = optax.cosine_decay_schedule(
        init_value=5e-4,
        decay_steps=500,  # Shorter training for MuJoCo demo
        alpha=0.02
    )
    
    # Optimizer with learning rate schedule
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_schedule)
    )
    
    # Training configuration
    cfg = TrainConfig(
        batch_size=64,
        num_epochs=500,
        t_span=(0.0, 4.0),
        print_data=True
    )
    
    # Train the controller
    nn_trained, loss_history = train(
        nn,
        cfg=cfg,
        loss_fn="combined_loss",
        optimiser=optimizer,
        init_state_fn="downward_initial_conditions",
        print_data=True
    )
    
    print(f"[INFO] Training completed. Final loss: {loss_history[-1]:.6f}")
    
    # JIT the controller for faster simulation
    nn_trained = nn_trained.jit()
    
    # Create controller function for MuJoCo
    controller_fn = lambda s, t: float(nn_trained(np.array(s, dtype=np.float32), float(t)))
    
    print("[INFO] Starting MuJoCo simulation...")
    run_interactive("cart_pole_minimal.xml", controller_fn, out_plot="nn_mujoco.png", disturbance_fn=_disturbance, setup_callback=_init_state)


if __name__ == "__main__":
    main()