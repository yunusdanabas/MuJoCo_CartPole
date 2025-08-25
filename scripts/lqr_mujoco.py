"""
Interactive MuJoCo simulation with LQR controller for cart-pole system.
Trains LQR controller first, then runs MuJoCo simulation with trained controller.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml
import time
from controller.lqr_controller import LQRController
from env.cartpole import CartPoleParams
from scripts.mujoco_core import run_interactive
from lib.training.lqr_training import train_lqr_controller, LQRTrainingConfig


def _load_disturbance_schedule(path="disturbance.yaml"):
    """Load disturbance schedule from YAML file."""
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


def main():
    """Train LQR controller and run MuJoCo simulation."""
    try:
        import mujoco
    except ImportError:
        print("MuJoCo not available. Install with: conda install -c conda-forge mujoco")
        return
    
    print("[INFO] Training Started for LQRController")
    
    # System parameters
    params = CartPoleParams()
    
    # Train LQR controller with detailed output
    t0 = time.perf_counter()
    lqr = train_lqr_controller(params, LQRTrainingConfig(print_data=True), print_data=True)
    t1 = time.perf_counter()
    print(f"[INFO] Training Finished for LQRController in {(t1 - t0):.2f} seconds")
    
    # JIT compile for faster simulation
    lqr = lqr.jit()
    
    print("[INFO] Starting MuJoCo simulation with trained controller...")
    
    # Create controller function for MuJoCo
    controller_fn = lambda s, t: float(lqr(np.array(s, dtype=np.float32), float(t)))
    
    # Run interactive simulation
    run_interactive(
        "cart_pole_minimal.xml", 
        controller_fn, 
        out_plot="lqr_mujoco.png", 
        disturbance_fn=_disturbance
    )


if __name__ == "__main__":
    main()