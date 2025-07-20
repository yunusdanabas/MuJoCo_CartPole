"""Visualization utilities for cart-pole experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


_RESULTS_DIR = Path("results")


def _ensure_dir() -> Path:
    """Create the results directory if it does not exist."""
    _RESULTS_DIR.mkdir(exist_ok=True)
    return _RESULTS_DIR


def save_cost_history(cost_history: Iterable[float], name: str = "cost.png", log_scale: bool = False) -> Path:
    """Plot and save training cost history."""
    path = _ensure_dir() / name
    plt.figure(figsize=(8, 5))
    data = np.array(cost_history)
    (plt.semilogy if log_scale else plt.plot)(data)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Training Cost")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def save_trajectory(
    t: jnp.ndarray,
    states: jnp.ndarray,
    name: str = "trajectory.png",
    labels: Optional[Iterable[str]] = None,
) -> Path:
    """Plot and save cart position and pole angle over time."""
    path = _ensure_dir() / name
    labels = labels or ["trajectory"]
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, states[:, 0])
    plt.title("Cart Position")
    plt.ylabel("x (m)")
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(t, np.rad2deg(np.array(states[:, 1])))
    plt.xlabel("Time (s)")
    plt.ylabel("theta (deg)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def save_comparison(
    t: jnp.ndarray,
    states_list: Iterable[jnp.ndarray],
    labels: Iterable[str],
    name: str = "comparison.png",
) -> Path:
    """Plot several trajectories for comparison."""
    path = _ensure_dir() / name
    plt.figure(figsize=(10, 6))
    for states, label in zip(states_list, labels):
        plt.plot(t, states[:, 0], label=f"x - {label}")
    plt.xlabel("Time (s)")
    plt.ylabel("x (m)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

