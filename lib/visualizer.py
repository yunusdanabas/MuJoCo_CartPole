"""Visualization utilities for cart-pole experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import jax
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


def plot_trajectory_comparison2(
    t: jnp.ndarray,
    states_list: Iterable[jnp.ndarray],
    labels: Optional[Iterable[str]] = None,
    title_prefix: str = "Trajectory Comparison",
    name: str = "trajectory_comparison.png",
) -> Path:
    """Save comparison of full state trajectories."""
    path = _ensure_dir() / name
    labels = list(labels) if labels is not None else [f"traj {i}" for i in range(len(states_list))]

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    titles = [
        "Cart Position (x)",
        "Pendulum Angle (theta)",
        "Cart Velocity (x_dot)",
        "Angular Velocity (theta_dot)",
    ]
    indices = [0, 1, 2, 3]
    ylabels = ["x (m)", "theta (deg)", "x_dot (m/s)", "theta_dot (rad/s)"]

    for ax, idx, title, ylabel in zip(axs.flatten(), indices, titles, ylabels):
        for states, label in zip(states_list, labels):
            y = states[:, idx]
            if idx == 1:
                y = np.rad2deg(np.array(y))
            ax.plot(t, y, label=label)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()

    fig.suptitle(title_prefix, y=1.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_energy(
    t: jnp.ndarray,
    states: jnp.ndarray,
    params: Tuple[float, float, float, float],
    title: str = "Energy Over Time",
    name: str = "energy.png",
) -> Path:
    """Save total system energy over time."""
    path = _ensure_dir() / name
    mc, mp, l, g = params

    def _energy(state):
        x, theta, x_dot, theta_dot = state
        kinetic = 0.5 * (mc + mp) * x_dot**2 - mp * l * np.cos(theta) * x_dot * theta_dot + 0.5 * mp * l**2 * theta_dot**2
        potential = mp * g * l * (1 - np.cos(theta))
        return kinetic + potential

    energies = jax.vmap(_energy)(states)

    plt.figure(figsize=(8, 5))
    plt.plot(t, np.array(energies))
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (J)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_cost_comparison(
    linear_cost: float,
    lqr_cost: float,
    title: str = "Trajectory Cost Comparison",
    name: str = "cost_comparison.png",
) -> Path:
    """Save bar plot comparing cost values."""
    path = _ensure_dir() / name
    plt.figure(figsize=(6, 4))
    plt.bar(["Trained Controller", "LQR"], [linear_cost, lqr_cost], color=["blue", "orange"])
    plt.title(title)
    plt.ylabel("Total Cost")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_control_forces_comparison(
    t: jnp.ndarray,
    controller1_forces: jnp.ndarray,
    controller2_forces: jnp.ndarray,
    labels: Tuple[str, str] = ("Trained Controller", "LQR"),
    title: str = "Control Forces Comparison",
    name: str = "control_forces_comparison.png",
) -> Path:
    """Save comparison plot of two control force histories."""
    path = _ensure_dir() / name
    plt.figure(figsize=(8, 4))
    plt.plot(t, np.array(controller1_forces), label=labels[0])
    plt.plot(t, np.array(controller2_forces), label=labels[1], linestyle="--")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

