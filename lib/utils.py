# lib/utils.py
# Description: Utility functions for sampling initial conditions, plotting, and computing costs and energies for a cart-pole system.

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple

###############################################################################
#                         Initial Condition Sampler                           #
###############################################################################

def sample_initial_conditions(
    num_samples,
    x_range=(-0.5, 0.5),
    theta_range=(-0.5, 0.5),
    xdot_range=(-0.5, 0.5),
    thetadot_range=(-0.5, 0.5),
    key=None
) -> jnp.ndarray:
    """
    Generates random initial conditions within specified ranges.

    Args:
        num_samples: Number of initial conditions to sample
        x_range: (min, max) range for cart position
        theta_range: (min, max) range for pendulum angle (radians)
        xdot_range: (min, max) range for cart velocity
        thetadot_range: (min, max) range for pendulum angular velocity
        key: JAX random key. If None, uses NumPy random

    Returns:
        Array of shape (num_samples, 4) containing [x, theta, x_dot, theta_dot]
    """
    if key is None:
        return _sample_with_numpy(num_samples, x_range, theta_range, 
                                 xdot_range, thetadot_range)
    else:
        return _sample_with_jax(num_samples, x_range, theta_range,
                               xdot_range, thetadot_range, key)

def _sample_with_numpy(num_samples, x_r, theta_r, xdot_r, thetadot_r):
    """NumPy-based sampling for non-JIT environments"""
    return jnp.array(np.column_stack([
        np.random.uniform(*x_r, num_samples),
        np.random.uniform(*theta_r, num_samples),
        np.random.uniform(*xdot_r, num_samples),
        np.random.uniform(*thetadot_r, num_samples)
    ]), dtype=jnp.float32)

def _sample_with_jax(num_samples, x_r, theta_r, xdot_r, thetadot_r, key):
    keys = jax.random.split(key, 4)
    return jnp.column_stack([
        jax.random.uniform(keys[0], (num_samples,), 
            minval=x_r[0], maxval=x_r[1]),
        jax.random.uniform(keys[1], (num_samples,), 
            minval=theta_r[0], maxval=theta_r[1]),
        jax.random.uniform(keys[2], (num_samples,), 
            minval=xdot_r[0], maxval=xdot_r[1]),
        jax.random.uniform(keys[3], (num_samples,), 
            minval=thetadot_r[0], maxval=thetadot_r[1]),
    ])

###############################################################################
#                           Visualization Utilities                           #
###############################################################################

def plot_cost(cost_history: jnp.ndarray, 
             title: str = "Cost Over Iterations", 
             log_scale: bool = False) -> None:
    """
    Plots training cost history.
    
    Args:
        cost_history: Array of cost values per iteration
        title: Plot title
        log_scale: Whether to use logarithmic y-axis
    """
    plt.figure(figsize=(8, 5))
    (plt.semilogy if log_scale else plt.plot)(cost_history)
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.show()

def plot_trajectory_comparison(
    t: jnp.ndarray,
    states_list: list[jnp.ndarray],
    labels: Optional[list[str]] = None,
    title_prefix: str = "Trajectory Comparison"
) -> None:
    """
    Compares multiple trajectories through cart position and pendulum angle.
    
    Args:
        t: Time vector shape (N,)
        states_list: List of trajectory arrays (each shape (N, 4))
        labels: Legend labels for each trajectory
        title_prefix: Title prefix for plots
    """
    labels = labels or [f"Trajectory {i}" for i in range(len(states_list))]
    
    # Plot cart positions
    _plot_comparison_sub(t, states_list, 0, "Cart Position (x)", "x (m)", labels, title_prefix)
    
    # Plot pendulum angles
    plt.figure(figsize=(10, 6))
    for states in states_list:
        theta = np.array(states[:, 1])  # Convert to NumPy for deg conversion
        plt.plot(t, np.rad2deg(theta))
    _format_plot("Pendulum Angle (theta)", "theta (degrees)", labels, title_prefix)

def _plot_comparison_sub(t, states_list, idx, ylabel, title_suffix, labels, prefix):
    """Helper for trajectory comparison subplots"""
    plt.figure(figsize=(10, 6))
    for states, label in zip(states_list, labels):
        plt.plot(t, states[:, idx], label=label)
    _format_plot(title_suffix, ylabel, labels, prefix)

def _format_plot(ylabel, title_suffix, labels, prefix):
    """Shared plot formatting"""
    plt.title(f"{prefix}: {title_suffix}")
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_trajectory_comparison2(
    t: jnp.ndarray,
    states_list: list[jnp.ndarray],
    labels: Optional[list[str]] = None,
    title_prefix: str = "Trajectory Comparison"
) -> None:
    """
    Comprehensive trajectory comparison across all state variables.
    
    Args:
        t: Time vector shape (N,)
        states_list: List of trajectory arrays (each shape (N, 4))
        labels: Legend labels
        title_prefix: Plot title prefix
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    titles = ["Cart Position (x)", "Pendulum Angle (theta)",
             "Cart Velocity (x_dot)", "Angular Velocity (theta_dot)"]
    indices = [0, 1, 2, 3]
    ylabels = ["x (m)", "theta (deg)", "x_dot (m/s)", "theta_dot (rad/s)"]

    for ax, idx, title, ylabel in zip(axs.flatten(), indices, titles, ylabels):
        for states, label in zip(states_list, labels or []):
            y = states[:, idx]
            if idx == 1:  # Convert angles to degrees
                y = np.rad2deg(np.array(y))
            ax.plot(t, y, label=label)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.suptitle(title_prefix, y=1.02)
    plt.show()

###############################################################################
#                           Analytical Utilities                             #
###############################################################################

def compute_trajectory_cost(
    Q: jnp.ndarray,
    states: jnp.ndarray,
    controller_func: callable,
    t: jnp.ndarray
) -> Tuple[float, jnp.ndarray]:
    """
    Computes LQR-style trajectory cost.
    
    Args:
        Q: State cost matrix (4x4 or 5x5)
        states: Array of states (N x state_dim)
        controller_func: Policy function (state, t) -> force
        t: Time vector (N,)
        
    Returns:
        total_cost: Scalar cost value
        forces: Array of control forces (N,)
    """
    dt = t[1] - t[0]
    forces = jax.vmap(controller_func)(states, t)
    state_costs = jnp.einsum('ni,ij,nj->n', states, Q, states)
    total_cost = jnp.sum(state_costs + forces**2) * dt
    return total_cost, forces

def compute_energy(
    states: jnp.ndarray,
    params: Tuple[float, float, float, float]
) -> jnp.ndarray:
    """
    Computes total energy for 4D states [x, theta, x_dot, theta_dot].
    
    Args:
        states: Array of states (N x 4)
        params: Tuple (mc, mp, l, g)
        
    Returns:
        energies: Array of total energies (N,)
    """
    mc, mp, l, g = params
    x, theta, x_dot, theta_dot = states.T
    kinetic = 0.5*(mc + mp)*x_dot**2 - mp*l*jnp.cos(theta)*x_dot*theta_dot + 0.5*mp*(l**2)*theta_dot**2
    potential = mp*g*l*(1 - jnp.cos(theta))
    return kinetic + potential

###############################################################################
#                           State Conversion                                 #
###############################################################################

def convert_4d_to_5d(state_4d: jnp.ndarray) -> jnp.ndarray:
    """
    Converts 4D state [x, theta, x_dot, theta_dot] to 
    5D [x, cos(theta), sin(theta), x_dot, theta_dot].
    """
    x, theta, x_dot, theta_dot = state_4d
    return jnp.array([
        x,
        jnp.cos(theta),
        jnp.sin(theta),
        x_dot,
        theta_dot
    ])

def convert_5d_to_4d(state_5d: jnp.ndarray) -> jnp.ndarray:
    """
    Converts 5D state [x, cos(theta), sin(theta), x_dot, theta_dot] 
    back to 4D [x, theta, x_dot, theta_dot].
    """
    x, cos_t, sin_t, x_dot, theta_dot = state_5d
    return jnp.array([
        x,
        jnp.arctan2(sin_t, cos_t),
        x_dot,
        theta_dot
    ])

###############################################################################
#                           Specialized Plots                                #
###############################################################################

def plot_energy_history(
    t: jnp.ndarray,
    states: jnp.ndarray,
    params: Tuple[float, float, float, float],
    desired_energy: Optional[float] = None,
    title: str = "Energy Over Time"
) -> None:
    """
    Plots energy components with optional desired energy reference.
    
    Args:
        t: Time vector (N,)
        states: State trajectories (N x 4)
        params: System parameters (mc, mp, l, g)
        desired_energy: Reference energy value
        title: Plot title
    """
    mc, mp, l, g = params
    energies = compute_energy(states, params)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, energies, label="Total Energy")
    if desired_energy is not None:
        plt.axhline(desired_energy, c='r', ls='--', label="Desired Energy")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (J)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_control_forces(
    t: jnp.ndarray,
    forces_list: list[jnp.ndarray],
    labels: list[str],
    title: str = "Control Forces Comparison"
) -> None:
    """
    Plots multiple control force histories.
    
    Args:
        t: Time vector (N,)
        forces_list: List of force arrays (each N,)
        labels: Legend labels
        title: Plot title
    """
    plt.figure(figsize=(10, 4))
    for forces, label in zip(forces_list, labels):
        plt.plot(t, forces, label=label)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # lib/utils.py (additional functions)

def plot_energy(
    t: jnp.ndarray,
    states: jnp.ndarray,
    params: Tuple[float, float, float, float],
    title: str = "Energy Over Time"
) -> None:
    """
    Plot total energy of the system over time.
    
    Args:
        t: Time vector (N,)
        states: State trajectories (N, 4) [x, theta, x_dot, theta_dot]
        params: Tuple (mc, mp, l, g)
        title: Plot title
    """
    mc, mp, l, g = params
    
    def compute_energy(state):
        x, theta, x_dot, theta_dot = state
        kinetic = 0.5*(mc + mp)*x_dot**2 - mp*l*jnp.cos(theta)*x_dot*theta_dot + 0.5*mp*(l**2)*theta_dot**2
        potential = mp*g*l*(1 - jnp.cos(theta))
        return kinetic + potential
    
    energies = jax.vmap(compute_energy)(states)
    energies = np.array(energies)  # Convert to NumPy for plotting
    
    plt.figure(figsize=(8, 5))
    plt.plot(t, energies)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (J)")
    plt.grid(True)
    plt.show()

def plot_cost_comparison(
    linear_cost: float,
    lqr_cost: float,
    title: str = "Trajectory Cost Comparison"
) -> None:
    """
    Bar plot comparing costs between two controllers.
    
    Args:
        linear_cost: Cost from linear/NN controller
        lqr_cost: Cost from LQR controller
        title: Plot title
    """
    plt.figure(figsize=(6, 4))
    plt.bar(["Trained Controller", "LQR"], [linear_cost, lqr_cost], color=["blue", "orange"])
    plt.title(title)
    plt.ylabel("Total Cost")
    plt.grid(True)
    plt.show()

def plot_control_forces_comparison(
    t: jnp.ndarray,
    controller1_forces: jnp.ndarray,
    controller2_forces: jnp.ndarray,
    labels: Tuple[str, str] = ("Trained Controller", "LQR"),
    title: str = "Control Forces Comparison"
) -> None:
    """
    Plot control forces from two controllers.
    
    Args:
        t: Time vector (N,)
        controller1_forces: Force array from first controller (N,)
        controller2_forces: Force array from second controller (N,)
        labels: Legend labels
        title: Plot title
    """
    plt.figure(figsize=(8, 4))
    plt.plot(t, np.array(controller1_forces), label=labels[0])
    plt.plot(t, np.array(controller2_forces), label=labels[1], linestyle="--")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compute_energy_nn(
    states: jnp.ndarray,
    params: Tuple[float, float, float, float]
) -> jnp.ndarray:
    """
    Compute total energy for 5D states [x, cosθ, sinθ, ẋ, θ̇]
    
    Args:
        states: (N, 5) array of states
        params: Tuple (mc, mp, l, g)
        
    Returns:
        Array of energies (N,)
    """
    mc, mp, l, g = params
    x, cosθ, sinθ, ẋ, θ̇ = states.T
    
    # Kinetic energy components
    trans_energy = 0.5 * (mc + mp) * ẋ**2
    rot_energy = 0.5 * mp * l**2 * θ̇**2
    coriolis = -mp * l * cosθ * ẋ * θ̇
    
    # Potential energy
    potential = mp * g * l * (1 - cosθ)
    
    return trans_energy + rot_energy + coriolis + potential