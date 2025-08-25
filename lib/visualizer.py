"""Visualization utilities for cart-pole experiments."""

from __future__ import annotations
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from typing import Optional, Union


_RESULTS_DIR = Path("results")


def _ensure_dir() -> Path:
    """Ensure results directory exists."""
    _RESULTS_DIR.mkdir(exist_ok=True)
    return _RESULTS_DIR


def _save_and_show_plot(fig: plt.Figure, save_path: Optional[str] = None, 
                        show_plot: bool = False) -> plt.Figure:
    """Helper function to save and/or show plots consistently."""
    if save_path:
        try:
            plt.savefig(_ensure_dir() / save_path, dpi=150, bbox_inches="tight")
        except Exception as e:
            print(f"Failed to save plot to {save_path}: {e}")
    
    if show_plot:
        try:
            plt.show()
        except Exception as e:
            print(f"Failed to show plot: {e}")
    
    return fig


def _create_time_grid(length: int, dt: float = 0.01) -> jnp.ndarray:
    """Create time grid for plotting."""
    return jnp.arange(length) * dt


def _extract_angles(trajectory: jnp.ndarray) -> jnp.ndarray:
    """Extract angles from cos/sin representation."""
    return jnp.arctan2(trajectory[:, 2], trajectory[:, 1])


def plot_trajectory(
    trajectory: Union[jnp.ndarray, np.ndarray],
    time_points: Optional[Union[jnp.ndarray, np.ndarray]] = None,
    title: str = "Cart-Pole Trajectory",
    save_path: Optional[str] = None,
    show_plot: bool = False
) -> plt.Figure:
    """
    Plot cart-pole trajectory with all state variables.
    
    Args:
        trajectory: Either state trajectory (N, 5) [x, cos(θ), sin(θ), ẋ, θ̇]
                    or generic y-series (N,) to plot vs given time_points.
        time_points: Time array, created automatically if None
        title: Plot title
        save_path: Path to save figure, displays if None
        show_plot: Whether to display the plot interactively
    
    Returns:
        matplotlib Figure object
    """
    if trajectory.ndim == 1:
        # Simple 1D line plot
        if time_points is None:
            time_points = _create_time_grid(len(trajectory))
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time_points, trajectory)
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Value')
        ax.grid(True)
        
        return _save_and_show_plot(fig, save_path, show_plot)
    
    # Multi-dimensional trajectory plot
    if time_points is None:
        time_points = _create_time_grid(len(trajectory))
    
    angles = _extract_angles(trajectory)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=16)
    
    # State variable plots
    plot_configs = [
        (0, 0, trajectory[:, 0], 'Cart Position', 'Position (m)'),
        (0, 1, jnp.degrees(angles), 'Pendulum Angle', 'Angle (degrees)'),
        (1, 0, trajectory[:, 3], 'Cart Velocity', 'Velocity (m/s)'),
        (1, 1, trajectory[:, 4], 'Angular Velocity', 'Angular Velocity (rad/s)')
    ]
    
    for row, col, data, plot_title, ylabel in plot_configs:
        axes[row, col].plot(time_points, data)
        axes[row, col].set_title(plot_title)
        axes[row, col].set_xlabel('Time (s)')
        axes[row, col].set_ylabel(ylabel)
        axes[row, col].grid(True)
    
    plt.tight_layout()
    return _save_and_show_plot(fig, save_path, show_plot)


def plot_training_history(
    history,
    title: str = "Training Progress",
    save_path: Optional[str] = None,
    show_plot: bool = False
) -> Optional[plt.Figure]:
    """
    Plot training cost history.
    
    Args:
        history: TrainingHistory object with costs
        title: Plot title
        save_path: Path to save figure
        show_plot: Whether to display the plot interactively
    
    Returns:
        matplotlib Figure object or None if no history
    """
    if not history.costs:
        print("No training history to plot")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = range(len(history.costs))
    ax.plot(iterations, history.costs, 'b-', linewidth=2)
    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    ax.grid(True, alpha=0.3)
    
    # Add improvement annotation
    if len(history.costs) > 1:
        try:
            improvement = history.get_improvement_ratio()
            ax.text(0.02, 0.98, f'Improvement: {improvement:.3f}x', 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        except AttributeError:
            # Handle case where improvement method doesn't exist
            pass
    
    # Log scale if useful
    if max(history.costs) / min(history.costs) > 100:
        ax.set_yscale('log')
        ax.set_ylabel('Cost (log scale)')
    
    plt.tight_layout()
    return _save_and_show_plot(fig, save_path, show_plot)


def plot_control_forces(
    forces: Union[jnp.ndarray, np.ndarray],
    time_points: Optional[Union[jnp.ndarray, np.ndarray]] = None,
    title: str = "Control Forces",
    save_path: Optional[str] = None,
    show_plot: bool = False
) -> plt.Figure:
    """
    Plot control force history.
    
    Args:
        forces: Control force array (N,)
        time_points: Time array, created automatically if None
        title: Plot title
        save_path: Path to save figure
        show_plot: Whether to display the plot interactively
    
    Returns:
        matplotlib Figure object
    """
    if time_points is None:
        time_points = _create_time_grid(len(forces))
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(time_points, forces, 'r-', linewidth=2)
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (N)')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    max_force = float(jnp.max(jnp.abs(forces)))
    rms_force = float(jnp.sqrt(jnp.mean(forces**2)))
    stats_text = f'Max: {max_force:.3f}N\nRMS: {rms_force:.3f}N'
    ax.text(0.02, 0.98, stats_text, 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    return _save_and_show_plot(fig, save_path, show_plot)


def plot_phase_portrait(
    trajectory: Union[jnp.ndarray, np.ndarray],
    title: str = "Phase Portrait",
    save_path: Optional[str] = None,
    show_plot: bool = False
) -> plt.Figure:
    """
    Plot phase portrait (angle vs angular velocity).
    
    Args:
        trajectory: State trajectory (N, 5) format
        title: Plot title  
        save_path: Path to save figure
        show_plot: Whether to display the plot interactively
    
    Returns:
        matplotlib Figure object
    """
    angles = _extract_angles(trajectory)
    angular_velocities = trajectory[:, 4]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot trajectory with color gradient
    scatter = ax.scatter(jnp.degrees(angles), angular_velocities, 
                        c=range(len(angles)), cmap='viridis', s=20)
    
    # Mark start and end points
    ax.plot(jnp.degrees(angles[0]), angular_velocities[0], 'go', markersize=10, label='Start')
    ax.plot(jnp.degrees(angles[-1]), angular_velocities[-1], 'ro', markersize=10, label='End')
    
    ax.set_title(title)
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Angular Velocity (rad/s)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.colorbar(scatter, label='Time Step')
    plt.tight_layout()
    
    return _save_and_show_plot(fig, save_path, show_plot)


def compare_trajectories(
    trajectories: list[Union[jnp.ndarray, np.ndarray]],
    labels: list[str],
    title: str = "Trajectory Comparison",
    save_path: Optional[str] = None,
    show_plot: bool = False
) -> plt.Figure:
    """
    Compare multiple trajectories on the same plots.
    
    Args:
        trajectories: List of trajectory arrays
        labels: List of labels for each trajectory
        title: Plot title
        save_path: Path to save figure
        show_plot: Whether to display the plot interactively
    
    Returns:
        matplotlib Figure object
    
    Raises:
        ValueError: If number of trajectories doesn't match number of labels
    """
    if len(trajectories) != len(labels):
        raise ValueError("Number of trajectories must match number of labels")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=16)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    
    for i, (traj, label) in enumerate(zip(trajectories, labels)):
        time_points = _create_time_grid(len(traj))
        angles = _extract_angles(traj)
        
        # Plot all state variables
        plot_data = [
            (traj[:, 0], 'Cart Position', 'Position (m)'),
            (jnp.degrees(angles), 'Pendulum Angle', 'Angle (degrees)'),
            (traj[:, 3], 'Cart Velocity', 'Velocity (m/s)'),
            (traj[:, 4], 'Angular Velocity', 'Angular Velocity (rad/s)')
        ]
        
        for j, (data, plot_title, ylabel) in enumerate(plot_data):
            row, col = j // 2, j % 2
            axes[row, col].plot(time_points, data, color=colors[i], label=label)
            axes[row, col].set_title(plot_title)
            axes[row, col].set_ylabel(ylabel)
            axes[row, col].grid(True)
            axes[row, col].legend()
    
    # Set x-labels for bottom plots
    for col in range(2):
        axes[1, col].set_xlabel('Time (s)')
    
    plt.tight_layout()
    return _save_and_show_plot(fig, save_path, show_plot)


def plot_mujoco_simulation(
    ts: Union[jnp.ndarray, np.ndarray],
    x: Union[jnp.ndarray, np.ndarray],
    theta: Union[jnp.ndarray, np.ndarray],
    xdot: Union[jnp.ndarray, np.ndarray],
    thdot: Union[jnp.ndarray, np.ndarray],
    u: Union[jnp.ndarray, np.ndarray],
    d: Union[jnp.ndarray, np.ndarray],
    title: str = "MuJoCo Cart-Pole Simulation Results",
    save_path: Optional[str] = None,
    show_plot: bool = False
) -> plt.Figure:
    """
    Plot MuJoCo simulation results as a comprehensive 2x3 grid.
    
    Args:
        ts: Time array
        x: Cart position array
        theta: Pendulum angle array (radians)
        xdot: Cart velocity array
        thdot: Angular velocity array
        u: Control force array
        d: Disturbance force array
        title: Plot title
        save_path: Path to save figure
        show_plot: Whether to display the plot interactively
    
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    fig.suptitle(title, fontsize=16)

    # Define plot configurations
    plot_configs = [
        (0, 0, x, "Cart Position", "x [m]"),
        (0, 1, np.degrees(theta), "Pendulum Angle", "theta [deg]"),
        (0, 2, xdot, "Cart Velocity", "xdot [m/s]"),
        (1, 0, thdot, "Angular Velocity", "thdot [rad/s]"),
        (1, 1, u, "Control Force (u)", "N"),
        (1, 2, d, "Disturbance (d)", "N")
    ]

    for row, col, data, plot_title, ylabel in plot_configs:
        axes[row, col].plot(ts, data)
        axes[row, col].set_title(plot_title)
        axes[row, col].set_xlabel("t [s]")
        axes[row, col].set_ylabel(ylabel)
        axes[row, col].grid(True)
    
    plt.tight_layout()
    return _save_and_show_plot(fig, save_path, show_plot)


# Alias for backward compatibility
save_mujoco_plot = plot_mujoco_simulation


def create_animation(
    trajectory: Union[jnp.ndarray, np.ndarray],
    dt: float = 0.01,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> Optional[plt.Figure]:
    """
    Create animated visualization of cart-pole motion.
    
    Args:
        trajectory: State trajectory (N, 5) format
        dt: Time step between frames
        save_path: Path to save animation
        show_plot: Whether to display the animation (default True for animations)
    
    Returns:
        matplotlib Figure object or None if animation creation fails
    """
    try:
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("Animation requires matplotlib.animation, skipping...")
        return None
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Cart and pendulum parameters for visualization
    cart_width = 0.3
    cart_height = 0.2
    pendulum_length = 1.0
    
    # Initialize empty plots
    cart_patch = plt.Rectangle((0, 0), cart_width, cart_height, fc='blue')
    pendulum_line, = ax.plot([], [], 'r-', linewidth=4)
    pendulum_mass = plt.Circle((0, 0), 0.1, fc='red')
    
    ax.add_patch(cart_patch)
    ax.add_patch(pendulum_mass)
    ax.add_line(pendulum_line)
    
    # Set axis limits
    x_positions = trajectory[:, 0]
    x_range = max(float(jnp.max(jnp.abs(x_positions))) + 2, 3)
    ax.set_xlim(-x_range, x_range)
    ax.set_ylim(-0.5, 2)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title('Cart-Pole Animation')
    
    def animate(frame):
        if frame >= len(trajectory):
            return cart_patch, pendulum_line, pendulum_mass
        
        # Get current state
        x, cos_theta, sin_theta, _, _ = trajectory[frame]
        
        # Update cart position
        cart_patch.set_xy((x - cart_width/2, 0))
        
        # Update pendulum
        pendulum_x = x + pendulum_length * sin_theta
        pendulum_y = pendulum_length * cos_theta
        
        pendulum_line.set_data([x, pendulum_x], [cart_height, cart_height + pendulum_y])
        pendulum_mass.set_center((pendulum_x, cart_height + pendulum_y))
        
        return cart_patch, pendulum_line, pendulum_mass
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(trajectory), 
                        interval=dt*1000, blit=True, repeat=True)
    
    if save_path:
        try:
            anim.save(_ensure_dir() / save_path, writer='pillow', fps=1/dt)
        except Exception as e:
            print(f"Failed to save animation: {e}")
            if show_plot:
                plt.show()
    elif show_plot:
        plt.show()
    
    return fig

