"""Visualization utilities for cart-pole experiments."""

from __future__ import annotations
import matplotlib.pyplot as plt
import jax.numpy as jnp
from pathlib import Path
from typing import Iterable, Optional, Tuple
import numpy as np


_RESULTS_DIR = Path("results")


def _ensure_dir() -> Path:
    """Ensure results directory exists."""
    _RESULTS_DIR.mkdir(exist_ok=True)
    return _RESULTS_DIR


def plot_trajectory(
    trajectory: jnp.ndarray,
    time_points: jnp.ndarray = None,
    title: str = "Cart-Pole Trajectory",
    save_path: Optional[str] = None
) -> None:
    """
    Plot cart-pole trajectory with all state variables.
    
    Args:
        trajectory: State trajectory (N, 5) format [x, cos(θ), sin(θ), ẋ, θ̇]
        time_points: Time array, created automatically if None
        title: Plot title
        save_path: Path to save figure, displays if None
    """
    if time_points is None:
        time_points = jnp.arange(len(trajectory)) * 0.01
    
    # Reconstruct angle from cos/sin representation
    angles = jnp.arctan2(trajectory[:, 2], trajectory[:, 1])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=16)
    
    # Position plot
    axes[0, 0].plot(time_points, trajectory[:, 0])
    axes[0, 0].set_title('Cart Position')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].grid(True)
    
    # Angle plot
    axes[0, 1].plot(time_points, jnp.degrees(angles))
    axes[0, 1].set_title('Pendulum Angle')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Angle (degrees)')
    axes[0, 1].grid(True)
    
    # Cart velocity plot
    axes[1, 0].plot(time_points, trajectory[:, 3])
    axes[1, 0].set_title('Cart Velocity')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Velocity (m/s)')
    axes[1, 0].grid(True)
    
    # Angular velocity plot
    axes[1, 1].plot(time_points, trajectory[:, 4])
    axes[1, 1].set_title('Angular Velocity')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Angular Velocity (rad/s)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(_ensure_dir() / save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def plot_training_history(
    history,
    title: str = "Training Progress",
    save_path: Optional[str] = None
) -> None:
    """
    Plot training cost history.
    
    Args:
        history: TrainingHistory object with costs
        title: Plot title
        save_path: Path to save figure, displays if None
    """
    if not history.costs:
        print("No training history to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = range(len(history.costs))
    ax.plot(iterations, history.costs, 'b-', linewidth=2)
    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    ax.grid(True, alpha=0.3)
    
    # Add improvement annotation
    if len(history.costs) > 1:
        improvement = history.get_improvement_ratio()
        ax.text(0.02, 0.98, f'Improvement: {improvement:.3f}x', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Log scale if useful
    if max(history.costs) / min(history.costs) > 100:
        ax.set_yscale('log')
        ax.set_ylabel('Cost (log scale)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(_ensure_dir() / save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def plot_control_forces(
    forces: jnp.ndarray,
    time_points: jnp.ndarray = None,
    title: str = "Control Forces",
    save_path: Optional[str] = None
) -> None:
    """
    Plot control force history.
    
    Args:
        forces: Control force array (N,)
        time_points: Time array, created automatically if None
        title: Plot title
        save_path: Path to save figure, displays if None
    """
    if time_points is None:
        time_points = jnp.arange(len(forces)) * 0.01
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(time_points, forces, 'r-', linewidth=2)
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (N)')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f'Max: {jnp.max(jnp.abs(forces)):.3f}N\nRMS: {jnp.sqrt(jnp.mean(forces**2)):.3f}N'
    ax.text(0.02, 0.98, stats_text, 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(_ensure_dir() / save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def plot_phase_portrait(
    trajectory: jnp.ndarray,
    title: str = "Phase Portrait",
    save_path: Optional[str] = None
) -> None:
    """
    Plot phase portrait (angle vs angular velocity).
    
    Args:
        trajectory: State trajectory (N, 5) format
        title: Plot title  
        save_path: Path to save figure, displays if None
    """
    # Reconstruct angle
    angles = jnp.arctan2(trajectory[:, 2], trajectory[:, 1])
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
    
    if save_path:
        plt.savefig(_ensure_dir() / save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def compare_trajectories(
    trajectories: list[jnp.ndarray],
    labels: list[str],
    title: str = "Trajectory Comparison",
    save_path: Optional[str] = None
) -> None:
    """
    Compare multiple trajectories on the same plots.
    
    Args:
        trajectories: List of trajectory arrays
        labels: List of labels for each trajectory
        title: Plot title
        save_path: Path to save figure, displays if None
    """
    if len(trajectories) != len(labels):
        raise ValueError("Number of trajectories must match number of labels")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=16)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    
    for i, (traj, label) in enumerate(trajectories, labels):
        time_points = jnp.arange(len(traj)) * 0.01
        angles = jnp.arctan2(traj[:, 2], traj[:, 1])
        
        # Position
        axes[0, 0].plot(time_points, traj[:, 0], color=colors[i], label=label)
        
        # Angle  
        axes[0, 1].plot(time_points, jnp.degrees(angles), color=colors[i], label=label)
        
        # Cart velocity
        axes[1, 0].plot(time_points, traj[:, 3], color=colors[i], label=label)
        
        # Angular velocity
        axes[1, 1].plot(time_points, traj[:, 4], color=colors[i], label=label)
    
    # Configure axes
    axes[0, 0].set_title('Cart Position')
    axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    axes[0, 1].set_title('Pendulum Angle')
    axes[0, 1].set_ylabel('Angle (degrees)')
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    
    axes[1, 0].set_title('Cart Velocity')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Velocity (m/s)')
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    
    axes[1, 1].set_title('Angular Velocity')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Angular Velocity (rad/s)')
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(_ensure_dir() / save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def create_animation(
    trajectory: jnp.ndarray,
    dt: float = 0.01,
    save_path: Optional[str] = None
) -> None:
    """
    Create animated visualization of cart-pole motion.
    
    Args:
        trajectory: State trajectory (N, 5) format
        dt: Time step between frames
        save_path: Path to save animation, displays if None
    """
    try:
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("Animation requires matplotlib.animation, skipping...")
        return
    
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
    x_range = max(jnp.max(jnp.abs(x_positions)) + 2, 3)
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
        
        pendulum_line.set_data([x, pendulum_x], [cart_height, pendulum_y])
        pendulum_mass.set_center((pendulum_x, pendulum_y))
        
        return cart_patch, pendulum_line, pendulum_mass
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(trajectory), 
                        interval=dt*1000, blit=True, repeat=True)
    
    if save_path:
        try:
            anim.save(_ensure_dir() / save_path, writer='pillow', fps=1/dt)
        except Exception as e:
            print(f"Failed to save animation: {e}")
            plt.show()
    else:
        plt.show()

