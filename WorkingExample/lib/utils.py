# lib/utils.py

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np



###############################################################################
#                         Initial Condition Sampler                           #
###############################################################################

def sample_initial_conditions(num_samples,
                              x_range=(-0.5, 0.5),
                              theta_range=(-0.5, 0.5),
                              xdot_range=(-0.5, 0.5),
                              thetadot_range=(-0.5, 0.5),
                              key=None):
    """
    Generates random initial conditions within specified ranges.

    Args:
        num_samples: Number of initial conditions to sample.
        x_range: (min, max) range for cart position.
        theta_range: (min, max) range for pendulum angle (rad).
        xdot_range: (min, max) range for cart velocity (m/s).
        thetadot_range: (min, max) range for pendulum angular velocity (rad/s).
        key: JAX random key. If None, uses NumPy random for sampling.

    Returns:
        jnp.array of shape (num_samples, 4), each row [x, theta, x_dot, theta_dot].
    """
    if key is None:
        # Use np.random
        x_vals = np.random.uniform(*x_range, size=(num_samples,))
        theta_vals = np.random.uniform(*theta_range, size=(num_samples,))
        xdot_vals = np.random.uniform(*xdot_range, size=(num_samples,))
        thetadot_vals = np.random.uniform(*thetadot_range, size=(num_samples,))
        init_conds = np.column_stack((x_vals, theta_vals, xdot_vals, thetadot_vals))
        return jnp.array(init_conds, dtype=jnp.float32)
    else:
        # Use JAX random
        x_keys = jax.random.split(key, 4)
        x_vals = jax.random.uniform(x_keys[0], shape=(num_samples,), minval=x_range[0], maxval=x_range[1])
        theta_vals = jax.random.uniform(x_keys[1], shape=(num_samples,), minval=theta_range[0], maxval=theta_range[1])
        xdot_vals = jax.random.uniform(x_keys[2], shape=(num_samples,), minval=xdot_range[0], maxval=xdot_range[1])
        thetadot_vals = jax.random.uniform(x_keys[3], shape=(num_samples,), minval=thetadot_range[0], maxval=thetadot_range[1])
        init_conds = jnp.column_stack((x_vals, theta_vals, xdot_vals, thetadot_vals))
        return init_conds
    

def plot_cost(cost_history, title="Cost Over Iterations", log_scale=False):
    """
    Plots the cost (or loss) history over training iterations.

    Args:
        cost_history: List or array of cost values at each iteration.
        title: Plot title string.
        log_scale: If True, use a semilog-y plot for cost values.
    """
    plt.figure(figsize=(8, 5))
    if log_scale:
        plt.semilogy(cost_history, label="Cost")
    else:
        plt.plot(cost_history, label="Cost")
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.legend()
    plt.show()



def plot_trajectory_comparison(t, states_list, labels=None, title_prefix="Trajectory Comparison"):
    """
    Compares multiple trajectories on a single plot (e.g., from different controllers).

    Args:
        t: 1D array of time points.
        states_list: List of 2D arrays, each shape (len(t), 4) representing [x, theta, x_dot, theta_dot].
        labels: List of labels for each trajectory in states_list.
        title_prefix: String prefix for the plot title.
    """
    if labels is None:
        labels = [f"Trajectory {i}" for i in range(len(states_list))]

    plt.figure(figsize=(10, 6))
    for i, states in enumerate(states_list):
        x = states[:, 0]
        plt.plot(t, x, label=labels[i])
    plt.title(f"{title_prefix}: Cart Position (x)")
    plt.xlabel("Time (s)")
    plt.ylabel("x (m)")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    for i, states in enumerate(states_list):
        theta = states[:, 1]
        plt.plot(t, np.rad2deg(theta), label=labels[i])
    plt.title(f"{title_prefix}: Pendulum Angle (theta)")
    plt.xlabel("Time (s)")
    plt.ylabel("theta (degrees)")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_trajectory_comparison2(t, states_list, labels=None, title_prefix="Trajectory Comparison"):
    if labels is None:
        labels = [f"Trajectory {i+1}" for i in range(len(states_list))]

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"{title_prefix}: Trajectory Comparison", fontsize=16)

    # Plot Cart Position (x)
    ax = axs[0, 0]
    for i, states in enumerate(states_list):
        x = states[:, 0]
        ax.plot(t, x, label=labels[i])
    ax.set_title("Cart Position (x)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("x (m)")
    ax.grid(True)
    ax.legend()

    # Plot Pendulum Angle (theta)
    ax = axs[0, 1]
    for i, states in enumerate(states_list):
        theta = states[:, 1]
        ax.plot(t, np.rad2deg(theta), label=labels[i])  # Convert to degrees for better interpretability
    ax.set_title("Pendulum Angle (theta)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("theta (degrees)")
    ax.grid(True)
    ax.legend()

    # Plot Cart Velocity (x_dot)
    ax = axs[1, 0]
    for i, states in enumerate(states_list):
        x_dot = states[:, 2]
        ax.plot(t, x_dot, label=labels[i])
    ax.set_title("Cart Velocity (x_dot)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("x_dot (m/s)")
    ax.grid(True)
    ax.legend()

    # Plot Pendulum Angular Velocity (theta_dot)
    ax = axs[1, 1]
    for i, states in enumerate(states_list):
        theta_dot = states[:, 3]
        ax.plot(t, theta_dot, label=labels[i])
    ax.set_title("Pendulum Angular Velocity (theta_dot)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("theta_dot (rad/s)")
    ax.grid(True)
    ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the suptitle
    plt.show()



def plot_energy(t, states, params, title="Energy Over Time"):
    mc, mp, l, g = params
    def compute_energy(state):
        x, theta, x_dot, theta_dot = state
        K = 0.5 * (mc + mp) * x_dot**2 - mp*l*jnp.cos(theta)*x_dot*theta_dot + 0.5*mp*(l**2)*theta_dot**2
        P = mp*g*l*(1.0 - jnp.cos(theta))
        return K + P

    E_vals = jax.vmap(compute_energy)(states)
    E_vals = np.array(E_vals)  # Convert to NumPy for plotting

    plt.figure(figsize=(8, 5))
    plt.plot(t, E_vals, label="Energy")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (J)")
    plt.grid(True)
    plt.legend()
    plt.show()

def compute_trajectory_cost(Q, states, controller_func, t):
    dt = t[1] - t[0]
    forces = jnp.array([controller_func(states[i], t[i]) for i in range(len(t))])
    cost = 0.0
    for i in range(len(states)):
        x = states[i]
        f = forces[i]
        cost += (x @ Q @ x + f**2) * dt
    return cost, forces


def plot_cost_comparison(linear_cost, lqr_cost, title="Trajectory Cost Comparison"):
    plt.figure(figsize=(6,4))
    plt.bar(["Trained Linear", "LQR"], [linear_cost, lqr_cost], color=["blue", "orange"])
    plt.title(title)
    plt.ylabel("Cost")
    plt.grid(True)
    plt.show()

def plot_control_forces_comparison(t, linear_forces, lqr_forces, title="Control Forces Comparison"):
    plt.figure(figsize=(6,4))
    plt.plot(t, linear_forces, label="Trained Linear")
    plt.plot(t, lqr_forces, label="LQR", linestyle="--")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_trajectories(t, traj, title_suffix=""):
    labels = ["x (Cart Position)", "cos(θ)", "sin(θ)", "ẋ (Cart Velocity)", "θ̇ (Angular Velocity)"]
    plt.figure(figsize=(12, 10))

    for i in range(5):
        plt.subplot(3, 2, i + 1)
        plt.plot(t, traj[:, i], label=labels[i])
        plt.xlabel("Time [s]")
        plt.ylabel(labels[i])
        plt.title(f"{labels[i]} over Time {title_suffix}")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()



def plot_trajectories3(t, traj, title_suffix=""):
    labels = ["x (Cart Position)", "ẋ (Cart Velocity)", "θ̇ (Angular Velocity)"]
    plt.figure(figsize=(12, 10))

    # First Subplot
    plt.subplot(3, 1, 1)
    plt.plot(t, traj[:, 0], label=labels[0])
    plt.xlabel("Time [s]")
    plt.ylabel(labels[0])
    plt.title(f"{labels[0]} over Time {title_suffix}")
    plt.grid(True)
    plt.legend()

    # Second Subplot
    plt.subplot(3, 1, 2)
    plt.plot(t, traj[:, 1], label=labels[1])
    plt.xlabel("Time [s]")
    plt.ylabel(labels[1])
    plt.title(f"{labels[1]} over Time {title_suffix}")
    plt.grid(True)
    plt.legend()

    # Third Subplot
    plt.subplot(3, 1, 3)
    plt.plot(t, traj[:, 2], label=labels[2])
    plt.xlabel("Time [s]")
    plt.ylabel(labels[2])
    plt.title(f"{labels[2]} over Time {title_suffix}")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()




def plot_energies(time, total_energy, kinetic_energy, potential_energy, desired_energy):
    """
    Plot the system's total, kinetic, and potential energies over time, along with a desired energy reference line.

    Parameters:
    ----------
    time : array-like
        Array of time points.
    total_energy : array-like
        Array of total energy values corresponding to each time point.
    kinetic_energy : array-like
        Array of kinetic energy values corresponding to each time point.
    potential_energy : array-like
        Array of potential energy values corresponding to each time point.
    desired_energy : float
        The desired energy level to be indicated as a reference line on the plot.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot Total Energy
    plt.plot(time, total_energy, label="Total Energy", linewidth=2, color="blue")
    
    # Plot Kinetic Energy
    plt.plot(time, kinetic_energy, label="Kinetic Energy", linestyle="--", color="green")
    
    # Plot Potential Energy
    plt.plot(time, potential_energy, label="Potential Energy", linestyle=":", color="orange")
    
    # Plot Desired Energy Reference Line
    plt.axhline(y=desired_energy, color="red", linestyle="--", label="Desired Energy")
    
    # Configure Plot Aesthetics
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Energy [J]", fontsize=12)
    plt.title("System Energy Over Time", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    # Display the Plot
    plt.show()

    

def plot_theta_over_time(time, theta_normalized):
    """
    Plot the pendulum angle (θ) over time, normalized to the range [0, 2π], with reference lines for upright positions.

    Parameters:
    ----------
    time : array-like
        Array of time points.
    theta_normalized : array-like
        Array of normalized pendulum angle values (θ) corresponding to each time point, normalized to [0, 2π].
    """
    plt.figure(figsize=(10, 6))
    
    # Plot Pendulum Angle
    plt.plot(time, theta_normalized, label=r"$\theta$ (Angle)", color="blue", linewidth=2)
    
    # Reference Line for Upright Position (θ = 0)
    plt.axhline(y=0, color="red", linestyle="--", label="Upright Position ($\\theta = 0$)")
    
    # Reference Line for Equivalent Upright Position (θ = 2π)
    plt.axhline(y=2 * jnp.pi, color="green", linestyle="--", label=r"Equivalent Upright Position ($\theta = 2\pi$)")
    
    # Configure Plot Aesthetics
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel(r"$\theta$ (Angle) [rad]", fontsize=12)
    plt.title("Pendulum Angle ($\\theta$) Over Time (Normalized to [0, $2\pi$])", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    # Display the Plot
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

def plot_combined_trajectories(t, traj, title_suffix=""):
    """
    Plot θ, cos(θ), and sin(θ) over time in a 3x1 subplot layout.

    Parameters:
    ----------
    t : array-like
        Array of time points.
    traj : array-like
        2D array where each column corresponds to:
            - traj[:, 0]: θ (Pendulum Angle) [rad]
            - traj[:, 1]: cos(θ)
            - traj[:, 2]: sin(θ)
    title_suffix : str, optional
        Suffix to append to each plot title for additional context.
    """
    # Validate input dimensions
    if traj.shape[1] < 3:
        raise ValueError("The 'traj' array must have at least three columns: θ, cos(θ), and sin(θ).")

    # Define labels and titles
    labels = [r"$\theta$ (Angle) [rad]", r"$\cos(\theta)$", r"$\sin(\theta)$"]
    y_labels = [labels[0], labels[1], labels[2]]

    plt.figure(figsize=(12, 15))  # Increased height for better readability

    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(t, traj[:, i], label=labels[i], color='blue' if i == 0 else 'green' if i == 1 else 'orange', linewidth=2)
        plt.xlabel("Time [s]", fontsize=12)
        plt.ylabel(y_labels[i], fontsize=12)
        plt.title(f"{y_labels[i]} over Time {title_suffix}", fontsize=14)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(fontsize=12)

        # If plotting θ, add reference lines
        if i == 0:
            plt.axhline(y=0, color="red", linestyle="--", label=r"Upright Position ($\theta = 0$)")
            plt.axhline(y=2 * np.pi, color="green", linestyle="--", label=r"Equivalent Upright Position ($\theta = 2\pi$)")
            plt.legend(fontsize=12)

    plt.tight_layout()
    plt.show()
