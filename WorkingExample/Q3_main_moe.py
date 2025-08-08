# main_moe.py

import jax
import jax.numpy as jnp
import jax.random as jr
from diffrax import ODETerm, Tsit5, SaveAt, diffeqsolve

import equinox as eqx

# Import MoE Controller Components
from controller.moe_controller import GatingNetwork, DeterministicGating, MoEController

# Local project imports (adjust paths if needed)
from controller.neuralnetwork_controller import MLP
from controller.lqr_controller import linearize_cartpole, compute_lqr_gain
from env.cartpole import cartpole_dynamics_nn
from env.closedloop import simulate_closed_loop
from lib.trainer import train_nn_controller, adaptive_scaling, compute_energy
from lib.utils import (
    plot_trajectory_comparison2,
    plot_trajectories,
    plot_theta_over_time,
    plot_energies,
    plot_combined_trajectories,
)

import matplotlib.pyplot as plt

# --- Add the following helper functions ---

def compute_energy_fixed(state, mass_cart, mass_pole, pole_length, gravity):
    """
    Computes the total, kinetic, and potential energy of the cart-pole system.
    Transforms a 4-element state into a 5-element state before computation.

    Args:
        state (jnp.ndarray): Current state [x, theta, x_dot, theta_dot].
        mass_cart (float): Mass of the cart.
        mass_pole (float): Mass of the pole.
        pole_length (float): Length of the pole.
        gravity (float): Acceleration due to gravity.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: Total energy, kinetic energy, potential energy.
    """
    x, theta, x_dot, theta_dot = state
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    state_5 = jnp.array([x, cos_theta, sin_theta, x_dot, theta_dot])
    return compute_energy(state_5, mass_cart, mass_pole, pole_length, gravity)


def adaptive_scaling_fixed(state, mass_cart, mass_pole, pole_length, gravity, desired_energy):
    """
    Computes an adaptive scaling factor based on the current energy.
    Transforms a 4-element state into a 5-element state before computation.

    Args:
        state (jnp.ndarray): Current state [x, theta, x_dot, theta_dot].
        mass_cart (float): Mass of the cart.
        mass_pole (float): Mass of the pole.
        pole_length (float): Length of the pole.
        gravity (float): Acceleration due to gravity.
        desired_energy (float): Desired energy level for the system.

    Returns:
        float: Scaling factor.
    """
    x, theta, x_dot, theta_dot = state
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    state_5 = jnp.array([x, cos_theta, sin_theta, x_dot, theta_dot])
    return adaptive_scaling(state_5, mass_cart, mass_pole, pole_length, gravity, desired_energy)


def main():
    """
    Main function that:
      1) Trains an NN for cart-pole swing-up.
      2) Sets up an LQR controller for stabilization.
      3) Initializes the MoE controller with the NN and LQR experts.
      4) Simulates the hybrid approach.
      5) Plots the results.
    """

    # -----------------------------------------------------------------------
    # A. Define System Parameters
    # -----------------------------------------------------------------------
    mass_cart = 1.0
    mass_pole = 0.2
    pole_length = 0.5
    gravity = 9.81
    env_params = (mass_cart, mass_pole, pole_length, gravity)

    # -----------------------------------------------------------------------
    # B. Initialize and Train the Neural Network Controller
    # -----------------------------------------------------------------------
    key = jr.PRNGKey(0)
    nn_model = MLP(
        in_size=5,         # [x, sin(theta), cos(theta), x_dot, theta_dot]
        hidden_sizes=[64, 64],
        out_size=1,
        key=key
    )

    print("\n=== Training the Neural Network Controller ===")
    num_iterations = 1500 # Increased training iterations
    learning_rate = 1e-3
    dt = 0.001
    simulation_time = 10.0

    trained_model, cost_history = train_nn_controller(
        env_params=env_params,
        model=nn_model,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        dt=dt,
        T=simulation_time
    )
    print("NN Training Complete!")

    # Plot training cost if desired
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Neural Network Controller Training Cost')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

    # -----------------------------------------------------------------------
    # C. Setup the LQR Controller
    # -----------------------------------------------------------------------
    print("\n=== Setting up the LQR Controller ===")
    A, B = linearize_cartpole(jnp.array(env_params))
    Q_lqr = jnp.diag(jnp.array([50.0, 50.0, 5.0, 10.0]))
    R_lqr = jnp.array([[0.1]])
    K = compute_lqr_gain(A, B, Q_lqr, R_lqr)  # Shape: (1,4)

    def lqr_controller(state: jnp.ndarray, t: float) -> float:
        """
        LQR controller function.

        Args:
            state (jnp.ndarray): Current state [x, theta, x_dot, theta_dot].
            t (float): Current time.

        Returns:
            float: Control force.
        """
        return -(K @ state)[0]

    # -----------------------------------------------------------------------
    # D. Initialize the Gating Network
    # -----------------------------------------------------------------------
    print("\n=== Initializing the Gating Network ===")
    # Option 1: Deterministic Gating for Testing
    #gating_net = DeterministicGating(
    #    angle_threshold=jnp.deg2rad(15.0),
    #    angular_vel_threshold=1500.0,
    #    x_threshold=1000.0,
    #    xdot_threshold=500.0
    #)

    # Option 2: Learned Gating Network
    # Uncomment the following lines to use a learned gating network
    
    gating_key = jr.PRNGKey(0)
    gating_net = GatingNetwork(in_dim=4, num_experts=2, key=gating_key, temperature=0.3)
    

    # -----------------------------------------------------------------------
    # E. Initialize the MoE Controller
    # -----------------------------------------------------------------------
    print("\n=== Initializing the Mixture-of-Experts (MoE) Controller ===")

    def nn_controller_fixed(state: jnp.ndarray, t: float) -> float:
        """
        Neural Network controller function using fixed adaptive scaling.

        Args:
            state (jnp.ndarray): Current state [x, theta, x_dot, theta_dot].
            t (float): Current time.

        Returns:
            float: Control force.
        """
        x, theta, x_dot, theta_dot = state
        # Construct a 5-element state: [x, cos(theta), sin(theta), x_dot, theta_dot]
        state_5 = jnp.array([
            x,
            jnp.cos(theta),
            jnp.sin(theta),
            x_dot,
            theta_dot
        ])
        raw_force = trained_model(state_5)
        # Compute desired energy
        desired_energy = 2.0 * mass_pole * gravity * pole_length
        # Apply adaptive scaling using the fixed function
        scaling_factor = adaptive_scaling_fixed(
            state,  # Pass the 4-element state
            mass_cart,
            mass_pole,
            pole_length,
            gravity,
            desired_energy
        )
        # Scale the force appropriately (adjust the scaling factor as needed)
        force = 10.0 * scaling_factor * raw_force  # Adjusted scaling factor
        return force

    # Define the LQR controller as an expert
    def lqr_ctrl(state: jnp.ndarray, t: float) -> float:
        return lqr_controller(state, t)

    # Create the MoE controller with NN and LQR experts
    moe_controller = MoEController(
        gating=gating_net,
        experts=[nn_controller_fixed, lqr_ctrl]
    )

    # -----------------------------------------------------------------------
    # F. Simulate the Hybrid MoE Controller
    # -----------------------------------------------------------------------
    print("\n=== Simulating the Hybrid MoE Controller ===")
    # Define simulation parameters
    t_start = 0.0
    t_end = 14.0  # Extended simulation time
    num_time_steps = 1401  # Increased number of time steps for higher resolution
    t_eval = jnp.linspace(t_start, t_end, num_time_steps)
    t_span = (t_start, t_end)

    # Define initial conditions for simulation
    initial_state_hybrid = jnp.array([0.0, jnp.pi, 0.0, 0.0])  # [x, theta, x_dot, theta_dot]

    # Simulate closed-loop system using the MoE controller
    solution_moe = simulate_closed_loop(
        controller=moe_controller,
        params=jnp.array(env_params),
        t_span=t_span,
        t=t_eval,
        initial_state=initial_state_hybrid
    )

    states_moe = solution_moe.ys  # Shape: (len(t_eval), 4)

    # -----------------------------------------------------------------------
    # G. Plot and Analyze the Results
    # -----------------------------------------------------------------------
    print("\n=== Plotting the Hybrid MoE Controller Results ===")
    print("Final State (Hybrid MoE):", states_moe[-1, :])

    # Plot trajectory comparison
    plot_trajectory_comparison2(
        t_eval,
        [states_moe],
        labels=["Hybrid MoE (NN + LQR)"],
        title_prefix="Hybrid MoE Controller"
    )

    # Plot energies over time using the fixed compute_energy function
    energies, kinetic_energies, potential_energies = jax.vmap(
        compute_energy_fixed,  # Use the fixed function
        in_axes=(0, None, None, None, None)
    )(states_moe, mass_cart, mass_pole, pole_length, gravity)

    desired_energy = 2.0 * mass_pole * gravity * pole_length

    plot_energies(
        solution_moe.ts,
        energies,
        kinetic_energies,
        potential_energies,
        desired_energy
    )

    # Plot pendulum angle over time
    theta_values = jnp.arctan2(states_moe[:, 2], states_moe[:, 1])  # tan(theta) = sin(theta)/cos(theta)
    theta_normalized = jnp.mod(theta_values, 2 * jnp.pi)

    plot_theta_over_time(
        solution_moe.ts,
        theta_normalized
    )

    print("\n=== Simulation Complete! ===")


if __name__ == "__main__":
    main()
