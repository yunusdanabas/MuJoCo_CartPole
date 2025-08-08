# main_nn_only.py
import jax
import jax.numpy as jnp
from diffrax import ODETerm, Tsit5, SaveAt, diffeqsolve
import equinox as eqx

from controller.neuralnetwork_controller import MLP

from env.cartpole import cartpole_dynamics_nn

from lib.trainer import train_nn_controller, adaptive_scaling, compute_energy

from lib.utils import plot_trajectories,plot_theta_over_time,plot_energies,plot_trajectories3,plot_combined_trajectories




def main():
    """
    Main function to train a neural network controller for the cart-pole system,
    simulate the closed-loop cartpole_dynamics_nn, and visualize the results.
    """
    
    # -----------------------------------------------------------------------
    # 1. Define System Parameters and Initial Conditions
    # -----------------------------------------------------------------------
    
    # Cart-pole system parameters
    mass_cart = 1.0      # Mass of the cart (m_c) in kg
    mass_pole = 0.2      # Mass of the pole (m_p) in kg
    pole_length = 0.5    # Length of the pole (l) in meters
    gravity = 9.81       # Acceleration due to gravity (g) in m/s²
    env_params = (mass_cart, mass_pole, pole_length, gravity)
    
    # Initial state: x cos(theta) sin(theta) x_dot theta_dot
    initial_state = jnp.array([0.0, -1.0, 0.0, 0.0, 0.0])


    # -----------------------------------------------------------------------
    # 2. Initialize the Neural Network Controller
    # -----------------------------------------------------------------------
    
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(0)
    
    # Initialize the Multi-Layer Perceptron (MLP) model
    model = MLP(
        in_size=5,
        hidden_sizes=[64, 64],
        out_size=1,
        key=key
    )


    # -----------------------------------------------------------------------
    # 3. Train the Neural Network Controller
    # -----------------------------------------------------------------------
    
    print("Training the neural network controller...")
    
    # Training hyperparameters
    num_iterations = 2000
    learning_rate = 1e-3
    dt = 0.001
    simulation_time = 10.0  # Total simulation time during training
    

    # Train the neural network controller
    trained_model, cost_history = train_nn_controller(
        env_params,
        model,
        num_iterations,
        learning_rate,
        dt,
        simulation_time
    )
    
    print("Training complete.")


    # Save the trained model
    eqx.tree_serialise_leaves("trained_nn_model.eqx", trained_model)
    print("Trained NN model saved as 'trained_nn_model.eqx'.")


    # -----------------------------------------------------------------------
    # 4. Define Desired Energy Level
    # -----------------------------------------------------------------------
    
    desired_energy = 2.0 * mass_pole * gravity * pole_length
    #print(f"Desired energy (E_desired): {desired_energy:.2f} J")
    

    # -----------------------------------------------------------------------
    # 5. Define the Control Force Function
    # -----------------------------------------------------------------------
    
    def force_function(t, state):
        """
        Compute the control force at a given time and state.

        Parameters:
        ----------
        t : float
            Current time.
        state : array-like
            Current state of the system.

        Returns:
        -------
        float
            Control force to be applied.
        """
        # Compute the adaptive scaling factor based on the current state
        scaling_factor = adaptive_scaling(
            state,
            mass_cart,
            mass_pole,
            pole_length,
            gravity,
            desired_energy
        )
        

        # Compute the raw force from the trained model
        raw_force = trained_model(state)
        
        # Scale the force appropriately
        force = 15 * scaling_factor * raw_force  
        
        return force


    # -----------------------------------------------------------------------
    # 6. Set Up and Solve the Differential Equation
    # -----------------------------------------------------------------------
    
    # Define the ODE term using the cart-pole cartpole_dynamics_nn
    ode_term = ODETerm(cartpole_dynamics_nn)
    
    # Choose the ODE solver (Tsit5 is a 5th-order Runge-Kutta method)
    solver = Tsit5()
    
    # Simulation time settings
    t_start = 0.0
    t_end = 10.0
    num_time_steps = 1001
    time_points = jnp.linspace(t_start, t_end, num_time_steps)

    # Solve the ODE using diffeqsolve
    solution = diffeqsolve(
        ode_term,
        solver,
        t0=t_start,
        t1=t_end,
        dt0=0.001,
        y0=initial_state,
        args=(env_params, force_function),
        saveat=SaveAt(ts=time_points),
        max_steps=50_000
    )

    # -----------------------------------------------------------------------
    # 7. Plot the Trajectories
    # -----------------------------------------------------------------------
    
    print("Plotting the trajectories...")
    plot_trajectories3(
        solution.ts,
        solution.ys,
        "(Trained Controller)"
    )
    
    # -----------------------------------------------------------------------
    # 8. Analyze and Plot Energy and Angle Over Time
    # -----------------------------------------------------------------------
    
    # Extract theta (angle) from simulation results
    sin_theta = solution.ys[:, 2]  # Assuming state[2] corresponds to sin(theta)
    cos_theta = solution.ys[:, 1]  # Assuming state[1] corresponds to cos(theta)
    theta_values = jax.vmap(jnp.arctan2)(sin_theta, cos_theta)
    
    # Normalize theta to the range [0, 2π]
    theta_normalized = jnp.mod(theta_values, 2 * jnp.pi)
    
    # Compute initial potential energy
    initial_potential_energy = mass_pole * gravity * pole_length * (1 - cos_theta[0])
    #print(f"Initial potential energy: {initial_potential_energy:.2f} J")
    #print(f"Desired energy: {desired_energy:.2f} J")
    
    # Compute total, kinetic, and potential energies over time
    energies, kinetic_energies, potential_energies = jax.vmap(
        compute_energy,
        in_axes=(0, None, None, None, None)
    )(solution.ys, mass_cart, mass_pole, pole_length, gravity)
    
    # # Plot Energies Over Time
    # plot_energies(
    #     solution.ts,
    #     energies,
    #     kinetic_energies,
    #     potential_energies,
    #     desired_energy
    # )
    
    # Plot Pendulum Angle Over Time
    plot_theta_over_time(
        solution.ts,
        theta_normalized
    )


    # Convert JAX arrays to NumPy arrays for plotting (if necessary)
    traj = jnp.stack([theta_normalized, cos_theta, sin_theta], axis=1)  # Shape: (N, 3)
    traj_np = jnp.array(traj)
    t_np = jnp.array(solution.ts)
    
    # Plot Combined Trajectories using the merged function
    plot_combined_trajectories(
        t_np,
        traj_np,
        title_suffix="(Trained Controller)"
    )
# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()