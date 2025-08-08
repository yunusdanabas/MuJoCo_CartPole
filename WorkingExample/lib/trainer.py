# lib/trainer.py

import time
from functools import partial
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from diffrax import ODETerm, Tsit5, SaveAt, diffeqsolve

# Local Module Imports
from env.cartpole import cartpole_dynamics_nn


# ---------------------------------------------------------------------------
# Energy Computation Functions
# ---------------------------------------------------------------------------

def compute_energy(state, mass_cart, mass_pole, pole_length, gravity):
    """
    Computes the total, kinetic, and potential energy of the cart-pole system.

    Parameters:
    ----------
    state : jax.numpy.ndarray
        The current state of the system [x, cos(theta), sin(theta), x_dot, theta_dot].
    mass_cart : float
        Mass of the cart (mc) in kilograms.
    mass_pole : float
        Mass of the pole (mp) in kilograms.
    pole_length : float
        Length of the pole (l) in meters.
    gravity : float
        Acceleration due to gravity (g) in m/s².

    Returns:
    -------
    tuple
        A tuple containing:
        - total_energy (float): Total energy of the system.
        - kinetic_energy (float): Kinetic energy of the system.
        - potential_energy (float): Potential energy of the system.
    """
    x, cos_theta, sin_theta, x_dot, theta_dot = state

    # Desired and maximum energy levels
    E_desired = 2.0 * mass_pole * gravity * pole_length
    E_max = 3.0 * E_desired  # Maximum energy to prevent excessive values

    # Potential Energy: Depends on the angle theta
    potential_energy = mass_pole * gravity * pole_length * (1 - cos_theta)

    # Kinetic Energy: Sum of translational and rotational kinetic energies
    kinetic_energy = (
        0.5 * (mass_cart + mass_pole) * (x_dot**2) +
        0.5 * mass_pole * (pole_length**2) * (theta_dot**2)
    )

    # Total Energy: Sum of kinetic and potential energies
    total_energy = kinetic_energy + potential_energy

    # Clip the total energy to the maximum energy level
    total_energy = jnp.minimum(total_energy, E_max)

    return total_energy, kinetic_energy, potential_energy


def adaptive_scaling(state, mass_cart, mass_pole, pole_length, gravity, E_desired):
    """
    Computes an adaptive scaling factor based on the current energy of the system.

    Parameters:
    ----------
    state : jax.numpy.ndarray
        The current state of the system [x, cos(theta), sin(theta), x_dot, theta_dot].
    mass_cart : float
        Mass of the cart (mc) in kilograms.
    mass_pole : float
        Mass of the pole (mp) in kilograms.
    pole_length : float
        Length of the pole (l) in meters.
    gravity : float
        Acceleration due to gravity (g) in m/s².
    E_desired : float
        Desired energy level for the system.

    Returns:
    -------
    float
        The adaptive scaling factor, clipped within a specified range.
    """
    energy, _, _ = compute_energy(state, mass_cart, mass_pole, pole_length, gravity)
    scaling_factor = jnp.clip(energy / E_desired, 0.5, 30.0)  # Clipped to [0.5, 30.0]
    return scaling_factor


# ---------------------------------------------------------------------------
# Initial Conditions Generation
# ---------------------------------------------------------------------------

def generate_initial_conditions(batch_size, key):
    """
    Generates a batch of random initial conditions for training.

    Parameters:
    ----------
    batch_size : int
        Number of initial conditions to generate.
    key : jax.random.PRNGKey
        Random key for JAX's random number generator.

    Returns:
    -------
    jax.numpy.ndarray
        A batch of initial states with shape (batch_size, 5).
    """
    # Randomly sample cart positions and velocities
    x = jax.random.uniform(key, shape=(batch_size,), minval=-10.0, maxval=10.0)      # Cart position
    x_dot = jax.random.uniform(key, shape=(batch_size,), minval=-2.0, maxval=2.0)   # Cart velocity

    # Randomly sample angles and angular velocities
    cos_theta = jax.random.uniform(key, shape=(batch_size,), minval=-1.0, maxval=1.0)
    sin_theta = jax.random.uniform(key, shape=(batch_size,), minval=-1.0, maxval=1.0)
    theta_dot = jax.random.uniform(key, shape=(batch_size,), minval=-4.0, maxval=4.0)  # Angular velocity

    # Normalize the cosine and sine of theta to ensure valid angles
    norm = jnp.sqrt(cos_theta**2 + sin_theta**2)
    cos_theta /= norm
    sin_theta /= norm

    # Stack all state variables into a single array
    initial_conditions = jnp.stack([x, cos_theta, sin_theta, x_dot, theta_dot], axis=1)
    return initial_conditions





# ---------------------------------------------------------------------------
# Cost Computation Functions
# ---------------------------------------------------------------------------

def instant_cost(state, desired_state, model, mass_cart, mass_pole, pole_length, gravity, E_desired):
    """
    Computes the instantaneous cost for a given state and control action.

    Parameters:
    ----------
    state : jax.numpy.ndarray
        The current state [x, cos(theta), sin(theta), x_dot, theta_dot].
    desired_state : jax.numpy.ndarray
        The desired state [x, cos(theta), sin(theta), x_dot, theta_dot].
    model : callable
        The neural network model (controller) that outputs control force.
    mass_cart : float
        Mass of the cart.
    mass_pole : float
        Mass of the pole.
    pole_length : float
        Length of the pole.
    gravity : float
        Gravitational acceleration.
    E_desired : float
        Desired energy for the upright position.

    Returns:
    -------
    float
        The computed instantaneous cost.
    """
    # Unpack current and desired states
    x, cos_theta, sin_theta, x_dot, theta_dot = state
    x_des, cos_theta_des, sin_theta_des, x_dot_des, theta_dot_des = desired_state

    # Compute control force from the neural network model
    model_output = model(state)

    # Energy-based cost: Penalize deviation from desired energy
    energy, _, _ = compute_energy(state, mass_cart, mass_pole, pole_length, gravity)
    energy_cost = 50.0 * jnp.abs(energy - E_desired)

    # State deviation cost: Penalize deviation from desired state
    state_error = (
        (50.0 * (x - x_des) ** 2) +
        (150.0 * (cos_theta - cos_theta_des) ** 2) +
        (1.0 * (sin_theta - sin_theta_des) ** 2) +
        (20.0 * (x_dot - x_dot_des) ** 2) +
        (250.0 * (theta_dot - theta_dot_des) ** 2)
    )

    # Oscillatory cost: Introduce additional cost based on state dynamics
    oscillatory_cost = 15.0 * jnp.cos(x) * jnp.sin(theta_dot)

    # Control effort cost: Penalize large control forces
    control_cost = 0.5 * model_output ** 2

    # Combine all cost components and normalize
    total_cost = (energy_cost + state_error + control_cost + oscillatory_cost) / 1_000_000.0
    return total_cost







def single_rollout_cost_nn(model, dynamics_func, env_params, T=5.0, dt=0.01, key=None):
    """
    Computes the total cost over a single rollout using the neural network controller.

    Parameters:
    ----------
    model : callable
        The neural network model (controller) that outputs control force.
    dynamics_func : callable
        The dynamics function defining the system's ODE.
    env_params : tuple
        Environment parameters (mass_cart, mass_pole, pole_length, gravity).
    T : float, optional
        Total simulation time for the rollout. Default is 5.0 seconds.
    dt : float, optional
        Time step size for the simulation. Default is 0.01 seconds.
    key : jax.random.PRNGKey, optional
        Random key for generating initial conditions. If None, a default key is used.

    Returns:
    -------
    float
        The average cost over all initial conditions in the rollout.
    """
    mass_cart, mass_pole, pole_length, gravity = env_params
    E_desired = 2.0 * mass_pole * gravity * pole_length  # Desired energy level


    # Define the desired (target) state for stabilization
    desired_state = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0])


    # Generate a batch of initial conditions for the rollout
    initial_conditions = generate_initial_conditions(batch_size=15, key=key)

    def force_func(t, state):
        """
        Computes the control force at a given time and state.

        Parameters:
        ----------
        t : float
            Current time.
        state : jax.numpy.ndarray
            Current state of the system.

        Returns:
        -------
        float
            Control force to be applied.
        """
        scaling_factor = adaptive_scaling(
            state, mass_cart, mass_pole, pole_length, gravity, E_desired
        )
        raw_force = model(state)
        scaled_force = 10.0 * scaling_factor * raw_force  # Scaling constant to adjust control force
        return scaled_force

    def rollout_cost(initial_state):
        """
        Simulates the system dynamics and computes the total cost for a single rollout.

        Parameters:
        ----------
        initial_state : jax.numpy.ndarray
            The initial state of the system.

        Returns:
        -------
        float
            The total cost for the rollout.
        """
        # Define the ODE term using the provided dynamics function
        ode_term = ODETerm(dynamics_func)
        solver = Tsit5()

        # Define simulation time points
        time_points = jnp.arange(0.0, T, dt)

        # Solve the ODE to simulate the system dynamics
        solution = diffeqsolve(
            ode_term,
            solver,
            t0=0.0,
            t1=T,
            dt0=0.005,
            y0=initial_state,
            args=(env_params, force_func),
            saveat=SaveAt(ts=time_points),
            max_steps=50_000
        )

        # Partial function to compute instantaneous cost with fixed parameters
        partial_instant_cost = partial(
            instant_cost,
            desired_state=desired_state,
            model=model,
            mass_cart=mass_cart,
            mass_pole=mass_pole,
            pole_length=pole_length,
            gravity=gravity,
            E_desired=E_desired,
        )

        # Compute costs at each time step using vectorized mapping
        costs = jax.vmap(partial_instant_cost)(solution.ys)
        total_cost = jnp.sum(costs) * dt  # Integrate cost over time
        return total_cost

    # Compute the average cost over all initial conditions in the batch
    rollout_costs = jax.vmap(rollout_cost)(initial_conditions)
    average_cost = jnp.mean(rollout_costs)
    return average_cost



# ---------------------------------------------------------------------------
# Training Function
# ---------------------------------------------------------------------------

def train_nn_controller(env_params, model, num_iterations=1000, learning_rate=1e-3, T=5.0, dt=0.01):
    """
    Trains the neural network controller by minimizing the swing-up cost.

    Parameters:
    ----------
    env_params : tuple
        Environment parameters (mass_cart, mass_pole, pole_length, gravity).
    model : eqx.Module
        The neural network model (controller) to be trained.
    num_iterations : int, optional
        Number of training iterations. Default is 1000.
    learning_rate : float, optional
        Learning rate for the optimizer. Default is 1e-3.
    T : float, optional
        Time horizon for each rollout. Default is 5.0 seconds.
    dt : float, optional
        Time step size for the simulation. Default is 0.01 seconds.

    Returns:
    -------
    eqx.Module
        The trained neural network model.
    list
        History of training costs recorded every 50 iterations.
    """
    # Define the optimizer with gradient clipping and Adam optimization
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip gradients to a global norm of 1.0
        optax.adam(learning_rate)         # Adam optimizer
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def compute_gradients(model, key):
        """
        Computes the gradients of the cost with respect to the model parameters.

        """

        cost = single_rollout_cost_nn(
            model=model,
            dynamics_func=cartpole_dynamics_nn,
            env_params=env_params,
            T=T,
            dt=dt,
            key=key
        )

        grads = jax.grad(lambda m: cost)(model)

        return cost, grads

    # Initialize random key for reproducibility
    key = jax.random.PRNGKey(0)

    # List to store cost history
    cost_history = []

    # Record the start time for monitoring training duration
    start_time = time.time()

    # Training loop
    for iteration in range(num_iterations):
        # Split the key to get a new subkey
        key, subkey = jax.random.split(key)

        # Compute cost and gradients
        cost, grads = compute_gradients(model, subkey)

        # Update the optimizer state and model parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Log progress every 50 iterations
        if iteration % 50 == 0:
            print(f"Iteration {iteration}: Cost = {cost:.6f} | Elapsed Time: {elapsed_time:.2f} s")
            cost_history.append(cost)

    return model, cost_history


def adaptive_scaling_5states(state, mass_cart, mass_pole, pole_length, gravity, desired_energy):
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