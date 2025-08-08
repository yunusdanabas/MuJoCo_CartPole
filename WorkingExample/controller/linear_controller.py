# controller/linear_controller.py

import jax
import jax.numpy as jnp
import optax
from functools import partial

from env.closedloop import simulate_closed_loop

################################################################################
#                                Linear Controller                             #
################################################################################

@jax.jit
def linear_control(state, w):
    """
    Computes the control force f as a linear function of the current state:
      f(x, theta, x_dot, theta_dot) = w1*x + w2*cos(theta) + w3*sin(theta) + w4*x_dot + w5*theta_dot

    Args:
        state: [x, theta, x_dot, theta_dot]
        w: jnp.array of length 5 (linear controller weights)

    Returns:
        Scalar control force f
    """
    x, theta, x_dot, theta_dot = state
    f = (w[0] * x
         + w[1] * jnp.cos(theta)
         + w[2] * jnp.sin(theta)
         + w[3] * x_dot
         + w[4] * theta_dot)
    return f

@partial(jax.jit, static_argnames=['t_span'])
def compute_trajectory_cost(w, params, t_span, t, initial_state, Q):
    """
    Computes the trajectory cost for a single initial condition using the linear controller w.

    Cost definition (discrete approximation):
      J(w) = sum_{k} [ (x_k^T Q x_k) + (f_k^2 ) ] * dt
    where x_k is the state at time step k, f_k is the computed force, and dt is the timestep.

    Args:
        w: jnp.array, linear controller weights (length 5).
        params: System parameters (mc, mp, l, g).
        t_span: (t0, t1) tuple for simulation time range.
        t: Array of time points for the simulation.
        initial_state: Initial state [x, theta, x_dot, theta_dot].
        Q: A 4x4 matrix penalizing state deviations in x, theta, x_dot, theta_dot.

    Returns:
        cost_total: Scalar cost of the trajectory for this initial condition.
    """
    # Controller wrapper to interface with simulate_closed_loop
    def controller(state, time):
        return linear_control(state, w)

    solution = simulate_closed_loop(controller, params, t_span, t, initial_state)
    states = solution.ys  # shape (len(t), 4)
    dt = t[1] - t[0]      # uniform timestep assumption

    # Define a function that computes cost per state sample
    def cost_step(state):
        # Quadratic state cost x^T Q x
        cost_state = state @ (Q @ state)
        # Control cost f^2
        f_val = linear_control(state, w)
        cost_control = f_val**2
        return cost_state + cost_control

    # Vectorize cost computation across all timesteps
    cost_trajectory = jax.vmap(cost_step)(states)
    cost_total = jnp.sum(cost_trajectory) * dt
    return cost_total

def train_linear_controller(params, t_span, t, initial_conditions, Q, opt_hyperparams):
    """
    Trains the linear controller weights w by minimizing the trajectory cost over one or more
    initial conditions. The cost function is:
       J_total(w) = mean( J(w) across all initial_conditions )

    Args:
        params: System parameters (mc, mp, l, g).
        t_span: (t0, t1) tuple for simulation time range.
        t: Array of time points for the simulation.
        initial_conditions: jnp.array of shape (N, 4) for N different initial states.
        Q: 4x4 matrix weighting state deviations (x, theta, x_dot, theta_dot).
        opt_hyperparams: Dictionary of optimization hyperparameters.
            - 'lr': Learning rate for the optimizer.
            - 'w_init': Initial guess for the weight vector w (length 5).
            - 'max_iters': Maximum training iterations.
            - 'tolerance': Convergence tolerance.

    Returns:
        w: Optimized linear controller weights.
        cost_history: List of cost values over the training iterations.
    """
    lr = opt_hyperparams.get('lr', 1e-3)
    w_init = jnp.array(opt_hyperparams.get('w_init', [0.0, 0.0, 0.0, 0.0, 0.0]))
    max_iters = opt_hyperparams.get('max_iters', 2_000)
    tolerance = opt_hyperparams.get('tolerance', 1e-6)

    # Define batched cost over multiple initial conditions
    @jax.jit
    def batched_cost(w):
        def single_ic_cost(ic):
            return compute_trajectory_cost(w, params, t_span, t, ic, Q)
        costs = jax.vmap(single_ic_cost)(initial_conditions)
        return jnp.mean(costs)

    # Setup optimizer
    optimizer = optax.adam(lr)
    w = w_init
    opt_state = optimizer.init(w)

    cost_history = []
    value_and_grad = jax.value_and_grad(batched_cost)

    for i in range(max_iters):
        cost_val, grads = value_and_grad(w)
        cost_history.append(cost_val)

        updates, opt_state = optimizer.update(grads, opt_state, w)
        w = optax.apply_updates(w, updates)

        # Print progress periodically
        if i % 200 == 0:
            print(f"Iteration {i}, Cost: {cost_val:.6f}, Weights: {w}")

        # Check for convergence
        if cost_val < tolerance:
            print(f"Converged at iteration {i}")
            break

    return w, cost_history
