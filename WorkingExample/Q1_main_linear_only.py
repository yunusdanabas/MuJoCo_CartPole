# main_linear_only.py

import jax.numpy as jnp

# Environment and simulation
from env.closedloop import simulate_closed_loop

# Controllers
from controller.linear_controller import train_linear_controller
from controller.lqr_controller import linearize_cartpole, compute_lqr_gain

# Utilities
from lib.utils import (
    plot_cost,
    sample_initial_conditions,
    plot_trajectory_comparison,
    plot_trajectory_comparison2,
    compute_trajectory_cost,
    plot_cost_comparison,
    plot_control_forces_comparison
)

# ---------------------------------------------------------------------------
# 1. Define System Parameters and Time Range
# ---------------------------------------------------------------------------

# Cart-pole system parameters
mass_cart = 1.0     # mc
mass_pole = 1.0     # mp
pole_length = 1.0   # l
gravity = 9.81      # g

params = jnp.array([mass_cart, mass_pole, pole_length, gravity])

# Simulation time settings
t_start = 0.0
t_end = 4.0
num_time_steps = 400
t = jnp.linspace(t_start, t_end, num_time_steps)
t_span = (t_start, t_end)

# ---------------------------------------------------------------------------
# 2. Generate Initial Conditions for Training the Linear Controller
# ---------------------------------------------------------------------------

initial_conditions = sample_initial_conditions(
    num_samples= 20,
    x_range=(-0.1, 0.1),
    theta_range=(-0.2, 0.2),
    xdot_range=(-0.2, 0.2),
    thetadot_range=(-0.2, 0.2)
)

# ---------------------------------------------------------------------------
# 3. Train the Linear Controller
# ---------------------------------------------------------------------------

# Define the state cost matrix Q
Q = jnp.diag(jnp.array([
    100.0,  # x position
    100.0,  # theta angle
    5.0,    # x velocity
    20.0    # theta angular velocity
]))

# Hyperparameters for linear controller optimization
linear_opt_hparams = {
    'lr': 0.0001,  # Learning rate
    'w_init': [
        1.07477045e+01,    # Weight for x
        -1.42533379e-03,   # Weight for cos(theta)
        -9.53456802e+01,   # Weight for sin(theta)
        1.32174416e+01,    # Weight for x_dot
        -2.78421364e+01    # Weight for theta_dot
    ],
    'max_iters': 300,      # Maximum number of iterations
    'tolerance': 1e-6       # Convergence tolerance
}

print("\n=== Training Linear Controller ONLY ===")
linear_weights, linear_cost_history = train_linear_controller(
    params,
    t_span,
    t,
    initial_conditions,
    Q,
    linear_opt_hparams
)

# Plot training cost
plot_cost(
    cost_history=linear_cost_history,
    title="Linear Controller Training Cost (Linear-Only)",
    log_scale=True
)

# ---------------------------------------------------------------------------
# 4. Test the Trained Linear Controller on a Specific Initial Condition
# ---------------------------------------------------------------------------

# Define a test initial condition
initial_condition_test = jnp.array([-3.0, jnp.deg2rad(20), 2.0, -3.0])  # [x, theta, x_dot, theta_dot]

def linear_controller(state, time):
    x, theta, x_dot, theta_dot = state
    control = (
        linear_weights[0] * x +
        linear_weights[1] * jnp.cos(theta) +
        linear_weights[2] * jnp.sin(theta) +
        linear_weights[3] * x_dot +
        linear_weights[4] * theta_dot
    )
    return control

# Simulate closed-loop system with the linear controller
solution_linear = simulate_closed_loop(
    linear_controller,
    params,
    t_span,
    t,
    initial_condition_test
)

states_linear = solution_linear.ys

# ---------------------------------------------------------------------------
# 5. Implementing the LQR Controller
# ---------------------------------------------------------------------------

# Define LQR cost matrices
Q_lqr = Q  # State cost matrix
R_lqr = jnp.array([[0.1]])  # Control effort cost

# Linearize the cart-pole system around the equilibrium
A, B = linearize_cartpole(params)

# Compute the LQR gain matrix K
K = compute_lqr_gain(A, B, Q_lqr, R_lqr)

def lqr_controller(state, time):
    control = -(K @ state)[0]
    return control

# Simulate closed-loop system with the LQR controller
solution_lqr = simulate_closed_loop(
    lqr_controller,
    params,
    t_span,
    t,
    initial_condition_test
)

states_lqr = solution_lqr.ys

# ---------------------------------------------------------------------------
# 6. Compute Trajectory Costs for Comparison
# ---------------------------------------------------------------------------

linear_cost, linear_control_forces = compute_trajectory_cost(
    Q,
    states_linear,
    linear_controller,
    t
)

lqr_cost, lqr_control_forces = compute_trajectory_cost(
    Q,
    states_lqr,
    lqr_controller,
    t
)

print("\nTrained Linear Weights:", linear_weights)
print("Final cost (trained linear):", linear_cost)
print("Final cost (LQR):", lqr_cost)

# ---------------------------------------------------------------------------
# 7. Plot Trajectories for Comparison
# ---------------------------------------------------------------------------

# Prepare list of state trajectories and labels
states_list = [states_linear, states_lqr]
labels = ["Trained Linear Controller", "LQR Controller"]

# Plot trajectory comparisons
plot_trajectory_comparison(
    t,
    states_list,
    labels,
    "Controller Comparison"
)

plot_trajectory_comparison2(
    t,
    states_list,
    labels,
    "Controller Comparison"
)

# Plot cost comparison
plot_cost_comparison(
    linear_cost,
    lqr_cost,
    "Trajectory Cost Comparison"
)

# Plot control forces comparison
plot_control_forces_comparison(
    t,
    linear_control_forces,
    lqr_control_forces,
    "Control Forces Comparison"
)
