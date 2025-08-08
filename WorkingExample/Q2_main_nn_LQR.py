# main_hybrid.py
# Neural Network for swing up and LQR takes over

import jax
import jax.numpy as jnp
import jax.random as jrandom
from diffrax import ODETerm, Tsit5, SaveAt, diffeqsolve

# Equinox for MLP
import equinox as eqx

# Local project imports (adjust paths as needed)
from controller.neuralnetwork_controller import MLP
from controller.lqr_controller import linearize_cartpole, compute_lqr_gain
from env.cartpole import cartpole_dynamics_nn
from env.closedloop import simulate_closed_loop

# Trainer, utilities, plotting
from lib.trainer import train_nn_controller, adaptive_scaling, compute_energy
from lib.utils import (
    plot_trajectories,
    plot_theta_over_time,
    plot_energies,
    plot_trajectory_comparison2,
)


###############################################################################
# 1. HYBRID CONTROLLER DEFINITION
###############################################################################

def make_hybrid_controller(
    nn_model,
    lqr_gain,
    angle_threshold=jnp.deg2rad(15.0),
    angular_vel_threshold=15.0,
    x_threshold=10.0,
    xdot_threshold=5.0
):
    """
    Returns a controller(state, t) -> force that switches from the NN to LQR
    if the system is 'near upright'.

    Args:
        nn_model: The trained MLP for swing-up.
                  Expects input shape [x, sin(theta), cos(theta), x_dot, theta_dot].
        lqr_gain: LQR gain matrix K (shape = [1, 4]).
        angle_threshold, angular_vel_threshold, x_threshold, xdot_threshold:
            Thresholds to define "near upright" region.
    """
    def hybrid_controller(state, t):
        # state is [x, theta, x_dot, theta_dot]
        x, theta, x_dot, theta_dot = state

        # Boolean: are we near upright?
        near_upright = (
            (jnp.abs(theta) < angle_threshold) &
            (jnp.abs(theta_dot) < angular_vel_threshold) &
            (jnp.abs(x) < x_threshold) &
            (jnp.abs(x_dot) < xdot_threshold)
        )

        # Branch 1: use LQR
        def use_lqr():
            force_lqr = -(lqr_gain @ state)[0]
            print("zaa lqr")
            return force_lqr

        # Branch 2: use NN
        def use_nn():
            # NN expects [x, sin(theta), cos(theta), x_dot, theta_dot]
            print("za nn")
            nn_input = jnp.array([
                x,
                jnp.sin(theta),
                jnp.cos(theta),
                x_dot,
                theta_dot
            ])
            return nn_model(nn_input)

        # jax.lax.cond is a JAX-compatible if-else
        force = jax.lax.cond(
            near_upright,
            use_lqr,
            use_nn
        )
        return force

    return hybrid_controller


###############################################################################
# 2. MAIN FUNCTION
###############################################################################
def main():
    """
    Main function that:
      1) Trains an NN for cart-pole swing-up (no saving/loading).
      2) Demonstrates an NN-only simulation for reference.
      3) Sets up an LQR controller.
      4) Defines a hybrid controller that switches from NN to LQR near upright.
      5) Simulates the hybrid approach and plots results.
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
    # B. Prepare Neural Network
    # -----------------------------------------------------------------------
    key = jrandom.PRNGKey(0)
    model = MLP(
        in_size=5,         # [x, sinθ, cosθ, x_dot, θ_dot]
        hidden_sizes=[64, 64],
        out_size=1,
        key=key
    )

    # -----------------------------------------------------------------------
    # C. Train the NN Controller
    # -----------------------------------------------------------------------
    print("\n=== Training the Neural Network Controller ===")
    num_iterations = 200
    learning_rate = 1e-3
    dt = 0.001
    simulation_time = 10.0

    trained_model, cost_history = train_nn_controller(
        env_params=env_params,
        model=model,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        dt=dt,
        T=simulation_time
    )
    print("NN Training Complete!")

    # -----------------------------------------------------------------------
    # D. NN-Only Simulation (Optional Demo)
    # -----------------------------------------------------------------------
    print("\n=== Simulating NN-Only Controller ===")
    # Initial state: 5-element array: [x, cos(theta), sin(theta), x_dot, theta_dot]
    # But our cartpole_dynamics_nn actually expects 4-element: [x, theta, x_dot, theta_dot]
    # and does its own sin/cos inside. We'll match that format by re-checking your code.

    # Actually, from the user code we see that it expects [x, cosθ, sinθ, x_dot, θ_dot].
    # We'll keep that for consistency with your trainer code. Then the "force_function" transforms it.
    initial_state_nn = jnp.array([0.0, -1.0, 0.0, 0.0, 0.0])

    # Define the force using the trained model (similar to your main.py)
    def force_function_nn(t, state):
        # adaptive scaling factor
        E_desired = 2.0 * mass_pole * gravity * pole_length
        scaling_factor = adaptive_scaling(
            state, mass_cart, mass_pole, pole_length, gravity, E_desired
        )
        raw_force = trained_model(state)
        return 15.0 * scaling_factor * raw_force

    # Solve ODE with NN-only
    ode_term_nn = ODETerm(cartpole_dynamics_nn)
    solver_nn = Tsit5()
    t0, t1 = 0.0, 10.0
    n_steps_nn = 1001
    ts_nn = jnp.linspace(t0, t1, n_steps_nn)

    solution_nn = diffeqsolve(
        ode_term_nn,
        solver_nn,
        t0=t0,
        t1=t1,
        dt0=0.001,
        y0=initial_state_nn,
        args=(env_params, force_function_nn),
        saveat=SaveAt(ts=ts_nn),
        max_steps=50_000
    )

    # Plot results (position, velocity, angles, etc.)
    print("Plotting NN-Only Trajectories...")
    plot_trajectories(solution_nn.ts, solution_nn.ys, title_suffix="(NN-Only)")

    # -----------------------------------------------------------------------
    # E. Construct LQR
    # -----------------------------------------------------------------------
    print("\n=== Setting up LQR Controller ===")
    A, B = linearize_cartpole(jnp.array(env_params))
    Q_lqr = jnp.diag(jnp.array([50.0, 50.0, 5.0, 10.0]))
    R_lqr = jnp.array([[0.1]])
    K = compute_lqr_gain(A, B, Q_lqr, R_lqr)  # shape (1,4)

    # -----------------------------------------------------------------------
    # F. Hybrid Simulation (NN + LQR)
    # -----------------------------------------------------------------------
    print("\n=== Simulating Hybrid Controller (NN + LQR) ===")
    # For the hybrid approach, we need an initial state for [x, theta, x_dot, theta_dot].
    # If your NN code uses a 5-element state, be mindful of the difference. We'll pick a 4-element
    # to match the "simulate_closed_loop" approach with the hybrid controller.

    # We'll do a standard cart-pole 4-element state: [x, theta, x_dot, theta_dot].
    initial_state_hybrid = jnp.array([0.0, jnp.pi, 0.0, 0.0])  # start inverted for a challenge

    # Build the hybrid controller
    # We define a function that transforms the 4-element state into the 5-element for the NN,
    # but let's rely on "make_hybrid_controller" to do that inside `use_nn()`.
    hybrid_controller = make_hybrid_controller(
        nn_model=trained_model,
        lqr_gain=K,
        angle_threshold=jnp.deg2rad(15.0),
        angular_vel_threshold=15.0,
        x_threshold=10.0,
        xdot_threshold=5.0
    )

    # Simulate closed-loop with hybrid
    solution_hybrid = simulate_closed_loop(
        controller=hybrid_controller,
        params=jnp.array(env_params),
        t_span=(0.0, 10.0),
        t=jnp.linspace(0.0, 10.0, 1000),
        initial_state=initial_state_hybrid
    )

    states_hybrid = solution_hybrid.ys  # shape (len(t), 4)

    # Plot or analyze the hybrid results
    print("Final State (Hybrid):", states_hybrid[-1, :])
    plot_trajectory_comparison2(
        solution_hybrid.ts,
        [states_hybrid],
        labels=["Hybrid (NN+LQR)"],
        title_prefix="Hybrid Controller"
    )

    print("\n=== All done! ===")


# 3. ENTRY POINT

if __name__ == "__main__":
    main()
