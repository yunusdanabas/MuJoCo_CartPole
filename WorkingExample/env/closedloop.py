# env/closedloop.py

import jax
import jax.numpy as jnp
from diffrax import SaveAt, Tsit5, ODETerm, diffeqsolve

# Import the new closed-loop dynamics from cartpole.py
from env.cartpole import cartpole_dynamics


def simulate_closed_loop(controller, params, t_span, t, initial_state=None):
    """
    Provides a standalone closed-loop simulation for the cart-pole system.

    Args:
        controller: A callable (state, time) -> force. 
                    Example: controller(state, t) returns a scalar force f.
        params: System parameters (mc, mp, l, g).
        t_span: Tuple specifying start and end times (t0, t1).
        t: Array of time points for simulation outputs.
        initial_state: Initial state [x, theta, x_dot, theta_dot].
                       Defaults to zeros if not provided.

    Returns:
        solution: A Diffrax solution object containing the simulated trajectory.
    """
    if initial_state is None:
        initial_state = jnp.zeros(4)

    # Define the closed-loop dynamics that references cartpole_dynamics_cl
    #@jax.jit
    def dynamics(t_step, state, args):
        # args is (params, controller)
        return cartpole_dynamics(t_step, state, args)

    term = ODETerm(dynamics)
    solver = Tsit5()
    t0, t1 = t_span

    solution = diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=0.01,
        y0=initial_state,
        args=(params, controller),
        saveat=SaveAt(ts=t),
        max_steps=10000
    )

    return solution
