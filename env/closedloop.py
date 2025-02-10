# closedloop.py

import jax.numpy as jnp
from diffrax import SaveAt, Tsit5, ODETerm, diffeqsolve

from env.cartpole import cartpole_dynamics

def simulate_closed_loop(controller, params, t_span, t, initial_state=None):
    """
    Provide a closed-loop simulation for the cart-pole system using diffrax.
    """
    if initial_state is None:
        initial_state = jnp.zeros(4)

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
