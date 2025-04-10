# env/closedloop.py

import jax.numpy as jnp
from diffrax import SaveAt, Tsit5, ODETerm, diffeqsolve
from env.cartpole import cartpole_dynamics, cartpole_dynamics_nn
from lib.utils import convert_4d_to_5d, convert_5d_to_4d
import jax


def simulate_closed_loop(controller, params, t_span, t, initial_state=None):
    """Original 4-state simulation (for LQR comparison)"""
    if initial_state is None:
        initial_state = jnp.zeros(4)

    def dynamics(t_step, state, args):
        return cartpole_dynamics(t_step, state, args)

    term = ODETerm(dynamics)
    solution = diffeqsolve(
        term,
        Tsit5(),
        t0=t_span[0],
        t1=t_span[1],
        dt0=0.01,
        y0=initial_state,
        args=(params, controller),
        saveat=SaveAt(ts=t),
        max_steps=10000
    )
    return solution

def simulate_closed_loop_nn(controller, params, t_span, t, initial_state=None):
    """Enhanced NN-compatible simulation with 5-element state"""
    # Default to downward position if no initial state provided
    if initial_state is None:
        initial_state = jnp.array([0.0, -1.0, 0.0, 0.0, 0.0])  # cos(π) = -1, sin(π) = 0

    # State conversion wrapper for JAX compatibility
    @jax.jit
    def dynamics(t_step, state, args):
        return cartpole_dynamics_nn(t_step, state, args)

    # Solve the ODE
    term = ODETerm(dynamics)
    solver = Tsit5()
    
    solution = diffeqsolve(
        term,
        solver,
        t0=t_span[0],
        t1=t_span[1],
        dt0=0.01,
        y0=initial_state,
        args=(params, controller),
        saveat=SaveAt(ts=t),
        max_steps=10_000,
        throw=False  # Prevent crashes from unstable simulations
    )
    
    # Post-process results
    return jax.tree_map(
        lambda x: x.astype(jnp.float32) if isinstance(x, jnp.ndarray) else x,
        solution
    )



def simulate_hybrid(controller_dict, params, t_span, t, initial_state):
    """
    Handles controller switching during simulation
    controller_dict: {
        'nn': neural network controller,
        'lqr': LQR controller,
        'threshold': angle threshold for switching (radians)
    }
    """
    def dynamics(t_step, state, args):
        params, controllers = args
        state_4d = convert_5d_to_4d(state)
        
        # Check if we should switch to LQR
        theta = state_4d[1] % (2*jnp.pi)
        near_upright = jnp.abs(theta) < controllers['threshold']
        
        force = jax.lax.cond(
            near_upright,
            lambda: controllers['lqr'](state_4d, t_step),
            lambda: controllers['nn'](state, t_step)
        )
        
        return cartpole_dynamics_nn(t_step, state, (params, lambda s, t: force))

    term = ODETerm(dynamics)
    solution = diffeqsolve(
        term,
        Tsit5(),
        t0=t_span[0],
        t1=t_span[1],
        dt0=0.01,
        y0=convert_4d_to_5d(initial_state),
        args=(params, controller_dict),
        saveat=SaveAt(ts=t),
        max_steps=10_000
    )
    return solution