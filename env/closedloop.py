"""
Closed-loop ODE wrappers for the cart-pole.

Key features:
* Works with either 4- or 5-state encodings
* Single-trajectory or batched roll-outs
* JIT-compiled once; no Python in the step loop
* Solver / tolerance kwargs surfaced for easy tuning
"""

from __future__ import annotations
from functools import partial
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from diffrax import Tsit5, ODETerm, SaveAt, diffeqsolve

from .cartpole import CartPoleParams, dynamics

# -----------------------------------------------------------------------------#
# Internal helpers                                                              #
# -----------------------------------------------------------------------------#

def _make_ode_term(params: CartPoleParams, controller):
    """Create an ODETerm whose rhs is JIT-compiled once."""
    @partial(jax.jit, static_argnums=(2,))
    def rhs(t, y, args):
        p, ctrl = args
        return dynamics(y, t, params=p, controller=ctrl)
    
    return ODETerm(rhs), (params, controller)


def _solve_ode(term, rhs_args, t_span, ts, y0, dt0, max_steps, rtol, atol):
    """Thin wrapper around diffeqsolve with sane defaults."""
    return diffeqsolve(
        term,
        Tsit5(),
        t0=t_span[0],
        t1=t_span[1],
        dt0=dt0,
        y0=y0,
        args=rhs_args,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        saveat=SaveAt(ts=ts),
    )

# -----------------------------------------------------------------------------#
# Public API                                                                    #
# -----------------------------------------------------------------------------#

def simulate(controller: Callable[[jnp.ndarray, float], float],
             params: CartPoleParams,
             t_span: tuple[float, float],
             ts: Sequence[float],
             initial_state: jnp.ndarray,
             *,
             dt0: float = 1e-2,
             max_steps: int = 10_000,
             rtol: float = 1e-6,
             atol: float = 1e-8):
    """
    Simulate **one** cart-pole trajectory.

    Parameters
    ----------
    controller : Callable
        Pure-JAX callable `u = controller(state, t)`
    params : CartPoleParams
        Physical parameters
    t_span : tuple
        (t0, t1) in seconds
    ts : Sequence
        1-D array of sampling times (must lie inside `t_span`)
    initial_state : array
        Shape (state_dim,) array (either 4 or 5 dims)
    """
    term, rhs_args = _make_ode_term(params, controller)
    return _solve_ode(term, rhs_args, t_span, ts, initial_state,
                      dt0, max_steps, rtol, atol)


def simulate_batch(controller: Callable[[jnp.ndarray, float], float],
                   params: CartPoleParams,
                   t_span: tuple[float, float],
                   ts: Sequence[float],
                   initial_states: jnp.ndarray,
                   **kwargs):
    """
    Vectorised rollout of *N* independent environments in parallel.

    Parameters
    ----------
    initial_states : array
        Array with shape `(N, state_dim)`
    """
    # Compile once, then vmap over the batch axis
    single_sim = partial(simulate, controller, params, t_span, ts, **kwargs)
    return jax.vmap(single_sim)(initial_states)


# -----------------------------------------------------------------------------#
# Legacy compatibility                                                          #
# -----------------------------------------------------------------------------#

def simulate_closed_loop(controller, params, t_span, t, initial_state=None):
    """Original 4-state simulation (for LQR comparison)."""
    if initial_state is None:
        initial_state = jnp.zeros(4)
    
    # Convert old parameter format if needed
    if isinstance(params, (tuple, list)):
        mc, mp, l, g = params
        params = CartPoleParams(mc=mc, mp=mp, l=l, g=g)
    
    return simulate(controller, params, t_span, t, initial_state)


def simulate_closed_loop_nn(controller, params, t_span, t, initial_state=None):
    """Enhanced NN-compatible simulation with 5-element state."""
    if initial_state is None:
        initial_state = jnp.array([0.0, -1.0, 0.0, 0.0, 0.0])  # downward position
    
    # Convert old parameter format if needed
    if isinstance(params, (tuple, list)):
        mc, mp, l, g = params
        params = CartPoleParams(mc=mc, mp=mp, l=l, g=g)
    
    return simulate(controller, params, t_span, t, initial_state)


# -----------------------------------------------------------------------------#
# Quick sanity check                                                            #
# -----------------------------------------------------------------------------#

if __name__ == "__main__":
    # Zero force controller
    null_ctrl = lambda s, t: 0.0

    p = CartPoleParams()
    t_grid = jnp.linspace(0.0, 2.0, 201)

    # Single run
    sol = simulate(null_ctrl, p, (0.0, 2.0), t_grid,
                   initial_state=jnp.array([0.0, jnp.pi - 0.01, 0.0, 0.0]))
    print("Single-traj OK, shape:", sol.ys.shape)

    # Batched run
    init = jnp.stack([jnp.array([0., jnp.pi,  0., 0.]),
                      jnp.array([0., 0.,      0., 0.])])
    sol_b = simulate_batch(null_ctrl, p, (0.0, 2.0), t_grid, init)
    print("Batch-traj OK, shape:", sol_b.ys.shape)
