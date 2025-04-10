# controller/linear_controller.py
# Description: This module implements a linear controller for a cart-pole system.
# It computes the control force as a linear function of the current state and
# optimizes the controller weights using gradient descent. The cost function is
# defined as the sum of the state cost and control effort over a trajectory.

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
      f = w0*x + w1*cos(theta) + w2*sin(theta) + w3*x_dot + w4*theta_dot
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
    Cost definition: 
      J(w) = ∑ [ x_k^T Q x_k + f_k^2 ] * dt
    """
    def controller(state, time):
        return linear_control(state, w)

    solution = simulate_closed_loop(controller, params, t_span, t, initial_state)
    states = solution.ys  # shape (len(t), 4)
    dt = t[1] - t[0]

    def cost_step(state):
        cost_state = state @ (Q @ state)
        f_val = linear_control(state, w)
        cost_control = f_val**2
        return cost_state + cost_control

    cost_trajectory = jax.vmap(cost_step)(states)
    cost_total = jnp.sum(cost_trajectory) * dt
    return cost_total, jax.vmap(lambda s: linear_control(s, w))(states)

def train_linear_controller(params, t_span, t, initial_conditions, Q, opt_hparams):
    """
    Minimize mean( J(w) ) over multiple initial conditions.
    """
    lr = opt_hparams.get('lr', 1e-3)
    w_init = jnp.array(opt_hparams.get('w_init', [0.0, 0.0, 0.0, 0.0, 0.0]))
    max_iters = opt_hparams.get('max_iters', 2000)
    tolerance = opt_hparams.get('tolerance', 1e-6)

    @jax.jit
    def batched_cost(w):
        def single_ic_cost(ic):
            c, _ = compute_trajectory_cost(w, params, t_span, t, ic, Q)
            return c
        costs = jax.vmap(single_ic_cost)(initial_conditions)
        return jnp.mean(costs)

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

        if i % 100 == 0:
            print(f"Iteration {i}, Cost: {cost_val:.6f}, Weights: {w}")

        if cost_val < tolerance:
            print(f"Converged at iteration {i}")
            break

    return w, cost_history
