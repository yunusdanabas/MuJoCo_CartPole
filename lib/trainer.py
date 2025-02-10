# lib/trainer.py

import jax
import jax.numpy as jnp
from diffrax import ODETerm, Tsit5, SaveAt, diffeqsolve
import optax
import equinox as eqx
import time

from env.cartpole import cartpole_dynamics_nn
from lib.utils import sample_initial_conditions


def compute_instant_cost(state, force, params):
    """
    Enhanced cost function:
      - Penalize energy error if you wish (E - E_des).
      - Also penalize angle^2.
      - Keep x near 0.
      - Penalize large force.
    """
    mc, mp, l, g = params
    x, cos_th, sin_th, x_dot, th_dot = state

    # Potential
    E_pot = mp*g*l*(1.0 - cos_th)
    # Kinetic (optionally including cross term)
    E_kin = 0.5*(mc+mp)*(x_dot**2) + 0.5*mp*(l**2)*(th_dot**2)
    # E_kin -= mp*l*cos_th*x_dot*th_dot  # optional cross term if you prefer
    E = E_pot + E_kin

    E_des = 0.0  # upright => zero potential by your definition
    cost_energy = (E - E_des)**2

    # Angle penalty
    theta = jnp.arctan2(sin_th, cos_th)
    cost_theta = theta**2

    # Weighted sum
    alpha = 0.3  # weight for energy error
    beta = 0.1   # weight for x^2
    gamma = 0.001  # weight for force
    delta = 5.0  # weight for angle^2

    return alpha*cost_energy + beta*(x**2) + gamma*(force**2) + delta*cost_theta


def rollout_once(nn_policy, params, init_state, T=8.0, dt=0.01):
    """
    Increase T to 8.0 for more time to swing up.
    """
    if init_state.shape[0] == 4:
        x, theta, x_dot, theta_dot = init_state
        init_state = jnp.array([x, jnp.cos(theta), jnp.sin(theta), x_dot, theta_dot])

    def control_func(t, state):
        return nn_policy(state)

    def dynamics(t, state, args):
        f = control_func(t, state)
        return cartpole_dynamics_nn(t, state, (params, lambda _t, _s: f))

    term = ODETerm(dynamics)
    solver = Tsit5()

    ts = jnp.linspace(0.0, T, int(T/dt) + 1)
    saveat = SaveAt(ts=ts)

    sol = diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=T,
        dt0=dt,
        y0=init_state,
        args=(params, control_func),
        saveat=saveat,
        max_steps=10_000
    )

    states = sol.ys
    times = sol.ts
    forces = jax.vmap(nn_policy)(states)

    # cost
    cost_fn = jax.vmap(lambda s, f: compute_instant_cost(s, f, params))
    instant_costs = cost_fn(states, forces)
    dt_array = times[1:] - times[:-1]
    total_cost = jnp.sum(instant_costs[:-1] * dt_array)
    return total_cost, states, times, forces


def rollout_batch(nn_policy, params, init_states, T=8.0, dt=0.01):
    """
    We'll also keep T=8.0 here.
    """
    costs = []
    for ic in init_states:
        c, _, _, _ = rollout_once(nn_policy, params, ic, T=T, dt=dt)
        costs.append(c)
    return jnp.mean(jnp.array(costs))


def train_nn_controller(
    nn_policy,
    params,
    num_iterations=2000,    # Increase the number of training iterations
    batch_size=16,
    T=8.0,
    dt=0.01,
    lr=1e-3,
    key=jax.random.PRNGKey(0)
):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(nn_policy, eqx.is_array))
    cost_history = []

    @jax.jit
    def loss_fn(model, ics):
        return rollout_batch(model, params, ics, T, dt)

    @jax.jit
    def step(model, opt_state, ics):
        val, grads = jax.value_and_grad(loss_fn)(model, ics)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, val

    current_model = nn_policy

    start_time = time.time()

    print("Check1")
    for i in range(num_iterations):
        subkey, key = jax.random.split(key)
        # Sample initial states near downward if you want to specifically train from pi
        init_states = sample_initial_conditions(
            x_range=(-0.5, 0.5),
            theta_range=(jnp.pi - 0.3, jnp.pi + 0.3),
            xdot_range=(-0.5, 0.5),
            thetadot_range=(-0.5, 0.5),
            key=subkey
        )
        current_time = time.time()

        current_model, opt_state, train_loss = step(current_model, opt_state, init_states)
        cost_history.append(train_loss)

        if i % 10 == 0:
            print(f"Iteration {i}, Cost = {train_loss:.4f}, Time = {(current_time - start_time):.2f}")

    return current_model, cost_history
