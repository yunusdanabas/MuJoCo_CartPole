########################################
# cartpole_nn_swingup.py
########################################

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

from diffrax import ODETerm, Tsit5, SaveAt, diffeqsolve


########################################
# 1) Cart-Pole Dynamics (Optimized)
########################################
def cartpole_dynamics(t, state, args):
    params, controller = args
    mc, mp, l, g = params

    x, cos_th, sin_th, x_dot, th_dot = state
    f = controller(state, t)  # External force

    # Mass matrix with regularization to avoid singularity
    M = jnp.array([
        [mc + mp, -mp * l * cos_th],
        [-mp * l * cos_th, mp * l**2]
    ]) + 1e-6 * jnp.eye(2)

    # Coriolis-like term
    C = jnp.array([
        [0, mp * l * sin_th * th_dot],
        [0, 0]
    ])
    # Gravity
    tau_g = jnp.array([0.0, mp * g * l * sin_th])
    B = jnp.array([1.0, 0.0])

    q_dot = jnp.array([x_dot, th_dot])
    rhs = tau_g + B * f - C @ q_dot

    q_ddot = jnp.linalg.solve(M, rhs)
    x_ddot, th_ddot = q_ddot

    # Derivatives of cosθ and sinθ
    cos_th_dot = -sin_th * th_dot
    sin_th_dot = cos_th * th_dot

    return jnp.array([x_dot, cos_th_dot, sin_th_dot, x_ddot, th_ddot])


def simulate_closed_loop(controller_func, params, t_span, t_eval, y0):
    """Integrate dynamics using JAX-compatible operations."""
    term = ODETerm(cartpole_dynamics)
    solver = Tsit5()
    sol = diffeqsolve(
        term,
        solver,
        t0=t_span[0],
        t1=t_span[1],
        dt0=0.01,
        y0=y0,
        saveat=SaveAt(ts=t_eval),
        args=(params, controller_func),
        max_steps=10_000
    )
    return sol.ys


########################################
# 2) Neural Network Controller (Optimized)
########################################
class NNController(eqx.Module):
    """Fixed MLP with manual layer definitions."""
    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear
    layer3: eqx.nn.Linear

    def __init__(self, key, width=128):
        key1, key2, key3 = jax.random.split(key, 3)
        self.layer1 = eqx.nn.Linear(5, width, key=key1)
        self.layer2 = eqx.nn.Linear(width, width, key=key2)
        self.layer3 = eqx.nn.Linear(width, 1, key=key3)

    def __call__(self, state, t=0.0):
        x = jax.nn.tanh(self.layer1(state))
        x = jax.nn.tanh(self.layer2(x))
        return self.layer3(x).squeeze()

########################################
# 3) Energy Calculations (Unchanged)
########################################

def total_energy(state, params):
    mc, mp, l, g = params
    x, cos_th, sin_th, xdot, thdot = state
    K = 0.5*(mc+mp)*xdot**2 - mp*l*cos_th*xdot*thdot + 0.5*mp*l**2*thdot**2
    P = mp*g*l*(1.0 - cos_th)
    return K + P


########################################
# 4) Vectorized Cost Function (Optimized)
########################################
def energy_swingup_cost(states, forces, params, dt):
    mc, mp, l, g = params
    E_desired = mp * g * l  # Precompute desired energy

    # Vectorized energy calculation
    E_vals = jax.vmap(total_energy, in_axes=(0, None))(states, params)
    
    # State penalties
    x = states[:, 0]
    cos_th = states[:, 1]
    sin_th = states[:, 2]
    xdot = states[:, 3]
    thdot = states[:, 4]

    # Weighted terms
    wE, wX, wF = 4.0, 0.7, 0.001
    E_err_sq = (E_vals - E_desired)**2
    state_pen = x**2 + (cos_th - 1.0)**2 + sin_th**2 + xdot**2 + thdot**2
    force_pen = forces**2

    return jnp.sum((wE*E_err_sq + wX*state_pen + wF*force_pen) * dt)


########################################
# 5) Training Loop (Optimized)
########################################
def train_nn_controller(key, nn_model, params, t_eval, init_states, max_epochs=3000, learning_rate=1e-2):
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(nn_model, eqx.is_array))
    dt = t_eval[1] - t_eval[0]
    t0, t1 = t_eval[0], t_eval[-1]

    # Vectorized rollout and cost calculation
    @jax.jit
    def single_rollout_cost(model, y0):
        traj = simulate_closed_loop(lambda s, t: model(s, t), params, (t0, t1), t_eval, y0)
        forces = jax.vmap(model)(traj, t_eval)  # Vectorized force calculation
        return energy_swingup_cost(traj, forces, params, dt)

    @jax.jit
    def loss_fn(model, init_states):
        costs = jax.vmap(lambda ic: single_rollout_cost(model, ic))(init_states)
        return jnp.mean(costs)

    @jax.jit
    def train_step(model, opt_state, init_states):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, init_states)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    loss_history = []
    model = nn_model
    for epoch in range(max_epochs):
        model, opt_state, loss_value = train_step(model, opt_state, init_states)
        loss_history.append(float(loss_value))
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss = {loss_value:.4f}")

    return model, loss_history

########################################
# 6) Main
########################################
def main():
    key = jax.random.PRNGKey(42)

    # Cart-pole params
    mc = 1.0
    mp = 1.0
    l  = 1.0
    g  = 9.81
    params = (mc, mp, l, g)

    # Time horizon
    T_final = 10.0
    t_train = jnp.linspace(0.0, T_final, 100)   # coarse
    t_eval  = jnp.linspace(0.0, T_final, 1000)  # finer

    # Build random initial states
    # Make the range a bit bigger to encourage actual swing
    num_inits = 32
    # Vectorized initial conditions with JAX RNG
    key, *subkeys = jax.random.split(key, 5)
    x0 = jax.random.uniform(subkeys[0], (num_inits,), minval=-0.4, maxval=0.4)
    theta0 = jnp.pi + jax.random.uniform(subkeys[1], (num_inits,), minval=-0.4, maxval=0.4)
    xdot0 = jax.random.uniform(subkeys[2], (num_inits,), minval=-0.2, maxval=0.2)
    thdot0 = jax.random.uniform(subkeys[3], (num_inits,), minval=-0.2, maxval=0.2)
    init_conds = jnp.stack([
        x0,
        jnp.cos(theta0),
        jnp.sin(theta0),
        xdot0,
        thdot0
    ], axis=1).astype(jnp.float32)

    # Create net with 2 hidden layers
    nn_model = NNController(key, width=128)

    # Train
    print("Starting training...")
    max_epochs    = 500
    learning_rate = 1e-3   # slightly larger
    nn_model_trained, loss_history = train_nn_controller(
        key=key,
        nn_model=nn_model,
        params=params,
        t_eval=t_train,
        init_states=init_conds,
        max_epochs=max_epochs,
        learning_rate=learning_rate
    )
    print("Training complete.")

    # Save
    eqx.tree_serialise_leaves("nn_swingup.eqx", nn_model_trained)
    print("Saved trained model to nn_swingup.eqx")

    # Evaluate on one test initial condition
    def controller_func(s, t):
        return nn_model_trained(s, t)

    test_ic = init_conds[0]
    traj = simulate_closed_loop(controller_func, params, (0.0, T_final), t_eval, test_ic)
    forces = jnp.array([controller_func(traj[i], t_eval[i]) for i in range(len(t_eval))])

    dt = t_eval[1] - t_eval[0]
    final_cost = energy_swingup_cost(traj, forces, params, dt)
    print(f"Final test cost = {final_cost:.2f}")

    # Plot training loss
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.title("Training Loss vs. Epoch")
    plt.show()

    # Plot x and theta
    x_arr = traj[:, 0]
    cos_arr = traj[:, 1]
    sin_arr = traj[:, 2]
    xdot_arr = traj[:, 3]
    thdot_arr = traj[:, 4]

    theta_arr = np.arctan2(sin_arr, cos_arr)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(t_eval, x_arr, label="x (cart)")
    plt.xlabel("Time [s]")
    plt.ylabel("x [m]")
    plt.grid(True)
    plt.title("Cart Position")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(t_eval, np.degrees(theta_arr), label="theta (deg)")
    plt.xlabel("Time [s]")
    plt.ylabel("θ [deg]")
    plt.grid(True)
    plt.title("Pendulum Angle")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
