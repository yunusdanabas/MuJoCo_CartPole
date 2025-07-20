import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from controller.nn_controller import CartPoleNN
from controller.linear_controller import linear_control
from controller.lqr_controller import create_lqr_controller, lqr_policy
from env.cartpole import cartpole_dynamics, cartpole_dynamics_nn
from env.closedloop import simulate_closed_loop, simulate_closed_loop_nn
from lib.utils import sample_initial_conditions, convert_4d_to_5d, convert_5d_to_4d
from lib.visualizer import save_cost_history
from lib.trainer import train_nn_controller


def test_nn_controller_forward():
    key = jax.random.PRNGKey(0)
    nn = CartPoleNN(key)
    output = nn(jnp.zeros(5))
    assert jnp.ndim(output) == 0


def test_cartpole_dynamics():
    params = (1.0, 0.1, 0.5, 9.81)
    def ctrl(s, t):
        return 0.0
    deriv4 = cartpole_dynamics(0.0, jnp.zeros(4), (params, ctrl))
    deriv5 = cartpole_dynamics_nn(0.0, jnp.array([0., 1., 0., 0., 0.]), (params, ctrl))
    assert deriv4.shape == (4,)
    assert deriv5.shape == (5,)


def test_closed_loop_simulations():
    """Ensure solvers can be imported without running full simulations."""
    pytest.importorskip("diffrax")


def test_linear_control_and_lqr():
    state = jnp.zeros(4)
    w = jnp.zeros(5)
    assert linear_control(state, w).shape == ()

    params = (1.0, 0.1, 0.5, 9.81)
    Q = jnp.eye(4)
    controller = create_lqr_controller(params, Q, 0.1)
    assert isinstance(controller(state), float)
    assert isinstance(lqr_policy(convert_4d_to_5d(state), 0.0), float)


def test_utils_and_visualizer(tmp_path):
    ic = sample_initial_conditions(1)
    s5 = convert_4d_to_5d(ic[0])
    assert convert_5d_to_4d(s5).shape == (4,)

    p = save_cost_history([1.0, 0.5, 0.25], name='tmp_cost.png')
    assert p.exists()
    p.unlink()


def test_training_loop_short():
    """Run a single epoch of the training loop to ensure it executes."""
    pytest.importorskip("diffrax")
    key = jax.random.PRNGKey(0)
    controller = CartPoleNN(key)
    t_eval = jnp.linspace(0.0, 0.1, 5)
    trained, history = train_nn_controller(
        controller,
        params_system=(1.0, 0.1, 0.5, 9.81),
        Q=jnp.eye(5),
        num_epochs=1,
        batch_size=1,
        t_span=(0.0, 0.1),
        t_eval=t_eval,
        key=key,
        learning_rate=1e-3,
    )
    assert history.shape[0] == 1
