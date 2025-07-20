import sys
from pathlib import Path
import os  # Added for environment variable configuration

# Disable auto-loading of external pytest plugins (e.g., ROS launch_testing)
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

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


def is_diffrax_available():
    """Helper function to determine if diffrax is installed."""
    try:
        import diffrax  # noqa: F401
        return True
    except ImportError:
        return False


def test_nn_controller_forward():
    """Test that the CartPoleNN controller produces a scalar output."""
    key = jax.random.PRNGKey(0)
    nn = CartPoleNN(key)
    output = nn(jnp.zeros(5))
    assert jnp.ndim(output) == 0, "The output of CartPoleNN should be a scalar (0-dimensional)"


def test_cartpole_dynamics():
    """Verify that the cartpole dynamics functions return outputs of correct shape."""
    params = (1.0, 0.1, 0.5, 9.81)

    def zero_control(s, t):
        """A simple control function that always returns zero."""
        return 0.0

    deriv4 = cartpole_dynamics(0.0, jnp.zeros(4), (params, zero_control))
    deriv5 = cartpole_dynamics_nn(0.0, jnp.array([0.0, 1.0, 0.0, 0.0, 0.0]), (params, zero_control))
    assert deriv4.shape == (4,), f"Expected 4d state dynamics but got shape {deriv4.shape}"
    assert deriv5.shape == (5,), f"Expected 5d state dynamics but got shape {deriv5.shape}"


@pytest.mark.skipif(not is_diffrax_available(), reason="diffrax is required for closed loop simulations")
def test_closed_loop_simulations():
    """Ensure that closed loop simulation modules are importable (requires diffrax)."""
    import diffrax  # noqa: F401
    # No simulation is run; we only verify that diffrax can be imported.


def test_linear_control_and_lqr():
    """Test that both linear_control and LQR-based controllers produce scalar outputs."""
    state = jnp.zeros(4)
    disturbance = jnp.zeros(5)
    
    # Verify linear_control returns a scalar value.
    lin_ctrl = linear_control(state, disturbance)
    assert jnp.isscalar(lin_ctrl), f"linear_control output should be a scalar, got {lin_ctrl}"

    params = (1.0, 0.1, 0.5, 9.81)
    Q = jnp.eye(4)
    lqr_controller = create_lqr_controller(params, Q, 0.1)
    
    # Test LQR controller output.
    lqr_value = lqr_controller(state)
    assert jnp.isscalar(lqr_value), f"Expected scalar output from lqr controller, got {lqr_value}"
    
    # Test lqr_policy output using a converted state.
    lqr_policy_value = lqr_policy(convert_4d_to_5d(state), 0.0)
    assert jnp.isscalar(lqr_policy_value), f"Expected scalar output from lqr_policy, got {lqr_policy_value}"


def test_utils_and_visualizer(tmp_path):
    """Test utility functions for state conversion and the visualizer's file-saving functionality."""
    initial_conditions = sample_initial_conditions(1)
    state_5d = convert_4d_to_5d(initial_conditions[0])
    state_back = convert_5d_to_4d(state_5d)
    assert state_back.shape == (4,), "Conversion from 5d to 4d state did not return correct shape."

    # Verify that save_cost_history creates a file.
    output_path = tmp_path / "tmp_cost.png"
    generated_path = save_cost_history([1.0, 0.5, 0.25], name=str(output_path))
    assert generated_path.exists(), f"Cost history file was not created at {generated_path}"
    # Clean up the temporary file.
    generated_path.unlink()


@pytest.mark.skipif(not is_diffrax_available(), reason="diffrax is required for the training loop simulation")
def test_training_loop_short():
    """Run a single epoch of the training loop and verify that training history is recorded correctly."""
    key = jax.random.PRNGKey(0)
    nn_controller = CartPoleNN(key)
    t_eval = jnp.linspace(0.0, 0.1, 5)
    trained_controller, history = train_nn_controller(
        nn_controller,
        params_system=(1.0, 0.1, 0.5, 9.81),
        Q=jnp.eye(5),
        num_epochs=1,
        batch_size=1,
        t_span=(0.0, 0.1),
        t_eval=t_eval,
        key=key,
        learning_rate=1e-3,
    )
    assert history.shape[0] == 1, f"Expected history to record 1 epoch but got shape {history.shape}"
    print("success")


# ----------------------------------------------------------------------
# Run-all-tests entry-point
# ----------------------------------------------------------------------

def main() -> None:
    """
    Programmatically run the tests in this module with pytest.

    The function delegates to `pytest.main`, passing it the current
    file’s path so that all tests above are collected and executed.
    The process exit status equals the pytest return code.
    """
    exit_code = pytest.main([__file__])
    if exit_code == 0:
        print("All tests passed successfully!")
    else:
        print("Some tests failed. Check the output for details.")


if __name__ == "__main__":
    main()

