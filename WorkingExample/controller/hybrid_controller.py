# controller/hybrid_controller.py

import jax
import jax.numpy as jnp

def make_hybrid_controller(
    nn_model,
    lqr_gain,
    angle_threshold=jnp.deg2rad(10.0),
    angular_vel_threshold=1.5,
    x_threshold=0.5,
    xdot_threshold=1.0
):
    """
    Returns a function `controller(state, t) -> force` that switches 
    from the neural network to LQR based on thresholds.

    Args:
        nn_model: Trained neural network (equinox MLP or similar).
        lqr_gain: Gain matrix K for LQR, shape (1, 4).
        angle_threshold: Threshold (in radians) to define 'near upright'.
        angular_vel_threshold: Threshold for angular velocity (rad/s).
        x_threshold: Threshold for cart position (m).
        xdot_threshold: Threshold for cart velocity (m/s).
    """

    def hybrid_controller(state, t):
        """
        Decide if we're close enough to upright. If yes, use LQR. Otherwise, use NN.
        """
        # State is assumed to be [x, theta, x_dot, theta_dot]
        x, theta, x_dot, theta_dot = state

        # Condition to check if we're 'close enough' to upright
        near_upright = (
            (jnp.abs(theta) < angle_threshold) &
            (jnp.abs(theta_dot) < angular_vel_threshold) &
            (jnp.abs(x) < x_threshold) &
            (jnp.abs(x_dot) < xdot_threshold)
        )

        # If near upright, switch to LQR
        if near_upright:
            # LQR control: f = -K @ [x, theta, x_dot, theta_dot]
            f_lqr = -(lqr_gain @ state)[0]
            return f_lqr
        else:
            # Otherwise, use neural network
            # If your NN expects [x, sin(theta), cos(theta), x_dot, theta_dot],
            # we need to convert the input accordingly:
            nn_input = jnp.array([
                x,
                jnp.sin(theta),
                jnp.cos(theta),
                x_dot,
                theta_dot
            ])
            raw_force = nn_model(nn_input)

            # Optional: scale the NN output if you did that during training
            # e.g., force = scaling_factor * raw_force
            force = raw_force  

            return force

    return hybrid_controller
