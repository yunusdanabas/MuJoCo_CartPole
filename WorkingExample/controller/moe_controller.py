# controller/moe_controller.py

import jax
import jax.numpy as jnp
from typing import Callable, List

import equinox as eqx
import jax.nn as jnn


class GatingNetwork(eqx.Module):
    """
    A gating network that outputs probabilities for each expert.
    This network determines how to weight each expert's contribution
    based on the current state.
    """
    linear: eqx.nn.Linear
    temperature: float = 1.0  # Controls the sharpness of the softmax

    def __init__(self, in_dim: int, num_experts: int, key: jax.random.PRNGKey, temperature: float = 1.0):
        """
        Initializes the Gating Network.

        Args:
            in_dim (int): Dimension of the input state.
            num_experts (int): Number of experts in the MoE.
            key (jax.random.PRNGKey): PRNG key for initializing weights.
            temperature (float): Temperature parameter for softmax.
        """
        super().__init__()
        self.linear = eqx.nn.Linear(in_features=in_dim, out_features=num_experts, key=key)
        self.temperature = temperature

    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the gating probabilities.

        Args:
            state (jnp.ndarray): Current state of the system.

        Returns:
            jnp.ndarray: Probabilities for each expert.
        """
        logits = self.linear(state) / self.temperature
        probs = jnn.softmax(logits)
        return probs


class DeterministicGating(eqx.Module):
    """
    A deterministic gating network that switches based on state thresholds.
    """
    angle_threshold: float
    angular_vel_threshold: float
    x_threshold: float
    xdot_threshold: float

    def __init__(
        self,
        angle_threshold: float,
        angular_vel_threshold: float,
        x_threshold: float,
        xdot_threshold: float
    ):
        super().__init__()
        self.angle_threshold = angle_threshold
        self.angular_vel_threshold = angular_vel_threshold
        self.x_threshold = x_threshold
        self.xdot_threshold = xdot_threshold

    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Determines expert probabilities based on state thresholds.

        Args:
            state (jnp.ndarray): Current state [x, theta, x_dot, theta_dot].

        Returns:
            jnp.ndarray: Probabilities for each expert.
        """
        x, theta, x_dot, theta_dot = state
        near_upright = (
            (jnp.abs(theta) < self.angle_threshold) &
            (jnp.abs(theta_dot) < self.angular_vel_threshold) &
            (jnp.abs(x) < self.x_threshold) &
            (jnp.abs(x_dot) < self.xdot_threshold)
        )
        # Expert 0: NN, Expert 1: LQR
        probs = jnp.array([
            1.0 - near_upright,  # p0
            near_upright          # p1
        ])
        return probs


class MoEController(eqx.Module):
    """
    Mixture-of-Experts (MoE) Controller that combines multiple expert controllers
    based on gating probabilities.
    """
    gating: Callable[[jnp.ndarray], jnp.ndarray]
    experts: List[Callable[[jnp.ndarray, float], float]]

    def __init__(self, gating: Callable[[jnp.ndarray], jnp.ndarray], experts: List[Callable[[jnp.ndarray, float], float]]):
        """
        Initializes the MoE Controller.

        Args:
            gating (Callable[[jnp.ndarray], jnp.ndarray]): The gating network function.
            experts (List[Callable[[jnp.ndarray, float], float]]): List of expert controller functions.
        """
        super().__init__()
        self.gating = gating
        self.experts = experts

    def __call__(self, state: jnp.ndarray, t: float) -> float:
        """
        Computes the control force by weighting each expert's output.

        Args:
            state (jnp.ndarray): Current state of the system.
            t (float): Current time.

        Returns:
            float: Combined control force.
        """
        probs = self.gating(state)  # Shape: (num_experts,)
        expert_outputs = jnp.array([expert(state, t) for expert in self.experts])  # Shape: (num_experts,)
        force = jnp.dot(probs, expert_outputs)
        return force

