"""nn_controller.py
Defines a neural network controller for the cart-pole system.
The controller is a simple multilayer perceptron mapping the
5D state [x, cos(theta), sin(theta), x_dot, theta_dot] to a
scalar force.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from env.closedloop import simulate_closed_loop_nn
from lib.utils import (
    sample_initial_conditions,
    convert_4d_to_5d,
    convert_5d_to_4d,
    compute_energy_nn,
)


class CartPoleNN(eqx.Module):
    """Multi-layer perceptron policy for cart-pole control."""

    layers: list[eqx.Module]
    activations: list = eqx.field(static=True)

    def __init__(self, key: jax.Array, in_dim: int = 5,
                 hidden_dims: tuple[int, ...] = (64, 64), out_dim: int = 1) -> None:
        keys = jax.random.split(key, len(hidden_dims) + 1)
        self.layers = []
        input_size = in_dim
        for i, h in enumerate(hidden_dims):
            self.layers.append(eqx.nn.Linear(input_size, h, key=keys[i]))
            input_size = h
        self.layers.append(eqx.nn.Linear(input_size, out_dim, key=keys[-1]))
        self.activations = [jax.nn.relu for _ in hidden_dims]

    def __call__(self, state: jnp.ndarray, t: Optional[float] = None) -> jnp.ndarray:
        x = state
        for layer, activation in zip(self.layers[:-1], self.activations):
            x = activation(layer(x))
        return self.layers[-1](x)[0]


###############################################################################
#                        Training and Evaluation Helpers                       #
###############################################################################

def _trajectory_loss(
    initial_state: jnp.ndarray,
    controller: eqx.Module,
    params: Tuple[float, float, float, float],
    Q: jnp.ndarray,
    t_span: Tuple[float, float],
    t_eval: jnp.ndarray,
    cost_weights: Tuple[float, float, float],
) -> float:
    """Compute loss for a single rollout."""
    mc, mp, l, g = params
    energy_w, state_w, control_w = cost_weights

    sol = simulate_closed_loop_nn(
        controller=controller,
        params=params,
        t_span=t_span,
        t=t_eval,
        initial_state=initial_state,
    )

    states = sol.ys
    forces = jax.vmap(controller)(states, sol.ts)

    desired_energy = 2 * mp * g * l
    current_energy = compute_energy_nn(states, params)
    energy_loss = jnp.mean((current_energy - desired_energy) ** 2)

    state_cost = jnp.mean(jnp.einsum("ni,ij,nj->n", states, Q, states))
    control_cost = jnp.mean(forces ** 2)

    return energy_w * energy_loss + state_w * state_cost + control_w * control_cost


def train_nn_controller(
    controller: CartPoleNN,
    params_system: Tuple[float, float, float, float],
    Q: jnp.ndarray,
    num_epochs: int,
    batch_size: int,
    t_span: Tuple[float, float],
    t_eval: jnp.ndarray,
    key: jax.Array,
    learning_rate: float = 1e-3,
    grad_clip: float = 1.0,
    cost_weights: Tuple[float, float, float] = (10.0, 1.0, 0.01),
    curriculum_learning: bool = False,
    curriculum_stages: int = 1,
) -> tuple[CartPoleNN, jnp.ndarray]:
    """Train the neural network swing-up controller using a simplified approach."""

    # Use the standard equinox filter approach but be more careful about the training loop
    diff_controller = eqx.filter(controller, eqx.is_inexact_array)
    static_controller = eqx.filter(controller, lambda x: not eqx.is_inexact_array(x))
    
    # Use only SGD to avoid the optax tree structure issues
    loss_history = []

    def _loss_fn(diff_controller, batch_states):
        # Reconstruct the controller
        model = eqx.combine(diff_controller, static_controller)
        
        per_traj = partial(
            _trajectory_loss,
            controller=model,
            params=params_system,
            Q=Q,
            t_span=t_span,
            t_eval=t_eval,
            cost_weights=cost_weights,
        )
        return jnp.mean(jax.vmap(per_traj)(batch_states))

    def sgd_update(model, batch_states, lr):
        """Simple SGD update using eqx functions"""
        loss, grads = eqx.filter_value_and_grad(_loss_fn)(model, batch_states)

        # Apply gradient clipping manually if needed
        if grad_clip > 0:
            grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
            if grad_norm > grad_clip:
                clip_factor = grad_clip / grad_norm
                grads = jax.tree_util.tree_map(lambda g: g * clip_factor, grads)
        
        # Manual SGD update
        # Only update array leaves; non-array leaves get `None`
        updates = jax.tree_util.tree_map(
            lambda g: -lr * g if eqx.is_inexact_array(g) else None,
            grads,
        )
        model = eqx.apply_updates(model, updates)
        
        return model, loss

    for epoch in range(num_epochs):
        key, subkey = jax.random.split(key)
        
        # Curriculum learning: gradually increase difficulty
        if curriculum_learning and curriculum_stages > 1:
            stage = min(epoch // (num_epochs // curriculum_stages), curriculum_stages - 1)
            # Start with smaller ranges and shorter times, gradually increase
            x_range_factor = 0.5 + 0.5 * (stage / (curriculum_stages - 1))
            theta_range_factor = 0.3 + 0.7 * (stage / (curriculum_stages - 1))
            current_x_range = (-2.0 * x_range_factor, 2.0 * x_range_factor)
            current_theta_range = (-jnp.pi * theta_range_factor, jnp.pi * theta_range_factor)
        else:
            current_x_range = (-2.0, 2.0)
            current_theta_range = (-jnp.pi, jnp.pi)
        
        batch_init = sample_initial_conditions(
            batch_size,
            x_range=current_x_range,
            theta_range=current_theta_range,
            key=subkey,
        )
        batch_5d = jax.vmap(convert_4d_to_5d)(batch_init)

        diff_controller, loss = sgd_update(diff_controller, batch_5d, learning_rate)
        loss_history.append(loss.item())

        if epoch % 50 == 0:  # Print less frequently for speed
            stage_info = ""
            if curriculum_learning and curriculum_stages > 1:
                stage = min(epoch // (num_epochs // curriculum_stages), curriculum_stages - 1)
                stage_info = f" | Stage: {stage+1}/{curriculum_stages}"
            print(f"Epoch {epoch:04d} | Loss: {loss:.4f}{stage_info}")

    # Reconstruct the final controller
    final_controller = eqx.combine(diff_controller, static_controller)
    return final_controller, jnp.array(loss_history)


def evaluate_controller(
    controller: CartPoleNN,
    params_system: Tuple[float, float, float, float],
    initial_state: jnp.ndarray,
    t_span: Tuple[float, float],
    t_eval: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate controller from a given initial 4D state."""
    init_5d = convert_4d_to_5d(initial_state)
    sol = simulate_closed_loop_nn(
        controller=controller,
        params=params_system,
        t_span=t_span,
        t=t_eval,
        initial_state=init_5d,
    )
    states_4d = jax.vmap(convert_5d_to_4d)(sol.ys)
    return sol.ts, states_4d


