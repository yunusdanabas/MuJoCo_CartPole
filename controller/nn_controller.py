"""nn_controller.py
Neural-network controller (MLP) built with Equinox.

Accepts either a ready-trained model or initialises a fresh,
random MLP of dimensions you specify.
Works on 5-state by design; will up-cast 4-state.
"""

from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from controller.base import Controller
from env.cartpole import CartPoleParams
from env.helpers import four_to_five, five_to_four
from lib.utils import sample_initial_conditions, convert_4d_to_5d, convert_5d_to_4d
from env.closedloop import simulate 


# ------------------------------------------------------------------- helpers
def _build_mlp(in_dim, hidden_dims=(64, 64), *, key):
    """Build MLP with tanh activations."""
    layers = []
    hkey = key
    dims = (in_dim, *hidden_dims, 1)
    for i, (m, n) in enumerate(zip(dims[:-1], dims[1:])):
        hkey, sub = jax.random.split(hkey)
        layers.append(eqx.nn.Linear(m, n, key=sub))
        if i < len(dims) - 2:
            layers.append(jax.nn.tanh)
    return eqx.nn.Sequential(layers)


# ------------------------------------------------------------------ class
@dataclass(frozen=True)
class NNController(Controller):
    net: eqx.Module = None

    @classmethod
    def init(cls, *, hidden_dims=(64, 64), key=jax.random.PRNGKey(0)):
        """Initialize with random weights."""
        net = _build_mlp(5, hidden_dims, key=key)
        return cls(net=net)

    def _force(self, state, _t):
        s5 = jax.lax.cond(
            state.shape[-1] == 4,
            four_to_five,
            lambda x: x,
            state
        )
        return jnp.squeeze(self.net(s5))


# Legacy CartPoleNN class for backward compatibility
class CartPoleNN(eqx.Module):
    """Legacy MLP policy for cart-pole control."""
    
    layers: list
    activations: list = eqx.field(static=True)

    def __init__(self, key: jax.Array, in_dim: int = 5,
                 hidden_dims: tuple = (64, 64), out_dim: int = 1):
        keys = jax.random.split(key, len(hidden_dims) + 1)
        self.layers = []
        input_size = in_dim
        for i, h in enumerate(hidden_dims):
            self.layers.append(eqx.nn.Linear(input_size, h, key=keys[i]))
            input_size = h
        self.layers.append(eqx.nn.Linear(input_size, out_dim, key=keys[-1]))
        self.activations = [jax.nn.relu for _ in hidden_dims]

    def __call__(self, state, t=None):
        x = state
        for layer, activation in zip(self.layers[:-1], self.activations):
            x = activation(layer(x))
        return self.layers[-1](x)[0]


# Legacy training functions
def _trajectory_loss(initial_state, controller, params, Q, t_span, t_eval, cost_weights):
    """Legacy trajectory loss computation."""
    from lib.utils import compute_energy_nn
    
    mc, mp, l, g = params
    energy_w, state_w, control_w = cost_weights

    sol = simulate(
        controller=controller,
        params=params,
        ts=t_eval,
        y0=initial_state,
    )

    states = sol.ys
    forces = jax.vmap(controller)(states, sol.ts)

    desired_energy = 2 * mp * g * l
    current_energy = compute_energy_nn(states, params)
    energy_loss = jnp.mean((current_energy - desired_energy) ** 2)

    state_cost = jnp.mean(jnp.einsum("ni,ij,nj->n", states, Q, states))
    control_cost = jnp.mean(forces ** 2)

    return energy_w * energy_loss + state_w * state_cost + control_w * control_cost


def train_nn_controller(controller, params_system, Q, num_epochs, batch_size, 
                       t_span, t_eval, key, **kwargs):
    """Legacy training function for backward compatibility."""
    
    
    # Extract parameters with defaults
    learning_rate = kwargs.get('learning_rate', 1e-3)
    cost_weights = kwargs.get('cost_weights', (10.0, 1.0, 0.01))
    curriculum_learning = kwargs.get('curriculum_learning', False)
    curriculum_stages = kwargs.get('curriculum_stages', 1)
    
    # Filter trainable vs static
    diff_controller = eqx.filter(controller, eqx.is_inexact_array)
    static_controller = eqx.filter(controller, lambda x: not eqx.is_inexact_array(x))
    
    # Use only SGD to avoid the optax tree structure issues
    loss_history = []
    grad_clip = kwargs.get('grad_clip', 0.0)

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
    sol = simulate(  # Changed from simulate_closed_loop_nn
        controller=controller,
        params=params_system,
        t_span=t_span,
        ts=t_eval,  # Changed parameter name
        y0=init_5d,  # Changed parameter name
    )
    states_4d = jax.vmap(convert_5d_to_4d)(sol.ys)
    return sol.ts, states_4d


