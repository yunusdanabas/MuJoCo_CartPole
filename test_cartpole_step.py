#!/usr/bin/env python3
"""Test cart-pole training step by step"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from functools import partial

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from controller.nn_controller import CartPoleNN, _trajectory_loss
from lib.utils import convert_4d_to_5d, sample_initial_conditions

def test_cart_pole_step():
    """Test each component of the cart-pole training"""
    
    key = jax.random.PRNGKey(42)
    controller = CartPoleNN(key=key)
    
    # Parameters exactly as in the main training
    params_system = (1.0, 0.1, 0.5, 9.81)
    Q_matrix = jnp.diag(jnp.array([0.1, 10.0, 10.0, 0.1, 0.1]))
    t_span = (0.0, 10.0)
    t_eval = jnp.linspace(0.0, 10.0, 100)
    cost_weights = (10.0, 1.0, 0.01)
    
    # Extract parameters exactly like in the new approach
    params = eqx.filter(controller, eqx.is_array)
    static = eqx.filter(controller, lambda x: not eqx.is_array(x))
    
    print("Params structure:", jax.tree_util.tree_structure(params))
    print("Static structure:", jax.tree_util.tree_structure(static))
    
    # Test parameter reconstruction
    reconstructed = eqx.combine(params, static)
    print("Reconstruction successful")
    
    # Create sample data
    key, subkey = jax.random.split(key)
    batch_init = sample_initial_conditions(2, (-2.0, 2.0), (-jnp.pi, jnp.pi), key=subkey)
    batch_5d = jax.vmap(convert_4d_to_5d)(batch_init)
    
    print(f"Batch 5d shape: {batch_5d.shape}")
    
    # Test the loss function with explicit parameters
    def _batch_loss(params, batch_states):
        # Reconstruct the controller from parameters and static parts
        controller = eqx.combine(params, static)
        
        per_traj = partial(
            _trajectory_loss,
            controller=controller,
            params=params_system,
            Q=Q_matrix,
            t_span=t_span,
            t_eval=t_eval,
            cost_weights=cost_weights,
        )
        return jnp.mean(jax.vmap(per_traj)(batch_states))
    
    # Test loss computation
    loss = _batch_loss(params, batch_5d)
    print(f"Loss: {loss}, finite: {jnp.isfinite(loss)}")
    
    # Test gradient computation
    loss, grads = jax.value_and_grad(_batch_loss)(params, batch_5d)
    print(f"Gradient loss: {loss}")
    print(f"Grads structure: {jax.tree_util.tree_structure(grads)}")
    
    # Examine gradient leaves
    leaves, tree_def = jax.tree_util.tree_flatten(grads)
    print(f"Number of gradient leaves: {len(leaves)}")
    for i, leaf in enumerate(leaves):
        if isinstance(leaf, jnp.ndarray):
            print(f"Leaf {i}: shape={leaf.shape}, dtype={leaf.dtype}, finite={jnp.isfinite(leaf).all()}")
        else:
            print(f"Leaf {i}: non-array type {type(leaf)} = {leaf}")
    
    # Test optimizer
    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(params)
    
    try:
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        print("Optimizer update successful!")
        new_params = optax.apply_updates(params, updates)
        print("Parameter application successful!")
    except Exception as e:
        print(f"Error in optimizer: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cart_pole_step()
