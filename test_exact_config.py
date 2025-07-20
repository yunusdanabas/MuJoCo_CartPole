#!/usr/bin/env python3
"""Test with exact same config as main script"""

import os
import yaml
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

def test_exact_config():
    """Test with the exact configuration from config.yaml"""
    
    # Load configuration exactly like the main script
    CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.yaml")
    with open(CONFIG_PATH, "r") as f:
        _CONFIG = yaml.safe_load(f) or {}

    nn_cfg = _CONFIG.get("nn_training", {})
    PARAMS_SYSTEM = tuple(nn_cfg.get("params_system", [1.0, 0.1, 0.5, 9.81]))
    Q_MATRIX = jnp.diag(jnp.array(nn_cfg.get("q_weights", [0.1, 10.0, 10.0, 0.1, 0.1])))
    _span = nn_cfg.get("t_span", [0.0, 10.0])
    _eval_pts = nn_cfg.get("t_eval_points", 100)
    TRAIN_CONFIG = {
        'batch_size': nn_cfg.get('batch_size', 32),
        't_span': tuple(_span),
        't_eval': jnp.linspace(_span[0], _span[1], _eval_pts),
        'learning_rate': nn_cfg.get('learning_rate', 3e-4),
        'grad_clip': nn_cfg.get('grad_clip', 1.0),
        'cost_weights': tuple(nn_cfg.get('cost_weights', [10.0, 1.0, 0.01]))
    }
    
    print(f"Using batch size: {TRAIN_CONFIG['batch_size']}")
    print(f"Using t_eval shape: {TRAIN_CONFIG['t_eval'].shape}")
    
    # Initialize exactly like the main script
    key = jax.random.PRNGKey(42)
    controller = CartPoleNN(key=key)
    
    # Extract parameters exactly like in the new training function
    params = eqx.filter(controller, eqx.is_array)
    static = eqx.filter(controller, lambda x: not eqx.is_array(x))
    
    optimizer = optax.sgd(TRAIN_CONFIG['learning_rate'])
    opt_state = optimizer.init(params)
    
    def _batch_loss(params, batch_states):
        # Reconstruct the controller from parameters and static parts
        controller = eqx.combine(params, static)
        
        per_traj = partial(
            _trajectory_loss,
            controller=controller,
            params=PARAMS_SYSTEM,
            Q=Q_MATRIX,
            t_span=TRAIN_CONFIG['t_span'],
            t_eval=TRAIN_CONFIG['t_eval'],
            cost_weights=TRAIN_CONFIG['cost_weights'],
        )
        return jnp.mean(jax.vmap(per_traj)(batch_states))
    
    # Create the exact same batch as the main training
    key, subkey = jax.random.split(key)
    batch_init = sample_initial_conditions(
        TRAIN_CONFIG['batch_size'],
        x_range=(-2.0, 2.0),
        theta_range=(-jnp.pi, jnp.pi),
        key=subkey,
    )
    batch_5d = jax.vmap(convert_4d_to_5d)(batch_init)
    
    print(f"Batch 5d shape: {batch_5d.shape}")
    
    # Test the loss
    try:
        loss = _batch_loss(params, batch_5d)
        print(f"Loss: {loss}")
    except Exception as e:
        print(f"Error in loss computation: {e}")
        return
    
    # Test gradients
    try:
        loss, grads = jax.value_and_grad(_batch_loss)(params, batch_5d)
        print(f"Gradient computation successful, loss: {loss}")
        
        leaves, tree_def = jax.tree_util.tree_flatten(grads)
        print(f"Number of gradient leaves: {len(leaves)}")
        for i, leaf in enumerate(leaves):
            if isinstance(leaf, jnp.ndarray):
                print(f"  Leaf {i}: shape={leaf.shape}, dtype={leaf.dtype}, finite={jnp.isfinite(leaf).all()}")
                print(f"    Range: [{jnp.min(leaf):.6f}, {jnp.max(leaf):.6f}]")
                # Check for unusual values
                if jnp.any(jnp.isinf(leaf)):
                    print(f"    Contains inf: {jnp.sum(jnp.isinf(leaf))}")
                if jnp.any(jnp.isnan(leaf)):
                    print(f"    Contains nan: {jnp.sum(jnp.isnan(leaf))}")
                # Check dtype specifics
                if leaf.dtype != jnp.float32:
                    print(f"    WARNING: Non-float32 dtype: {leaf.dtype}")
                # Check for zero-dimensional arrays
                if leaf.ndim == 0:
                    print(f"    WARNING: Zero-dimensional array with value: {leaf}")
            else:
                print(f"  Leaf {i}: non-array {type(leaf)} = {leaf}")
        
        # Also check the parameter structure to ensure they match
        param_leaves, param_tree_def = jax.tree_util.tree_flatten(params)
        print(f"\nParameter leaves: {len(param_leaves)}")
        for i, leaf in enumerate(param_leaves):
            if isinstance(leaf, jnp.ndarray):
                print(f"  Param {i}: shape={leaf.shape}, dtype={leaf.dtype}")
            else:
                print(f"  Param {i}: non-array {type(leaf)} = {leaf}")
        
        # Check if structures match
        if jax.tree_util.tree_structure(grads) != jax.tree_util.tree_structure(params):
            print("WARNING: Gradient and parameter structures don't match!")
        else:
            print("Gradient and parameter structures match.")
            
        # Test a direct multiplication to see where the issue occurs
        print("\nTesting direct gradient operations...")
        try:
            test_scale = jnp.array(0.001)  # Test scalar
            for i, grad_leaf in enumerate(leaves):
                if isinstance(grad_leaf, jnp.ndarray):
                    scaled = test_scale * grad_leaf
                    print(f"  Leaf {i} scaling successful: {scaled.shape}")
        except Exception as e:
            print(f"Direct scaling failed on leaf {i}: {e}")
        
    except Exception as e:
        print(f"Error in gradient computation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test optimizer - try manual implementation first
    print("\nTesting manual optimizer update...")
    try:
        learning_rate = TRAIN_CONFIG['learning_rate']
        manual_updates = jax.tree_util.tree_map(lambda g: -learning_rate * g, grads)
        print("Manual tree_map successful!")
        
        manual_new_params = jax.tree_util.tree_map(lambda p, u: p + u, params, manual_updates)
        print("Manual parameter update successful!")
        
        # Now test optax
        print("Testing with optax.sgd...")
        optimizer = optax.sgd(learning_rate)
        opt_state = optimizer.init(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        print("Optax SGD successful!")
        
    except Exception as e:
        print(f"Error in optimizer test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_exact_config()
