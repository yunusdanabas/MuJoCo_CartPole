#!/usr/bin/env python3
"""Debug script to isolate the training issue"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from functools import partial
import yaml
import os

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from controller.nn_controller import CartPoleNN, _trajectory_loss
from lib.utils import convert_4d_to_5d, sample_initial_conditions

def debug_actual_training_step():
    """Debug the actual training step that's failing"""
    
    # Load the same configuration as the main script
    CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.yaml")
    with open(CONFIG_PATH, "r") as f:
        _CONFIG = yaml.safe_load(f) or {}

    nn_cfg = _CONFIG.get("nn_training", {})
    PARAMS_SYSTEM = tuple(nn_cfg.get("params_system", [1.0, 0.1, 0.5, 9.81]))
    Q_MATRIX = jnp.diag(jnp.array(nn_cfg.get("q_weights", [0.1, 10.0, 10.0, 0.1, 0.1])))
    _span = nn_cfg.get("t_span", [0.0, 10.0])
    _eval_pts = nn_cfg.get("t_eval_points", 100)
    TRAIN_CONFIG = {
        'num_epochs': nn_cfg.get('num_epochs', 500),
        'batch_size': nn_cfg.get('batch_size', 32),
        't_span': tuple(_span),
        't_eval': jnp.linspace(_span[0], _span[1], _eval_pts),
        'learning_rate': nn_cfg.get('learning_rate', 3e-4),
        'grad_clip': nn_cfg.get('grad_clip', 1.0),
        'cost_weights': tuple(nn_cfg.get('cost_weights', [10.0, 1.0, 0.01]))
    }
    
    # Initialize exactly like the main script
    key = jax.random.PRNGKey(42)
    controller = CartPoleNN(key=key)
    
    print("=== Reproducing the exact training setup ===")
    print(f"Params: {PARAMS_SYSTEM}")
    print(f"Q shape: {Q_MATRIX.shape}")
    print(f"t_span: {TRAIN_CONFIG['t_span']}")
    print(f"t_eval shape: {TRAIN_CONFIG['t_eval'].shape}")
    print(f"Batch size: {TRAIN_CONFIG['batch_size']}")
    print(f"Cost weights: {TRAIN_CONFIG['cost_weights']}")
    
    # Create the same batch as the main script
    key, subkey = jax.random.split(key)
    batch_init = sample_initial_conditions(
        TRAIN_CONFIG['batch_size'],
        x_range=(-2.0, 2.0),
        theta_range=(-jnp.pi, jnp.pi),
        key=subkey,
    )
    batch_5d = jax.vmap(convert_4d_to_5d)(batch_init)
    
    print(f"Batch 5d shape: {batch_5d.shape}")
    print(f"Batch 5d sample:\n{batch_5d[0]}")
    
    # Create the same optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(TRAIN_CONFIG['grad_clip']),
        optax.adam(TRAIN_CONFIG['learning_rate']),
    )
    opt_state = optimizer.init(eqx.filter(controller, eqx.is_array))
    
    # Test the batch loss function exactly as in the training
    def _batch_loss(ctrl, batch_states):
        per_traj = partial(
            _trajectory_loss,
            controller=ctrl,
            params=PARAMS_SYSTEM,
            Q=Q_MATRIX,
            t_span=TRAIN_CONFIG['t_span'],
            t_eval=TRAIN_CONFIG['t_eval'],
            cost_weights=TRAIN_CONFIG['cost_weights'],
        )
        return jnp.mean(jax.vmap(per_traj)(batch_states))
    
    print("\n=== Testing batch loss ===")
    try:
        loss = _batch_loss(controller, batch_5d)
        print(f"Batch loss: {loss}")
        print(f"Loss finite: {jnp.isfinite(loss)}")
    except Exception as e:
        print(f"Error in batch loss: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n=== Testing gradient computation with eqx.filter_value_and_grad ===")
    try:
        loss, grads = eqx.filter_value_and_grad(_batch_loss)(controller, batch_5d)
        print(f"Gradient computation successful, loss: {loss}")
        
        # Check gradient structure and values
        def check_gradients(node, path=""):
            if isinstance(node, jnp.ndarray):
                finite_mask = jnp.isfinite(node)
                print(f"  {path}: shape={node.shape}, finite={finite_mask.all()}, "
                      f"nan={jnp.isnan(node).sum()}, inf={jnp.isinf(node).sum()}")
                if not finite_mask.all():
                    print(f"    Min: {jnp.min(node)}, Max: {jnp.max(node)}")
                    print(f"    Sample non-finite: {node[~finite_mask][:5] if (~finite_mask).any() else 'None'}")
            elif hasattr(node, '_fields'):  # namedtuple or similar
                for field in node._fields:
                    check_gradients(getattr(node, field), f"{path}.{field}")
            elif hasattr(node, '__dict__'):  # regular object
                for attr, value in node.__dict__.items():
                    if not attr.startswith('_'):
                        check_gradients(value, f"{path}.{attr}")
            elif isinstance(node, (list, tuple)):
                for i, item in enumerate(node):
                    check_gradients(item, f"{path}[{i}]")
            else:
                print(f"  {path}: {type(node)} = {node}")
        
        print("Gradient structure:")
        check_gradients(grads)
        
    except Exception as e:
        print(f"Error in gradient computation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n=== Testing optimizer update ===")
    try:
        print("About to call optimizer.update...")
        print(f"Grads type: {type(grads)}")
        print(f"Opt state type: {type(opt_state)}")
        print(f"Controller type: {type(controller)}")
        
        updates, new_opt_state = optimizer.update(grads, opt_state, eqx.filter(controller, eqx.is_array))
        print("Optimizer update successful!")
        
    except Exception as e:
        print(f"Error in optimizer update: {e}")
        import traceback
        traceback.print_exc()
        
        # Let's try examining the gradients leaf by leaf
        print("\n=== Detailed gradient examination ===")
        leaves, tree_def = jax.tree_util.tree_flatten(grads)
        print(f"Number of gradient leaves: {len(leaves)}")
        
        for i, leaf in enumerate(leaves):
            print(f"Leaf {i}:")
            if isinstance(leaf, jnp.ndarray):
                print(f"  Shape: {leaf.shape}, Dtype: {leaf.dtype}")
                print(f"  Finite: {jnp.isfinite(leaf).all()}")
                print(f"  Min: {jnp.min(leaf)}, Max: {jnp.max(leaf)}")
                if not jnp.isfinite(leaf).all():
                    print(f"  Non-finite count: {(~jnp.isfinite(leaf)).sum()}")
                    print(f"  Sample values: {leaf.flatten()[:10]}")
            else:
                print(f"  Type: {type(leaf)}, Value: {leaf}")

if __name__ == "__main__":
    debug_actual_training_step()
