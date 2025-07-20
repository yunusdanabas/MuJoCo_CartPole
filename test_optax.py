#!/usr/bin/env python3
"""Minimal test to isolate the optax issue"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

# Create a simple neural network 
class SimpleNN(eqx.Module):
    layers: list
    
    def __init__(self, key):
        keys = jax.random.split(key, 2)
        self.layers = [
            eqx.nn.Linear(2, 4, key=keys[0]),
            eqx.nn.Linear(4, 1, key=keys[1])
        ]
    
    def __call__(self, x):
        x = jax.nn.relu(self.layers[0](x))
        return self.layers[1](x)[0]

def test_optax():
    key = jax.random.PRNGKey(42)
    
    # Create model
    model = SimpleNN(key)
    
    # Create optimizer
    optimizer = optax.adam(3e-4)
    
    # Extract parameters
    params = eqx.filter(model, eqx.is_array)
    static = eqx.filter(model, lambda x: not eqx.is_array(x))
    
    opt_state = optimizer.init(params)
    
    def loss_fn(params, x, y):
        model = eqx.combine(params, static)
        pred = model(x)
        return (pred - y) ** 2
    
    # Test data
    x = jnp.array([1.0, 2.0])
    y = jnp.array(3.0)
    
    # Test gradient computation
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    print(f"Loss: {loss}")
    print(f"Grads structure: {jax.tree_util.tree_structure(grads)}")
    
    # Examine gradients
    leaves, tree_def = jax.tree_util.tree_flatten(grads)
    print(f"Number of gradient leaves: {len(leaves)}")
    for i, leaf in enumerate(leaves):
        print(f"Leaf {i}: shape={leaf.shape}, dtype={leaf.dtype}, finite={jnp.isfinite(leaf).all()}")
    
    # Test optimizer update
    try:
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        print("Optimizer update successful!")
        new_params = optax.apply_updates(params, updates)
        print("Parameter update successful!")
    except Exception as e:
        print(f"Error in optimizer update: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_optax()
