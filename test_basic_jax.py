# test_training_minimal.py
# Minimal training test to isolate the issue

import jax
import jax.numpy as jnp

def test_basic_operations():
    """Test basic JAX operations that seem to be failing"""
    
    print("Testing basic JAX operations...")
    
    # Test basic array creation
    x = jnp.array([1.0, 2.0, 3.0])
    print(f"Created array: {x}")
    
    # Test scalar multiplication
    try:
        lr = 0.01
        y = lr * x
        print(f"Scalar multiplication successful: {y}")
    except Exception as e:
        print(f"Error in scalar multiplication: {e}")
        return False
    
    # Test array subtraction
    try:
        z = x - y
        print(f"Array subtraction successful: {z}")
    except Exception as e:
        print(f"Error in array subtraction: {e}")
        return False
    
    # Test gradient computation
    try:
        def simple_fn(params):
            return jnp.sum(params ** 2)
        
        grads = jax.grad(simple_fn)(x)
        print(f"Gradient computation successful: {grads}")
    except Exception as e:
        print(f"Error in gradient computation: {e}")
        return False
    
    # Test SGD step
    try:
        def sgd_step(params, grads, lr):
            return params - lr * grads
        
        new_params = sgd_step(x, grads, lr)
        print(f"SGD step successful: {new_params}")
    except Exception as e:
        print(f"Error in SGD step: {e}")
        return False
    
    print("All basic operations successful!")
    return True

if __name__ == "__main__":
    success = test_basic_operations()
    if success:
        print("JAX operations are working correctly.")
    else:
        print("JAX operations are failing - there may be an installation issue.")
