"""Debug script to identify the cost computation issue."""

import jax.numpy as jnp
from controller.linear_controller import LinearController
from env.closedloop import simulate
from env.cartpole import CartPoleParams
from lib.cost_functions import create_cost_matrices, compute_trajectory_cost

def debug_cost_computation():
    """Debug the cost computation step by step."""
    print("Debugging cost computation...")
    
    # Test parameters
    K = jnp.array([1.0, -10.0, 10.0, 1.0, 1.0])
    initial_state = jnp.array([0.1, 0.95, 0.31, 0.0, 0.0])
    t_span = (0.0, 3.0)
    dt = 0.01
    params = CartPoleParams()
    
    print(f"K: {K}")
    print(f"Initial state: {initial_state}")
    print(f"Time span: {t_span}")
    
    # Step 1: Test controller function
    try:
        def controller_fn(state, t):
            return -jnp.dot(K, state)
        
        test_force = controller_fn(initial_state, 0.0)
        print(f"✓ Controller function works: force = {test_force}")
    except Exception as e:
        print(f"✗ Controller function failed: {e}")
        return False
    
    # Step 2: Test time array
    try:
        ts = jnp.arange(t_span[0], t_span[1] + dt, dt)
        print(f"✓ Time array created: length = {len(ts)}, range = [{ts[0]:.3f}, {ts[-1]:.3f}]")
    except Exception as e:
        print(f"✗ Time array failed: {e}")
        return False
    
    # Step 3: Test simulation
    try:
        sol = simulate(controller_fn, params, t_span, ts, initial_state)
        print(f"✓ Simulation completed")
        print(f"  Solution type: {type(sol)}")
        print(f"  Has 'ys' attribute: {hasattr(sol, 'ys')}")
        
        if hasattr(sol, 'ys'):
            print(f"  Trajectory shape: {sol.ys.shape}")
            print(f"  First state: {sol.ys[0]}")
            print(f"  Last state: {sol.ys[-1]}")
        else:
            print(f"  Solution attributes: {dir(sol)}")
            return False
            
    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Test force computation
    try:
        import jax
        forces = jax.vmap(lambda state: controller_fn(state, 0.0))(sol.ys)
        print(f"✓ Force computation works: shape = {forces.shape}")
        print(f"  Force range: [{jnp.min(forces):.3f}, {jnp.max(forces):.3f}]")
    except Exception as e:
        print(f"✗ Force computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Test cost matrices
    try:
        Q = create_cost_matrices()
        R = 0.1
        print(f"✓ Cost matrices created: Q shape = {Q.shape}")
    except Exception as e:
        print(f"✗ Cost matrices failed: {e}")
        return False
    
    # Step 6: Test trajectory cost
    try:
        cost = compute_trajectory_cost(sol.ys, forces, Q, R, dt)
        print(f"✓ Trajectory cost computed: {cost}")
        print(f"  Cost is finite: {jnp.isfinite(cost)}")
    except Exception as e:
        print(f"✗ Trajectory cost failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = debug_cost_computation()
    print(f"\nDebug result: {'SUCCESS' if success else 'FAILED'}")