"""
examples/controllers/linear_example.py

Linear Controller Training Examples
Demonstrates training workflows for linear cart-pole controllers.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import time
import traceback

from lib.training.linear_training import (
    train_linear_controller,
    grid_search_linear_gains,
    LinearTrainingConfig,
)

basic_train = train_linear_controller
advanced_train = train_linear_controller

from lib.stability import quick_stability_check
from controller.linear_controller import LinearController


def basic_training_example():
    """Basic linear controller training with minimal settings."""
    print("Basic Linear Controller Training Example")
    print("=" * 50)

    try:
        initial_state = jnp.array([0.1, 0.95, 0.31, 0.0, 0.0])
        initial_K = jnp.array([1.0, -10.0, 10.0, 1.0, 1.0])

        config = LinearTrainingConfig(learning_rate=0.02, num_iterations=50, trajectory_length=2.0)
        controller, history = basic_train(initial_K, initial_state, config)

        print("Cost trend:")
        for i, c in enumerate(history.costs):
            if i % 10 == 0 or i == len(history.costs) - 1:
                print(f"  iter {i:02d}: {c:.6f}")

        test_state = jnp.array([0.1, 0.95, 0.31, 0.0, 0.0])
        jit_force = controller(test_state, 0.0)
        eager_force = controller.eager(test_state, 0.0)
        print(f"JIT force: {jit_force}, Eager force: {eager_force}")
        print(f"Match: {jnp.allclose(jit_force, eager_force)}")

        return controller, history

    except Exception as e:
        print(f"Training failed: {e}")
        traceback.print_exc()
        return None, None


def grid_search_example():
    """Find good initial gains using grid search."""
    print("\nGrid Search Example")
    print("=" * 50)
    
    try:
        initial_state = jnp.array([0.1, 0.95, 0.31, 0.0, 0.0])
        
        # Search ranges for [x, cos(θ), sin(θ), ẋ, θ̇]
        gain_ranges = (
            (0.5, 3.0),      # x position
            (-30.0, -5.0),   # cos(θ) - negative
            (5.0, 30.0),     # sin(θ) - positive
            (0.2, 2.0),      # x velocity
            (0.2, 2.0)       # angular velocity
        )
        
        best_controller = grid_search_linear_gains(
            initial_state=initial_state,
            gain_ranges=gain_ranges,
            n_points=3
        )
        
        print(f"Best gains: {best_controller.K}")
        return best_controller
        
    except Exception as e:
        print(f"Grid search failed: {e}")
        return None


def multi_scenario_training():
    """Train on multiple scenarios for robustness."""
    print("\nMulti-Scenario Training Example")
    print("=" * 50)
    
    try:
        # Test scenarios
        scenarios = [
            jnp.array([0.1, 0.95, 0.31, 0.0, 0.0]),    # Small angle
            jnp.array([0.0, 0.7, 0.7, 0.0, 0.0]),      # 45° angle
            jnp.array([0.2, 0.8, 0.6, 0.1, -0.1]),     # With velocity
            jnp.array([-0.1, 0.9, -0.436, 0.0, 0.0])   # Negative angle
        ]
        
        # Initial gains from grid search
        best_initial = grid_search_linear_gains(scenarios[0], n_points=2)
        
        # Robust training config
        config = LinearTrainingConfig(
            learning_rate=0.005,
            num_iterations=50,
            trajectory_length=3.0,
            optimizer='adam',
        )

        controller, history = advanced_train(best_initial.K, scenarios[2], config)
        
        # Test on all scenarios
        print("\nPerformance Evaluation:")
        print(f"Gains: {controller.K}")
        
        for i, scenario in enumerate(scenarios):
            stable = quick_stability_check(controller, scenario)
            print(f"  Scenario {i+1}: {'STABLE' if stable else 'UNSTABLE'}")
        
        return controller, history
        
    except Exception as e:
        print(f"Multi-scenario training failed: {e}")
        return None, None


def comparison_study():
    """Compare different training configurations."""
    print("\nConfiguration Comparison Study")
    print("=" * 50)
    
    try:
        initial_state = jnp.array([0.1, 0.95, 0.31, 0.0, 0.0])
        initial_K = jnp.array([1.0, -10.0, 10.0, 1.0, 1.0])
        
        configs = [
            ('Fast', LinearTrainingConfig(learning_rate=0.02, num_iterations=20)),
            ('Medium', LinearTrainingConfig(learning_rate=0.01, num_iterations=40)),
            ('Careful', LinearTrainingConfig(learning_rate=0.005, num_iterations=60))
        ]
        
        results = {}
        
        for name, config in configs:
            print(f"\nTesting {name} configuration...")
            controller, history = advanced_train(initial_K, initial_state, config)
            
            if controller is not None and history is not None and len(history.costs) > 0:
                results[name] = {
                    'controller': controller,
                    'history': history,
                    'final_cost': history.costs[-1] if history.costs else float('inf'),
                    'improvement': history.get_improvement_ratio() if history.costs else 1.0,
                }
            else:
                results[name] = {
                    'controller': None,
                    'history': None,
                    'final_cost': float('inf'),
                    'improvement': 1.0,
                }
        
        # Print comparison
        print(f"\n{'Configuration':<12} {'Final Cost':<12} {'Improvement':<12}")
        print("-" * 40)
        for name, result in results.items():
            final_cost = result['final_cost']
            improvement = result['improvement']
            print(f"{name:<12} {final_cost:<12.4f} {improvement:<12.2f}x")
        
        return results
        
    except Exception as e:
        print(f"Comparison study failed: {e}")
        traceback.print_exc()
        return {}


def jit_comparison_example():
    """Compare JIT vs eager mode performance."""
    print("\nJIT vs Eager Performance Comparison")
    print("=" * 50)
    
    try:
        # Create controller
        K = jnp.array([1.0, -10.0, 10.0, 1.0, 1.0])
        controller = LinearController(K=K)
        
        # Test data
        single_state = jnp.array([0.1, 0.95, 0.31, 0.0, 0.0])
        batch_size = 1000
        batch_states = jnp.tile(single_state, (batch_size, 1))
        
        # Warmup
        _ = controller(single_state, 0.0)
        _ = controller.eager(single_state, 0.0)
        _ = controller(batch_states, 0.0)
        _ = controller.eager(batch_states, 0.0)
        
        print("Running performance tests...")
        
        # Single state test
        n_tests = 100
        
        start_time = time.time()
        for _ in range(n_tests):
            force = controller(single_state, 0.0)
            jax.device_get(force)
        jit_time = time.time() - start_time
        
        start_time = time.time()
        for _ in range(n_tests):
            force = controller.eager(single_state, 0.0)
            jax.device_get(force)
        eager_time = time.time() - start_time
        
        # Batch test
        start_time = time.time()
        force = controller(batch_states, 0.0)
        jax.device_get(force)
        jit_batch_time = time.time() - start_time
        
        start_time = time.time()
        force = controller.eager(batch_states, 0.0)
        jax.device_get(force)
        eager_batch_time = time.time() - start_time
        
        # Profile test
        _, jit_latency = controller(single_state, 0.0, profile=True)
        _, eager_latency = controller.eager(single_state, 0.0, profile=True)
        
        # Results
        print("\nPerformance Results:")
        print(f"{'Test':<20} {'Eager (ms)':<15} {'JIT (ms)':<15} {'Speedup':<10}")
        print("-" * 60)
        
        ms_single_eager = eager_time * 1000 / n_tests
        ms_single_jit = jit_time * 1000 / n_tests
        print(f"Single State      {ms_single_eager:.4f}       {ms_single_jit:.4f}       {eager_time/jit_time:.1f}x")
        
        ms_profile_eager = eager_latency * 1000
        ms_profile_jit = jit_latency * 1000
        print(f"Profile           {ms_profile_eager:.4f}       {ms_profile_jit:.4f}       {eager_latency/jit_latency:.1f}x")
        
        ms_batch_eager = eager_batch_time * 1000
        ms_batch_jit = jit_batch_time * 1000
        print(f"Batch ({batch_size})     {ms_batch_eager:.4f}       {ms_batch_jit:.4f}       {eager_batch_time/jit_batch_time:.1f}x")
        
        # Validation
        jit_force = controller(single_state, 0.0)
        eager_force = controller.eager(single_state, 0.0)
        match = jnp.allclose(jit_force, eager_force)
        
        print(f"\nValidation: {'PASS' if match else 'FAIL'}")
        print(f"JIT:   {jit_force:.6f}")
        print(f"Eager: {eager_force:.6f}")
        
        return {
            'controller': controller,
            'speedup_single': eager_time / jit_time,
            'speedup_batch': eager_batch_time / jit_batch_time
        }
        
    except Exception as e:
        print(f"JIT comparison failed: {e}")
        traceback.print_exc()
        return None


def interactive_example():
    """Interactive training with custom parameters."""
    print("\nInteractive Training Example")
    print("=" * 50)
    
    try:
        # Custom parameters
        initial_K = jnp.array([1.5, -15.0, 15.0, 1.2, 1.2])
        initial_state = jnp.array([0.05, 0.98, 0.2, 0.0, 0.0])
        
        config = LinearTrainingConfig(
            learning_rate=0.02,
            num_iterations=50,
            trajectory_length=3.0,
            optimizer='adam',
        )

        controller, history = advanced_train(initial_K, initial_state, config)
        
        # Test challenging state
        test_state = jnp.array([0.2, 0.92, -0.39, 0.1, -0.1])
        stable = quick_stability_check(controller, test_state)
        
        print(f"\nFinal Test: {'PASS' if stable else 'FAIL'}")
        
        return controller, history
        
    except Exception as e:
        print(f"Interactive example failed: {e}")
        return None, None


def run_single_example(example_name: str):
    """Run specific example by name."""
    examples = {
        'basic': basic_training_example,
        'grid': grid_search_example,
        'multi': multi_scenario_training,
        'comparison': comparison_study,
        'jit': jit_comparison_example,
        'interactive': interactive_example
    }
    
    if example_name not in examples:
        print(f"Available: {list(examples.keys())}")
        return None
    
    print(f"Running {example_name} example...")
    return examples[example_name]()


def main():
    """Run all examples."""
    print("Linear Controller Training Examples")
    print("=" * 60)
    
    examples = [
        ('basic', basic_training_example),
        ('grid_search', grid_search_example),
        ('robust', multi_scenario_training),
        ('comparison', comparison_study),
        ('jit', jit_comparison_example),
        ('interactive', interactive_example)
    ]
    
    controllers = {}
    
    for name, func in examples:
        print(f"\n{'='*20} {name} {'='*20}")
        result = func()
        
        if result:
            controllers[name] = result[0] if isinstance(result, tuple) else result
            print(f"{name} completed successfully!")
        else:
            controllers[name] = None
            print(f"{name} failed!")
    
    # Summary
    successful = sum(1 for c in controllers.values() if c is not None)
    total = len(controllers)
    
    print(f"\n{'='*60}")
    print("EXECUTION SUMMARY")
    print(f"{'='*60}")
    
    for name, controller in controllers.items():
        status = "SUCCESS" if controller else "FAILED"
        print(f"  {name:.<20} {status}")
    
    print(f"\nOverall: {successful}/{total} examples completed")
    
    return controllers


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        run_single_example(sys.argv[1])
    else:
        main()