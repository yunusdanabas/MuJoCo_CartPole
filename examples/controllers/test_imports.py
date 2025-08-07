"""
Test script to identify import issues for linear_example.py
"""

def test_basic_imports():
    """Test basic Python and JAX imports"""
    print("Testing basic imports...")
    try:
        import jax
        import jax.numpy as jnp
        print("✓ JAX imports successful")
    except ImportError as e:
        print(f"✗ JAX import error: {e}")
        return False
    return True

def test_lib_imports():
    """Test lib module imports"""
    print("\nTesting lib imports...")
    
    # Test training imports
    try:
        from lib.training.linear_training import (
            train_linear_controller, 
            grid_search_linear_gains,
            LinearTrainingConfig,
            validate_linear_training_setup
        )
        print("✓ lib.training.linear_training imports successful")
    except ImportError as e:
        print(f"✗ lib.training.linear_training import error: {e}")
        return False
    
    # Test stability imports
    try:
        from lib.stability import quick_stability_check
        print("✓ lib.stability import successful")
    except ImportError as e:
        print(f"✗ lib.stability import error: {e}")
        return False
    
    # Test training_utils imports
    try:
        from lib.training_utils import print_training_summary
        print("✓ lib.training_utils import successful")
    except ImportError as e:
        print(f"✗ lib.training_utils import error: {e}")
        return False
    
    return True

def test_controller_imports():
    """Test controller imports (dependencies)"""
    print("\nTesting controller imports...")
    try:
        from controller.linear_controller import LinearController
        print("✓ LinearController import successful")
    except ImportError as e:
        print(f"✗ LinearController import error: {e}")
        return False
    return True

def test_env_imports():
    """Test environment imports (dependencies)"""
    print("\nTesting environment imports...")
    try:
        from env.cartpole import CartPoleParams
        print("✓ CartPoleParams import successful")
    except ImportError as e:
        print(f"✗ CartPoleParams import error: {e}")
        return False
    
    try:
        from env.closedloop import simulate
        print("✓ simulate import successful")
    except ImportError as e:
        print(f"✗ simulate import error: {e}")
        return False
    
    return True

def test_file_existence():
    """Check if required files exist"""
    print("\nChecking file existence...")
    import os
    
    files_to_check = [
        "lib/training/linear_training.py",
        "lib/stability.py", 
        "lib/training_utils.py",
        "lib/cost_functions.py",
        "controller/linear_controller.py",
        "env/cartpole.py",
        "env/closedloop.py"
    ]
    
    missing_files = []
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def test_basic_functionality():
    """Test basic functionality works"""
    print("\nTesting basic functionality...")
    try:
        import jax.numpy as jnp
        from lib.training.linear_training import LinearTrainingConfig
        
        # Test array creation
        K = jnp.array([1.0, -10.0, 10.0, 1.0, 1.0])
        state = jnp.array([0.1, 0.95, 0.31, 0.0, 0.0])
        print("✓ JAX arrays created successfully")
        
        # Test config creation
        config = LinearTrainingConfig()
        print("✓ LinearTrainingConfig created successfully")
        print(f"  Default learning_rate: {config.learning_rate}")
        print(f"  Default num_iterations: {config.num_iterations}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def main():
    """Run all import tests"""
    print("=" * 60)
    print("IMPORT DIAGNOSTIC TEST")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("File Existence", test_file_existence), 
        ("Controller Imports", test_controller_imports),
        ("Environment Imports", test_env_imports),
        ("Lib Imports", test_lib_imports),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running: {test_name}")
        print(f"{'-' * 40}")
        results[test_name] = test_func()
    
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All imports working! You can run the linear_example.py")
    else:
        print("✗ Some imports failed. Fix the issues above before running examples.")
        print("\nCommon fixes:")
        print("1. Ensure all lib/ modules are created and populated")
        print("2. Check controller/ and env/ modules exist") 
        print("3. Verify Python path includes the project directory")
        print("4. Install missing dependencies (JAX, optax, etc.)")
    
    return passed == total

if __name__ == "__main__":
    success = main()