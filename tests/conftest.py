# tests/conftest.py
"""
Shared fixtures & path hack so `import controller` works even when the repo
isn't installed as a package.
"""
import sys
import pathlib
import os
import pytest
import jax

# project root on sys.path ---------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# default JAX platform  ------------------------------------------------------
jax.config.update("jax_platform_name", os.getenv("JAX_PLATFORM", "cpu"))


# common fixtures -----------------------------------------------------------
@pytest.fixture(scope="session")
def params():
    from env.cartpole import CartPoleParams
    return CartPoleParams()

@pytest.fixture(scope="session")
def random_key():
    import jax
    return jax.random.PRNGKey(0)


def test_project_root_identification():
    """Test that ROOT correctly identifies the project root directory"""
    assert ROOT.exists()
    assert ROOT.is_dir()
    # Should contain expected project structure
    expected_dirs = ['env', 'controller', 'lib', 'tests']
    for dirname in expected_dirs:
        assert (ROOT / dirname).exists()


def test_root_added_to_syspath():
    """Test that project root is added to sys.path"""
    root_str = str(ROOT)
    assert root_str in sys.path


def test_can_import_project_modules():
    """Test that project modules can be imported after path setup"""
    try:
        import env.cartpole
        import controller.base
        import lib.utils
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import project modules: {e}")