import jax.numpy as jnp
from env.closedloop import simulate
from controller.lqr_controller import LQRController
from controller.linear_controller import LinearController
from env.cartpole import CartPoleParams
from env.helpers import four_to_five
import unittest

class TestIntegration(unittest.TestCase):
    def test_module_interaction(self):
        """Test that different modules work together properly"""
        params = CartPoleParams()
        tgrid = jnp.linspace(0.0, 0.1, 11)
        
        # Test LQR controller with environment (4-state ICs are supported)
        lqr = LQRController.from_linearisation(params).jit()
        init_state = jnp.array([0.01, 0.01, 0.0, 0.0])
        sol = simulate(lqr, params, (0., 0.1), tgrid, init_state)
        
        assert sol.ys.shape == (len(tgrid), 4)
        assert jnp.all(jnp.isfinite(sol.ys))

    def test_hybrid_stabilises(self):
        """Test that LQR controller stabilizes the system from hanging position"""
        params = CartPoleParams()
        lqr = LQRController.from_linearisation(params).jit()

        tgrid = jnp.linspace(0., 2., 201)
        init = jnp.array([0., jnp.pi, 0., 0.])  # hanging down
        sol = simulate(lqr, params, (0., 2.), tgrid, init)
        final_angle = sol.ys[-1, 1]  # θ
        assert abs(final_angle) < 0.1  # ~upright

    def test_controller_switching(self):
        """Test switching between different controllers"""
        params = CartPoleParams()
        tgrid = jnp.linspace(0.0, 0.2, 21)
        
        # Linear controller (requires 5-state gains)
        lin = LinearController(K=jnp.array([10., 0., 5., 0., 0.])).jit()
        # LQR controller  
        lqr = LQRController.from_linearisation(params).jit()
        
        init_state4 = jnp.array([0.05, 0.05, 0.0, 0.0])
        init_state5 = four_to_five(init_state4)
        
        # Both should be able to handle the same initial condition representation
        sol_lin = simulate(lin, params, (0., 0.2), tgrid, init_state5)
        sol_lqr = simulate(lqr, params, (0., 0.2), tgrid, init_state4)
        
        assert jnp.all(jnp.isfinite(sol_lin.ys))
        assert jnp.all(jnp.isfinite(sol_lqr.ys))

if __name__ == '__main__':
    unittest.main()