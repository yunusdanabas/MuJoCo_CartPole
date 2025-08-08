"""
controller/linear_controller.py

Linear feedback controller for cart-pole systems.
Implements u = -K @ state with JIT compilation by default.
"""

from __future__ import annotations
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import time
from controller.base import Controller
from env.helpers import four_to_five


@dataclass(frozen=True)
class LinearController(Controller):
    """Linear feedback controller: u = -K · state (clipped)."""
    K: jnp.ndarray          # shape (5,) preferred; (4,) supported for 4D

    def __post_init__(self):
        """Initialize JIT functions and validate dimensions."""
        if self.K.shape not in ((5,), (4,)):
            raise ValueError(f"K must have shape (5,) or (4,), got {self.K.shape}")
        
        # JIT compile force functions
        object.__setattr__(self, '_jit_single', jax.jit(self._force_impl))
        object.__setattr__(self, '_jit_batch', jax.jit(jax.vmap(self._force_impl, in_axes=(0, None))))
        
        super().__post_init__()
    
    def _force_impl(self, state: jnp.ndarray, t: float) -> jnp.ndarray:
        if state.shape[-1] == 4:
            if self.K.shape == (4,):
                raw = -jnp.dot(self.K, state)
            else:
                # Up-cast state to 5D for 5D gains
                s5 = four_to_five(state)
                raw = -jnp.dot(self.K, s5)
        else:
            if self.K.shape == (5,):
                raw = -jnp.dot(self.K, state)
            else:
                # Promote 4D gains to 5D by inserting zero angle terms
                K5 = jnp.array([self.K[0], 0.0, 0.0, self.K[1], self.K[2]]) if self.K.shape == (3,) else jnp.array([self.K[0], 0.0, 0.0, self.K[2], self.K[3]])
                raw = -jnp.dot(K5, state)
        return jnp.clip(raw, -100.0, 100.0)
    
    def _force(self, state: jnp.ndarray, t: float) -> jnp.ndarray:
        """Required by base Controller class."""
        return self._jit_single(state, t)
    
    def __call__(self, state, t, profile=False):
        """Main interface - JIT by default."""
        if profile:
            start_time = time.perf_counter()
        
        # Choose JIT function based on input shape
        if state.ndim == 1:
            force = self._jit_single(state, t)
        else:
            force = self._jit_batch(state, t)
        
        if profile:
            jax.block_until_ready(force)
            latency = time.perf_counter() - start_time
            return force, latency
        
        return force
    
    def eager(self, state, t, profile=False):
        """Non-JIT version for debugging."""
        if profile:
            start_time = time.perf_counter()
        
        if state.ndim == 1:
            force = self._force_impl(state, t)
        else:
            force = jax.vmap(self._force_impl, in_axes=(0, None))(state, t)
        
        if profile:
            latency = time.perf_counter() - start_time
            return force, latency
        
        return force


def create_pd_controller(kp_pos=1.0, kd_pos=1.0, kp_angle=20.0, kd_angle=2.0):
    """Create PD controller with specified gains."""
    # Target: [x=0, cos(θ)=1, sin(θ)=0, ẋ=0, θ̇=0]
    K = jnp.array([
        kp_pos,    # x position
        -kp_angle, # cos(θ) - negative to push toward 1
        kp_angle,  # sin(θ) - positive to push toward 0  
        kd_pos,    # ẋ damping
        kd_angle   # θ̇ damping
    ])
    return LinearController(K=K)


def create_zero_controller():
    """Controller that outputs zero force."""
    return LinearController(K=jnp.zeros(5))


def analyze_controller_gains(controller: LinearController) -> dict[str, float]:
    """Analyze controller gain properties."""
    K = controller.K
    
    return {
        'position_gain': float(K[0]),
        'cos_theta_gain': float(K[1]), 
        'sin_theta_gain': float(K[2]),
        'velocity_gain': float(K[3]),
        'angular_velocity_gain': float(K[4]),
        'total_magnitude': float(jnp.linalg.norm(K)),
        'proportional_magnitude': float(jnp.linalg.norm(K[:3])),
        'derivative_magnitude': float(jnp.linalg.norm(K[3:]))
    }