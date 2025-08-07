"""
controller/base.py

JAX-compatible base class for cart-pole controllers.
Provides JIT compilation, batch processing, and optional profiling.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import time
import jax
import jax.numpy as jnp

# --------------------------------------------------------------------------- #
# Class-level compilation cache                                               #
# --------------------------------------------------------------------------- #

_jit_cache: dict = {}  # Cache compiled functions to avoid recompilation


# --------------------------------------------------------------------------- #
# Base class                                                                  #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Controller(ABC):
    """Abstract base: subclasses implement `_force(state, t) -> float`."""

    _compiled: Callable | None = field(default=None, init=False, repr=False)
    _batch_force: Callable | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Initialize cached batch function."""
        batch_force = jax.vmap(self._force, in_axes=(0, None))
        object.__setattr__(self, '_batch_force', batch_force)

    @abstractmethod
    def _force(self, state: jnp.ndarray, t: float) -> jnp.ndarray: ...

    def __call__(self,
                 state: jnp.ndarray,
                 t: float,
                 *,
                 profile: bool = False) -> jnp.ndarray | tuple[jnp.ndarray, float]:
        """
        Compute control force with optional performance profiling.
        
        Args:
            state: System state vector (1D or batched 2D)
            t: Current time
            profile: If True, return (force, latency_seconds) tuple
            
        Returns:
            force: Control force (scalar for 1D state, array for batched)
            OR (force, latency) tuple if profile=True
        """
        # Select execution mode: compiled (fast) vs eager (flexible)
        execution_fn = self._compiled if self._compiled is not None else self._eager
        
        if not profile:
            # Standard execution - just return the force
            return execution_fn(state, t)
        
        # Profiling mode - measure execution time
        start_time_ns = time.perf_counter_ns()
        force = execution_fn(state, t)
        
        # Ensure computation is complete before measuring end time
        jax.block_until_ready(force)
        
        end_time_ns = time.perf_counter_ns()
        latency_seconds = (end_time_ns - start_time_ns) / 1e9
        
        return force, latency_seconds

    def jit(self) -> "Controller":
        """Return JIT-compiled version with class-level caching."""
        if self._compiled is None:
            # Use simpler, faster cache key
            cache_key = (type(self), repr(self))
            
            if cache_key in _jit_cache:
                compiled = _jit_cache[cache_key]
            else:
                # Compile core function and create batched version
                jit_force = jax.jit(self._force)
                batched_force = jax.vmap(jit_force, in_axes=(0, None))
                
                # JIT-compile dispatcher for zero Python overhead
                @jax.jit
                def compiled(state, t):
                    return jit_force(state, t) if state.ndim == 1 else batched_force(state, t)
                
                _jit_cache[cache_key] = compiled
                compiled = _jit_cache[cache_key]
            
            # Use object.__setattr__ to bypass frozen restriction
            object.__setattr__(self, '_compiled', compiled)
        
        return self

    def batched(self) -> Callable:
        """Return callable optimized for batch inputs."""
        return self.jit().__call__

    def _eager(self, state: jnp.ndarray, t: float) -> jnp.ndarray:
        """Eager execution with automatic batch handling."""
        if state.ndim == 1:
            return self._force(state, t)
        else:
            return self._batch_force(state, t)