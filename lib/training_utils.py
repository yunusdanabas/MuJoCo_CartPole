"""
lib/training_utils.py

Common training utilities and configurations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import jax.numpy as jnp
import numpy as np


@dataclass
class BaseTrainingConfig:
    """Base configuration for controller training."""
    learning_rate: float = 0.01
    num_iterations: int = 500
    trajectory_length: float = 5.0
    convergence_tol: float = 1e-6
    verbose: bool = True


class TrainingHistory:
    """Track training progress."""
    
    def __init__(self):
        self.costs = []
        self.parameters = []
    
    def update(self, cost: float, params: Any = None):
        """Record training step."""
        self.costs.append(float(cost))
        if params is not None:
            if hasattr(params, '__array__'):
                self.parameters.append(np.array(params))
            else:
                self.parameters.append(params)
    
    def get_improvement_ratio(self) -> float:
        """Final to initial cost ratio."""
        if len(self.costs) < 2:
            return 1.0
        return self.costs[-1] / self.costs[0]
    
    def summary(self) -> dict[str, Any]:
        """Training statistics."""
        if not self.costs:
            return {}
        
        return {
            'initial_cost': self.costs[0],
            'final_cost': self.costs[-1],
            'improvement_ratio': self.get_improvement_ratio(),
            'total_iterations': len(self.costs)
        }


def create_standard_test_states() -> jnp.ndarray:
    """Standard test states for validation."""
    return jnp.array([
        [0.0, 1.0, 0.0, 0.0, 0.0],      # Upright
        [0.1, 0.95, 0.31, 0.0, 0.0],    # Small angle
        [0.0, 0.0, 1.0, 0.0, 0.0],      # Horizontal
        [-0.2, 0.8, 0.6, 0.1, -0.1]     # Large angle with motion
    ])


def print_training_summary(history: TrainingHistory, config: BaseTrainingConfig) -> None:
    """Print training summary."""
    if not history or len(history.costs) == 0:
        print("No training data available")
        return
    
    summary = history.summary()
    
    print(f"\n{'='*50}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*50}")
    print(f"Learning rate:    {config.learning_rate}")
    print(f"Max iterations:   {config.num_iterations}")
    print(f"Trajectory length: {config.trajectory_length:.1f}s")
    print(f"Initial cost:     {summary['initial_cost']:.6f}")
    print(f"Final cost:       {summary['final_cost']:.6f}")
    print(f"Improvement:      {summary['improvement_ratio']:.3f}x")
    print(f"Total iterations: {summary['total_iterations']}")
    print(f"{'='*50}")