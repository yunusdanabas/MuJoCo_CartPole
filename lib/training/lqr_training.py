"""
LQR Training Utilities

Provides automated cost matrix tuning, stability analysis, and performance optimization
for LQR control of cart-pole systems.
"""

from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Dict, List, Tuple

import numpy as np
import jax.numpy as jnp

from controller.lqr_controller import LQRController, _linearise
from env.cartpole import CartPoleParams

__all__ = ["LQRTrainingConfig", "train_lqr_controller", "tune_lqr_costs"]


@dataclass
class LQRTrainingConfig:
    """Configuration for LQR controller training."""
    
    # State cost weights (4-state: [x, θ, ẋ, θ̇])
    position_weight: float = 20.0      # Cart position penalty
    angle_weight: float = 50.0         # Pole angle penalty  
    velocity_weight: float = 5.0       # Velocity penalties
    control_weight: float = 5.0        # Control effort penalty
    
    # Performance targets
    min_damping: float = 0.8           # Minimum damping ratio
    max_frequency: float = 3.0         # Maximum natural frequency (Hz)
    cost_scaling: float = 1.0          # Global scaling factor
    
    print_data: bool = False


def _create_cost_matrices(config: LQRTrainingConfig) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Create cost matrices Q and R for LQR control."""
    # State cost matrix Q (4x4) - diagonal weights
    Q = jnp.diag(jnp.array([
        config.position_weight,    # x position
        config.angle_weight,       # θ angle
        config.velocity_weight,    # ẋ velocity
        config.velocity_weight     # θ̇ angular velocity
    ]))
    
    # Control cost matrix R (1x1)
    R = jnp.array([[config.control_weight]])
    
    # Apply global scaling
    Q = Q * config.cost_scaling
    R = R * config.cost_scaling
    
    return Q, R


def _analyze_stability(controller: LQRController, params: CartPoleParams) -> Dict:
    """Analyze closed-loop stability and performance metrics."""
    A, B = _linearise(params)
    K = controller.K
    
    # Closed-loop dynamics: A_cl = A - B*K
    A_cl = A - B @ K[None, :]
    
    # Eigenvalue analysis
    eigvals = jnp.linalg.eigvals(A_cl)
    real_parts = jnp.real(eigvals)
    
    # Stability metrics
    max_real_part = float(np.max(np.array(real_parts)))
    is_stable = max_real_part < 0
    
    # Damping and frequency analysis
    damping_ratios = []
    natural_frequencies = []
    
    # Convert to numpy for analysis
    eigvals_np = np.array(eigvals).flatten()
    
    for eigval in eigvals_np:
        imag_part = float(np.imag(eigval))
        if imag_part != 0:  # Complex conjugate pair
            omega_n = float(np.abs(eigval))
            zeta = -float(np.real(eigval)) / omega_n
            damping_ratios.append(zeta)
            natural_frequencies.append(omega_n)
    
    return {
        "is_stable": is_stable,
        "max_real_part": max_real_part,
        "damping_ratios": damping_ratios,
        "natural_frequencies": natural_frequencies,
        "eigenvalues": eigvals
    }


def _auto_tune_costs(config: LQRTrainingConfig, params: CartPoleParams) -> LQRTrainingConfig:
    """Automatically tune cost matrices for optimal performance."""
    Q, R = _create_cost_matrices(config)
    ctrl = LQRController.from_linearisation(params, Q, R)
    stability = _analyze_stability(ctrl, params)
    
    # Tune for stability and damping
    if not stability["is_stable"] or any(d < config.min_damping for d in stability["damping_ratios"]):
        if config.print_data:
            print("  [TUNE] Auto-tuning for stability...")
        
        config.control_weight *= 1.5
        config.cost_scaling *= 1.2
    
    # Tune for smoothness
    if (stability["natural_frequencies"] and 
        any(f > config.max_frequency for f in stability["natural_frequencies"])):
        if config.print_data:
            print("  [TUNE] Auto-tuning for smoothness...")
        
        config.control_weight *= 1.3
        config.angle_weight *= 0.8
    
    return config


def _print_header(title: str, char: str = "=", width: int = 60):
    """Print formatted section header."""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")


def _print_matrix(name: str, matrix: jnp.ndarray, indent: str = "  "):
    """Print matrix with proper formatting."""
    print(f"{indent}{name}:")
    for row in matrix:
        print(f"{indent}  [{', '.join(f'{x:8.3f}' for x in row)}]")


def _print_stability_analysis(stability: Dict, indent: str = "  "):
    """Print formatted stability analysis results."""
    print(f"{indent}Stability: {'Yes' if stability['is_stable'] else 'No'}")
    print(f"{indent}Max real part: {stability['max_real_part']:8.4f}")
    
    if stability['damping_ratios']:
        damping_str = ', '.join(f'{d:.3f}' for d in stability['damping_ratios'])
        freq_str = ', '.join(f'{f:.3f}' for f in stability['natural_frequencies'])
        print(f"{indent}Damping ratios: [{damping_str}]")
        print(f"{indent}Natural frequencies: [{freq_str}] Hz")


def train_lqr_controller(
    params: CartPoleParams = CartPoleParams(),
    cfg: LQRTrainingConfig = LQRTrainingConfig(),
    *,
    print_data: bool = True,
) -> LQRController:
    """
    Train LQR controller with automated cost matrix tuning.
    
    Returns:
        Trained LQR controller with stability analysis
    """
    cfg.print_data = bool(print_data)
    start_time = time.perf_counter()
    
    if cfg.print_data:
        _print_header("LQR Controller Training")
        print(f"  Parameters:")
        print(f"    Position weight: {cfg.position_weight:6.1f}")
        print(f"    Angle weight:    {cfg.angle_weight:6.1f}")
        print(f"    Velocity weight: {cfg.velocity_weight:6.1f}")
        print(f"    Control weight:  {cfg.control_weight:6.1f}")
        print(f"    Cost scaling:    {cfg.cost_scaling:6.1f}")
    
    # Auto-tune cost matrices
    cfg = _auto_tune_costs(cfg, params)
    Q, R = _create_cost_matrices(cfg)
    
    if cfg.print_data:
        print(f"\n  Final Cost Matrices:")
        _print_matrix("Q (4x4)", Q)
        _print_matrix("R (1x1)", R)
    
    # Train controller
    controller = LQRController.from_linearisation(params, Q, R)
    stability = _analyze_stability(controller, params)
    
    elapsed = time.perf_counter() - start_time
    
    if cfg.print_data:
        print(f"\n  Training completed in {elapsed:.3f}s")
        print(f"  Stability Analysis:")
        _print_stability_analysis(stability)
        
        # Handle different possible shapes of controller.K
        K_array = np.array(controller.K).flatten()
        gains = [f'{k:.3f}' for k in K_array]
        print(f"  Final gains: [{', '.join(gains)}]")
    
    return controller


def tune_lqr_costs(
    params: CartPoleParams = CartPoleParams(),
    target_damping: float = 0.8,
    target_frequency: float = 3.0,
    max_iterations: int = 10
) -> LQRTrainingConfig:
    """
    Automatically tune LQR cost matrices for target performance.
    
    Args:
        target_damping: Target damping ratio (0.7-0.9 recommended)
        target_frequency: Target natural frequency in Hz
        max_iterations: Maximum tuning iterations
    
    Returns:
        Optimized LQR configuration
    """
    print(f"Auto-tuning LQR for damping={target_damping}, frequency={target_frequency}Hz")
    
    # Start with conservative configuration
    config = LQRTrainingConfig(
        position_weight=20.0,
        angle_weight=40.0,
        velocity_weight=5.0,
        control_weight=8.0,
        cost_scaling=1.0
    )
    
    best_config = config
    best_error = float('inf')
    
    for iteration in range(max_iterations):
        controller = train_lqr_controller(params, config, print_data=False)
        stability = _analyze_stability(controller, params)
        
        if stability['damping_ratios'] and stability['natural_frequencies']:
            # Calculate performance error
            damping_error = abs(stability['damping_ratios'][0] - target_damping)
            freq_error = abs(stability['natural_frequencies'][0] - target_frequency)
            total_error = damping_error + freq_error
            
            if total_error < best_error:
                best_error = total_error
                best_config = config
            
            print(f"  Iter {iteration+1:2d}: damping={stability['damping_ratios'][0]:.3f}, "
                  f"freq={stability['natural_frequencies'][0]:.3f}Hz, error={total_error:.3f}")
            
            # Early convergence
            if total_error < 0.1:
                break
        
        # Adaptive parameter adjustment
        if stability['damping_ratios'] and stability['damping_ratios'][0] < target_damping:
            config.control_weight *= 0.8  # More aggressive
        else:
            config.control_weight *= 1.2  # More conservative
        
        config.cost_scaling *= 0.9
    
    print(f"Best configuration found with error: {best_error:.3f}")
    return best_config