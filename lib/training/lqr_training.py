"""
LQR training utilities with improved cost matrix tuning for smooth, less noisy control.
Provides stability analysis, pole placement, and cost matrix optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
import jax.numpy as jnp
from controller.lqr_controller import LQRController, _linearise
from env.cartpole import CartPoleParams

__all__ = ["LQRTrainingConfig", "train_lqr_controller", "tune_lqr_costs"]


@dataclass
class LQRTrainingConfig:
    """Configuration for LQR controller training with improved cost matrices."""
    
    # Cost matrix weights (4-state: [x, θ, ẋ, θ̇])
    position_weight: float = 50.0      # Cart position weight
    angle_weight: float = 100.0        # Pole angle weight (high for stability)
    velocity_weight: float = 10.0      # Velocity weights (moderate)
    control_weight: float = 1.0        # Control penalty (higher = smoother)
    
    # Tuning parameters
    cost_scaling: float = 1.0          # Global cost scaling factor
    min_damping: float = 0.7           # Minimum damping ratio for poles
    max_frequency: float = 5.0         # Maximum natural frequency
    
    print_data: bool = False


def _create_cost_matrices(config: LQRTrainingConfig) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Create well-tuned cost matrices for smooth LQR control (4-state format)."""
    
    # State cost matrix Q (4x4) - balanced weights for stability
    Q = jnp.diag(jnp.array([
        config.position_weight,    # x position - moderate weight
        config.angle_weight,       # θ - high weight for stability
        config.velocity_weight,    # ẋ - moderate weight
        config.velocity_weight     # θ̇ - moderate weight
    ]))
    
    # Control cost matrix R (1x1) - higher penalty = less noise
    R = jnp.array([[config.control_weight]])
    
    # Apply global scaling
    Q = Q * config.cost_scaling
    R = R * config.cost_scaling
    
    return Q, R


def _analyze_stability(controller: LQRController, params: CartPoleParams) -> dict:
    """Analyze closed-loop stability and performance."""
    A, B = _linearise(params)
    
    # Controller gains are already 4-state
    K4 = controller.K
    
    # Closed-loop dynamics: A_cl = A - B*K
    A_cl = A - B @ K4[None, :]
    
    # Eigenvalue analysis
    eigvals = jnp.linalg.eigvals(A_cl)
    real_parts = jnp.real(eigvals)
    
    # Stability metrics
    max_real_part = jnp.max(real_parts)
    is_stable = max_real_part < 0
    
    # Damping and frequency analysis
    damping_ratios = []
    natural_frequencies = []
    
    for eigval in eigvals:
        if jnp.imag(eigval) != 0:  # Complex conjugate pair
            omega_n = jnp.abs(eigval)
            zeta = -jnp.real(eigval) / omega_n
            damping_ratios.append(float(zeta))
            natural_frequencies.append(float(omega_n))
    
    return {
        "is_stable": bool(is_stable),
        "max_real_part": float(max_real_part),
        "damping_ratios": damping_ratios,
        "natural_frequencies": natural_frequencies,
        "eigenvalues": eigvals
    }


def _auto_tune_costs(config: LQRTrainingConfig, params: CartPoleParams) -> LQRTrainingConfig:
    """Automatically tune cost matrices for better performance."""
    
    # Create initial cost matrices
    Q, R = _create_cost_matrices(config)
    
    # Test stability with current costs
    ctrl = LQRController.from_linearisation(params, Q, R)
    stability = _analyze_stability(ctrl, params)
    
    # Auto-tune if needed for stability or damping
    if not stability["is_stable"] or any(d < config.min_damping for d in stability["damping_ratios"]):
        if config.print_data:
            print("[TUNE] Auto-tuning cost matrices for stability...")
        
        # Increase control penalty for more conservative control
        config.control_weight *= 2.0
        config.cost_scaling *= 1.5
        
        if config.print_data:
            print(f"[TUNE] Adjusted control_weight: {config.control_weight:.2f}")
            print(f"[TUNE] Adjusted cost_scaling: {config.cost_scaling:.2f}")
    
    return config


def train_lqr_controller(
    params: CartPoleParams = CartPoleParams(),
    cfg: LQRTrainingConfig = LQRTrainingConfig(),
    *,
    print_data: bool = True,
) -> LQRController:
    """
    Train LQR controller with improved cost matrix tuning for smooth, less noisy control.
    
    Args:
        params: Cart-pole system parameters
        cfg: LQR training configuration
        print_data: Whether to print training progress
    
    Returns:
        Trained LQR controller with stability analysis
    """
    # Sync logging flag
    cfg.print_data = bool(print_data)
    
    # ==================== Initialization ====================
    start_total = time.perf_counter()
    
    if cfg.print_data:
        print("[TRAIN] LQRController started")
        print("[TRAIN] LQR Parameters:")
        print(f"  position_weight: {cfg.position_weight}")
        print(f"  angle_weight: {cfg.angle_weight}")
        print(f"  velocity_weight: {cfg.velocity_weight}")
        print(f"  control_weight: {cfg.control_weight}")
        print(f"  cost_scaling: {cfg.cost_scaling}")
    
    # ==================== Cost Matrix Tuning ====================
    # Auto-tune cost matrices for better performance
    cfg = _auto_tune_costs(cfg, params)
    
    # Create final cost matrices
    Q, R = _create_cost_matrices(cfg)
    
    if cfg.print_data:
        print("[TRAIN] Final Cost Matrices:")
        print("  Q (4x4):")
        print(Q)
        print("  R (1x1):")
        print(R)
    
    # ==================== LQR Computation ====================
    # Linearise system and solve for optimal gains
    ctrl = LQRController.from_linearisation(params, Q, R)
    
    # ==================== Stability Analysis ====================
    stability = _analyze_stability(ctrl, params)
    
    elapsed = time.perf_counter() - start_total
    
    # ==================== Results and Logging ====================
    if cfg.print_data:
        print(f"[TRAIN] LQRController finished in {elapsed:.6f}s")
        print("[TRAIN] Stability Analysis:")
        print(f"  Stable: {'Yes' if stability['is_stable'] else 'No'}")
        print(f"  Max real part: {stability['max_real_part']:.4f}")
        
        if stability['damping_ratios']:
            print(f"  Damping ratios: {[f'{d:.3f}' for d in stability['damping_ratios']]}")
            print(f"  Natural frequencies: {[f'{f:.3f}' for f in stability['natural_frequencies']]}")
        
        print(f"[TRAIN] Final gains (4-state): {ctrl.K}")
    
    return ctrl


def tune_lqr_costs(
    params: CartPoleParams = CartPoleParams(),
    target_damping: float = 0.8,
    target_frequency: float = 3.0,
    max_iterations: int = 10
) -> LQRTrainingConfig:
    """
    Automatically tune LQR cost matrices for target performance.
    
    Args:
        params: Cart-pole system parameters
        target_damping: Target damping ratio
        target_frequency: Target natural frequency
        max_iterations: Maximum tuning iterations
    
    Returns:
        Tuned LQR configuration
    """
    print(f"[TUNE] Auto-tuning LQR for damping={target_damping}, frequency={target_frequency}")
    
    # Start with conservative gains
    config = LQRTrainingConfig(
        position_weight=30.0,
        angle_weight=60.0,
        velocity_weight=8.0,
        control_weight=2.0,
        cost_scaling=1.0
    )
    
    best_config = config
    best_error = float('inf')
    
    for iteration in range(max_iterations):
        # Train controller with current config
        controller = train_lqr_controller(params, config, print_data=False)
        
        # Analyze performance
        stability = _analyze_stability(controller, params)
        
        if stability['damping_ratios'] and stability['natural_frequencies']:
            # Calculate error from target
            damping_error = abs(stability['damping_ratios'][0] - target_damping)
            freq_error = abs(stability['natural_frequencies'][0] - target_frequency)
            total_error = damping_error + freq_error
            
            if total_error < best_error:
                best_error = total_error
                best_config = config
            
            print(f"[TUNE] Iter {iteration+1}: damping={stability['damping_ratios'][0]:.3f}, "
                  f"freq={stability['natural_frequencies'][0]:.3f}, error={total_error:.3f}")
            
            # Early convergence
            if total_error < 0.1:
                break
        
        # Adjust parameters for next iteration
        if stability['damping_ratios'] and stability['damping_ratios'][0] < target_damping:
            config.control_weight *= 0.8  # More aggressive control
        else:
            config.control_weight *= 1.2  # More conservative control
        
        config.cost_scaling *= 0.9  # Gradually reduce scaling
    
    print(f"[TUNE] Best configuration found with error: {best_error:.3f}")
    return best_config