"""
lib/training/lqr_training.py
LQR 'training' utilities (single-shot Riccati solve) with logging.

This module matches the logging interface used by other trainers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
import jax.numpy as jnp
from controller.lqr_controller import LQRController, _linearise
from env.cartpole import CartPoleParams

__all__ = ["LQRTrainingConfig", "train_lqr_controller"]


@dataclass
class LQRTrainingConfig:
    Q: jnp.ndarray = field(default_factory=lambda: jnp.diag(jnp.array([10.0, 10.0, 1.0, 1.0])))
    R: jnp.ndarray = field(default_factory=lambda: jnp.array([[0.1]]))
    print_data: bool = False


def train_lqr_controller(
    params: CartPoleParams = CartPoleParams(),
    cfg: LQRTrainingConfig = LQRTrainingConfig(),
    *,
    print_data: bool = True,
) -> LQRController:
    """Compute LQR gain once and return controller, with optional logging."""
    # Sync logging flag
    cfg.print_data = bool(print_data)

    start_total = time.perf_counter()
    if cfg.print_data:
        print("[TRAIN] LQRController started")
        print("[TRAIN] LQR Parameters:")
        np_Q = jnp.asarray(cfg.Q)
        np_R = jnp.asarray(cfg.R)
        print("  Q:")
        print(np_Q)
        print("  R:")
        print(np_R)

    # Linearise and solve for K
    A, B = _linearise(params)
    ctrl = LQRController.from_linearisation(params, cfg.Q, cfg.R)

    elapsed = time.perf_counter() - start_total
    if cfg.print_data:
        print(f"[TRAIN] LQRController finished in {elapsed:.6f}s")

    return ctrl