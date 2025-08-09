"""LQR 'training' utilities (single-shot Riccati solve) with logging.

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
) -> LQRController:
    """Compute LQR gain once and return controller, with optional logging."""
    start_total = time.perf_counter()
    if cfg.print_data:
        print("[TRAIN] LQRController started")

    # Linearise and solve for K
    A, B = _linearise(params)
    ctrl = LQRController.from_linearisation(params, cfg.Q, cfg.R)

    elapsed = time.perf_counter() - start_total
    if cfg.print_data:
        # Single-shot; emit one iter line
        print(f"[TRAIN] iter=0 time={elapsed:.6f}s loss=0.000000")
        print(f"[TRAIN] LQRController finished in {elapsed:.6f}s")

    return ctrl