"""Tests for advanced linear training options."""

from __future__ import annotations

import time

import jax.numpy as jnp

from lib.training.linear_training import LinearTrainingConfig, train_linear_controller


def test_advanced_training_runs_fast():
    initial_state = jnp.array([0.1, 0.95, 0.31, 0.0, 0.0])
    config = LinearTrainingConfig(
        num_iterations=5,
        trajectory_length=1.0,
        batch_size=4,
        lr_schedule="none",
        stability_weight=0.1,
        lqr_warm_start=True,
    )

    initial_K = jnp.array([1.0, -10.0, 10.0, 1.0, 1.0])
    _ = train_linear_controller(initial_K, initial_state, config)
    start = time.time()
    _, history = train_linear_controller(initial_K, initial_state, config)
    duration = time.time() - start

    assert duration < 15.0
    assert len(history.costs) > 0
    assert jnp.isfinite(jnp.array(history.costs)).all()


__all__ = ["test_advanced_training_runs_fast"]

