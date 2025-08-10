"""
scripts/linear_mujoco.py
Interactive script to run a linear controller on a MuJoCo CartPole environment.
"""


from __future__ import annotations

import mujoco  # used inside setup callback
import numpy as np
import yaml
from controller.linear_controller import create_pd_controller
from scripts.mujoco_core import run_interactive


def _load_disturbance_schedule(path="disturbance.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config["disturbances"]


def _disturbance(t: float) -> float:
    # Load once and cache
    if not hasattr(_disturbance, "_schedule"):
        _disturbance._schedule = _load_disturbance_schedule()
    for d in _disturbance._schedule:
        if d["start"] <= t < d["end"]:
            return d["value"]
    return 0.0


def _init_state(model, data):
    """Set a reproducible initial condition (similar to your old script)."""
    # Safe-guard against models with different joint indexing: assume qpos[0]=cart, qpos[1]=pole
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    if model.nq >= 2:
        data.qpos[0] = 0.0     # cart x
        data.qpos[1] = 0.5     # pole angle (rad)
    if model.nv >= 2:
        data.qvel[0] = -0.15   # cart xdot
        data.qvel[1] = -0.35   # pole thdot
    mujoco.mj_forward(model, data)


def main():  # pragma: no cover - interactive
    ctrl = create_pd_controller().jit()
    controller_fn = lambda s, t: float(ctrl(np.array(s, dtype=np.float32), float(t)))
    run_interactive(
        "cart_pole.xml",
        controller_fn,
        out_plot="linear_mujoco.png",
        disturbance_fn=_disturbance,
        setup_callback=_init_state,
    )


if __name__ == "__main__":
    main()