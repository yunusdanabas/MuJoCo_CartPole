from __future__ import annotations

import numpy as np
import yaml
from controller.lqr_controller import LQRController
from env.cartpole import CartPoleParams
from scripts.mujoco_core import run_interactive


def _load_disturbance_schedule(path="disturbance.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config["disturbances"]


def _disturbance(t: float) -> float:
    if not hasattr(_disturbance, "_schedule"):
        _disturbance._schedule = _load_disturbance_schedule()
    for d in _disturbance._schedule:
        if d["start"] <= t < d["end"]:
            return d["value"]
    return 0.0


def main():  # pragma: no cover - interactive
    ctrl = LQRController.from_linearisation(CartPoleParams()).jit()
    controller_fn = lambda s, t: float(ctrl(np.array(s, dtype=np.float32), float(t)))
    run_interactive("cart_pole.xml", controller_fn, out_plot="lqr_mujoco.png", disturbance_fn=_disturbance)


if __name__ == "__main__":
    main()