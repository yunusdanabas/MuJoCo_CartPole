from __future__ import annotations

import numpy as np
from controller.lqr_controller import LQRController
from env.cartpole import CartPoleParams
from scripts.mujoco_core import run_interactive


def main():  # pragma: no cover - interactive
    ctrl = LQRController.from_linearisation(CartPoleParams()).jit()
    controller_fn = lambda s, t: float(ctrl(np.array(s, dtype=np.float32), float(t)))
    run_interactive("cart_pole.xml", controller_fn, out_plot="lqr_mujoco.png")


if __name__ == "__main__":
    main()