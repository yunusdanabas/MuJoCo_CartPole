from __future__ import annotations

import numpy as np
from controller.linear_controller import create_pd_controller
from scripts.mujoco_core import run_interactive


def main():  # pragma: no cover - interactive
    ctrl = create_pd_controller().jit()
    controller_fn = lambda s, t: float(ctrl(np.array(s, dtype=np.float32), float(t)))
    run_interactive("cart_pole.xml", controller_fn, out_plot="linear_mujoco.png")


if __name__ == "__main__":
    main()