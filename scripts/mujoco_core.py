"""Common MuJoCo helpers: model loading, viewer setup, callbacks, logging, plotting."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    import mujoco
    import glfw  # type: ignore
except Exception:  # pragma: no cover - optional
    mujoco = None  # type: ignore
    glfw = None  # type: ignore

import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class SimConfig:
    model_path: str
    horizon: float = 10.0
    dt: float = 0.01
    paused: bool = False


def load_model(model_path: str):
    if mujoco is None:
        raise RuntimeError("MuJoCo not available")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    return model, data


def default_key_callback(window, key, scancode, action, mods):  # pragma: no cover
    if action != glfw.PRESS:
        return
    state = glfw.get_window_user_pointer(window)
    if key == glfw.KEY_SPACE:
        state.paused = not state.paused
    elif key == glfw.KEY_R:
        state.paused = True
        # Signal for external reset if needed


def run_interactive(model_path: str,
                    controller_fn: Callable[[np.ndarray, float], float],
                    out_plot: str,
                    setup_callback: Optional[Callable] = None):  # pragma: no cover
    if mujoco is None or glfw is None:
        print("MuJoCo/glfw not available; skipping interactive run.")
        return

    model, data = load_model(model_path)
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    window = glfw.create_window(1200, 900, "Cart-Pole", None, None)
    glfw.make_context_current(window)

    cfg = SimConfig(model_path=model_path)
    glfw.set_window_user_pointer(window, cfg)
    glfw.set_key_callback(window, default_key_callback)

    ts = []
    xs = []
    angles = []

    t = 0.0
    while not glfw.window_should_close(window):
        glfw.poll_events()
        if not cfg.paused:
            # Build 5-state from MuJoCo data (approximate mapping)
            x = float(data.qpos[0])
            th = float(data.qpos[1])
            xdot = float(data.qvel[0])
            thdot = float(data.qvel[1])
            state5 = np.array([x, np.cos(th), np.sin(th), xdot, thdot], dtype=np.float32)
            u = float(controller_fn(state5, t))
            data.ctrl[0] = u
            mujoco.mj_step(model, data)
            t += model.opt.timestep

            ts.append(t)
            xs.append(x)
            angles.append(th)

        mujoco.mjv_updateScene(model, data, None, None, mujoco.mjtCatBit.mjCAT_ALL.value, None)
        mujoco.mjv_defaultCamera()
        mujoco.mjr_render(mujoco.MjrRect(0, 0, 1200, 900), None, None)

        if setup_callback is not None:
            setup_callback(model, data)

        glfw.swap_buffers(window)

    glfw.terminate()
    save_plot(out_plot, np.array(ts), np.array(xs), np.array(angles))


def save_plot(path: str, ts: np.ndarray, x: np.ndarray, theta: np.ndarray):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ts, x)
    ax[0].set_title("Cart Position")
    ax[0].set_xlabel("t [s]")
    ax[0].set_ylabel("x [m]")

    ax[1].plot(ts, np.degrees(theta))
    ax[1].set_title("Pendulum Angle")
    ax[1].set_xlabel("t [s]")
    ax[1].set_ylabel("theta [deg]")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / path, dpi=150, bbox_inches="tight")
    plt.close(fig)