"""
scripts/mujoco_core.py
Common MuJoCo helpers: model loading, viewer setup, callbacks, logging, plotting.
"""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
import time

import numpy as np

try:
    import mujoco
    import glfw  # type: ignore
except Exception:  # pragma: no cover - optional
    mujoco = None  # type: ignore
    glfw = None  # type: ignore

# Prefer MuJoCo's built-in viewer if available (no GLFW dependency needed)
try:  # mujoco>=3
    import mujoco.viewer as mjviewer  # type: ignore
except Exception:
    mjviewer = None  # type: ignore

import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class SimConfig:
    model_path: str
    horizon: float = 10.0
    dt: float = 0.01    
    paused: bool = False
    should_quit: bool = False
    should_reset: bool = False
    disturbance: float = 0.0
    last_control: float = 0.0


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
        state.should_reset = True
    elif key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
        state.should_quit = True
    elif key == glfw.KEY_LEFT:
        state.disturbance -= 15.0
    elif key == glfw.KEY_RIGHT:
        state.disturbance += 15.0
    elif key == glfw.KEY_UP:
        # Nudge pole angle +15 deg
        # We do not have access to model/data here; set a flag and handle next frame
        state._nudge_up = True
    elif key == glfw.KEY_DOWN:
        state._nudge_down = True
    elif key == glfw.KEY_C:
        state.disturbance = 0.0
    elif key == glfw.KEY_P:
        state.paused = not state.paused


def _make_scene_objects(model):  # pragma: no cover - interactive only
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultCamera(cam)
    mujoco.mjv_defaultOption(opt)

    scene = mujoco.MjvScene(model, maxgeom=2000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    return cam, opt, scene, context


def _render_overlay(context, viewport, gridpos, left: str, right: str):
    mujoco.mjr_overlay(
        mujoco.mjtFontScale.mjFONTSCALE_150,
        gridpos,
        viewport,
        left,
        right,
        context,
    )


def run_interactive(model_path: str,
                    controller_fn: Callable[[np.ndarray, float], float],
                    out_plot: str,
                    setup_callback: Optional[Callable[[object, object], None]] = None,
                    *,
                    disturbance_fn: Optional[Callable[[float], float]] = None,
                    prefer_glfw: bool = True):  # pragma: no cover
    """
    Interactive sim with logging and a saved plot.
    Prefers mujoco.viewer (no GLFW req). Falls back to raw GLFW renderer.
    - controller_fn: expects 5-state [x, cosθ, sinθ, ẋ, θ̇] and time t -> force u
    - setup_callback(model, data): called once at start and on resets (set qpos/qvel, etc.)
    - disturbance_fn(t): optional external force added to u
    """
    if mujoco is None:
        print("MuJoCo not available; skipping interactive run.")
        return

    model, data = load_model(model_path)

    # ------------------------ Fast path: built-in viewer --------------------- #
    if mjviewer is not None and not prefer_glfw:
        # Logs
        ts, xs, thetas, xdots, thdots, us, ds = [], [], [], [], [], [], []
        dt = float(model.opt.timestep)
        t = 0.0
        user_disturbance = 0.0

        # Initial conditions
        if setup_callback is not None:
            setup_callback(model, data)
            mujoco.mj_forward(model, data)

        with mjviewer.launch_passive(model, data) as viewer:
            # Optional camera convenience (helps when XML has odd defaults)
            try:
                viewer.cam.distance = 4.0
                viewer.cam.elevation = -20.0
                viewer.cam.azimuth = 90.0
            except Exception:
                pass

            start = time.time()
            while viewer.is_running():
                step_start = time.time()

                if t >= SimConfig(model_path).horizon:
                    mujoco.mj_resetData(model, data)
                    if setup_callback is not None:
                        setup_callback(model, data)
                        mujoco.mj_forward(model, data)
                    t = 0.0
                    viewer.sync()
                    continue

                # Build 5-state from MuJoCo
                x = float(data.qpos[0])
                th = float(data.qpos[1])
                xdot = float(data.qvel[0])
                thdot = float(data.qvel[1])
                state5 = np.array([x, np.cos(th), np.sin(th), xdot, thdot], dtype=np.float32)

                u = float(controller_fn(state5, t))
                sched_d = float(disturbance_fn(t)) if disturbance_fn is not None else 0.0
                d = sched_d + user_disturbance
                data.ctrl[0] = u + d

                mujoco.mj_step(model, data)
                t += dt

                # Overlay text (limited in passive viewer; we can only print via console or ignore)
                # We keep internal tracking only; overlays are fully implemented in GLFW branch.

                # Log
                ts.append(t); xs.append(x); thetas.append(th)
                xdots.append(xdot); thdots.append(thdot); us.append(u); ds.append(d)

                # Render & keep real-time
                viewer.sync()
                time.sleep(max(0.0, dt - (time.time() - step_start)))

        save_plot(out_plot,
                  np.array(ts), np.array(xs), np.array(thetas),
                  np.array(xdots), np.array(thdots),
                  np.array(us), np.array(ds))
        return

    # ------------------------ Fallback: GLFW pipeline ------------------------ #
    if glfw is None:
        print("Neither mujoco.viewer nor glfw available; install `mujoco` (>=3) or `glfw`.")
        return

    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    window = glfw.create_window(1200, 900, "Cart-Pole", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)  # vsync

    cfg = SimConfig(model_path=model_path)
    glfw.set_window_user_pointer(window, cfg)
    # Initialize nudge flags
    setattr(cfg, "_nudge_up", False)
    setattr(cfg, "_nudge_down", False)
    glfw.set_key_callback(window, default_key_callback)

    cam, opt, scene, context = _make_scene_objects(model)

    # Call user setup once now
    if setup_callback is not None:
        setup_callback(model, data)
        mujoco.mj_forward(model, data)

    ts: list[float] = []
    xs: list[float] = []
    thetas: list[float] = []
    xdots: list[float] = []
    thdots: list[float] = []
    us: list[float] = []
    ds: list[float] = []

    # Simulation timestep from model
    dt = float(model.opt.timestep)
    t = 0.0

    while not glfw.window_should_close(window):
        glfw.poll_events()

        if cfg.should_quit:
            break

        if cfg.should_reset:
            mujoco.mj_resetData(model, data)
            # Re-run user setup after reset
            if setup_callback is not None:
                setup_callback(model, data)
                mujoco.mj_forward(model, data)
            ts.clear(); xs.clear(); thetas.clear(); xdots.clear(); thdots.clear(); us.clear(); ds.clear()
            t = 0.0
            cfg.should_reset = False

        if not cfg.paused:
            # Optional angle nudge
            if getattr(cfg, "_nudge_up", False) and model.nq >= 2:
                data.qpos[1] += np.deg2rad(15.0)
                cfg._nudge_up = False
                mujoco.mj_forward(model, data)
            if getattr(cfg, "_nudge_down", False) and model.nq >= 2:
                data.qpos[1] -= np.deg2rad(15.0)
                cfg._nudge_down = False
                mujoco.mj_forward(model, data)

            # Build 5-state from MuJoCo data
            x = float(data.qpos[0])
            th = float(data.qpos[1])
            xdot = float(data.qvel[0])
            thdot = float(data.qvel[1])
            state5 = np.array([x, np.cos(th), np.sin(th), xdot, thdot], dtype=np.float32)

            u = float(controller_fn(state5, t))
            sched_d = float(disturbance_fn(t)) if disturbance_fn is not None else 0.0
            d = sched_d + cfg.disturbance

            cfg.last_control = u
            data.ctrl[0] = u + d
            mujoco.mj_step(model, data)
            t += dt

            ts.append(t)
            xs.append(x)
            thetas.append(th)
            xdots.append(xdot)
            thdots.append(thdot)
            us.append(u)
            ds.append(d)

        # Render
        width, height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(viewport, scene, context)

        # Overlays: hints + status
        left_text = (
            "Controls:\n"
            "SPACE: pause/resume\n"
            "R: reset\n"
            "Q/ESC: quit\n"
            "LEFT/RIGHT: disturbance -15/+15 N\n"
            "UP/DOWN: nudge pole angle ±15°\n"
            "C: clear disturbance\n"
        )
        right_text = (
            f"t = {t:6.2f} s\n"
            f"u = {cfg.last_control:7.2f} N\n"
            f"d = {cfg.disturbance:7.2f} N\n"
            f"θ = {np.degrees(float(data.qpos[1])):6.2f} deg\n"
            f"x = {float(data.qpos[0]):6.2f} m\n"
        )
        _render_overlay(context, viewport, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, left_text, " ")
        _render_overlay(context, viewport, mujoco.mjtGridPos.mjGRID_TOPRIGHT, "Status", right_text)

        glfw.swap_buffers(window)

    glfw.terminate()
    save_plot(out_plot,
              np.array(ts), np.array(xs), np.array(thetas),
              np.array(xdots), np.array(thdots),
              np.array(us), np.array(ds))


def save_plot(path: str,
              ts: np.ndarray,
              x: np.ndarray,
              theta: np.ndarray,
              xdot: np.ndarray,
              thdot: np.ndarray,
              u: np.ndarray,
              d: np.ndarray):
    # Create a 2x3 grid of plots
    fig, ax = plt.subplots(2, 3, figsize=(14, 6))

    ax[0, 0].plot(ts, x)
    ax[0, 0].set_title("Cart Position")
    ax[0, 0].set_xlabel("t [s]")
    ax[0, 0].set_ylabel("x [m]")

    ax[0, 1].plot(ts, np.degrees(theta))
    ax[0, 1].set_title("Pendulum Angle")
    ax[0, 1].set_xlabel("t [s]")
    ax[0, 1].set_ylabel("theta [deg]")

    ax[0, 2].plot(ts, xdot)
    ax[0, 2].set_title("Cart Velocity")
    ax[0, 2].set_xlabel("t [s]")
    ax[0, 2].set_ylabel("xdot [m/s]")

    ax[1, 0].plot(ts, thdot)
    ax[1, 0].set_title("Angular Velocity")
    ax[1, 0].set_xlabel("t [s]")
    ax[1, 0].set_ylabel("thdot [rad/s]")

    ax[1, 1].plot(ts, u)
    ax[1, 1].set_title("Control Force (u)")
    ax[1, 1].set_xlabel("t [s]")
    ax[1, 1].set_ylabel("N")

    ax[1, 2].plot(ts, d)
    ax[1, 2].set_title("Disturbance (d)")
    ax[1, 2].set_xlabel("t [s]")
    ax[1, 2].set_ylabel("N")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / path, dpi=150, bbox_inches="tight")
    plt.close(fig)