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

# Import plotting function from visualizer
from lib.visualizer import plot_mujoco_simulation

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class SimConfig:
    """Configuration for MuJoCo simulation."""
    model_path: str
    horizon: float = 45.0
    dt: float = 0.01    
    paused: bool = False
    should_quit: bool = False
    should_reset: bool = False


def load_model(model_path: str):
    """Load MuJoCo model and data."""
    if mujoco is None:
        raise RuntimeError("MuJoCo not available")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    return model, data


def default_key_callback(window, key, scancode, action, mods):  # pragma: no cover
    """Handle keyboard input for GLFW window."""
    if action != glfw.PRESS:
        return
    state = glfw.get_window_user_pointer(window)
    if key == glfw.KEY_SPACE:
        state.paused = not state.paused
    elif key == glfw.KEY_R:
        state.should_reset = True
    elif key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
        state.should_quit = True


def _make_scene_objects(model):  # pragma: no cover - interactive only
    """Create MuJoCo scene objects for GLFW rendering."""
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultCamera(cam)
    mujoco.mjv_defaultOption(opt)

    scene = mujoco.MjvScene(model, maxgeom=2000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    return cam, opt, scene, context


def _draw_disturbance_text(viewer, disturbance_value: float, t: float):
    """Draw disturbance information as text overlay in MuJoCo viewer."""
    try:
        # Format disturbance value with sign
        if abs(disturbance_value) < 0.01:
            dist_text = "Dist: 0.00"
        else:
            dist_text = f"Dist: {disturbance_value:+.2f}"
        
        # Add time information
        time_text = f"Time: {t:.1f}s"
        
        # Draw text overlay in top-left corner
        viewer.add_overlay(
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            f"{dist_text} | {time_text}"
        )
    except Exception:
        # Fallback if overlay not available
        pass


def run_interactive(model_path: str,
                    controller_fn: Callable[[np.ndarray, float], float],
                    out_plot: str,
                    setup_callback: Optional[Callable[[object, object], None]] = None,
                    *,
                    disturbance_fn: Optional[Callable[[float], float]] = None,
                    debug_mode: bool = False):  # pragma: no cover
    """
    Run interactive MuJoCo simulation with logging and plotting.
    
    Args:
        model_path: Path to MuJoCo XML model
        controller_fn: Function that takes 5-state [x, cosθ, sinθ, ẋ, θ̇] and time t -> force u
        out_plot: Output plot filename
        setup_callback: Called at start and on resets to set qpos/qvel
        disturbance_fn: Optional external force function
        debug_mode: Skip training and use pre-determined gains
    """
    if mujoco is None:
        print("MuJoCo not available; skipping interactive run.")
        return

    model, data = load_model(model_path)

    # ------------------------ Fast path: built-in viewer --------------------- #
    if mjviewer is not None:
        # Initialize logging arrays
        ts, xs, thetas, xdots, thdots, us, ds = [], [], [], [], [], [], []
        dt = float(model.opt.timestep)
        t = 0.0

        # Set initial conditions
        if setup_callback is not None:
            setup_callback(model, data)
            mujoco.mj_forward(model, data)

        with mjviewer.launch_passive(model, data) as viewer:
            # Set camera position for better view
            try:
                viewer.cam.distance = 10.0
                viewer.cam.elevation = -20.0
                viewer.cam.azimuth = 90.0
            except Exception:
                pass

            start = time.time()
            while viewer.is_running():
                step_start = time.time()

                # Auto-reset at horizon limit
                if t >= SimConfig(model_path).horizon:
                    mujoco.mj_resetData(model, data)
                    if setup_callback is not None:
                        setup_callback(model, data)
                        mujoco.mj_forward(model, data)
                    t = 0.0
                    viewer.sync()
                    continue

                # Extract current state from MuJoCo
                x = float(data.qpos[0])
                th = float(data.qpos[1])
                xdot = float(data.qvel[0])
                thdot = float(data.qvel[1])
                state5 = np.array([x, np.cos(th), np.sin(th), xdot, thdot], dtype=np.float32)

                # Apply controller and disturbance
                u = float(controller_fn(state5, t))
                d = float(disturbance_fn(t)) if disturbance_fn is not None else 0.0
                # Ensure control is within actuator limits (u_max enforced in controller)
                data.ctrl[0] = u + d

                # Show disturbance info in viewer
                if disturbance_fn is not None:
                    _draw_disturbance_text(viewer, d, t)

                # Step simulation
                mujoco.mj_step(model, data)
                t += dt

                # Log data for plotting
                ts.append(t); xs.append(x); thetas.append(th)
                xdots.append(xdot); thdots.append(thdot); us.append(u); ds.append(d)

                # Render and maintain real-time
                viewer.sync()
                time.sleep(max(0.0, dt - (time.time() - step_start)))

        # Generate final plot
        plot_mujoco_simulation(
            np.array(ts), np.array(xs), np.array(thetas),
            np.array(xdots), np.array(thdots),
            np.array(us), np.array(ds),
            save_path=out_plot,
            show_plot=True
        )
        return

    # ------------------------ Fallback: GLFW pipeline ------------------------ #
    if glfw is None:
        print("Neither mujoco.viewer nor glfw available; install `mujoco` (>=3) or `glfw`.")
        return

    # Initialize GLFW with larger default window
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    
    window = glfw.create_window(1600, 1200, "Cart-Pole Simulation", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
    
    glfw.make_context_current(window)
    glfw.swap_interval(1)  # Enable vsync

    # Setup simulation configuration
    cfg = SimConfig(model_path=model_path)
    glfw.set_window_user_pointer(window, cfg)
    glfw.set_key_callback(window, default_key_callback)

    # Create scene objects
    cam, opt, scene, context = _make_scene_objects(model)

    # Initialize simulation
    if setup_callback is not None:
        setup_callback(model, data)
        mujoco.mj_forward(model, data)

    # Initialize logging arrays
    ts: list[float] = []
    xs: list[float] = []
    thetas: list[float] = []
    xdots: list[float] = []
    thdots: list[float] = []
    us: list[float] = []
    ds: list[float] = []

    # Simulation parameters
    dt = float(model.opt.timestep)
    t = 0.0

    # Main simulation loop
    while not glfw.window_should_close(window):
        glfw.poll_events()

        if cfg.should_quit:
            break

        # Handle reset
        if cfg.should_reset:
            mujoco.mj_resetData(model, data)
            if setup_callback is not None:
                setup_callback(model, data)
                mujoco.mj_forward(model, data)
            # Clear logged data
            ts.clear(); xs.clear(); thetas.clear(); xdots.clear(); thdots.clear(); us.clear(); ds.clear()
            t = 0.0
            cfg.should_reset = False

        # Simulation step
        if not cfg.paused:
            # Extract current state
            x = float(data.qpos[0])
            th = float(data.qpos[1])
            xdot = float(data.qvel[0])
            thdot = float(data.qvel[1])
            state5 = np.array([x, np.cos(th), np.sin(th), xdot, thdot], dtype=np.float32)

            # Apply controller and disturbance
            u = float(controller_fn(state5, t))
            d = float(disturbance_fn(t)) if disturbance_fn is not None else 0.0

            # Ensure control is within actuator limits (u_max enforced in controller)
            data.ctrl[0] = u + d
            mujoco.mj_step(model, data)
            t += dt

            # Log data
            ts.append(t); xs.append(x); thetas.append(th)
            xdots.append(xdot); thdots.append(thdot); us.append(u); ds.append(d)

        # Render scene
        width, height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(viewport, scene, context)

        # Update window title with disturbance info
        if disturbance_fn is not None:
            d = float(disturbance_fn(t))
            if abs(d) > 0.01:
                title = f"Cart-Pole | Dist: {d:+.2f} | Time: {t:.1f}s"
            else:
                title = f"Cart-Pole | Dist: 0.00 | Time: {t:.1f}s"
            glfw.set_window_title(window, title)

        glfw.swap_buffers(window)

    # Cleanup and generate plot
    glfw.terminate()
    plot_mujoco_simulation(
        np.array(ts), np.array(xs), np.array(thetas),
        np.array(xdots), np.array(thdots),
        np.array(us), np.array(ds),
        save_path=out_plot,
        show_plot=True
    )


