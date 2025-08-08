# mujoco_linear_control.py
# Description: This script implements a linear controller for a cart-pole system using JAX and MuJoCo.

import jax
import jax.numpy as jnp
import optax
import numpy as np
import time
import warnings
import os
import sys
from pathlib import Path
import yaml
from collections import deque
import pickle

sys.path.append(str(Path(__file__).resolve().parents[1]))

import mujoco
try:
    import mujoco_viewer
except ImportError:
    print("Warning: mujoco_viewer not found. Running in headless mode.")
    mujoco_viewer = None

from controller.linear_controller import LinearController
from lib.training.linear_training import train_linear_controller

# Set up configuration
CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.yaml")
try:
    with open(CONFIG_PATH, "r") as f:
        _CFG = yaml.safe_load(f) or {}
except FileNotFoundError:
    print(f"Warning: Config file {CONFIG_PATH} not found. Using defaults.")
    _CFG = {}

warnings.filterwarnings("ignore", category=UserWarning, module="glfw")


def find_sensor_indices(model):
    """Find sensor indices by name to avoid hard-coding."""
    sensor_map = {}
    for i in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        sensor_map[name] = i
    
    # Expected sensor names - adjust based on your XML
    indices = {
        'cart_pos': sensor_map.get('cart_pos', 0),
        'cart_vel': sensor_map.get('cart_vel', 1), 
        'pole_ang': sensor_map.get('pole_ang', 2),
        'pole_angvel': sensor_map.get('pole_angvel', 3)
    }
    return indices


def disturbance_schedule(elapsed_time):
    """Compute disturbance force based on simulation time."""
    if elapsed_time > 17:
        return -20
    elif elapsed_time > 12:
        return 0
    elif elapsed_time > 7:
        return 20
    else:
        return 0


# JIT-compiled step function for high-frequency control
@jax.jit
def control_step(state_vec, controller):
    """JIT-compiled control computation for minimal latency."""
    return controller(state_vec, 0.0)


def add_debug_info(viewer, data, force, disturbance):
    """Add debug information to the viewer."""
    # Skip if viewer doesn't support overlays
    if not hasattr(viewer, 'add_overlay'):
        return
    
    # Add info
    viewer.add_overlay(mujoco.mjtGridPos.mjGRID_TOPLEFT, 
                      "Linear Controller",
                      f"Time: {data.time:.2f}s\nForce: {force:.2f}N\nDisturbance: {disturbance:.1f}N")


def main():
    ###############################################################################
    # 1. TRAIN OR LOAD THE LINEAR CONTROLLER
    ###############################################################################
    from env.cartpole import CartPoleParams
    from lib.training.linear_training import LinearTrainingConfig, create_cost_matrices
    
    # Cache file path for controller weights
    cache_path = Path(__file__).resolve().parent / "cached_controller.pkl"
    ctrl = None
    cost_history = []
    
    # Try to load cached controller
    retrain = bool(os.environ.get("RETRAIN", _CFG.get("retrain", False)))
    if not retrain and cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
                ctrl_K = cached_data.get("K")
                cost_history = cached_data.get("cost_history", [])
                print(f"\n=== Loaded cached controller with weights: {ctrl_K} ===")
                ctrl = LinearController(K=jnp.array(ctrl_K))
        except Exception as e:
            print(f"Failed to load cached controller: {e}")
            ctrl = None
    
    # Train controller if needed
    if ctrl is None:
        # Create CartPole parameters
        params = CartPoleParams(mc=1.0, mp=1.0, l=1.0, g=9.81)

        # Create cost matrix
        Q = create_cost_matrices(pos_weight=50.0, angle_weight=300.0, vel_weight=5.0, angvel_weight=20.0)

        # Generate a single initial condition for training
        key = jax.random.PRNGKey(42)
        initial_state = jnp.array([0.1, 0.1, 0.0, 0.0])  # Small perturbation from equilibrium

        # Training hyperparameters
        lin_cfg = _CFG.get("linear_training", {})
        # Option A: user supplies explicit K
        initial_K = None          # set to vector to BYPASS warm-start

        config = LinearTrainingConfig(
            learning_rate=lin_cfg.get('lr', 0.01),
            num_iterations=lin_cfg.get('max_iters', 300),
            trajectory_length=3.0,
            state_weight=1.0,
            control_weight=0.1,
            convergence_tol=lin_cfg.get('tolerance', 1e-6),
            batch_size=lin_cfg.get('batch_size', 32),
            lr_schedule='cosine',
            stability_weight=lin_cfg.get('stability_weight', 0.0),
            perturb_std=lin_cfg.get('perturb_std', 0.05),
            seed=lin_cfg.get('seed', 0),
            lqr_warm_start=lin_cfg.get('lqr_warm_start', True),
        )

        print("\n=== Training the Linear Controller ===")
        ctrl, cost_history = train_linear_controller(
            initial_K,                    # None → optional warm-start
            jnp.zeros(4),
            config,
            Q,
            params,
        )
        print("\nTraining completed.")
        print("Optimized Weights:", ctrl.K)
        
        # Cache controller weights
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "K": np.array(ctrl.K),
                    "cost_history": cost_history
                }, f)
            print(f"Saved controller weights to {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save controller: {e}")

    ###############################################################################
    # 2. CREATE CONTROLLER AND SETUP
    ###############################################################################
    
    # JIT-compile the controller once
    ctrl = ctrl.jit()
    
    # Create a closure for control_step that only depends on state
    control_step = jax.jit(ctrl.__call__)  # No "controller" arg, no retracing
    
    # MuJoCo setup
    XML_FILE = os.environ.get("MODEL_XML", _CFG.get("model_xml", "cart_pole_minimal.xml"))
    
    # Improve XML file path resolution
    project_root = Path(__file__).resolve().parents[1]
    possible_paths = [
        Path(XML_FILE),                  # Direct path as provided
        project_root / "models" / XML_FILE,  # Check in models directory
        project_root / XML_FILE,         # Check in project root
    ]
    
    xml_str = None
    for path in possible_paths:
        try:
            with open(path, "r") as f:
                xml_str = f.read()
                print(f"Found model XML at: {path}")
                break
        except FileNotFoundError:
            continue
            
    if xml_str is None:
        raise FileNotFoundError(f"Could not find model XML file '{XML_FILE}'. "
                               f"Tried paths: {[str(p) for p in possible_paths]}")

    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Find sensor indices dynamically
    sensor_idx = find_sensor_indices(model)

    # Determine DOF indices based on model type
    cart_pos_idx = 0  # Cart is typically the first DOF
    pole_ang_idx = 1 if "minimal" in str(XML_FILE) else 5  # Different in full vs minimal model

    # Pre-allocate logs using deque for efficiency
    max_steps = int(_CFG.get("sim_duration", 30.0) / model.opt.timestep) + 100
    time_log = deque(maxlen=max_steps)
    force_log = deque(maxlen=max_steps)
    x_log = deque(maxlen=max_steps)
    theta_log = deque(maxlen=max_steps)
    xdot_log = deque(maxlen=max_steps)
    thetadot_log = deque(maxlen=max_steps)

    # Camera Settings with fallback
    try:
        if mujoco_viewer is None:
            raise ImportError("mujoco_viewer module not available")
            
        viewer = mujoco_viewer.MujocoViewer(model, data, title='Yunus Emre Danabas CartPole Linear Controller')
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -25.0
        viewer.cam.distance = 7.0
        has_viewer = True
    except Exception as e:
        print(f"Warning: Could not initialize MuJoCo viewer: {e}")
        print("Running in headless mode (no visualization)")
        viewer = None
        has_viewer = False

    sim_time = 0.0
    sim_duration = float(os.environ.get("SIM_DURATION", _CFG.get("sim_duration", 30.0)))
    start_time = time.time()

    # Set initial conditions
    data.qpos[cart_pos_idx] = 2.5       # Cart x position
    data.qpos[pole_ang_idx] = 0.5       # Pole hinge angle (theta) rad
    data.qvel[cart_pos_idx] = -0.15     # Cart x_dot
    data.qvel[pole_ang_idx] = -0.35     # Pole theta_dot
    mujoco.mj_forward(model, data)

    ###############################################################################
    # 3. MAIN SIMULATION LOOP
    ###############################################################################

    def mujoco_linear_controller(data):
        """
        Unified controller using the same LinearController interface.
        Returns force and state components for logging.
        """
        # Read sensor data
        x = data.sensordata[sensor_idx['cart_pos']]
        x_dot = data.sensordata[sensor_idx['cart_vel']]
        raw_theta = data.sensordata[sensor_idx['pole_ang']]
        theta_dot = data.sensordata[sensor_idx['pole_angvel']]
        
        # Wrap angle to [-pi, pi] using JAX math to keep computation on device
        theta = jnp.mod(raw_theta + jnp.pi, 2 * jnp.pi) - jnp.pi
        
        # Create state vector and compute force using the trained controller
        state_vec = jnp.array([x, theta, x_dot, theta_dot])
        force = float(control_step(state_vec, 0.0))
        
        return force, x, theta, x_dot, theta_dot

    print("\n=== Starting MuJoCo Simulation ===")
    
    # Render the initial state if viewer available
    if has_viewer:
        viewer.render()
        time.sleep(2)  # Pause for 2 seconds to show initial state
    
    while True:
        current_time = time.time()
        elapsed = current_time - start_time
        
        if elapsed > sim_duration:
            break

        step_start = current_time
        
        # 1) Compute control force
        force, x, theta, x_dot, theta_dot = mujoco_linear_controller(data)
        
        # 2) Apply disturbance and control
        disturbance = disturbance_schedule(elapsed)
        total_force = force + disturbance
        data.ctrl[0] = total_force
        
        # Add debug info overlay if available
        if has_viewer:
            add_debug_info(viewer, data, force, disturbance)
        
        # Zero out additional actuators if present
        if model.nu > 1:
            data.ctrl[1:] = 0.0

        # 3) Step the simulation
        mujoco.mj_step(model, data)
        sim_time += model.opt.timestep

        # 4) Log data efficiently
        time_log.append(sim_time)
        force_log.append(force)
        x_log.append(x)
        theta_log.append(theta)
        xdot_log.append(x_dot)
        thetadot_log.append(theta_dot)

        # 5) Render if viewer available
        if has_viewer:
            viewer.render()
            if not viewer.is_alive:
                break

        # 6) Real-time sync
        elapsed_step = time.time() - step_start
        if elapsed_step < model.opt.timestep:
            time.sleep(model.opt.timestep - elapsed_step)

    if has_viewer:
        viewer.close()

    ###############################################################################
    # 4. PLOTTING
    ###############################################################################
    
    # Convert deques to numpy arrays for plotting
    time_array = np.array(time_log)
    force_array = np.array(force_log)
    x_array = np.array(x_log)
    theta_array = np.array(theta_log)
    xdot_array = np.array(xdot_log)
    thetadot_array = np.array(thetadot_log)

    try:
        # Import matplotlib only when needed for plotting
        import matplotlib.pyplot as plt

        # Plot training cost
        plt.figure()
        plt.plot(np.array(cost_history), marker='o')
        plt.title("Linear Controller Training Cost")
        plt.xlabel("Iteration (checkpoints)")
        plt.ylabel("Cost")
        plt.grid(True)
        plt.show()

        # Plot control force
        plt.figure()
        plt.plot(time_array, force_array, 'b')
        plt.title("Linear Control Force vs Time (MuJoCo)")
        plt.xlabel("Time (s)")
        plt.ylabel("Force (N)")
        plt.grid(True)
        plt.show()

        # Plot system states
        fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
        fig.suptitle("Cart-Pole States (MuJoCo w/ Trained Linear Controller)")

        axs[0, 0].plot(time_array, x_array, 'b')
        axs[0, 0].set_title("Cart Position x(t)")
        axs[0, 0].set_ylabel("x (m)")
        axs[0, 0].grid(True)

        axs[0, 1].plot(time_array, theta_array, 'r')
        axs[0, 1].set_title("Pole Angle theta(t)")
        axs[0, 1].set_ylabel("Angle (rad)")
        axs[0, 1].grid(True)

        axs[1, 0].plot(time_array, xdot_array, 'g')
        axs[1, 0].set_xlabel("Time (s)")
        axs[1, 0].set_ylabel("x_dot (m/s)")
        axs[1, 0].set_title("Cart Velocity")
        axs[1, 0].grid(True)

        axs[1, 1].plot(time_array, thetadot_array, 'm')
        axs[1, 1].set_xlabel("Time (s)")
        axs[1, 1].set_ylabel("theta_dot (rad/s)")
        axs[1, 1].set_title("Pole Angular Velocity")
        axs[1, 1].grid(True)

        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Warning: matplotlib not available. Skipping plots.")
        # Save data to CSV instead
        np.savetxt('cartpole_simulation_results.csv', 
                  np.column_stack((time_array, x_array, theta_array, xdot_array, thetadot_array, force_array)),
                  delimiter=',',
                  header='time,x,theta,x_dot,theta_dot,force')
        print("Data saved to cartpole_simulation_results.csv")
    
    print("MuJoCo linear-controller simulation finished.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulation terminated by user.")
    except Exception as e:
        print(f"\nError: {e}")
        raise
