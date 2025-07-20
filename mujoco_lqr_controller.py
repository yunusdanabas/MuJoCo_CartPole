# mujoco_lqr_controller.py
# Description: This script implements a cart-pole swing-up controller using a linear quadratic regulator (LQR) approach.

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
import warnings
import os
import yaml

# MuJoCo + viewer
import mujoco
import mujoco_viewer

# Import your LQR helpers
from controller.lqr_controller import linearize_cartpole, compute_lqr_gain

CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    _CFG = yaml.safe_load(f) or {}

warnings.filterwarnings("ignore", category=UserWarning, module="glfw")


###############################################################################
# 1. System & LQR Configuration
###############################################################################
mc = 1.0
mp = 1.0
l = 1.0
g = 9.81
params_jax = jnp.array([mc, mp, l, g])

# Cost matrices
Q_lqr = jnp.diag(jnp.array([50.0, 100.0, 5.0, 20.0]))  # penalize x, theta, x_dot, theta_dot
R_lqr = jnp.array([[0.1]])                            # penalize input force

# Linearize around upright equilibrium
A, B = linearize_cartpole(params_jax)
K = compute_lqr_gain(A, B, Q_lqr, R_lqr)  # shape (1,4)
print("LQR gain K =", K)


###############################################################################
# 2. Load MuJoCo Model
###############################################################################
XML_FILE = os.environ.get("MODEL_XML", _CFG.get("model_xml", "cart_pole.xml"))
with open(XML_FILE, "r") as f:
    xml_str = f.read()

model = mujoco.MjModel.from_xml_string(xml_str)
data = mujoco.MjData(model)

# Optionally override MuJoCo's time step if needed:
# model.opt.timestep = 0.01

# Initialize the model
mujoco.mj_forward(model, data)

# Indices for sensor data (assuming same order as your XML):
SENSOR_CART_POS = 0
SENSOR_CART_VEL = 1
SENSOR_POLE_ANG = 2
SENSOR_POLE_ANGVEL = 3


###############################################################################
# 3. Define LQR Controller using Sensor Data
###############################################################################
def mujoco_lqr_controller(data, K):
    """
    1) Read state [x, theta, x_dot, theta_dot] from sensor data.
    2) Compute control:  u = -(K @ state).
    3) Return u as the force on the cart.
    """
    # Read sensor data
    x = data.sensordata[SENSOR_CART_POS]
    x_dot = data.sensordata[SENSOR_CART_VEL]
    theta_raw = data.sensordata[SENSOR_POLE_ANG]
    theta = ((theta_raw + jnp.pi) % (2 * jnp.pi)) - jnp.pi
    theta_dot = data.sensordata[SENSOR_POLE_ANGVEL]

    # If your MuJoCo model uses a different zero angle for upright,
    # shift the sensor reading here, e.g.:
    # theta = theta - np.pi/2   # if sensor=+1.57 rad means "upright"

    state = np.array([x, theta, x_dot, theta_dot])
    
    # LQR control law:  u = -(K state)
    force = -(K @ state)[0]
    return force


###############################################################################
# 4. Set Initial Conditions (optional)
###############################################################################
# If you only have 1 DOF for cart x and 1 DOF for pole hinge, their qpos indices might be:
data.qpos[0] = 2.5   # cart x at 0
data.qpos[5] = 0.5   # pole hinge angle (slightly tilted)
data.qvel[0] = -0.15
data.qvel[5] = -0.35

# Re-forward for consistency
mujoco.mj_forward(model, data)

###############################################################################
# 5. Simulation Loop (Real-Time)
###############################################################################
viewer = mujoco_viewer.MujocoViewer(model, data, title='Yunus Emre Danabas CartPool LQR Controller')
viewer.cam.azimuth = 90
viewer.cam.elevation = -25.0
viewer.cam.distance = 7.0


# Render the initial state so you can see it clearly
viewer.render()

# Pause for 3 seconds after the initial render
time.sleep(2)

sim_time = 0.0
sim_duration = float(os.environ.get("SIM_DURATION", _CFG.get("sim_duration", 30.0)))
start_time = time.time()

time_log = []
force_log = []
x_log = []
theta_log = []
xdot_log = []
thetadot_log = []


while True:
    current_time = time.time()
    if (current_time - start_time) > sim_duration:
        break

    step_start = current_time
    force = mujoco_lqr_controller(data, K)

    if (current_time - start_time) > 17:
        
        disturbance = -40
    
    elif (current_time - start_time) > 12:

        disturbance = 0

    elif (current_time - start_time) > 7:
        
        disturbance = 40
    
    else:

        disturbance = 0


    data.ctrl[0] = force + disturbance
    if model.nu > 1:
        data.ctrl[1] = 0.0  # zero out second actuator if unused

    # 2) Step forward
    mujoco.mj_step(model, data)
    sim_time += model.opt.timestep

    # 3) Log data
    time_log.append(sim_time)
    force_log.append(force)
    x_log.append(data.sensordata[SENSOR_CART_POS])
    theta_log.append(data.sensordata[SENSOR_POLE_ANG])
    xdot_log.append(data.sensordata[SENSOR_CART_VEL])
    thetadot_log.append(data.sensordata[SENSOR_POLE_ANGVEL])


    # 4) Render in viewer
    viewer.render()



    if not viewer.is_alive:
        break

    # 5) Real-time sync
    elapsed = time.time() - step_start
    if elapsed < model.opt.timestep:
        time.sleep(model.opt.timestep - elapsed)

viewer.close()


###############################################################################
# 6. Plot the Results
###############################################################################
time_log = np.array(time_log)
force_log = np.array(force_log)
x_log = np.array(x_log)
theta_log = np.array(theta_log)
xdot_log = np.array(xdot_log)
thetadot_log = np.array(thetadot_log)

# Force vs Time
plt.figure()
plt.plot(time_log, force_log, label="Force")
plt.title("LQR Cart Force vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.grid(True)
plt.legend()
plt.show()

# 4 Subplots for x, theta, x_dot, theta_dot
fig, axs = plt.subplots(2, 2, figsize=(10,6), sharex=True)
fig.suptitle("Cart-Pole States (LQR)")

axs[0,0].plot(time_log, x_log, 'b')
axs[0,0].set_ylabel("x (m)")
axs[0,0].set_title("Cart Position")
axs[0,0].grid(True)

axs[0,1].plot(time_log, theta_log, 'r')
axs[0,1].set_ylabel("theta (rad)")
axs[0,1].set_title("Pole Angle")
axs[0,1].grid(True)

axs[1,0].plot(time_log, xdot_log, 'g')
axs[1,0].set_xlabel("Time (s)")
axs[1,0].set_ylabel("x_dot (m/s)")
axs[1,0].set_title("Cart Velocity")
axs[1,0].grid(True)

axs[1,1].plot(time_log, thetadot_log, 'm')
axs[1,1].set_xlabel("Time (s)")
axs[1,1].set_ylabel("theta_dot (rad/s)")
axs[1,1].set_title("Pole Angular Velocity")
axs[1,1].grid(True)

plt.tight_layout()
plt.show()

print("LQR-based simulation finished.")
