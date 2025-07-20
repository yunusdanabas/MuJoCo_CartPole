# mujoco_linear_control.py
# Description: This script implements a linear controller for a cart-pole system using JAX and MuJoCo.

import jax
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import os
import yaml

import mujoco
import mujoco_viewer

from controller.linear_controller import train_linear_controller

CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    _CFG = yaml.safe_load(f) or {}

warnings.filterwarnings("ignore", category=UserWarning, module="glfw")


###############################################################################
# 1. TRAIN THE LINEAR CONTROLLER IN JAX
###############################################################################
mc = 1.0
mp = 1.0
l = 1.0
g = 9.81
params_jax = jnp.array([mc, mp, l, g])

# State cost matrix
Q = jnp.diag(jnp.array([50.0, 300.0, 5.0, 20.0]))

# Time settings for the JAX simulation used in training
t_start = 0.0
t_end = 3.0
num_time_steps = 300
t_eval = jnp.linspace(t_start, t_end, num_time_steps)
t_span = (t_start, t_end)

# Generate small random initial states for training
key = jax.random.PRNGKey(42)
def sample_ic(key, N=10):
    x_range = (-0.1, 0.1)
    th_range = (-0.2, 0.2)
    xd_range = (-0.2, 0.2)
    thd_range = (-0.2, 0.2)
    keys = jax.random.split(key, 4)
    x_init = jax.random.uniform(keys[0], (N,), minval=x_range[0], maxval=x_range[1])
    th_init = jax.random.uniform(keys[1], (N,), minval=th_range[0], maxval=th_range[1])
    xd_init = jax.random.uniform(keys[2], (N,), minval=xd_range[0], maxval=xd_range[1])
    thd_init = jax.random.uniform(keys[3], (N,), minval=thd_range[0], maxval=thd_range[1])
    return jnp.stack([x_init, th_init, xd_init, thd_init], axis=1)

initial_conditions = sample_ic(key, 10)

# Training hyperparameters
lin_cfg = _CFG.get("linear_training", {})
train_opts = {
    'lr': lin_cfg.get('lr', 1e-4),
    'w_init': lin_cfg.get('w_init', [
        1.07477045e+01,
        -1.42533379e-03,
        -9.53456802e+01,
        1.32174416e+01,
        -2.78421364e+01
    ]),
    'max_iters': lin_cfg.get('max_iters', 100),
    'tolerance': lin_cfg.get('tolerance', 1e-6),
}

print("\n=== Training the Linear Controller in JAX ===")
w_opt, cost_history = train_linear_controller(
    params_jax,
    t_span,
    t_eval,
    initial_conditions,
    Q,
    train_opts
)
print("\nTraining completed.")
print("Optimized Weights:", w_opt)

# Optional: Plot training cost
plt.figure()
plt.plot(np.array(cost_history), marker='o')
plt.title("Linear Controller Training Cost")
plt.xlabel("Iteration (checkpoints)")
plt.ylabel("Cost")
plt.grid(True)
plt.show()

###############################################################################
# 2. CONTROLLER
###############################################################################

def mujoco_linear_controller(data):
    """
    1) Read sensor data [x, theta, x_dot, theta_dot].
    2) Wrap angle to [-pi, pi].
    3) Compute linear control using w_opt the same way as in 'main_linear_only.py'.
    """
    # cart states
    x = data.sensordata[SENSOR_CART_POS]
    x_dot = data.sensordata[SENSOR_CART_VEL]

    # raw pole angle from the sensor
    raw_theta = data.sensordata[SENSOR_POLE_ANG]
    # wrap angle to [-pi, pi]
    theta = ((raw_theta + jnp.pi) % (2.0 * jnp.pi)) - jnp.pi
    theta_dot = data.sensordata[SENSOR_POLE_ANGVEL]

    # Force 
    f = (w_opt[0] * x
         + w_opt[1] * jnp.cos(theta)
         + w_opt[2] * jnp.sin(theta)
         + w_opt[3] * x_dot
         + w_opt[4] * theta_dot)
    
    return f, x, theta, x_dot, theta_dot

###############################################################################
# 3. MUJOCO SETUP & SIMULATION
###############################################################################
XML_FILE = os.environ.get("MODEL_XML", _CFG.get("model_xml", "cart_pole.xml"))  # your MuJoCo model
with open(XML_FILE, "r") as f:
    xml_str = f.read()

model = mujoco.MjModel.from_xml_string(xml_str)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)


# Sensor indices
SENSOR_CART_POS = 0
SENSOR_CART_VEL = 1
SENSOR_POLE_ANG = 2
SENSOR_POLE_ANGVEL = 3

# Log for plotting
time_log = []
force_log = []
x_log = []
theta_log = []
xdot_log = []
thetadot_log = []

# Camera Settings
viewer = mujoco_viewer.MujocoViewer(model, data,title='Yunus Emre Danabas CartPool Linear Controller')
viewer.cam.azimuth = 90
viewer.cam.elevation = -25.0
viewer.cam.distance = 7.0

sim_time = 0.0
sim_duration = float(os.environ.get("SIM_DURATION", _CFG.get("sim_duration", 30.0)))
start_time = time.time()

# You can adjust inital positions from here
data.qpos[0] = 2.5       # Cart x at 0
data.qpos[5] = 0.5       # Pole hinge angle (theta) rad
data.qvel[0] = -0.15     # Cart x_dot
data.qvel[5] = -0.35     # Cart theta_dot
mujoco.mj_forward(model, data)


###############################################################################
# 4. MAIN LOOP
###############################################################################

while True:
    current_time = time.time()
    if (current_time - start_time) > sim_duration:
        break

    step_start = current_time
    # 1) Compute force from linear controller
    force, x, theta, x_dot, theta_dot = mujoco_linear_controller(data)

    if (current_time - start_time) > 17:
        
        disturbance = -20
    
    elif (current_time - start_time) > 12:

        disturbance = 0

    elif (current_time - start_time) > 7:
        
        disturbance = 20
    
    else:

        disturbance = 0


    data.ctrl[0] = force + disturbance # Apply force and disturbance
    if model.nu > 1:
        data.ctrl[1] = 0.0  # zero out second actuator if not used

    # 2) Step the simulation
    mujoco.mj_step(model, data)
    sim_time += model.opt.timestep

    # 3) Log data
    time_log.append(sim_time)
    force_log.append(force)
    x_log.append(x)
    theta_log.append(theta)
    xdot_log.append(x_dot)
    thetadot_log.append(theta_dot)

    # 4) Render
    viewer.render()
    if not viewer.is_alive:
        break

    # 5) Real-time sync
    elapsed = time.time() - step_start
    if elapsed < model.opt.timestep:
        time.sleep(model.opt.timestep - elapsed)

viewer.close()



###############################################################################
# 5. PLOTTING
###############################################################################

time_log = np.array(time_log)
force_log = np.array(force_log)
x_log = np.array(x_log)
theta_log = np.array(theta_log)
xdot_log = np.array(xdot_log)
thetadot_log = np.array(thetadot_log)

plt.figure()
plt.plot(time_log, force_log, 'b')
plt.title("Linear Control Force vs Time (MuJoCo)")
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.grid(True)
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(10,6), sharex=True)
fig.suptitle("Cart-Pole States (MuJoCo w/ Trained Linear)")

axs[0,0].plot(time_log, x_log, 'b')
axs[0,0].set_title("Cart Position x(t)")
axs[0,0].set_ylabel("x (m)")
axs[0,0].grid(True)

axs[0,1].plot(time_log, theta_log, 'r')
axs[0,1].set_title("Pole Angle theta(t)")
axs[0,1].set_ylabel("Angle (rad)")
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

print("MuJoCo linear-controller simulation finished.")
