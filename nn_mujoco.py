# file: nn_mujoco.py

import time
import warnings
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt

import mujoco
import mujoco_viewer

# Import your model (MLP policy) definition
from controller.neuralnetwork_controller import CartPolePolicy

warnings.filterwarnings("ignore", category=UserWarning, module="glfw")


###############################################################################
# 1. Load the Trained NN Model
###############################################################################
# Must instantiate a "dummy" model with the same structure as we had at training
dummy_model = CartPolePolicy(
    key=jax.random.PRNGKey(0),
    in_dim=5,
    hidden_dims=(64, 64),  # must match the dims used during training
    out_dim=1
)
trained_model = eqx.tree_deserialise_leaves("trained_nn_model.eqx", dummy_model)
print("Loaded trained NN model from 'trained_nn_model.eqx'.")


###############################################################################
# 2. MuJoCo Cart-Pole Setup
###############################################################################
XML_FILE = "cart_pole.xml"   # path to your XML
with open(XML_FILE, "r") as f:
    xml_str = f.read()

model = mujoco.MjModel.from_xml_string(xml_str)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# Sensor indices (check your XML sensor names and order!)
SENSOR_CART_POS = 0
SENSOR_CART_VEL = 1
SENSOR_POLE_ANG = 2
SENSOR_POLE_ANGVEL = 3

###############################################################################
# 3. Control Scaling (Optional) 
###############################################################################
CONTROL_SCALE = 20.0  # Adjust as needed

###############################################################################
# 4. Define the NN-based Swing-Up Controller
###############################################################################
def nn_swingup_controller(data, model):
    """
    Build the 5-element state: [x, cos(theta), sin(theta), x_dot, theta_dot].
    Then call the NN, scale the output force, return it.
    """
    x = data.sensordata[SENSOR_CART_POS]
    x_dot = data.sensordata[SENSOR_CART_VEL]

    theta_raw = data.sensordata[SENSOR_POLE_ANG]
    # Wrap angle to [-pi, pi]
    theta = ((theta_raw + np.pi) % (2 * np.pi)) - np.pi

    theta_dot = data.sensordata[SENSOR_POLE_ANGVEL]
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)
    state_5 = jnp.array([x, cos_th, sin_th, x_dot, theta_dot])

    raw_force = model(state_5)
    force = float(CONTROL_SCALE * raw_force)
    return force

###############################################################################
# 5. Set Initial Conditions
###############################################################################
# For swing-up, set pole near downward
data.qpos[0] = 0.0    # cart x
data.qpos[5] = np.pi  # pole hinge ~ downward
data.qvel[0] = 0.0
data.qvel[5] = 0.0
mujoco.mj_forward(model, data)

###############################################################################
# 6. Run the MuJoCo Simulation
###############################################################################
viewer = mujoco_viewer.MujocoViewer(model, data)
viewer.cam.azimuth = 90
viewer.cam.elevation = -35.0
viewer.cam.distance = 8.0

time_log = []
force_log = []
x_log = []
theta_log = []
xdot_log = []
thetadot_log = []

sim_time = 0.0
sim_duration = 30.0
start_time = time.time()

while True:
    current_time = time.time()
    if (current_time - start_time) > sim_duration:
        break

    step_start = current_time

    # 1) Compute force from NN
    force = nn_swingup_controller(data, trained_model)
    data.ctrl[0] = force   # apply control to the cart joint
    # If you have a second actuator for the pole, you might set data.ctrl[1] = 0

    # 2) Step simulation
    mujoco.mj_step(model, data)
    sim_time += model.opt.timestep

    # 3) Logging
    time_log.append(sim_time)
    force_log.append(force)
    x_log.append(data.sensordata[SENSOR_CART_POS])
    raw_theta = data.sensordata[SENSOR_POLE_ANG]
    wrapped_theta = ((raw_theta + np.pi) % (2 * np.pi)) - np.pi
    theta_log.append(wrapped_theta)
    xdot_log.append(data.sensordata[SENSOR_CART_VEL])
    thetadot_log.append(data.sensordata[SENSOR_POLE_ANGVEL])

    # 4) Render
    viewer.render()
    if not viewer.is_alive:
        break

    # 5) Real-time sync
    elapsed = time.time() - step_start
    if elapsed < model.opt.timestep:
        time.sleep(model.opt.timestep - elapsed)

viewer.close()
print("MuJoCo NN swing-up simulation finished.")

###############################################################################
# 7. Plot the Results
###############################################################################
time_log = np.array(time_log)
force_log = np.array(force_log)
x_log = np.array(x_log)
theta_log = np.array(theta_log)
xdot_log = np.array(xdot_log)
thetadot_log = np.array(thetadot_log)

plt.figure()
plt.plot(time_log, force_log, label="Control Force")
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.title("NN Swing-Up Control Force vs. Time")
plt.grid(True)
plt.legend()
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
fig.suptitle("Cart-Pole States (NN Swing-Up)")

axs[0, 0].plot(time_log, x_log, 'b')
axs[0, 0].set_ylabel("x (m)")
axs[0, 0].set_title("Cart Position")
axs[0, 0].grid(True)

axs[0, 1].plot(time_log, theta_log, 'r')
axs[0, 1].set_ylabel("theta (rad)")
axs[0, 1].set_title("Pole Angle")
axs[0, 1].grid(True)

axs[1, 0].plot(time_log, xdot_log, 'g')
axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].set_ylabel("x_dot (m/s)")
axs[1, 0].set_title("Cart Velocity")
axs[1, 0].grid(True)

axs[1, 1].plot(time_log, thetadot_log, 'm')
axs[1, 1].set_xlabel("Time (s)")
axs[1, 1].set_ylabel("theta_dot (rad/s)")
axs[1, 1].set_title("Pole Angular Velocity")
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()

print("Done. NN swing-up visualization in MuJoCo is complete.")