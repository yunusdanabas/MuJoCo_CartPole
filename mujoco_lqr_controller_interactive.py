# mujoco_lqr_controller_interactive.py
# Description: This script implements a cart-pole swing-up controller using a linear quadratic regulator (LQR).

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
import warnings

# MuJoCo + viewer
import mujoco
from mujoco.glfw import glfw

# Import your LQR helpers
from controller.lqr_controller import linearize_cartpole, compute_lqr_gain

warnings.filterwarnings("ignore", category=UserWarning, module="glfw")


###############################################################################
# 0. Callback Configuration
###############################################################################

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

_overlay = {}
def add_overlay(gridpos, text1, text2):

    if gridpos not in _overlay:
        _overlay[gridpos] = ["", ""]
    _overlay[gridpos][0] += text1 + "\n"
    _overlay[gridpos][1] += text2 + "\n"

#HINT1: add the overlay here
def create_overlay(model,data):

    global disturbance
    global force

    topleft = mujoco.mjtGridPos.mjGRID_TOPLEFT
    topright = mujoco.mjtGridPos.mjGRID_TOPRIGHT
    bottomleft = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT
    bottomright = mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT
    bottom = mujoco.mjtGridPos.mjGRID_BOTTOM
    top = mujoco.mjtGridPos.mjGRID_TOP

    add_overlay(
        bottomleft,
        "Restart",'r' ,
         )
    
    add_overlay(
        bottomleft,
        "Time",'%.2f' % data.time,
         )
    
    add_overlay(
       topleft,
       "Use Left/Right Arrow to increase/decrease disturbance -15/+15 N", " "
        )

    add_overlay(
       topleft,
       "Disturbance: ",'%.2f' % disturbance,
        )
    
    add_overlay(
       topleft,
       "Use Up/Down arrow keys to instantly change the pole's angle. +15/-15 degree", " "
        )

    add_overlay(
       topleft,
       "Pole Angle: ",'%.2f' % jnp.rad2deg(data.qpos[5]),
        )

    add_overlay(
       topright,
       "Current Control Force",'%.2f' % force,
        )


#HINT2: add the logics for key press here
def keyboard(window, key, scancode, act, mods):

    global disturbance

    if (act == glfw.PRESS and key == glfw.KEY_R):
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

    if act == glfw.PRESS and key == glfw.KEY_LEFT:
        disturbance -= 15

    if act == glfw.PRESS and key == glfw.KEY_RIGHT:
        disturbance += 15

    if act == glfw.PRESS and key == glfw.KEY_UP:
        data.qpos[5] += jnp.deg2rad(15)

    if act == glfw.PRESS and key == glfw.KEY_DOWN:
        data.qpos[5] -= jnp.deg2rad(15)


def waitBeforeStart(sleeptime):
    create_overlay(model,data)
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
    # Update scene and render
    mujoco.mjv_updateScene(model, data, opt, None, cam,mujoco.mjtCatBit.mjCAT_ALL.value, scene)
    mujoco.mjr_render(viewport, scene, context)
    # overlay items
    for gridpos, [t1, t2] in _overlay.items():

        mujoco.mjr_overlay(
            mujoco.mjtFontScale.mjFONTSCALE_250,
            gridpos,
            viewport,
            t1,
            t2,
            context)
    # clear overlay
    _overlay.clear()
    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)
    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()
    time.sleep(sleeptime)

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
XML_FILE = "cart_pole.xml"
with open(XML_FILE, "r") as f:
    xml_str = f.read()

model = mujoco.MjModel.from_xml_string(xml_str)
data = mujoco.MjData(model)
cam = mujoco.MjvCamera()                        # Abstract camera
opt = mujoco.MjvOption()                        # visualization options

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
data.qpos[5] = jnp.deg2rad(25)   # pole hinge angle (slightly tilted)
data.qvel[0] = -0.15
data.qvel[5] = -0.35

# Re-forward for consistency
mujoco.mj_forward(model, data)


###############################################################################
# 5. Simulation Loop (Real-Time)
###############################################################################
# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1900, 1000, "Yunus Emre Danabas Interactive CartPool LQR Controller", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)


# initialize visualization data structures
mujoco.mjv_defaultCamera(cam)
mujoco.mjv_defaultOption(opt)
scene = mujoco.MjvScene(model, maxgeom=10000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)


cam.azimuth = 90
cam.elevation = -25
cam.distance = 6.75
cam.lookat = np.array([0.0, 1.0, 0])


sim_time = 0.0
sim_duration = 60.0
start_time = time.time()

time_log = []
force_log = []
x_log = []
theta_log = []
xdot_log = []
thetadot_log = []

global disturbance 
global force 

disturbance = 0
force = 0


waitBeforeStart(3)

while not glfw.window_should_close(window):


    current_time = time.time()
    if (current_time - start_time) > sim_duration:
        break


    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

    #create overlay
    create_overlay(model,data)

    step_start = current_time
    force = mujoco_lqr_controller(data, K)
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


    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)


    # Update scene and render
    mujoco.mjv_updateScene(model, data, opt, None, cam,
                       mujoco.mjtCatBit.mjCAT_ALL.value, scene)
    mujoco.mjr_render(viewport, scene, context)

        # overlay items
    for gridpos, [t1, t2] in _overlay.items():

        mujoco.mjr_overlay(
            mujoco.mjtFontScale.mjFONTSCALE_250,
            gridpos,
            viewport,
            t1,
            t2,
            context)

    # clear overlay
    _overlay.clear()


    # 5) Real-time sync
    elapsed = time.time() - step_start
    if elapsed < model.opt.timestep:
        time.sleep(model.opt.timestep - elapsed)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()


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
