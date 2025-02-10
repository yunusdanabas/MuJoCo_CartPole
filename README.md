# Cart-Pole Swing-Up Control with JAX and MuJoCo

This repository implements controllers for the classic cart-pole swing-up task using both classical control methods and modern deep-learning techniques. We train a neural network controller via differentiable simulation to swing up the pole from a downward or arbitrary configuration to near upright—so that a simpler linear/LQR controller can take over stabilization. Real-time simulations are performed with MuJoCo and mujoco_viewer.

---

## Overview

The cart-pole system consists of a cart moving horizontally along a track and a pole hinged to the cart. The physical objective is to use control forces applied to the cart to “pump” the pole, accumulating enough kinetic and potential energy to drive it to the upright equilibrium while keeping the cart near the center (x = 0). This project compares several approaches:

- **Linear Controller:** Trained in a differentiable JAX simulation with a quadratic cost.
- **LQR Controller:** Derived from a linearization of the nonlinear dynamics.
- **Neural Network Controller:** An MLP policy trained with an energy-shaping plus cart-deviation cost function.

---

## Techniques & Libraries

- **JAX:** Provides automatic differentiation and JIT compilation.
- **Equinox:** Used to build and manage our neural network modules in a functional style.
- **Optax:** Supplies gradient-based optimizers for training our neural network.
- **Diffrax:** Enables differentiable ODE integration for rollout simulations.
- **MuJoCo & mujoco_viewer:** Deliver high-fidelity physics simulation and real-time interactive visualization.
- **Classical Control Methods:** Linear and LQR controllers for baseline performance and comparison.

---

## Physical System & Objectives

The physical system is a cart-pole, where:
- The **cart** (mass \(M\)) moves along a horizontal track.
- The **pole** (mass \(m\) and length \(l\)) swings about a hinge on the cart.
- The goal is to swing up the pole from a downward (or arbitrary) state to an upright state.
- We measure the state as \([x, \cos\theta, \sin\theta, \dot{x}, \dot{\theta}]\) to avoid discontinuities.
- Our NN cost function penalizes the squared difference between the current total energy (kinetic plus potential) and the desired energy for the upright configuration, plus a penalty for cart deviation from zero and large control forces.

---

## Project Structure

- **`controller/`**  
  - `linear_controller.py`: Implements a linear controller and its training routine.  
  - `lqr_controller.py`: Contains functions to linearize the cart-pole system and compute the LQR gain.  
  - `neuralnetwork_controller.py`: Defines the MLP (or CartPolePolicy) used for neural network control.
  
- **`env/`**  
  - `cartpole.py`: Contains the nonlinear dynamics of the cart-pole.  
  - `closedloop.py`: Provides a wrapper for closed-loop simulation using Diffrax.
  
- **`lib/`**  
  - `trainer.py`: Implements the training loop for the neural network controller using an energy-shaping plus cart-deviation loss.  
  - `utils.py`: Contains utility functions for sampling initial conditions and plotting trajectories, energies, and costs.
  
- **Main Scripts:**  
  - `main_linear_only.py`: Trains and simulates the linear controller and LQR in a JAX environment.  
  - `mujoco_lqr_controller_interactive.py`: Runs an interactive LQR controller in MuJoCo (with overlays, etc.).  
  - `mujoco_linear_control.py`: Implements the trained linear controller in MuJoCo.  
  - `nn_mujoco.py`: Runs the neural network (NN) controller in MuJoCo.  
  - `train_nn_controller.py`: Trains the NN controller using differentiable simulation and rollout cost minimization.

---

## How It Works

1. **Differentiable Simulation & Training:**  
   - We simulate the cart-pole dynamics using Diffrax.  
   - A cost function penalizes deviations of the system’s energy from the desired upright energy, deviations of the cart position from zero, and excessive control force.
   - The NN (an MLP) is trained via gradient descent (using Optax) to minimize this cost across multiple rollout simulations from random initial conditions.

2. **Neural Network Policy:**  
   - The policy receives a 5D state \([x, \cos\theta, \sin\theta, \dot{x}, \dot{\theta}]\) and outputs a scalar control force.
   - The training objective is to “pump” the energy into the pole until it reaches near-upright, while keeping the cart near the center.

3. **Classical Controllers:**  
   - In parallel, a linear controller and an LQR controller are implemented for comparison. The LQR is computed from a linearized model of the system.

4. **Real-Time Simulation with MuJoCo:**  
   - Controllers are deployed in MuJoCo for high-fidelity simulation and visualization.  
   - The interactive LQR script includes overlays (e.g., current disturbance force) and real-time keyboard controls.

---

## Screenshot

![LQR Interactive](mujoco_lqr_controller_interactive.png)  
*Screenshot from `mujoco_lqr_controller_interactive.py` showing the interactive MuJoCo simulation with control overlays.*

---

## Installation & Requirements

- **Python 3.8+**
- **JAX** (CPU/GPU version as needed)
- **Equinox**
- **Optax**
- **Diffrax**
- **MuJoCo 3.x** and **mujoco_viewer**
- **Matplotlib** and **NumPy**

Install required packages via pip:
```bash
pip install jax jaxlib equinox optax diffrax mujoco mujoco_viewer matplotlib numpy
```
Also, ensure MuJoCo is installed and licensed.

---

## Usage

- **Training the NN Controller:**  
  Run `train_nn_controller.py` to train the neural network controller for swing-up.
  
- **Testing & Simulation:**  
  Use `main_linear_only.py`, `mujoco_lqr_controller_interactive.py`, `mujoco_linear_control.py`, and `nn_mujoco.py` to test various controllers in simulation.

- **Visualization:**  
  Utility functions in `lib/utils.py` provide tools to plot trajectories, energy profiles, and control costs.

---

## Future Work

- **Controller Handoff:**  
  Integrate a switching mechanism to hand over to an LQR controller when the system is near upright.
- **Advanced Cost Functions:**  
  Experiment with additional loss terms or phase-based training strategies.
- **Robustness:**  
  Enhance performance under disturbances and across a wider range of initial conditions.
