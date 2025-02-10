# Cart-Pole Swing-Up Control with JAX and MuJoCo

This repository implements various controllers for the classic cart-pole swing-up task. We combine classical control methods (linear controllers and LQR) with modern deep learning techniques (neural network policies trained via differentiable simulation) using JAX. The system is simulated both in a JAX environment and in real time with MuJoCo and mujoco_viewer.

---

## Overview

The **cart-pole system** consists of a cart that moves along a horizontal track and a pole hinged to the cart that can swing freely. The goal is to develop a controller that can swing the pole from a downward (or arbitrary) configuration to an upright equilibrium and keep it near that equilibrium so that a simpler (e.g., LQR) controller can take over for fine stabilization.

### Key Objectives

- **Swing-Up Control:**  
  Train a neural network controller that “pumps” energy into the system, ensuring that the total energy (a combination of kinetic and potential) reaches the desired level corresponding to the upright configuration.  
- **Cart Regulation:**  
  Ensure that the cart’s position remains near the center (x = 0) so that the pole can be stabilized without the cart drifting off track.
- **Controller Comparison:**  
  Implement and compare different control strategies:
  - A **linear controller** trained using gradient descent in a differentiable simulation.
  - A classical **LQR controller** derived by linearizing the system dynamics.
  - A **neural network (MLP) controller** trained with an energy‐shaping cost function.

---

## Techniques & Libraries

- **JAX:**  
  Provides automatic differentiation and JIT compilation to efficiently compute gradients and simulate the system.
  
- **Equinox:**  
  Used to build and manage neural network modules (MLPs) in a functional style.

- **Optax:**  
  A gradient-based optimization library for JAX that we use to train our neural network controllers.

- **Diffrax:**  
  A differential equation solver library that enables differentiable ODE integration, essential for rollout-based cost evaluation and training.

- **MuJoCo & mujoco_viewer:**  
  MuJoCo is used for high-fidelity physics simulation and real-time visualization. The mujoco_viewer library allows for interactive rendering and overlays (e.g., displaying disturbance forces).

- **Classical Control:**  
  - A **linear controller** is trained using gradient descent with a quadratic cost on the state and control.
  - An **LQR controller** is derived by linearizing the cart-pole dynamics and solving the continuous-time algebraic Riccati equation to compute the optimal gain matrix.

---

## Project Structure

- **`controller/`**  
  Contains implementations of the controllers:
  - `linear_controller.py` implements the linear controller and training for it.
  - `lqr_controller.py` implements the linearization of the cart-pole system and LQR gain computation.
  - `neuralnetwork_controller.py` defines the MLP (neural network) architecture (also named here as `CartPolePolicy` in some files).

- **`env/`**  
  Contains the system dynamics:
  - `cartpole.py` includes the nonlinear dynamics used for simulation.
  - `closedloop.py` provides a wrapper to simulate the system in closed-loop using Diffrax.

- **`lib/`**  
  Contains the training utilities and cost functions:
  - `trainer.py` includes functions for computing energy, cost, rollout simulation, and the training loop for the NN controller.
  - `utils.py` has helper functions for plotting trajectories, cost, energy, and for sampling initial conditions.

- **Main scripts:**  
  - `main_linear_only.py` trains and tests the linear controller (and LQR) in the differentiable JAX environment.
  - `mujoco_lqr_controller.py` runs the LQR controller in MuJoCo for real-time visualization.
  - `mujoco_linear_control.py` attempts to run the trained linear controller in MuJoCo.
  - `train_nn_controller.py` trains the neural network controller and simulates its performance.

---

## How It Works

1. **Differentiable Simulation & Training:**  
   - The training code uses Diffrax to simulate the cart-pole dynamics over a rollout.  
   - A cost function (loss) is defined that penalizes deviations of the system’s energy from the desired upright energy, deviations of the cart position from zero, and excessive control force.  
   - The neural network (MLP) parameters are updated via gradient descent using Optax.

2. **Neural Network Policy:**  
   - The NN receives a 5-dimensional state \([x, \cos(\theta), \sin(\theta), \dot{x}, \dot{\theta}]\) and outputs a scalar control force.  
   - The cost function is designed to encourage the system to “pump” energy into the pole (swing-up) while keeping the cart close to the center.

3. **Real-Time Simulation with MuJoCo:**  
   - Once trained, controllers (linear, LQR, or NN) can be deployed in a MuJoCo simulation.  
   - MuJoCo and mujoco_viewer are used to render the simulation in real time and to overlay additional information (like disturbance forces) on the display.

4. **Controller Comparison:**  
   - The repository includes examples of comparing the trajectories, costs, and control forces from the trained linear controller versus the LQR controller, as well as testing the NN controller for swing-up.

---

## Installation & Requirements

- **Python 3.8+**
- **JAX** (with your preferred backend, e.g., CPU or GPU)  
- **Equinox**
- **Optax**
- **Diffrax**
- **MuJoCo 3.x** (and mujoco_viewer)  
- **Matplotlib** and **NumPy**

Install the necessary Python packages (e.g., via pip):
```bash
pip install jax jaxlib equinox optax diffrax mujoco mujoco_viewer matplotlib numpy
```
Ensure that MuJoCo is installed and licensed correctly on your system.

---

## Usage

- **Training the Neural Network Controller:**  
  Run `train_nn_controller.py` to train the NN swing-up controller. The script will log training costs and save the trained model to disk.

- **Testing and Visualization:**  
  Use the provided MuJoCo scripts (e.g., `mujoco_linear_control.py`, `mujoco_lqr_controller.py`, `nn_mujoco.py`) to run real-time simulations and visualize controller performance.

- **Controller Comparison:**  
  The repository includes examples that compare the trajectories and costs of different controllers (linear, LQR, and NN).

---

## Future Work

- **Controller Handoff:**  
  Once the NN drives the system near the upright configuration, a linear or LQR controller may be used for final stabilization.
- **Advanced Loss Functions:**  
  Experiment with different cost terms or phase-based training where the loss changes as the system approaches the upright state.
- **Robustness:**  
  Test and tune the controllers under various disturbances and different initial conditions.
