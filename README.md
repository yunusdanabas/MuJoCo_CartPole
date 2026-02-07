# MuJoCo CartPole Control

A JAX-based implementation of cart-pole swing-up control using classical control methods and neural networks. Train controllers via differentiable simulation and deploy them in high-fidelity MuJoCo environments.

## What It Does

The cart-pole system has a cart moving horizontally with a pole attached that can swing freely. Controllers are designed to:
- Swing up the pole from hanging position to upright
- Stabilize around the upright equilibrium
- Keep the cart near the center (x = 0)

## Control Approaches

- **Linear** – PD control with quadratic cost, simple stabilization
- **LQR** – Linear-quadratic regulator, optimal linear control
- **Neural Network** – MLP trained via differentiable simulation, energy-based swing-up

## Project Structure

```
MuJoCo_CartPole/
├── controller/          # Control algorithms
├── env/                 # Cart-pole dynamics & simulation
├── lib/                 # Training, utilities, visualization
├── examples/            # Quick start demonstrations
├── scripts/             # MuJoCo simulation scripts
├── tests/               # Test suite
└── config.yaml          # Configuration & parameters
```

## Installation

```bash
pip install jax jaxlib equinox optax diffrax mujoco matplotlib numpy mujoco-python-viewer
```

Ensure MuJoCo is properly installed on your system.

## Quick Start

Run examples:
```bash
python examples/linear.py
python examples/lqr.py
python examples/nn.py
python examples/combo_linear_lqr_nn.py
```

Train the neural network controller:
```bash
python scripts/train_nn_controller.py
```

MuJoCo simulation:
```bash
python scripts/linear_mujoco.py
python scripts/lqr_mujoco.py
python scripts/nn_mujoco.py
```

## Configuration

Edit `config.yaml` to adjust training parameters, system parameters, cost weights, and time horizons.

## Testing

```bash
./scripts/run_tests.sh
# or
pytest tests/
```

## How It Works

- **Differentiable Simulation** – Diffrax for ODE integration, JAX for automatic differentiation; controllers trained by minimizing cost over trajectories
- **Neural Network** – 5D state input `[x, cosθ, sinθ, ẋ, θ̇]`, control force output, energy-based loss, Adam optimizer
- **MuJoCo Deployment** – High-fidelity physics, real-time visualization, interactive controls

## Key Features

- JIT compilation for fast execution
- Batch processing for efficient training
- Energy-based loss for swing-up tasks
- Modular design
- Full test coverage

Built with JAX, MuJoCo, Equinox, Optax, Diffrax
