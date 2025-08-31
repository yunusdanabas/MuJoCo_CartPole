# MuJoCo CartPole Control

A JAX-based implementation of cart-pole swing-up control using classical control methods and neural networks. Train controllers via differentiable simulation and deploy them in high-fidelity MuJoCo environments.

## 🎯 What It Does

The cart-pole system has a cart moving horizontally with a pole attached that can swing freely. The goal is to design controllers that can:

1. **Swing up** the pole from hanging position to upright
2. **Stabilize** the system around the upright equilibrium
3. **Keep the cart** near the center (x = 0)

## 🚀 Three Control Approaches

| Controller | Method | Purpose |
|------------|---------|---------|
| **Linear** | PD control with quadratic cost | Simple stabilization |
| **LQR** | Linear-quadratic regulator | Optimal linear control |
| **Neural Network** | MLP trained via differentiable simulation | Energy-based swing-up |

## 🏗️ Project Structure

```
MuJoCo_CartPole/
├── controller/          # Control algorithms
├── env/                # Cart-pole dynamics & simulation
├── lib/                # Training, utilities, visualization
├── examples/           # Quick start demonstrations
├── scripts/            # MuJoCo simulation scripts
├── tests/              # Comprehensive test suite
└── config.yaml         # Configuration & parameters
```

## 🛠️ Installation

```bash
pip install jax jaxlib equinox optax diffrax mujoco matplotlib numpy mujoco-python-viewer
```

**Note**: Ensure MuJoCo is properly installed and licensed on your system.

## 🎮 Quick Start

### Run Examples
```bash
# Individual controllers
python examples/linear.py      # Linear PD control
python examples/lqr.py         # LQR control
python examples/nn.py          # Neural network control

# Compare all three
python examples/combo_linear_lqr_nn.py
```

### Train Neural Network
```bash
python scripts/train_nn_controller.py
```

### MuJoCo Simulation
```bash
# Linear controller in MuJoCo
python scripts/mujoco_linear_control.py

# LQR controller in MuJoCo  
python scripts/lqr_mujoco.py

# Neural network in MuJoCo
python scripts/nn_mujoco.py
```

## 🔧 Configuration

Edit `config.yaml` to adjust:
- Training parameters (epochs, batch size, learning rate)
- System parameters (masses, lengths, gravity)
- Cost function weights
- Time horizons

## 🧪 Testing

```bash
# Run all tests
./scripts/run_tests.sh

# Or use pytest directly
pytest tests/
```

## 🔬 How It Works

### 1. Differentiable Simulation
- Uses **Diffrax** for ODE integration
- **JAX** provides automatic differentiation
- Train controllers by minimizing cost over trajectories

### 2. Neural Network Training
- **Input**: 5D state `[x, cosθ, sinθ, ẋ, θ̇]`
- **Output**: Control force
- **Loss**: Energy-based + position penalty
- **Optimizer**: Adam via Optax

### 3. MuJoCo Deployment
- High-fidelity physics simulation
- Real-time visualization
- Interactive controls and disturbance testing

## 📊 Key Features

- **JIT Compilation** - Fast execution with JAX
- **Batch Processing** - Efficient training and evaluation
- **Energy-Based Loss** - Specialized for swing-up tasks
- **Modular Design** - Clean separation of concerns
- **Full Test Coverage** - Comprehensive testing suite

## 📚 Documentation

- **Project Report**: `ProjectReport.pdf` - Complete methodology and results
- **Code Examples**: `examples/` directory for learning
- **Configuration**: `config.yaml` for parameter tuning

## 🔮 Future Work

- Controller handoff (NN → LQR) for optimal performance
- Enhanced cost functions with phase-aware terms
- Robustness testing under various disturbances
- Real-time performance optimization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

---

**Built with**: JAX, MuJoCo, Equinox, Optax, Diffrax
