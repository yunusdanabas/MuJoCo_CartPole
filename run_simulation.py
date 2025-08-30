"""
Unified Command-Line Runner for Cart-Pole Simulations

Provides a single entry point for running different controller types
with configurable parameters and execution modes.
"""

import argparse
import importlib
import os
import sys
from typing import Dict

import yaml


def load_config(path: str) -> dict:
    """Load simulation configuration from YAML file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file '{path}' not found")
    
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def run_module(module_name: str, env: Dict[str, str]) -> None:
    """Import module and execute its main function if present."""
    os.environ.update(env)

    # Remove .py extension if present
    if module_name.endswith(".py"):
        module_name = module_name[:-3]

    # Import or reload module
    if module_name not in sys.modules:
        module = importlib.import_module(module_name)
    else:
        module = importlib.reload(sys.modules[module_name])

    # Execute main function if available
    if hasattr(module, "main") and callable(module.main):
        module.main()


def setup_environment(config: dict, config_path: str) -> Dict[str, str]:
    """Setup environment variables for child scripts."""
    env = os.environ.copy()
    
    # Extract simulation settings
    sim_config = config.get("simulation", {})
    models_config = config.get("models", {})
    
    env.update({
        "CONFIG_PATH": config_path,
        "MODEL_XML": str(sim_config.get("model_xml", "cart_pole.xml")),
        "SIM_DURATION": str(sim_config.get("duration", 30.0)),
        "NN_MODEL_PATH": str(models_config.get("nn_controller", "saved_models/nn_controller.eqx")),
    })
    
    return env


def main() -> None:
    """Main entry point for cart-pole simulation runner."""
    parser = argparse.ArgumentParser(
        description="Unified CartPole simulation runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_simulation.py --controller lqr --mode simulate
  python run_simulation.py --controller nn --mode train
  python run_simulation.py --controller linear --config custom_config.yaml
        """
    )
    
    parser.add_argument(
        "--controller",
        choices=["nn", "linear", "lqr", "lqr-interactive"],
        required=True,
        help="Controller type to run"
    )
    
    parser.add_argument(
        "--mode",
        choices=["simulate", "train"],
        default="simulate",
        help="Execution mode (default: simulate)"
    )
    
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)"
    )
    
    args = parser.parse_args()

    # Load configuration
    config_path = os.path.abspath(args.config)
    config = load_config(config_path)
    
    # Setup environment
    env = setup_environment(config, config_path)

    # Route to appropriate module based on controller type
    if args.controller == "nn":
        if args.mode == "train":
            run_module("scripts.train_nn_controller", env)
        else:
            run_module("scripts.nn_mujoco", env)
    elif args.controller == "linear":
        run_module("scripts.linear_mujoco", env)
    elif args.controller == "lqr":
        run_module("scripts.lqr_mujoco", env)
    elif args.controller == "lqr-interactive":
        run_module("scripts.mujoco_lqr_controller_interactive", env)
    else:
        raise ValueError(f"Unknown controller type '{args.controller}'")


if __name__ == "__main__":
    main()