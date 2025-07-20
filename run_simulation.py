"""Unified command-line runner for Cart-Pole simulations."""

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
    """Import ``module_name`` and execute its ``main`` function if present."""
    os.environ.update(env)

    if module_name.endswith(".py"):
        module_name = module_name[:-3]

    if module_name not in sys.modules:
        module = importlib.import_module(module_name)
    else:
        module = importlib.reload(sys.modules[module_name])

    if hasattr(module, "main") and callable(module.main):
        module.main()


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified CartPole simulation runner")
    parser.add_argument(
        "--controller",
        choices=["nn", "linear", "lqr", "lqr-interactive"],
        required=True,
        help="Controller type to run",
    )
    parser.add_argument(
        "--mode",
        choices=["simulate", "train"],
        default="simulate",
        help="Execution mode (if applicable)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    config = load_config(config_path)

    # Environment variables for child scripts
    env = os.environ.copy()
    env.update({
        "CONFIG_PATH": config_path,
        "MODEL_XML": str(config.get("model_xml", "cart_pole.xml")),
        "SIM_DURATION": str(config.get("sim_duration", 30.0)),
        "NN_MODEL_PATH": str(config.get("nn_model_path", "saved_models/nn_controller.eqx")),
    })

    if args.controller == "nn":
        if args.mode == "train":
            run_module("scripts.train_nn_controller", env)
        else:
            run_module("scripts.nn_mujoco", env)
    elif args.controller == "linear":
        run_module("scripts.mujoco_linear_control", env)
    elif args.controller == "lqr":
        run_module("scripts.mujoco_lqr_controller", env)
    elif args.controller == "lqr-interactive":
        run_module("scripts.mujoco_lqr_controller_interactive", env)
    else:
        raise ValueError(f"Unknown controller type '{args.controller}'")


if __name__ == "__main__":
    main()
