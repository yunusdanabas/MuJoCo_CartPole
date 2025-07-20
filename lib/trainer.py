"""Deprecated training utilities.

This module now imports helper functions from ``controller.nn_controller`` for
backwards compatibility. New code should import directly from that module.
"""

from controller.nn_controller import train_nn_controller, evaluate_controller
