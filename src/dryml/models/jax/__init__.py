"""Experimental JAX, Flax NNX, and Optax model-state APIs.

These APIs are intentionally experimental and may change based on use. Importing
this facade remains lightweight; JAX, Flax, and Optax load only when a selected
feature constructs or executes its native runtime.
"""

from .base import EarlyStoppingTraining, Model, Optimizer, Training, TrainFunction, pure_training_transition
from .flax import FlaxModel, NNXModel
from .state import JaxStateError

__all__ = ["EarlyStoppingTraining", "FlaxModel", "JaxStateError", "Model", "NNXModel", "Optimizer", "Training", "TrainFunction", "pure_training_transition"]
