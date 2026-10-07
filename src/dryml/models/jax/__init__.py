"""Experimental JAX, Flax NNX, and Optax model-state APIs.

These APIs are intentionally experimental and may change based on use. Importing
this facade remains lightweight; JAX, Flax, and Optax load only when a selected
feature constructs or executes its native runtime.
"""

from .base import Model, Optimizer, TrainFunction, pure_training_transition
from .flax import FlaxModel, NNXModel
from .state import JaxStateError

__all__ = ["FlaxModel", "JaxStateError", "Model", "NNXModel", "Optimizer", "TrainFunction", "pure_training_transition"]
