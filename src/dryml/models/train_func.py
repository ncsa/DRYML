from __future__ import annotations

from dryml.code import Method


class TrainFunction(Method):
    """A Method that mutates an Experiment through one training procedure."""

    def __call__(self, exp):
        raise NotImplementedError("TrainFunction subclasses must implement __call__(exp).")


__all__ = ["TrainFunction"]
