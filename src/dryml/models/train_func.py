from __future__ import annotations

from abc import abstractmethod

from dryml.methods import Method


class TrainFunction(Method):
    """Abstract Method that mutates one Experiment through a training procedure."""

    @abstractmethod
    def __call__(self, exp):
        """Train ``exp`` and return its implementation-defined result.

        Args:
            exp: The Experiment supplying model, data, and training state.

        Returns:
            An implementation-defined training result.

        Raises:
            Implementations may raise backend or training failures.
        """


__all__ = ["TrainFunction"]
