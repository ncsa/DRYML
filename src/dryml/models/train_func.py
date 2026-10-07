from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Sequence

from dryml.methods import Method


class TrainFunction(Method):
    """Abstract Method that mutates one Experiment through a training procedure."""

    supports_safe_points = True
    """Whether this trainer can invoke truthful optimizer-update callbacks.

    ``Experiment.train`` uses this capability to request intermediate managed
    checkpoints. Trainers that only expose one opaque fit operation leave it
    false; Experiment still creates and evaluates its terminal checkpoint.
    """

    supports_observers = False
    """Whether this trainer accepts Experiment invocation telemetry sessions."""

    def _validate_observer_session(self, observer_session) -> None:
        """Apply the default host-callable observer admission contract."""

        observer_session.require_callables(type(self).__name__)

    @abstractmethod
    def __call__(self, exp, *, callbacks: Sequence[Callable[[], None]] = ()):
        """Train ``exp`` and return its implementation-defined result.

        Args:
            exp: The Experiment supplying model, data, and training state.
            callbacks: Process-local zero-argument safe-point callbacks invoked
                only after a successful optimizer update has retained truthful
                state and accounting.

        Returns:
            An implementation-defined training result.

        Raises:
            TypeError: If an implementation does not support the callback form.
            Implementations may raise backend or training failures.
        """


__all__ = ["TrainFunction"]
