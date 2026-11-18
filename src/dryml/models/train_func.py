from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from dryml.methods import Method

if TYPE_CHECKING:
    from dryml.core import Repo
    from .experiment import Experiment
    from .training_transaction import TrainingInputSession


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

    def input_session(self, exp: Experiment, *, positive: bool = True,
                      exact: bool = True, repo: Repo | None = None) -> TrainingInputSession:
        """Create invocation-local input control for this Experiment's TrainState.

        Args:
            exp: Experiment supplying the existing Dataset and state-owner graph.
            positive: Exact bool; False performs source-inert zero-work admission.
            exact: Exact bool selecting qualified recovery for fresh input. False
                explicitly selects positional compatibility; started runs retain
                their lane. Backend admission must exclude transient native owners
                or unsupported execution options before requesting exact input.
            repo: Optional Repo for owner evidence/invalidation, otherwise the
                active Repo or an invocation-owned in-memory Repo.
        Returns:
            A context manager with ``traversal``, ``update(candidate)``,
            ``sync_boundary()`` and ``capture()``. Enter before preparing native
            work; perform mutation and record_train_update inside its update guard,
            then invoke callbacks after leaving that guard.
        Raises:
            TypeError: Invalid booleans or state.
            ValueError: Stale/missing exact retry record or incompatible progress.
            RepoSaveError: Invalidated, changed, or concurrently reserved owners.
            TraversalError: Unsupported exact input, ownership, or capture failure.
        Side Effects:
            Positive entry prepares/opens input lazily and selects traversal-owned
            conditional replay. Saves and normal exact completion capture on
            demand. Every exit retires the transient provider and closes input;
            uncertain native mutations invalidate owners without claiming rollback.
            No cursor/provider is stored on this reusable TrainFunction.
        """
        from .training_transaction import TrainingInputSession
        return TrainingInputSession(exp, positive=positive, exact=exact, repo=repo)

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
