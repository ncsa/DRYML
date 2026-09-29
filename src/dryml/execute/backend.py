"""Abstract contract implemented by generic Execute backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

from .future import ExecutionFuture

if TYPE_CHECKING:
    from dryml.environments import EnvironmentRequirement
    from dryml.environments.selection import ResolvedEnvironmentSelection
    from dryml.worlds import WorldRequirement

    from .models import DiscoverySnapshot, ResourceSnapshot, SubmittedCall
    from .output import ExecutionOutput


T = TypeVar("T")


class Backend(ABC):
    """Define execution ownership for one configured backend.

    Backends create the concrete Future before acceptance, then own native setup,
    admission, workload transport, resource observations, and per-submission
    cleanup. They never select another backend or accept legacy Store transport.
    """

    @abstractmethod
    def start(self) -> None:
        """Initialize backend resources without accepting a workload.

        Returns:
            ``None`` after the backend is ready for discovery or accepted calls.

        Raises:
            ExecutionError: If the configured backend cannot be initialized or is
                already closed.
            BackendUnavailableError: If an optional selected backend is absent or
                cannot reach its required existing service.

        Side Effects:
            May connect to an existing service or initialize backend-local state.
            It must not launch a workload, create a spool, or reserve resources.
        """

    @abstractmethod
    def capabilities(self) -> frozenset[str]:
        """Return supported control capability names without claiming admission.

        Returns:
            Immutable stable capability names. Presence means this backend can
            attempt the control; its supplied requirement can still be rejected.

        Side Effects:
            None. This query must not initialize native services or launch work.
        """

    @abstractmethod
    def create_future(
        self, submission_id: str, output: "ExecutionOutput"
    ) -> ExecutionFuture[Any]:
        """Create one inert concrete Future without launching or binding output.

        Args:
            submission_id: Nonempty executor-generated correlation identifier.
            output: Exact unbound caller output holder for this submission.

        Returns:
            An inert concrete Future with the supplied identity and output holder.

        Raises:
            TypeError: If inputs have invalid types.
            ValueError: If the submission ID is invalid.
            ExecutionError: If this backend cannot produce its required Future.

        Side Effects:
            Does not start native work, bind output, or retain accepted cleanup.
        """

    @abstractmethod
    def submit(
        self, call: "SubmittedCall[T]", *, future: ExecutionFuture[T]
    ) -> None:
        """Schedule one accepted call on its exact supplied Future.

        Args:
            call: Immutable executor-accepted spool and admission metadata.
            future: The exact inert concrete Future created for ``call``.

        Returns:
            ``None``. Implementations retain and publish on ``future`` rather
            than returning another Future.

        Raises:
            ExecutionError: If the call or Future does not belong to this backend.
            RuntimeError: If the backend lifecycle cannot accept the call.

        Side Effects:
            Begins backend-owned asynchronous admission and native work. Terminal
            failures after acceptance are published on ``future``.
        """

    @abstractmethod
    def discover(
        self,
        *,
        environment: "EnvironmentRequirement | None" = None,
        environment_spec: "ResolvedEnvironmentSelection | None" = None,
        world: "WorldRequirement | None" = None,
        timeout: float,
    ) -> "DiscoverySnapshot":
        """
        Return a bounded discovery snapshot without invoking workload code.

                Args:
                    environment: Optional environment requirement used for
                    candidate and
                        feasible-plan evidence.
                    environment_spec: Optional frozen exact selector. Backends
                    inspect only
                        this selected target and must not enumerate fallback
                        candidates.
                    world: Optional world requirement used for feasible-plan
                    evidence.
                    timeout: Positive maximum observation time in seconds.

                Returns:
                    A non-reserving bounded discovery snapshot, possibly
                    incomplete.

                Raises:
                    TypeError: If a requirement or timeout has an invalid type.
                    ValueError: If ``timeout`` is not finite and positive.
                    TimeoutError: If initialization or observation exceeds
                    ``timeout``.
                    BackendUnavailableError: If the selected backend cannot
                    observe its
                        existing target.

                Side Effects:
                    May initialize the backend and perform bounded probe or
                    service I/O;
                    it never launches submitted workload code or reserves
                    capacity.
        """

    @abstractmethod
    def resources(self, *, timeout: float) -> "ResourceSnapshot":
        """Return this backend's bounded resource observation.

        Args:
            timeout: Positive maximum observation time in seconds.

        Returns:
            A backend-scoped logical resource snapshot, possibly incomplete.

        Raises:
            TypeError: If ``timeout`` has an invalid type.
            ValueError: If ``timeout`` is not finite and positive.
            TimeoutError: If observation exceeds ``timeout``.
            BackendUnavailableError: If current capacity cannot be observed.

        Side Effects:
            May initialize the backend and perform bounded native observation I/O.
            It does not reserve capacity or launch workload code.
        """

    @abstractmethod
    def reconcile_cleanup(self, submission_id: str, *, timeout: float) -> None:
        """Retry cleanup for one known submission without affecting other work.

        Args:
            submission_id: Exact accepted backend submission identifier.
            timeout: Positive bounded cleanup time in seconds.

        Returns:
            ``None`` after qualified native cleanup and bookkeeping release.

        Raises:
            TypeError: If arguments have invalid types.
            ValueError: If ``timeout`` is not finite and positive.
            RuntimeError: If the submission has not reached terminality.
            CleanupError: If native release cannot be confirmed within the bound.
            ExecutionError: If the submission is unknown to this backend.

        Side Effects:
            May terminate or close only the identified backend-owned native work.
            Failure preserves retry ownership rather than reporting cleanup done.
        """

    @abstractmethod
    def close(self, *, cancel: bool, timeout: float | None) -> None:
        """Release backend-owned resources without touching caller-owned services.

        Args:
            cancel: Whether to request cancellation of outstanding owned work.
            timeout: Optional finite positive cleanup bound; ``None`` selects the
                configuration's termination timeout.

        Returns:
            ``None`` after all backend-owned resources are released.

        Raises:
            TypeError: If ``cancel`` or ``timeout`` has an invalid type.
            ValueError: If ``timeout`` is not finite and positive when supplied.
            CleanupError: If owned work, handles, or connection release remains
                incomplete; retry ownership is retained.

        Side Effects:
            Stops backend admission and may request cancellation. It never stops
            caller-owned services or removes caller-owned directories.
        """


__all__ = ["Backend"]
