"""Abstract contract implemented by generic Execute backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

from .future import ExecutionFuture

if TYPE_CHECKING:
    from dryml.environments import EnvironmentRequirement
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

        Raises:
            ExecutionError: If the configured backend cannot be initialized.
        """

    @abstractmethod
    def capabilities(self) -> frozenset[str]:
        """Return supported control capability names without claiming admission."""

    @abstractmethod
    def create_future(
        self, submission_id: str, output: "ExecutionOutput"
    ) -> ExecutionFuture[Any]:
        """Create one inert concrete Future without launching or binding output."""

    @abstractmethod
    def submit(
        self, call: "SubmittedCall[T]", *, future: ExecutionFuture[T]
    ) -> None:
        """Schedule one accepted call on its exact supplied Future.

        Raises:
            ExecutionError: If the call or Future does not belong to this backend.
        """

    @abstractmethod
    def discover(
        self,
        *,
        environment: "EnvironmentRequirement | None" = None,
        world: "WorldRequirement | None" = None,
        timeout: float,
    ) -> "DiscoverySnapshot":
        """Return a bounded discovery snapshot without invoking workload code."""

    @abstractmethod
    def resources(self, *, timeout: float) -> "ResourceSnapshot":
        """Return this backend's bounded resource observation."""

    @abstractmethod
    def reconcile_cleanup(self, submission_id: str, *, timeout: float) -> None:
        """Retry cleanup for one known submission without affecting other work."""

    @abstractmethod
    def close(self, *, cancel: bool, timeout: float | None) -> None:
        """Release backend-owned resources without touching caller-owned services."""


__all__ = ["Backend"]
