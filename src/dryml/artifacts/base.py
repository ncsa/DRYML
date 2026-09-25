from __future__ import annotations

from abc import abstractmethod
from typing import Any

from dryml.core import Serializable
from dryml.managed import ManagedContext, managed_operation


class Artifact(Serializable):
    """Abstract repo-backed computed payload with subclass-owned content.

    Concrete subclasses implement one managed ``compute`` operation and a
    boolean ``ready`` property. Artifact does not discover results, rerun work,
    validate payloads, or own input references; those policies remain with the
    concrete domain class.
    """

    @managed_operation()
    @abstractmethod
    def compute(self, *args: Any, managed: ManagedContext, **kwargs: Any) -> Any:
        """Compute this Artifact's content through DRYML's managed lifecycle.

        Args:
            *args: Domain-specific positional arguments.
            managed: Framework-provided managed operation context.
            **kwargs: Domain-specific keyword arguments.

        Returns:
            The domain-specific operation result.

        Raises:
            dryml.managed.ManagedError: If managed lifecycle handling fails.

        Side Effects:
            Concrete implementations may update their own content and publish
            immutable state through the managed-operation lifecycle.
        """

    @property
    @abstractmethod
    def ready(self) -> bool:
        """Return whether this Artifact's current content is usable.

        Returns:
            ``True`` only when the subclass considers its current payload
            complete and usable.

        Side Effects:
            Implementations must not compute or materialize input references
            merely to answer readiness.
        """

Artifact.__module__ = "dryml.artifacts"
