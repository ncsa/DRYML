"""Immutable caller policy and invocation snapshots for managed operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from dryml.core.repo import Repo
from dryml.core.store.dir import DirStore
from dryml.core.store.store import Store

from .errors import ManagedConfigError, ManagedStoreError


Callback = Callable[[object, object], object]


@dataclass(frozen=True, kw_only=True)
class ManagedConfig:
    """Caller-owned policy accepted through a managed bound method.

    Args:
        state_repo: Optional borrowed :class:`~dryml.core.repo.Repo` or existing
            :class:`~dryml.core.store.store.Store` selected for Object state.
        control_store: Optional supported DirStore selected for control authority.
        rerun: Exact bool requesting a fresh attempt when lifecycle support is
            available.
        callbacks: ``None`` or a list of at most 64 callable checkpoint observers.

    Raises:
        ManagedConfigError: If ``rerun`` or ``callbacks`` violates the closed
            caller grammar.
        ManagedStoreError: If state authority is not a Repo or Store, or control
            authority is not an exact DirStore.

    Side Effects:
        None. The callback list remains caller-owned; :meth:`snapshot` creates
        the invocation-local immutable copy.
    """

    state_repo: Repo | Store | None = None
    control_store: DirStore | None = None
    rerun: bool = False
    callbacks: list[Callback] | None = None

    def __post_init__(self) -> None:
        """Reject unsupported policy before managed invocation can begin."""

        if self.state_repo is not None and not isinstance(self.state_repo, (Repo, Store)):
            raise ManagedStoreError(message="state_repo must be a Repo or Store")
        if self.control_store is not None and type(self.control_store) is not DirStore:
            raise ManagedStoreError(message="control_store must be a DirStore")
        if type(self.rerun) is not bool:
            raise ManagedConfigError(message="rerun must be an exact bool")
        if self.callbacks is not None:
            if type(self.callbacks) is not list:
                raise ManagedConfigError(message="callbacks must be a list or None")
            if len(self.callbacks) > 64:
                raise ManagedConfigError(message="callbacks may contain at most 64 entries")
            if any(not callable(callback) for callback in self.callbacks):
                raise ManagedConfigError(message="callbacks entries must be callable")

    def snapshot(self) -> "_ResolvedManagedConfig":
        """Create one immutable invocation-local view of this caller policy.

        Returns:
            A private immutable policy value with callbacks copied to a tuple for
            the current invocation.

        Raises:
            ManagedConfigError: If a caller mutated the retained callback list to
                an unsupported value after construction.

        Side Effects:
            None. The original callback list is neither changed nor executed.
        """

        callbacks = self.callbacks
        if callbacks is not None:
            if type(callbacks) is not list:
                raise ManagedConfigError(message="callbacks must remain a list")
            if len(callbacks) > 64:
                raise ManagedConfigError(message="callbacks may contain at most 64 entries")
            if any(not callable(callback) for callback in callbacks):
                raise ManagedConfigError(message="callbacks entries must be callable")
            callback_snapshot = tuple(callbacks)
        else:
            callback_snapshot = ()
        return _ResolvedManagedConfig(
            state_repo=self.state_repo,
            control_store=self.control_store,
            rerun=self.rerun,
            callbacks=callback_snapshot,
        )


@dataclass(frozen=True, slots=True)
class _ResolvedManagedConfig:
    """Private immutable policy snapshot passed to one lifecycle invocation."""

    state_repo: Repo | Store | None
    control_store: DirStore | None
    rerun: bool
    callbacks: tuple[Callback, ...]
