"""Selected DirStore authority for managed control and Object state.

This module deliberately resolves only caller-selected Stores or the current core
session Repo.  It neither records a locator nor searches control Stores.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dryml.core.reference_values import StateRef
from dryml.core.repo import Repo
from dryml.core.session import current_repo
from dryml.core.store.dir import DirStore

from .errors import ManagedRecoveryError, ManagedStoreError


@dataclass(frozen=True, slots=True)
class ResolvedStores:
    """Immutable selected state and control authority for one managed operation.

    Attributes:
        state_store: DirStore holding exact Object-state references.
        control_store: DirStore holding managed current-operation authority.

    The caller retains ownership of both Store handles.  Resolution opens no
    query indexes and does not create managed control files.
    """

    state_store: DirStore
    control_store: DirStore


def resolve_stores(
        obj: object,
        *,
        state_store: DirStore | None = None,
        control_store: DirStore | None = None,
        require_writable: bool = True) -> ResolvedStores:
    """Resolve managed Store authority before an operation mutates anything.

    Args:
        obj: Live Object whose exact last completed StateRef may disambiguate a
            multi-Store session Repo.
        state_store: Explicit DirStore for state, if supplied.
        control_store: Explicit DirStore for control, if supplied.
        require_writable: Whether both selected Stores must support publication.

    Returns:
        The selected state/control Store pair.  Omitted control authority uses
        the selected state Store.

    Raises:
        ManagedStoreError: If a Store is unsupported, session discovery is
            missing or ambiguous, publication is unavailable, or an exact saved
            StateRef cannot be validated without materializing an Object.

    Side Effects:
        Validation reads immutable StateRef authority only.  It never creates a
        Repo, opens a query index, writes a Store, or migrates state.
    """

    if state_store is not None and type(state_store) is not DirStore:
        raise ManagedStoreError(message="state_store must be an exact DirStore")
    if control_store is not None and type(control_store) is not DirStore:
        raise ManagedStoreError(message="control_store must be an exact DirStore")

    selected_state = state_store
    if selected_state is None:
        selected_state = _resolve_session_state_store(obj)
    selected_control = selected_state if control_store is None else control_store
    if require_writable:
        _require_publication(selected_state, "managed state publication", local_state=True)
        _require_publication(selected_control, "managed control publication", local_state=False)
    return ResolvedStores(selected_state, selected_control)


def validate_state_ref(store: DirStore, state_ref: StateRef) -> StateRef:
    """Validate one exact StateRef closure in one selected Store without hooks.

    Args:
        store: DirStore that must contain the complete immutable closure.
        state_ref: Exact StateRef to validate.

    Returns:
        The same exact StateRef after authority-only preflight succeeds.

    Raises:
        ManagedRecoveryError: If the record is absent, differs from the exact
            reference, or its local-state closure cannot be read.

    Side Effects:
        Creates and closes a private non-owning Repo view.  It never
        materializes, restores, indexes, or mutates user Objects or Stores.
    """

    if type(store) is not DirStore or type(state_ref) is not StateRef:
        raise ManagedRecoveryError("invalid_state_reference", "managed state validation requires exact DirStore and StateRef")
    try:
        record = store.read_state_ref_record(state_ref.digest())
        if record is None or record.state_ref != state_ref:
            raise ManagedRecoveryError("missing_state_reference", "selected state Store lacks the exact StateRef")
        from dryml.core.materialization import build_exact_state_load_plan

        view = Repo._for_state_io((store,))
        try:
            build_exact_state_load_plan(view, state_ref)
        finally:
            view.close()
    except ManagedRecoveryError:
        raise
    except BaseException as error:
        raise ManagedRecoveryError("invalid_state_reference", "selected state Store cannot validate the exact StateRef closure") from error
    return state_ref


def _resolve_session_state_store(obj: object) -> DirStore:
    """Choose the sole physical Store or a unique exact-current StateRef match."""

    repo = current_repo()
    if repo is None:
        raise ManagedStoreError("explicit_state_store_required", "no current Repo; supply state_store explicitly")
    stores = tuple(getattr(repo, "stores", ()))
    if not stores:
        raise ManagedStoreError("explicit_state_store_required", "current Repo has no Stores; supply state_store explicitly")
    if any(type(store) is not DirStore for store in stores):
        raise ManagedStoreError("explicit_state_store_required", "current Repo has a non-DirStore authority; supply state_store explicitly")
    physical = _deduplicate_roots(stores)
    if len(physical) == 1:
        return physical[0]
    state_ref = getattr(obj, "last_state_ref", None)
    if type(state_ref) is not StateRef:
        raise ManagedStoreError("explicit_state_store_required", "multiple Stores require one exact current StateRef match")
    matches = []
    for store in physical:
        try:
            validate_state_ref(store, state_ref)
        except ManagedRecoveryError as error:
            if error.__cause__ is not None:
                raise ManagedStoreError("state_store_unreadable", "could not inspect a current Repo Store") from error
            continue
        except BaseException as error:
            raise ManagedStoreError("state_store_unreadable", "could not inspect a current Repo Store") from error
        matches.append(store)
    if len(matches) != 1:
        raise ManagedStoreError("explicit_state_store_required", "multiple Stores require exactly one full exact StateRef match")
    return matches[0]


def _deduplicate_roots(stores: tuple[DirStore, ...]) -> tuple[DirStore, ...]:
    """Return one handle per canonical physical DirStore root in stable order."""

    selected = []
    seen = set()
    for store in stores:
        try:
            root = os.path.realpath(store.base_dir)
            stat_result = os.stat(root)
        except OSError as error:
            raise ManagedStoreError("state_store_unreadable", "could not inspect current Repo Store root") from error
        key = (root, stat_result.st_dev, stat_result.st_ino)
        if key not in seen:
            seen.add(key)
            selected.append(store)
    return tuple(selected)


def _require_publication(store: DirStore, operation: str, *, local_state: bool) -> None:
    """Translate Store capability failures to a managed boundary error."""

    try:
        store.preflight_publication(operation, local_state=local_state)
    except BaseException as error:
        raise ManagedStoreError("store_capability_unavailable", f"{operation} is unavailable") from error


__all__ = ["ResolvedStores", "resolve_stores", "validate_state_ref"]
