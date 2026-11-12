"""Metadata-only managed-operation inspection for query results and live Objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dryml.core import Node, Object, Repo
    from dryml.core.store.dir import DirStore
    from .model import ManagedStatus


def status(receiver: Node | Object, member: str, *, repo: Repo | None = None,
           control_store: DirStore | None = None) -> ManagedStatus:
    """Inspect a named managed operation without materializing its receiver.

    Args:
        receiver: Detached concrete Node with bound stateful ObjectIds, or a
            live Object. Query ObjectRefs and SavedNodes are accepted; recipes
            lacking required identities are rejected.
        member: Python member name, such as "train". No declaration is imported
            or resolved; an absent operation, including an unknown member name,
            reports not_started.
        repo: Existing Repo, or None for the current session Repo.
        control_store: Optional exact DirStore override. Otherwise use the active
            worker control default, then repo.default_store. This lookup never
            federates records across other connected Stores.

    Returns:
        Immutable ManagedStatus with state, failure_code, checkpoint and final
        metadata. Compare state with "completed" to check successful completion.
        Proven owner loss reports failed/owner_lost.

    Raises:
        ManagedConfigError: If receiver/member metadata is invalid.
        ManagedStoreError: If selected resources are missing or unsupported.
        ManagedControlError: If authority is corrupt or pending reconciliation.
        ManagedConflictError: If complete identity evidence conflicts or control
            changes during owner observation.

    Side Effects:
        Fresh metadata reads and nonblocking owner observation only. It does not
        create namespaces, import the receiver class, load state payloads, invoke
        work, repair records, or close borrowed resources. Returned status is a
        snapshot, not a reservation or a promise that saved bytes remain usable.
    """
    from .control import ManagedControlStore

    return ManagedControlStore.from_repo(repo, control_store=control_store).status_for(
        receiver, member,
    )


__all__ = ["status"]
