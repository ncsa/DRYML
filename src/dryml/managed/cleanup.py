"""Explicit fenced cleanup for retained managed realization history."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from dryml.core2.repo import make_store
from dryml.core2.store.store import Store
from dryml.formats.canonical import canonical_json_bytes
from dryml.records import (
    RealizationRecord,
    scan_record_refs,
    scan_spec_refs,
    spec_family_for_id,
    validate_realization_id,
)

from .errors import (
    ManagedCleanupRefusedError,
    ManagedLeaseConflictError,
    ManagedStateError,
)
from .locking import PlatformFileLock
from .refs import ManagedOutputRef
from .state import OperationKey, validate_declaration_fingerprint
from .store import ManagedOperationStore, _read_json, _write_json


CLEANUP_SCHEMA = "dryml.managed.cleanup.v1"


@dataclass(frozen=True, slots=True)
class CleanupPlan:
    """Immutable dry-run deletion intent derived from authoritative scans."""

    cleanup_id: str
    store_ref: str
    producer_cdef_id: str
    method: str
    declaration_fingerprint: str
    realization_ids: tuple[str, ...]
    paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.store_ref, str) or not self.store_ref:
            raise ManagedStateError("cleanup store_ref is malformed")
        OperationKey(self.producer_cdef_id, self.method)
        validate_declaration_fingerprint(self.declaration_fingerprint)
        if (
            not isinstance(self.cleanup_id, str)
            or re.fullmatch(r"cleanup-v1-[0-9a-f]{64}", self.cleanup_id) is None
        ):
            raise ManagedStateError("cleanup ID is malformed")
        if not isinstance(self.realization_ids, (list, tuple)):
            raise ManagedStateError("cleanup realization IDs are malformed")
        realization_ids = tuple(self.realization_ids)
        if not realization_ids or len(set(realization_ids)) != len(realization_ids):
            raise ManagedStateError("cleanup realization IDs are malformed")
        for realization_id in realization_ids:
            validate_realization_id(realization_id)
        if not isinstance(self.paths, (list, tuple)):
            raise ManagedStateError("cleanup paths are malformed")
        paths = tuple(self.paths)
        if len(set(paths)) != len(paths):
            raise ManagedStateError("cleanup paths contain duplicates")
        for value in paths:
            if not isinstance(value, str) or not value:
                raise ManagedStateError("cleanup path is malformed")
            path = Path(value)
            if (
                "\\" in value
                or path.is_absolute()
                or any(part in {"", ".", ".."} for part in path.parts)
            ):
                raise ManagedStateError("cleanup path is malformed")
        object.__setattr__(self, "realization_ids", realization_ids)
        object.__setattr__(self, "paths", paths)
        if _cleanup_id(self._identity()) != self.cleanup_id:
            raise ManagedStateError("cleanup intent identity is inconsistent")

    def to_json(self) -> dict[str, Any]:
        """Return the strict persisted deletion intent."""

        return {
            "schema": CLEANUP_SCHEMA,
            "schema_version": 1,
            "cleanup_id": self.cleanup_id,
            "store_ref": self.store_ref,
            "producer_cdef_id": self.producer_cdef_id,
            "method": self.method,
            "declaration_fingerprint": self.declaration_fingerprint,
            "realization_ids": list(self.realization_ids),
            "paths": list(self.paths),
        }

    @classmethod
    def from_json(cls, value: Any) -> "CleanupPlan":
        """Validate and decode one persisted cleanup intent."""

        fields = {
            "schema",
            "schema_version",
            "cleanup_id",
            "store_ref",
            "producer_cdef_id",
            "method",
            "declaration_fingerprint",
            "realization_ids",
            "paths",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ManagedStateError("cleanup intent fields are malformed")
        if value.get("schema") != CLEANUP_SCHEMA or value.get("schema_version") != 1:
            raise ManagedStateError("cleanup intent schema is unsupported")
        if not isinstance(value.get("realization_ids"), list) or not isinstance(
            value.get("paths"), list
        ):
            raise ManagedStateError("cleanup intent arrays are malformed")
        plan = cls(
            cleanup_id=value["cleanup_id"],
            store_ref=value["store_ref"],
            producer_cdef_id=value["producer_cdef_id"],
            method=value["method"],
            declaration_fingerprint=value["declaration_fingerprint"],
            realization_ids=tuple(value["realization_ids"]),
            paths=tuple(value["paths"]),
        )
        return plan

    def _identity(self) -> dict[str, Any]:
        return {
            "store_ref": self.store_ref,
            "producer_cdef_id": self.producer_cdef_id,
            "method": self.method,
            "declaration_fingerprint": self.declaration_fingerprint,
            "realization_ids": list(self.realization_ids),
            "paths": list(self.paths),
        }


@dataclass(frozen=True, slots=True)
class CleanupReport:
    """Result of an idempotent explicit cleanup execution."""

    cleanup_id: str
    realization_ids: tuple[str, ...]
    paths_deleted: tuple[str, ...]


def plan_cleanup(
    store: Store | str | Path,
    target: ManagedOutputRef,
    *,
    realization_ids: tuple[str, ...] | list[str],
) -> CleanupPlan:
    """Dry-run cleanup and refuse any currently protected realization.

    Protection is derived from direct managed control plus a full authoritative
    record scan. Active/running work, leased operations, checkpoint-bearing
    work and records referenced outside the selected closure are hard stops.
    """

    ids = tuple(realization_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("realization_ids must be unique and non-empty")
    selected = make_store(store)
    operation = _resolve_operation(selected, target, ids)
    try:
        with PlatformFileLock(operation._lock_path):
            paths = _derive_cleanup_paths(selected, operation, ids)
    except ManagedCleanupRefusedError:
        raise
    except ManagedLeaseConflictError as exc:
        raise ManagedCleanupRefusedError(
            "cleanup refuses a leased managed operation"
        ) from exc
    identity = {
        "store_ref": selected.records._store_ref(),
        "producer_cdef_id": operation.key.producer_cdef_id,
        "method": operation.key.method,
        "declaration_fingerprint": operation.declaration_fingerprint,
        "realization_ids": list(ids),
        "paths": list(paths),
    }
    return CleanupPlan(
        cleanup_id=_cleanup_id(identity),
        store_ref=identity["store_ref"],
        producer_cdef_id=identity["producer_cdef_id"],
        method=identity["method"],
        declaration_fingerprint=identity["declaration_fingerprint"],
        realization_ids=ids,
        paths=paths,
    )


def execute_cleanup(
    store: Store | str | Path,
    plan: CleanupPlan,
) -> CleanupReport:
    """Persist and execute only the supplied fenced deletion intent.

    The first execution revalidates all protection under the operation lock and
    the record publication lock. Once the immutable intent exists, retries only
    resume its exact path list; they never broaden cleanup from a changed scan.
    """

    if not isinstance(plan, CleanupPlan):
        raise TypeError("plan must be a CleanupPlan")
    selected = make_store(store)
    if selected.records._store_ref() != plan.store_ref:
        raise ManagedStateError("cleanup plan belongs to a different Store")
    managed = ManagedOperationStore(selected)
    key = OperationKey(plan.producer_cdef_id, plan.method)
    operation = managed.operation(key, plan.declaration_fingerprint)
    intent_path = _intent_path(managed, plan.cleanup_id)
    done_path = intent_path.with_suffix(".done.json")
    try:
        with PlatformFileLock(operation._lock_path):
            with selected.records._ref_index_mutation_lock():
                if intent_path.exists():
                    persisted = CleanupPlan.from_json(
                        _read_json(intent_path, "cleanup intent")
                    )
                    if persisted != plan:
                        raise ManagedStateError(
                            "cleanup intent conflicts with supplied plan"
                        )
                    _revalidate_persisted_intent(selected, operation, plan)
                else:
                    observed = _derive_cleanup_paths(
                        selected, operation, plan.realization_ids
                    )
                    if observed != plan.paths:
                        raise ManagedCleanupRefusedError(
                            "cleanup plan is stale; run a new dry-run"
                        )
                    _write_json(intent_path, plan.to_json(), immutable=True)
                control = operation._read_control()
                assert control is not None
                control_updates = {}
                if control.pending_realization_id in plan.realization_ids:
                    control_updates.update(
                        pending_realization_id=None,
                        current_attempt_id=None,
                        checkpoint_head=None,
                        diagnostics=(),
                        progress=None,
                    )
                if control.latest_realization_id in plan.realization_ids:
                    retained = tuple(
                        state
                        for state in operation.history()
                        if state.realization_id not in plan.realization_ids
                    )
                    latest = max(
                        retained,
                        key=lambda state: (state.sequence, state.realization_id),
                        default=None,
                    )
                    control_updates["latest_realization_id"] = (
                        None if latest is None else latest.realization_id
                    )
                if control_updates:
                    _write_json(
                        operation.control_path,
                        replace(control, **control_updates).to_json(),
                    )
                selected.records.mark_ref_index_dirty()
                for relative_path in plan.paths:
                    _delete_declared_path(selected.records.base_dir, relative_path)
                _write_json(
                    done_path,
                    {
                        "schema": CLEANUP_SCHEMA,
                        "schema_version": 1,
                        "cleanup_id": plan.cleanup_id,
                        "status": "complete",
                    },
                    immutable=True,
                )
    except ManagedLeaseConflictError as exc:
        raise ManagedCleanupRefusedError(
            "cleanup refuses a leased managed operation"
        ) from exc
    return CleanupReport(plan.cleanup_id, plan.realization_ids, plan.paths)


def resume_cleanup(store: Store | str | Path, cleanup_id: str) -> CleanupReport:
    """Resume one previously persisted cleanup intent idempotently."""

    selected = make_store(store)
    managed = ManagedOperationStore(selected)
    plan = CleanupPlan.from_json(
        _read_json(_intent_path(managed, cleanup_id), "cleanup intent")
    )
    return execute_cleanup(selected, plan)


def _resolve_operation(
    store: Store,
    target: ManagedOutputRef,
    realization_ids: tuple[str, ...],
):
    if not isinstance(target, ManagedOutputRef):
        raise TypeError("target must be a ManagedOutputRef")
    managed = ManagedOperationStore(store)
    key = OperationKey.from_producer(target.producer, target.method)
    namespace = managed._read_namespace(key, missing_ok=True)
    if namespace is None:
        raise ManagedStateError("managed operation has no retained state")
    candidates = tuple(
        operation
        for fingerprint in namespace.generations
        for operation in (managed.operation(key, fingerprint),)
        if all(
            operation._realization_path(realization_id).is_file()
            for realization_id in realization_ids
        )
    )
    if len(candidates) != 1:
        raise ManagedStateError(
            "cleanup realization IDs do not resolve to one declaration generation"
        )
    return candidates[0]


def _derive_cleanup_paths(store: Store, operation: Any, ids: tuple[str, ...]) -> tuple[str, ...]:
    selected = set(ids)
    states = []
    namespace = operation.managed_store._read_namespace(operation.key)
    assert namespace is not None
    if namespace.current_declaration_fingerprint == operation.declaration_fingerprint:
        active = operation.active(missing_ok=True)
        if active is not None and active.realization_id in selected:
            raise ManagedCleanupRefusedError("cleanup refuses active realization state")
    for realization_id in ids:
        state = operation._read_realization(realization_id)
        if state.status == "running":
            raise ManagedCleanupRefusedError("cleanup refuses active running state")
        if state.checkpoint_head is not None:
            raise ManagedCleanupRefusedError(
                "cleanup refuses checkpoint-referenced state"
            )
        states.append(state)

    selected_records = set()
    for state in states:
        if state.realization_record_id is None:
            continue
        realization = RealizationRecord.from_envelope(
            store.records.read_record(state.realization_record_id)
        )
        selected_records.add(state.realization_record_id)
        selected_records.add(realization.execution_record_id)
        selected_records.update(output.record_id for output in realization.outputs)

    _refuse_external_references(store, selected_records)

    root = store.records.base_dir
    paths = []
    for record_id in sorted(selected_records):
        record_path = store.records._record_path(record_id)
        if record_path.is_file():
            paths.append(record_path.relative_to(root).as_posix())
        product_root = store.records.product_root(record_id)
        if product_root.exists():
            paths.extend(
                path.relative_to(root).as_posix()
                for path in sorted(product_root.rglob("*"))
                if path.is_file()
            )
    for event in operation._activation_events():
        if event.realization_id in selected:
            paths.append(operation._activation_path(event).relative_to(root).as_posix())
    generation_active = operation.active_event(missing_ok=True)
    if (
        generation_active is not None
        and generation_active.realization_id in selected
    ):
        paths.append(operation.active_pointer_path.relative_to(root).as_posix())
    for state in states:
        for attempt_id in state.attempt_ids:
            if operation.attempts_dir.exists():
                for workspace in operation.attempts_dir.iterdir():
                    if workspace.is_dir() and workspace.name.endswith(attempt_id):
                        paths.extend(
                            path.relative_to(root).as_posix()
                            for path in sorted(workspace.rglob("*"))
                            if path.is_file()
                        )
        paths.append(
            operation._realization_path(state.realization_id)
            .relative_to(root)
            .as_posix()
        )
    return tuple(dict.fromkeys(paths))


def _revalidate_persisted_intent(
    store: Store,
    operation: Any,
    plan: CleanupPlan,
) -> None:
    selected = set(plan.realization_ids)
    namespace = operation.managed_store._read_namespace(operation.key)
    assert namespace is not None
    if namespace.current_declaration_fingerprint == operation.declaration_fingerprint:
        active = operation.active_event(missing_ok=True)
        if active is not None and active.realization_id in selected:
            raise ManagedCleanupRefusedError(
                "cleanup refuses newly active realization state"
            )
    for realization_id in plan.realization_ids:
        path = operation._realization_path(realization_id)
        if not path.exists():
            continue
        state = operation._read_realization(realization_id)
        if state.status == "running":
            raise ManagedCleanupRefusedError(
                "cleanup refuses newly active running state"
            )
        if state.checkpoint_head is not None:
            raise ManagedCleanupRefusedError(
                "cleanup refuses newly checkpoint-referenced state"
            )
    selected_records = {
        Path(relative_path).stem
        for relative_path in plan.paths
        if relative_path.startswith("records/items/")
        and relative_path.endswith(".json")
    }
    _refuse_external_references(store, selected_records)


def _refuse_external_references(store: Store, selected_records: set[str]) -> None:
    for record in store.records.iter_records():
        if record["id"] in selected_records:
            continue
        for mention in scan_record_refs(record):
            if mention.target_id in selected_records:
                raise ManagedCleanupRefusedError(
                    "cleanup refuses externally referenced or consumed state"
                )
    for spec in store.records.iter_specs():
        family = spec_family_for_id(spec["id"])
        for mention in scan_spec_refs(spec, family=family):
            if mention.target_id in selected_records:
                raise ManagedCleanupRefusedError(
                    "cleanup refuses spec-referenced state"
                )


def _delete_declared_path(root: Path, relative_path: str) -> None:
    path = root / relative_path
    try:
        path.unlink()
    except FileNotFoundError:
        return
    current = path.parent
    while current != root:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _cleanup_id(identity: dict[str, Any]) -> str:
    return "cleanup-v1-" + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def _intent_path(managed: ManagedOperationStore, cleanup_id: str) -> Path:
    if not isinstance(cleanup_id, str) or not cleanup_id.startswith("cleanup-v1-"):
        raise ManagedStateError("invalid cleanup ID")
    return managed.root / "cleanup-intents" / f"{cleanup_id}.json"


__all__ = [
    "CleanupPlan",
    "CleanupReport",
    "execute_cleanup",
    "plan_cleanup",
    "resume_cleanup",
]
