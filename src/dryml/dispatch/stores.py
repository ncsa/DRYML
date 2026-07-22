"""Same-host store marshalling helpers for local subprocess dispatch."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dryml.core2.store.dir import DirStore

from .errors import DispatchPlanningError, WorkerHandshakeError
from .protocol import WorkerStoreRef


@dataclass(frozen=True, slots=True)
class StoreMarshalPlan:
    """Parent-side store transfer decision for one dispatch."""

    strategy: str
    store_refs: tuple[WorkerStoreRef, ...]
    diagnostics: tuple[dict[str, Any], ...] = ()


def worker_store_ref_from_dir_store(store: DirStore, *, role: str = "shared", mode: str = "readwrite", label: str = "main", query_index: str | None = None) -> WorkerStoreRef:
    """Create a launch-time ``WorkerStoreRef`` for a local ``DirStore``."""

    return WorkerStoreRef(
        kind="dir_store",
        role=role,
        path=os.path.abspath(os.fspath(store.base_dir)),
        mode=mode,
        query_index=query_index or str(getattr(store, "query_index_policy", "auto")),
        label=label,
        capabilities={"objects": True, "records": True, "products": True, "query_index": True},
    )


def same_host_dir_store(store: Any) -> bool:
    """Return whether *store* can use same-host DirStore marshalling."""

    return isinstance(store, DirStore) and os.path.isabs(os.path.abspath(os.fspath(store.base_dir)))


def select_marshal_plan(store: Any, *, query_index: str | None = None) -> StoreMarshalPlan:
    """Choose the local subprocess marshalling strategy for a store."""

    if same_host_dir_store(store):
        return StoreMarshalPlan("shared_dir_store", (worker_store_ref_from_dir_store(store, query_index=query_index),))
    return StoreMarshalPlan(
        "unsupported",
        (),
        ({"message": "local subprocess dispatch requires same-host DirStore marshalling", "store_type": type(store).__name__},),
    )


def open_worker_store(ref: WorkerStoreRef) -> DirStore:
    """Open a worker store ref as a ``DirStore`` after handshake validation."""

    if ref.kind != "dir_store":
        raise WorkerHandshakeError("unsupported store ref kind", context={"kind": ref.kind})
    if not os.path.isdir(ref.path):
        raise WorkerHandshakeError("worker store path is not accessible", context={"path": ref.path, "label": ref.label})
    if ref.mode == "read":
        raise WorkerHandshakeError("read-only worker store refs are not yet enforced by DirStore", context={"path": ref.path, "label": ref.label, "mode": ref.mode})
    return DirStore(ref.path, query_index=ref.query_index)


def validate_worker_store_access(ref: WorkerStoreRef) -> dict[str, Any]:
    """Return structured store access status or raise for handshake failure."""

    exists = os.path.isdir(ref.path)
    if not exists:
        raise WorkerHandshakeError("worker store path is missing", context={"path": ref.path, "label": ref.label})
    readable = os.access(ref.path, os.R_OK)
    writable = os.access(ref.path, os.W_OK)
    if ref.mode in {"read", "readwrite"} and not readable:
        raise WorkerHandshakeError("worker store path is not readable", context={"path": ref.path, "label": ref.label})
    if ref.mode in {"write", "readwrite"} and not writable:
        raise WorkerHandshakeError("worker store path is not writable", context={"path": ref.path, "label": ref.label})
    return {"label": ref.label, "path": ref.path, "role": ref.role, "mode": ref.mode, "readable": readable, "writable": writable}


def require_supported_plan(plan: StoreMarshalPlan) -> None:
    """Raise when the selected marshalling plan cannot be executed."""

    if plan.strategy == "unsupported":
        raise DispatchPlanningError("unsupported store marshalling strategy", context={"diagnostics": plan.diagnostics})


__all__ = [
    "StoreMarshalPlan",
    "open_worker_store",
    "require_supported_plan",
    "same_host_dir_store",
    "select_marshal_plan",
    "validate_worker_store_access",
    "worker_store_ref_from_dir_store",
]
