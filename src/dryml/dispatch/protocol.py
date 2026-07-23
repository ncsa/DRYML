"""JSON protocol models for local subprocess dispatch workers.

The protocol is intentionally file based for the first backend: request,
handshake, and response JSON files are separate from child stdout/stderr so user
prints cannot corrupt machine-readable worker messages.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from dryml.formats import CanonicalJSONError, json_ready
from dryml.records.policy import normalize_record_policy

from .errors import WorkerProtocolError


DISPATCH_WORKER_PROTOCOL_SCHEMA = "dryml.dispatch.worker_protocol.v1"
DISPATCH_WORKER_PROTOCOL_VERSION = 1
EXECUTION_ENVELOPE_SCHEMA = "dryml.execution_envelope.v1"
EXECUTION_ENVELOPE_SCHEMA_VERSION = 1

TRANSFER_STRATEGIES = frozenset({"shared_dir_store", "pickle_small", "unsupported"})
RESPONSE_STATUSES = frozenset({"ok", "failed", "cancelled", "timeout", "unsupported"})


@dataclass(frozen=True, slots=True)
class WorkerStoreRef:
    """Launch-time reference to a same-host store visible to the worker."""

    kind: Literal["dir_store"]
    role: Literal["input", "work", "output", "shared"]
    path: str
    mode: Literal["read", "write", "readwrite"] = "readwrite"
    query_index: Literal["auto", "sqlite", "memory", "none"] = "auto"
    label: str = "main"
    capabilities: Mapping[str, bool] = field(default_factory=lambda: {"objects": True, "records": True, "products": True, "query_index": True})

    def __post_init__(self) -> None:
        if self.kind != "dir_store":
            raise WorkerProtocolError("unsupported worker store kind", context={"kind": self.kind})
        if self.role not in {"input", "work", "output", "shared"}:
            raise WorkerProtocolError("invalid worker store role", context={"role": self.role})
        if self.mode not in {"read", "write", "readwrite"}:
            raise WorkerProtocolError("invalid worker store mode", context={"mode": self.mode})
        if self.query_index not in {"auto", "sqlite", "memory", "none"}:
            raise WorkerProtocolError("invalid worker store query_index", context={"query_index": self.query_index})
        if not isinstance(self.path, str) or not os.path.isabs(self.path):
            raise WorkerProtocolError("dir_store worker refs require an absolute path", context={"path": self.path})
        if not isinstance(self.label, str) or not self.label:
            raise WorkerProtocolError("worker store label must be a non-empty string")
        if not isinstance(self.capabilities, Mapping):
            raise WorkerProtocolError("worker store capabilities must be a mapping")
        object.__setattr__(self, "path", os.path.abspath(self.path))
        object.__setattr__(self, "capabilities", {str(key): bool(value) for key, value in self.capabilities.items()})

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "WorkerStoreRef":
        """Build a store ref from JSON data."""

        if not isinstance(data, Mapping):
            raise WorkerProtocolError("worker store ref must be a mapping", context={"type": type(data).__name__})
        unknown = set(data) - {"kind", "role", "path", "mode", "query_index", "label", "capabilities"}
        if unknown:
            raise WorkerProtocolError("worker store ref contains unknown fields", context={"fields": sorted(unknown)})
        return cls(
            kind=data.get("kind"),
            role=data.get("role"),
            path=data.get("path"),
            mode=data.get("mode", "readwrite"),
            query_index=data.get("query_index", "auto"),
            label=data.get("label", "main"),
            capabilities=data.get("capabilities") or {"objects": True, "records": True, "products": True, "query_index": True},
        )

    def to_json(self) -> dict[str, Any]:
        """Return the canonical JSON form of this launch-time store ref."""

        return {
            "kind": self.kind,
            "role": self.role,
            "path": self.path,
            "mode": self.mode,
            "query_index": self.query_index,
            "label": self.label,
            "capabilities": dict(sorted(self.capabilities.items())),
        }


@dataclass(frozen=True, slots=True)
class ExecutionEnvelope:
    """Launch-only worker request data.

    Unlike DispatchSpec and ExecutionRecipe, this envelope may contain absolute
    local paths, raw environment/runtime/world configuration, and other
    non-identity launch details. ``launch`` may also carry separately named
    projected world/allocation specs for provenance publication; raw launch
    specs must not be written to the Store.
    """

    dispatch_spec: Mapping[str, Any]
    execution_recipe: Mapping[str, Any]
    operation_spec: Mapping[str, Any]
    environment_spec: Mapping[str, Any] = field(default_factory=dict)
    runtime_spec: Mapping[str, Any] = field(default_factory=dict)
    allocation_view: Mapping[str, Any] = field(default_factory=dict)
    store_refs: tuple[WorkerStoreRef, ...] = ()
    transfer: Mapping[str, Any] = field(default_factory=lambda: {"strategy": "shared_dir_store"})
    result_policy: Mapping[str, Any] = field(default_factory=lambda: {"return": "canonical_or_refs"})
    record_policy: str = "descriptive"
    reporting: Mapping[str, Any] = field(default_factory=dict)
    handshake: Mapping[str, Any] = field(default_factory=lambda: {"min_protocol": 1, "required_features": ["operation.function_call", "store.dir", "runtime.worker"]})
    launch: Mapping[str, Any] = field(default_factory=dict)
    schema: str = EXECUTION_ENVELOPE_SCHEMA
    schema_version: int = EXECUTION_ENVELOPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != EXECUTION_ENVELOPE_SCHEMA or self.schema_version != EXECUTION_ENVELOPE_SCHEMA_VERSION:
            raise WorkerProtocolError("unsupported execution envelope schema", context={"schema": self.schema, "schema_version": self.schema_version})
        for name in ("dispatch_spec", "execution_recipe", "operation_spec", "environment_spec", "runtime_spec", "allocation_view", "transfer", "result_policy", "reporting", "handshake", "launch"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise WorkerProtocolError(f"{name} must be a mapping", context={"type": type(value).__name__})
        strategy = self.transfer.get("strategy")
        if strategy not in TRANSFER_STRATEGIES:
            raise WorkerProtocolError("unsupported transfer strategy", context={"strategy": strategy})
        try:
            object.__setattr__(self, "record_policy", normalize_record_policy(self.record_policy))
        except Exception as exc:
            raise WorkerProtocolError("invalid record_policy", context=getattr(exc, "context", {})) from exc
        object.__setattr__(self, "store_refs", _store_ref_tuple(self.store_refs))
        _validate_coordination(self.launch.get("coordination"), self.allocation_view)

    @property
    def operation_id(self) -> str | None:
        """Return the embedded operation ID when available."""

        return self.operation_spec.get("id") or self.dispatch_spec.get("payload", {}).get("operation_id")

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "ExecutionEnvelope":
        """Validate and build an execution envelope from JSON data."""

        if not isinstance(data, Mapping):
            raise WorkerProtocolError("execution envelope must be a mapping", context={"type": type(data).__name__})
        unknown = set(data) - {
            "schema",
            "schema_version",
            "dispatch_spec",
            "execution_recipe",
            "operation_spec",
            "environment_spec",
            "runtime_spec",
            "allocation_view",
            "store_refs",
            "transfer",
            "result_policy",
            "record_policy",
            "reporting",
            "handshake",
            "launch",
        }
        if unknown:
            raise WorkerProtocolError("execution envelope contains unknown fields", context={"fields": sorted(unknown)})
        for required in ("dispatch_spec", "execution_recipe", "operation_spec"):
            if required not in data:
                raise WorkerProtocolError("execution envelope missing required field", context={"field": required})
        return cls(
            schema=data.get("schema", EXECUTION_ENVELOPE_SCHEMA),
            schema_version=data.get("schema_version", EXECUTION_ENVELOPE_SCHEMA_VERSION),
            dispatch_spec=data["dispatch_spec"],
            execution_recipe=data["execution_recipe"],
            operation_spec=data["operation_spec"],
            environment_spec=data.get("environment_spec") or {},
            runtime_spec=data.get("runtime_spec") or {},
            allocation_view=data.get("allocation_view") or {},
            store_refs=data.get("store_refs") or (),
            transfer=data.get("transfer") or {"strategy": "shared_dir_store"},
            result_policy=data.get("result_policy") or {"return": "canonical_or_refs"},
            record_policy=data.get("record_policy", "descriptive"),
            reporting=data.get("reporting") or {},
            handshake=data.get("handshake") or {"min_protocol": 1, "required_features": ["operation.function_call", "store.dir", "runtime.worker"]},
            launch=data.get("launch") or {},
        )

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-ready request mapping."""

        return _json_ready(
            {
                "schema": self.schema,
                "schema_version": self.schema_version,
                "dispatch_spec": self.dispatch_spec,
                "execution_recipe": self.execution_recipe,
                "operation_spec": self.operation_spec,
                "environment_spec": self.environment_spec,
                "runtime_spec": self.runtime_spec,
                "allocation_view": self.allocation_view,
                "store_refs": [ref.to_json() for ref in self.store_refs],
                "transfer": self.transfer,
                "result_policy": self.result_policy,
                "record_policy": self.record_policy,
                "reporting": self.reporting,
                "handshake": self.handshake,
                "launch": self.launch,
            },
            "execution envelope",
        )


@dataclass(frozen=True, slots=True)
class WorkerHandshakeRequest:
    """Parent-requested worker protocol constraints."""

    min_protocol: int = 1
    required_features: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, data: Mapping[str, Any] | None) -> "WorkerHandshakeRequest":
        """Build handshake requirements from envelope JSON."""

        data = data or {}
        if not isinstance(data, Mapping):
            raise WorkerProtocolError("handshake request must be a mapping")
        required = _string_tuple(data.get("required_features") or (), "required_features")
        return cls(min_protocol=int(data.get("min_protocol", 1)), required_features=required)

    def to_json(self) -> dict[str, Any]:
        """Return JSON-ready handshake requirements."""

        return {"min_protocol": self.min_protocol, "required_features": list(self.required_features)}


@dataclass(frozen=True, slots=True)
class WorkerHandshakeResponse:
    """Worker handshake facts validated before trusting execution results."""

    status: Literal["ok", "unsupported", "failed"]
    protocol_schema: str
    protocol_version: int
    dryml_version: str | None
    python_version: str
    platform: str
    pid: int
    features: tuple[str, ...]
    operation_kinds: tuple[str, ...]
    call_transports: tuple[str, ...]
    store_ref_kinds: tuple[str, ...]
    record_schemas: Mapping[str, int]
    runtime_modes: tuple[str, ...]
    environment_kind: str | None = None
    process_group: bool = False
    store_status: Mapping[str, Any] = field(default_factory=dict)
    world_id: str | None = None
    world_allocation_id: str | None = None
    worker_key: Mapping[str, Any] | None = None
    diagnostics: tuple[Mapping[str, Any], ...] = ()

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "WorkerHandshakeResponse":
        """Validate a worker handshake response."""

        if not isinstance(data, Mapping):
            raise WorkerProtocolError("handshake response must be a mapping", context={"type": type(data).__name__})
        return cls(
            status=data.get("status"),
            protocol_schema=data.get("protocol_schema"),
            protocol_version=int(data.get("protocol_version", 0)),
            dryml_version=data.get("dryml_version"),
            python_version=str(data.get("python_version", "")),
            platform=str(data.get("platform", "")),
            pid=int(data.get("pid", 0)),
            features=_string_tuple(data.get("features") or (), "features"),
            operation_kinds=_string_tuple(data.get("operation_kinds") or (), "operation_kinds"),
            call_transports=_string_tuple(data.get("call_transports") or (), "call_transports"),
            store_ref_kinds=_string_tuple(data.get("store_ref_kinds") or (), "store_ref_kinds"),
            record_schemas=dict(data.get("record_schemas") or {}),
            runtime_modes=_string_tuple(data.get("runtime_modes") or (), "runtime_modes"),
            environment_kind=data.get("environment_kind"),
            process_group=bool(data.get("process_group", False)),
            store_status=data.get("store_status") or {},
            world_id=data.get("world_id"),
            world_allocation_id=data.get("world_allocation_id"),
            worker_key=data.get("worker_key"),
            diagnostics=tuple(_diagnostics(data.get("diagnostics") or ())),
        )

    def __post_init__(self) -> None:
        if self.status not in {"ok", "unsupported", "failed"}:
            raise WorkerProtocolError("invalid handshake status", context={"status": self.status})
        if self.protocol_schema != DISPATCH_WORKER_PROTOCOL_SCHEMA or self.protocol_version != DISPATCH_WORKER_PROTOCOL_VERSION:
            raise WorkerProtocolError("unsupported worker protocol", context={"schema": self.protocol_schema, "version": self.protocol_version})
        if self.worker_key is not None and not isinstance(self.worker_key, Mapping):
            raise WorkerProtocolError("handshake worker_key must be a mapping", context={"type": type(self.worker_key).__name__})

    def to_json(self) -> dict[str, Any]:
        """Return the JSON-ready handshake response."""

        return _json_ready(
            {
                "status": self.status,
                "protocol_schema": self.protocol_schema,
                "protocol_version": self.protocol_version,
                "dryml_version": self.dryml_version,
                "python_version": self.python_version,
                "platform": self.platform,
                "pid": self.pid,
                "features": list(self.features),
                "operation_kinds": list(self.operation_kinds),
                "call_transports": list(self.call_transports),
                "store_ref_kinds": list(self.store_ref_kinds),
                "record_schemas": self.record_schemas,
                "runtime_modes": list(self.runtime_modes),
                "environment_kind": self.environment_kind,
                "process_group": self.process_group,
                "store_status": self.store_status,
                "world_id": self.world_id,
                "world_allocation_id": self.world_allocation_id,
                "worker_key": self.worker_key,
                "diagnostics": list(self.diagnostics),
            },
            "handshake response",
        )


@dataclass(frozen=True, slots=True)
class WorkerResponse:
    """Compact worker result with canonical values and store-owned refs."""

    status: Literal["ok", "failed", "cancelled", "timeout", "unsupported"]
    operation_id: str | None = None
    dispatch_id: str | None = None
    recipe_id: str | None = None
    result_canonical: Any = None
    result_cdef_ids: tuple[str, ...] = ()
    updated_cdef_ids: tuple[str, ...] = ()
    produced_record_ids: tuple[str, ...] = ()
    execution_record_id: str | None = None
    stdout_ref: Mapping[str, Any] | None = None
    stderr_ref: Mapping[str, Any] | None = None
    diagnostics: tuple[Mapping[str, Any], ...] = ()
    error: Mapping[str, Any] | None = None
    cancellation: Mapping[str, Any] | None = None
    managed_result: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.status not in RESPONSE_STATUSES:
            raise WorkerProtocolError("invalid worker response status", context={"status": self.status})
        for field_name in ("result_cdef_ids", "updated_cdef_ids", "produced_record_ids"):
            object.__setattr__(self, field_name, _string_tuple(getattr(self, field_name), field_name))
        object.__setattr__(self, "diagnostics", tuple(_diagnostics(self.diagnostics)))
        if self.error is not None and not isinstance(self.error, Mapping):
            raise WorkerProtocolError("worker response error must be a mapping", context={"type": type(self.error).__name__})
        if self.cancellation is not None and not isinstance(self.cancellation, Mapping):
            raise WorkerProtocolError("worker response cancellation must be a mapping", context={"type": type(self.cancellation).__name__})
        if self.managed_result is not None:
            if not isinstance(self.managed_result, Mapping):
                raise WorkerProtocolError("worker response managed_result must be a mapping", context={"type": type(self.managed_result).__name__})
            object.__setattr__(self, "managed_result", _validate_managed_result(self.managed_result))
        _validate_response_context(self)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "WorkerResponse":
        """Validate and build a worker response from JSON."""

        if not isinstance(data, Mapping):
            raise WorkerProtocolError("worker response must be a mapping", context={"type": type(data).__name__})
        unknown = set(data) - {
            "status",
            "operation_id",
            "dispatch_id",
            "recipe_id",
            "result_canonical",
            "result_cdef_ids",
            "updated_cdef_ids",
            "produced_record_ids",
            "execution_record_id",
            "stdout_ref",
            "stderr_ref",
            "diagnostics",
            "error",
            "cancellation",
            "managed_result",
        }
        if unknown:
            raise WorkerProtocolError("worker response contains unknown fields", context={"fields": sorted(unknown)})
        return cls(
            status=data.get("status"),
            operation_id=data.get("operation_id"),
            dispatch_id=data.get("dispatch_id"),
            recipe_id=data.get("recipe_id"),
            result_canonical=data.get("result_canonical"),
            result_cdef_ids=data.get("result_cdef_ids") or (),
            updated_cdef_ids=data.get("updated_cdef_ids") or (),
            produced_record_ids=data.get("produced_record_ids") or (),
            execution_record_id=data.get("execution_record_id"),
            stdout_ref=data.get("stdout_ref"),
            stderr_ref=data.get("stderr_ref"),
            diagnostics=tuple(data.get("diagnostics") or ()),
            error=data.get("error"),
            cancellation=data.get("cancellation"),
            managed_result=data.get("managed_result"),
        )

    def to_json(self) -> dict[str, Any]:
        """Return response JSON without object-state bytes."""

        data = {
            "status": self.status,
            "operation_id": self.operation_id,
            "dispatch_id": self.dispatch_id,
            "recipe_id": self.recipe_id,
            "result_canonical": self.result_canonical,
            "result_cdef_ids": list(self.result_cdef_ids),
            "updated_cdef_ids": list(self.updated_cdef_ids),
            "produced_record_ids": list(self.produced_record_ids),
            "execution_record_id": self.execution_record_id,
            "stdout_ref": self.stdout_ref,
            "stderr_ref": self.stderr_ref,
            "diagnostics": list(self.diagnostics),
            "error": self.error,
            "cancellation": self.cancellation,
            "managed_result": self.managed_result,
        }
        return _json_ready({key: value for key, value in data.items() if value not in (None, (), [])}, "worker response")


@dataclass(frozen=True, slots=True)
class DispatchResult:
    """Stable first-pass public result returned by ``dryml.dispatch``."""

    status: str
    operation_id: str | None = None
    dispatch_id: str | None = None
    recipe_id: str | None = None
    execution_record_id: str | None = None
    result_canonical: Any = None
    result_cdef_ids: tuple[str, ...] = ()
    produced_record_ids: tuple[str, ...] = ()
    updated_cdef_ids: tuple[str, ...] = ()
    stdout_ref: Mapping[str, Any] | None = None
    stderr_ref: Mapping[str, Any] | None = None
    diagnostics: tuple[Mapping[str, Any], ...] = ()
    error: Mapping[str, Any] | None = None
    cancellation: Mapping[str, Any] | None = None
    managed_result: Mapping[str, Any] | None = None

    @classmethod
    def from_worker_response(cls, response: WorkerResponse) -> "DispatchResult":
        """Build a public result from a compact worker response."""

        return cls(
            status=response.status,
            operation_id=response.operation_id,
            dispatch_id=response.dispatch_id,
            recipe_id=response.recipe_id,
            execution_record_id=response.execution_record_id,
            result_canonical=response.result_canonical,
            result_cdef_ids=response.result_cdef_ids,
            produced_record_ids=response.produced_record_ids,
            updated_cdef_ids=response.updated_cdef_ids,
            stdout_ref=response.stdout_ref,
            stderr_ref=response.stderr_ref,
            diagnostics=response.diagnostics,
            error=response.error,
            cancellation=response.cancellation,
            managed_result=response.managed_result,
        )

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-ready public result mapping."""

        return _json_ready(
            {
                "status": self.status,
                "operation_id": self.operation_id,
                "dispatch_id": self.dispatch_id,
                "recipe_id": self.recipe_id,
                "execution_record_id": self.execution_record_id,
                "result_canonical": self.result_canonical,
                "result_cdef_ids": list(self.result_cdef_ids),
                "produced_record_ids": list(self.produced_record_ids),
                "updated_cdef_ids": list(self.updated_cdef_ids),
                "stdout_ref": self.stdout_ref,
                "stderr_ref": self.stderr_ref,
                "diagnostics": list(self.diagnostics),
                "error": self.error,
                "cancellation": self.cancellation,
                "managed_result": self.managed_result,
            },
            "dispatch result",
        )


def read_json_file(path: str) -> Any:
    """Read one UTF-8 JSON protocol file."""

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_file(path: str, data: Any) -> None:
    """Atomically write one UTF-8 JSON protocol file."""

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")
    deadline = time.monotonic() + 5.0
    while True:
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def load_envelope(path: str) -> ExecutionEnvelope:
    """Load and validate an execution envelope from disk."""

    return ExecutionEnvelope.from_json(read_json_file(path))


def save_envelope(path: str, envelope: ExecutionEnvelope) -> None:
    """Write an execution envelope to disk."""

    write_json_file(path, envelope.to_json())


def _store_ref_tuple(value: Sequence[Any]) -> tuple[WorkerStoreRef, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise WorkerProtocolError("store_refs must be a JSON array", context={"type": type(value).__name__})
    return tuple(item if isinstance(item, WorkerStoreRef) else WorkerStoreRef.from_json(item) for item in value)


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise WorkerProtocolError(f"{field_name} must be a JSON array", context={"type": type(value).__name__})
    result = tuple(value)
    if any(not isinstance(item, str) for item in result):
        raise WorkerProtocolError(f"{field_name} items must be strings")
    return result


def _diagnostics(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise WorkerProtocolError("diagnostics must be a JSON array", context={"type": type(value).__name__})
    result = []
    for item in value:
        if not isinstance(item, Mapping):
            raise WorkerProtocolError("diagnostics items must be mappings", context={"type": type(item).__name__})
        result.append(dict(item))
    return tuple(result)


def _validate_response_context(response: WorkerResponse) -> None:
    if response.status == "ok":
        if response.error is not None:
            raise WorkerProtocolError("ok worker responses must not include error")
        if response.cancellation is not None:
            raise WorkerProtocolError("ok worker responses must not include cancellation")
        return
    if response.status == "cancelled":
        if response.cancellation is None:
            raise WorkerProtocolError("cancelled worker responses require cancellation")
        if response.error is not None:
            raise WorkerProtocolError("cancelled worker responses must not include error")
        return
    if response.cancellation is not None:
        raise WorkerProtocolError("cancellation is only valid on cancelled worker responses")
    if response.status in {"failed", "timeout", "unsupported"} and response.error is None and not response.diagnostics:
        raise WorkerProtocolError("failed, timeout, and unsupported worker responses require error or diagnostics", context={"status": response.status})


def _validate_managed_result(value: Mapping[str, Any]) -> dict[str, Any]:
    data = _json_ready(dict(value), "managed_result")
    if data.get("schema") != "dryml.managed.operation_result.v1" or data.get("schema_version") != 1:
        raise WorkerProtocolError("worker response managed_result schema is unsupported")
    if data.get("status") not in {"ok", "failed", "interrupted", "cancelled", "timeout"}:
        raise WorkerProtocolError("worker response managed_result status is unsupported")
    effects = data.get("effects")
    if effects is not None and (not isinstance(effects, Mapping) or len(effects) > 256):
        raise WorkerProtocolError("worker response managed_result effects are malformed")
    representations = data.get("representations")
    if representations is not None and (
        not isinstance(representations, list) or len(representations) > 256
    ):
        raise WorkerProtocolError("worker response managed_result representations are malformed")
    checkpoint = data.get("checkpoint_head")
    if checkpoint is not None and not isinstance(checkpoint, str):
        raise WorkerProtocolError("worker response managed_result checkpoint is malformed")
    return data


def _validate_coordination(coordination: Any, allocation_view: Mapping[str, Any]) -> None:
    if coordination is None:
        return
    if not isinstance(coordination, Mapping):
        raise WorkerProtocolError("coordination metadata must be a mapping", context={"type": type(coordination).__name__})
    unknown = set(coordination) - {"group_id", "worker_key", "start_path", "cancel_path", "heartbeat_path", "start_timeout"}
    if unknown:
        raise WorkerProtocolError("coordination metadata contains unknown fields", context={"fields": sorted(unknown)})
    for field_name in ("start_path", "cancel_path", "heartbeat_path"):
        value = coordination.get(field_name)
        if value is not None and (not isinstance(value, str) or not os.path.isabs(value)):
            raise WorkerProtocolError("coordination paths must be absolute", context={"field": field_name, "path": value})
    key = coordination.get("worker_key")
    if key is None:
        return
    if not isinstance(key, Mapping):
        raise WorkerProtocolError("coordination worker_key must be a mapping")
    for field_name in ("role", "replica", "rank", "local_rank"):
        if key.get(field_name) != allocation_view.get(field_name):
            raise WorkerProtocolError("coordination worker_key does not match allocation view", context={"field": field_name, "worker_key": key.get(field_name), "allocation_view": allocation_view.get(field_name)})


def _json_ready(value: Any, field_name: str) -> Any:
    try:
        return json_ready(value)
    except CanonicalJSONError as exc:
        raise WorkerProtocolError(f"{field_name} is not JSON-ready", context=exc.context) from exc


__all__ = [
    "DISPATCH_WORKER_PROTOCOL_SCHEMA",
    "DISPATCH_WORKER_PROTOCOL_VERSION",
    "EXECUTION_ENVELOPE_SCHEMA",
    "EXECUTION_ENVELOPE_SCHEMA_VERSION",
    "DispatchResult",
    "ExecutionEnvelope",
    "WorkerHandshakeRequest",
    "WorkerHandshakeResponse",
    "WorkerResponse",
    "WorkerStoreRef",
    "load_envelope",
    "read_json_file",
    "save_envelope",
    "write_json_file",
]
