"""Closed version-local JSON transport for fixed Dispatch declaration probes.

The protocol transports a source-free inspection snapshot and owner-encoded
declaration views.  It never transports user callables, receivers, arguments,
state, imports, or Store authority.
"""

from __future__ import annotations

import contextvars
import hashlib
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dryml.code import KernelCall, StaticDependenciesKernel, probe
from dryml.code.inspection import (
    InspectionCall,
    InspectionRecord,
    InspectionSnapshot,
    InspectionTarget,
)
from dryml.code.targets import TargetInfo
from dryml.environments.kernel import (
    EnvironmentRequirementsKernel,
)
from dryml.requirements import (
    RequirementIssue,
    RequirementReport,
    RequirementResult,
    RequirementSource,
)
from dryml.worlds.kernel import (
    WorldRequirementsKernel,
)

from dryml.formats import (
    CanonicalJSONError,
    canonical_json_bytes,
    canonical_json_load_bytes,
)

_REQUEST_SCHEMA = "dryml.dispatch.probe_request.v1"
_RESULT_SCHEMA = "dryml.dispatch.probe_result.v1"
_COLLECTOR_VERSION = 1
_MAX_BYTES = 4 * 1024 * 1024
_MAX_DIAGNOSTICS = 64
_MAX_DIAGNOSTIC_TEXT = 512
_MAX_TARGETS = 4_096
_MAX_CALLS = 16_384
_MAX_TRAVERSAL_DEPTH = 128
_SAFE_DIAGNOSTICS = frozenset(
    {"static.depth_limit", "static.target_limit", "static.unresolved"}
)
_JSON_BOUNDS = {
    "max_depth": 64,
    "max_nodes": 200_000,
    "max_entries": 200_000,
    "max_string": 65_536,
    "max_int_bits": 4096,
}


class ProbeProtocolError(ValueError):
    """Report malformed, mismatched, or oversized private probe transport.

    This error is deliberately value-free. Worker execution reports its type
    through Execute rather than retaining arbitrary request or diagnostic data.
    """


@dataclass(frozen=True, slots=True)
class ProbeRequest:
    """One canonical probe request retained through one probe operation.

    Args:
        data: Canonical JSON bytes passed to the fixed worker callable.
        digest: SHA-256 association digest of the request body.
        root_id: Snapshot-local root identifier required by the result.

    Side Effects:
        None. This carrier has no live target, declaration, Store, or backend
        ownership.
    """

    data: bytes
    digest: str
    root_id: str


@dataclass(frozen=True, slots=True)
class ProbeWireResult:
    """Validated detached result of one fixed projected declaration probe.

    Args:
        complete: Whether bounded static resolution proved every selected edge.
        diagnostics: Bounded coverage categories with no source/value text.
        environment: Existing environment requirement combination result.
        world: Existing world requirement combination result.

    Side Effects:
        None. Domain values remain immutable owner-owned result values.
    """

    complete: bool
    diagnostics: tuple[str, ...]
    environment: RequirementResult[Any]
    world: RequirementResult[Any]


def _canonical(value: object) -> bytes:
    """Encode one closed protocol value under fixed aggregate limits."""

    try:
        data = canonical_json_bytes(value, **_JSON_BOUNDS)
    except CanonicalJSONError as exc:
        raise ProbeProtocolError("probe envelope is invalid") from exc
    if len(data) > _MAX_BYTES:
        raise ProbeProtocolError("probe envelope exceeds transport limit")
    return data


def _load(data: object) -> Mapping[str, Any]:
    """Decode canonical JSON without accepting alternate spellings."""

    if type(data) is not bytes or len(data) > _MAX_BYTES:
        raise ProbeProtocolError("probe envelope is invalid")
    try:
        value = canonical_json_load_bytes(data, **_JSON_BOUNDS)
    except CanonicalJSONError as exc:
        raise ProbeProtocolError("probe envelope is invalid") from exc
    if not isinstance(value, Mapping) or _canonical(value) != data:
        raise ProbeProtocolError("probe envelope is not canonical")
    return value


def _mutable(value: object) -> object:
    """Thaw canonical JSON for owner codecs requiring exact containers."""

    if isinstance(value, Mapping):
        return {key: _mutable(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_mutable(item) for item in value]
    return value


def _info_data(info: TargetInfo) -> dict[str, object]:
    """Project detached target metadata without source or import provenance."""

    return {
        "descriptor_kind": info.descriptor_kind,
        "kind": info.kind,
        "module": info.module,
        "name": info.name,
        "owner_module": info.owner_module,
        "owner_qualname": info.owner_qualname,
        "qualname": info.qualname,
    }


def _info_from_data(data: object) -> TargetInfo:
    """Decode one closed source-free target metadata record."""

    if not isinstance(data, Mapping) or set(data) != {
        "descriptor_kind",
        "kind",
        "module",
        "name",
        "owner_module",
        "owner_qualname",
        "qualname",
    }:
        raise ProbeProtocolError("probe target metadata is invalid")
    try:
        return TargetInfo(
            data["kind"],
            data["name"],
            data["module"],
            data["qualname"],
            data["owner_module"],
            data["owner_qualname"],
            data["descriptor_kind"],
            None,
            None,
            None,
        )
    except Exception as exc:
        raise ProbeProtocolError("probe target metadata is invalid") from exc


def _snapshot_data(target: InspectionTarget) -> dict[str, object]:
    """Encode the immutable source-free snapshot in deterministic order."""

    return {
        "root_id": target.snapshot.root_id,
        "version": target.snapshot.version,
        "records": [
            {
                "calls": [call.target_id for call in record.calls],
                "id": record.target_id,
                "incomplete": record.incomplete,
                "info": _info_data(record.info),
            }
            for record in target.snapshot.records
        ],
    }


def _snapshot_from_data(data: object) -> InspectionTarget:
    """Decode a closed snapshot before domain or scheduler allocation."""

    if not isinstance(data, Mapping) or set(data) != {
        "root_id",
        "version",
        "records",
    }:
        raise ProbeProtocolError("probe snapshot is invalid")
    if (
        type(data["version"]) is not int
        or isinstance(data["version"], bool)
        or type(data["root_id"]) is not str
        or not data["root_id"]
    ):
        raise ProbeProtocolError("probe snapshot is invalid")
    records = data["records"]
    if type(records) is not tuple and type(records) is not list:
        raise ProbeProtocolError("probe snapshot is invalid")
    if not records or len(records) > _MAX_TARGETS:
        raise ProbeProtocolError("probe snapshot is invalid")
    values: list[InspectionRecord] = []
    call_count = 0
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "calls",
            "id",
            "incomplete",
            "info",
        }:
            raise ProbeProtocolError("probe snapshot record is invalid")
        calls = record["calls"]
        if type(calls) is not tuple and type(calls) is not list:
            raise ProbeProtocolError("probe snapshot calls are invalid")
        call_count += len(calls)
        if call_count > _MAX_CALLS:
            raise ProbeProtocolError("probe snapshot calls are invalid")
        if (
            type(record["id"]) is not str
            or not record["id"]
            or type(record["incomplete"]) is not bool
            or any(
                value is not None and (type(value) is not str or not value)
                for value in calls
            )
        ):
            raise ProbeProtocolError("probe snapshot record is invalid")
        try:
            values.append(
                InspectionRecord(
                    record["id"],
                    _info_from_data(record["info"]),
                    tuple(InspectionCall(value) for value in calls),
                    record["incomplete"],
                )
            )
        except Exception as exc:
            raise ProbeProtocolError(
                "probe snapshot record is invalid"
            ) from exc
    try:
        snapshot = InspectionSnapshot(
            data["version"], data["root_id"], tuple(values)
        )
        return InspectionTarget(snapshot, snapshot.root_id)
    except Exception as exc:
        raise ProbeProtocolError("probe snapshot is invalid") from exc


def _source_data(source: RequirementSource) -> dict[str, str | None]:
    """Encode a non-identifying source category for a conflict report."""

    return {
        "label": "declaration",
        "module": None,
        "qualname": None,
    }


def _source_from_data(data: object) -> RequirementSource:
    """Decode an owner requirement source without arbitrary attributes."""

    if not isinstance(data, Mapping) or set(data) != {
        "label",
        "module",
        "qualname",
    }:
        raise ProbeProtocolError("probe requirement source is invalid")
    if data != {"label": "declaration", "module": None, "qualname": None}:
        raise ProbeProtocolError("probe requirement source is invalid")
    return RequirementSource("declaration")


def _result_data(result: RequirementResult[Any]) -> dict[str, object]:
    """Encode a domain result with its value codec and conflict report."""

    if type(result) is not RequirementResult:
        raise ProbeProtocolError("probe domain result is invalid")
    value = result.value
    if value is not None and not callable(getattr(value, "to_data", None)):
        raise ProbeProtocolError("probe domain result is invalid")
    issues = result.report.issues
    if len(issues) > _MAX_DIAGNOSTICS or (value is not None and issues):
        raise ProbeProtocolError("probe domain result is invalid")
    encoded_issues: list[dict[str, object]] = []
    for issue in issues:
        if (
            type(issue) is not RequirementIssue
            or type(issue.code) is not str
            or not issue.code.isascii()
            or len(issue.code) > _MAX_DIAGNOSTIC_TEXT
            or len(issue.sources) > _MAX_DIAGNOSTICS
        ):
            raise ProbeProtocolError("probe domain issue is invalid")
        # Requirement values remain owner-coded. Conflict text is reduced
        # to stable categories so source/path text cannot escape workers.
        encoded_issues.append(
            {
                "code": issue.code,
                "message": "requirement conflict",
                "path": None,
                "sources": [_source_data(source) for source in issue.sources],
            }
        )
    return {
        "issues": encoded_issues,
        "value": None if value is None else value.to_data(),
    }


def _result_from_data(
    data: object, value_type: type[Any]
) -> RequirementResult[Any]:
    """Decode one owner result without hiding known conflicts."""

    if not isinstance(data, Mapping) or set(data) != {"issues", "value"}:
        raise ProbeProtocolError("probe domain result is invalid")
    issues = data["issues"]
    if (type(issues) is not tuple and type(issues) is not list) or len(
        issues
    ) > _MAX_DIAGNOSTICS:
        raise ProbeProtocolError("probe domain result is invalid")
    decoded: list[RequirementIssue] = []
    for issue in issues:
        if not isinstance(issue, Mapping) or set(issue) != {
            "code",
            "message",
            "path",
            "sources",
        }:
            raise ProbeProtocolError("probe domain issue is invalid")
        sources = issue["sources"]
        if (
            (type(sources) is not tuple and type(sources) is not list)
            or len(sources) > _MAX_DIAGNOSTICS
            or type(issue["code"]) is not str
            or len(issue["code"]) > _MAX_DIAGNOSTIC_TEXT
            or type(issue["message"]) is not str
            or len(issue["message"]) > _MAX_DIAGNOSTIC_TEXT
            or issue["message"] != "requirement conflict"
            or issue["path"] is not None
        ):
            raise ProbeProtocolError("probe domain issue is invalid")
        try:
            decoded.append(
                RequirementIssue(
                    issue["code"],
                    issue["message"],
                    path=issue["path"],
                    sources=tuple(_source_from_data(item) for item in sources),
                )
            )
        except Exception as exc:
            raise ProbeProtocolError("probe domain issue is invalid") from exc
    try:
        value = (
            None
            if data["value"] is None
            else value_type.from_data(data["value"])
        )
        if value is not None and decoded:
            raise ProbeProtocolError("probe domain result is invalid")
        return RequirementResult(value, RequirementReport(tuple(decoded)))
    except Exception as exc:
        raise ProbeProtocolError("probe domain result is invalid") from exc


def build_request(
    target: InspectionTarget,
    *,
    environment_view: object,
    world_view: object,
    max_targets: int,
    max_depth: int,
) -> ProbeRequest:
    """Create a bounded canonical request for the fixed probe callable.

    Args:
        target: Detached root target from a local inspection capture.
        environment_view: Environment owner-encoded declaration projection.
        world_view: World owner-encoded declaration projection.
        max_targets: Positive user traversal target budget.
        max_depth: Positive user traversal depth budget.

    Returns:
        Canonical request bytes plus strict association fields.

    Raises:
        ProbeProtocolError: If projections or transport bounds are invalid.

    Side Effects:
        None. This function neither imports user code nor starts Execute.
    """

    if type(target) is not InspectionTarget:
        raise ProbeProtocolError("probe target is invalid")
    body = {
        "collector_version": _COLLECTOR_VERSION,
        "environment": environment_view,
        "max_depth": max_depth,
        "max_targets": max_targets,
        "root_id": target.target_id,
        "schema": _REQUEST_SCHEMA,
        "snapshot": _snapshot_data(target),
        "world": world_view,
    }
    digest = hashlib.sha256(_canonical(body)).hexdigest()
    data = _canonical({**body, "digest": digest})
    # Validate the exact serialized bytes before a backend can be initialized.
    _decode_request(data)
    return ProbeRequest(data, digest, target.target_id)


def _decode_request(
    data: bytes,
) -> tuple[
    InspectionTarget,
    EnvironmentRequirementsKernel,
    WorldRequirementsKernel,
    int,
    int,
    str,
]:
    """Validate and bind one request before the worker creates kernels."""

    envelope = _load(data)
    expected = {
        "collector_version",
        "digest",
        "environment",
        "max_depth",
        "max_targets",
        "root_id",
        "schema",
        "snapshot",
        "world",
    }
    if (
        set(envelope) != expected
        or envelope.get("schema") != _REQUEST_SCHEMA
        or type(envelope.get("collector_version")) is not int
        or envelope.get("collector_version") != _COLLECTOR_VERSION
    ):
        raise ProbeProtocolError("probe request version is invalid")
    digest = envelope["digest"]
    body = {key: value for key, value in envelope.items() if key != "digest"}
    if (
        type(digest) is not str
        or len(digest) != 64
        or hashlib.sha256(_canonical(body)).hexdigest() != digest
    ):
        raise ProbeProtocolError("probe request digest is invalid")
    target = _snapshot_from_data(envelope["snapshot"])
    if envelope["root_id"] != target.target_id:
        raise ProbeProtocolError("probe request root is invalid")
    for name in ("max_targets", "max_depth"):
        value = envelope[name]
        maximum = (
            _MAX_TARGETS if name == "max_targets" else _MAX_TRAVERSAL_DEPTH
        )
        if (
            type(value) is not int
            or isinstance(value, bool)
            or not 0 < value <= maximum
        ):
            raise ProbeProtocolError("probe traversal limits are invalid")
    try:
        environment = EnvironmentRequirementsKernel._from_data(
            target, _mutable(envelope["environment"])
        )
        world = WorldRequirementsKernel._from_data(
            target, _mutable(envelope["world"])
        )
    except Exception as exc:
        raise ProbeProtocolError(
            "probe declaration projection is invalid"
        ) from exc
    return (
        target,
        environment,
        world,
        envelope["max_targets"],
        envelope["max_depth"],
        digest,
    )


def _safe_diagnostics(values: object) -> tuple[str, ...]:
    """Retain bounded framework categories, never messages or values."""

    if type(values) is not tuple or len(values) > _MAX_DIAGNOSTICS:
        raise ProbeProtocolError("probe diagnostics are invalid")
    if any(
        type(value) is not str
        or not value
        or len(value) > _MAX_DIAGNOSTIC_TEXT
        or value not in _SAFE_DIAGNOSTICS
        for value in values
    ):
        raise ProbeProtocolError("probe diagnostics are invalid")
    return values


def _encode_result(
    *,
    target: InspectionTarget,
    digest: str,
    complete: bool,
    diagnostics: tuple[str, ...],
    environment: RequirementResult[Any],
    world: RequirementResult[Any],
) -> bytes:
    """Encode a result only after every scheduler outcome succeeds."""

    if type(complete) is not bool:
        raise ProbeProtocolError("probe coverage is invalid")
    return _canonical(
        {
            "collector_version": _COLLECTOR_VERSION,
            "coverage": {
                "complete": complete,
                "diagnostics": list(_safe_diagnostics(diagnostics)),
            },
            "digest": digest,
            "environment": _result_data(environment),
            "root_id": target.target_id,
            "schema": _RESULT_SCHEMA,
            "world": _result_data(world),
        }
    )


def _check_deadline(deadline: float | None) -> None:
    """Check a coordinator-supplied cooperative deadline when present."""

    if deadline is not None and time.monotonic() > deadline:
        raise TimeoutError("inline probe exceeded its cooperative timeout")


def execute_probe(request: bytes, *, deadline: float | None = None) -> bytes:
    """Run the fixed snapshot probe callable inside a fresh Context.

    Args:
        request: Canonical request bytes created by :func:`build_request`.
        deadline: Optional local deadline for inline use. It is checked between
            scheduler boundaries and cannot interrupt trusted work.

    Returns:
        Canonical result bytes associated with exactly the supplied request.

    Raises:
        ProbeProtocolError: If request/result contracts or scheduler outcomes
            are invalid.

    Side Effects:
        Executes the generic static scheduler and domain kernel DAG in a fresh
        logical context. It never imports or reconstructs user code.
    """

    (
        target,
        environment_kernel,
        world_kernel,
        max_targets,
        max_depth,
        digest,
    ) = _decode_request(request)
    started = time.monotonic()

    def collect() -> bytes:
        """Run the scheduler inside the new context-local state."""

        result = probe(
            target,
            (
                KernelCall(
                    StaticDependenciesKernel(
                        max_targets=max_targets, max_depth=max_depth
                    ),
                    None,
                ),
                KernelCall(environment_kernel, None),
                KernelCall(world_kernel, None),
            ),
        )
        _check_deadline(deadline)
        dependencies = result.require(StaticDependenciesKernel)
        environment = result.require(EnvironmentRequirementsKernel)
        world = result.require(WorldRequirementsKernel)
        _check_deadline(deadline)
        if (
            type(dependencies.complete) is not bool
            or type(environment) is not RequirementResult
            or type(world) is not RequirementResult
            or any(
                outcome.status != "succeeded" for outcome in result.outcomes
            )
        ):
            raise ProbeProtocolError("probe scheduler outcome is invalid")
        diagnostics = tuple(dict.fromkeys(dependencies.diagnostics))
        return _encode_result(
            target=target,
            digest=digest,
            complete=dependencies.complete,
            diagnostics=diagnostics,
            environment=environment,
            world=world,
        )

    _check_deadline(deadline)
    result = contextvars.Context().run(collect)
    if time.monotonic() < started:
        raise ProbeProtocolError("probe clock is invalid")
    return result


def decode_result(data: bytes, request: ProbeRequest) -> ProbeWireResult:
    """Validate a terminal result against one coordinator-retained request.

    Args:
        data: Canonical result bytes returned after Execute retrieves it.
        request: Original request association retained by the coordinator.

    Returns:
        Detached bounded coverage and typed domain results.

    Raises:
        ProbeProtocolError: If result version, digest, root, owner codec, or
            canonical bounds do not match the exact request.

    Side Effects:
        None. Result decoding imports no user module or live target.
    """

    if type(request) is not ProbeRequest:
        raise ProbeProtocolError("probe request association is invalid")
    envelope = _load(data)
    expected = {
        "collector_version",
        "coverage",
        "digest",
        "environment",
        "root_id",
        "schema",
        "world",
    }
    if (
        set(envelope) != expected
        or envelope.get("schema") != _RESULT_SCHEMA
        or type(envelope.get("collector_version")) is not int
        or envelope.get("collector_version") != _COLLECTOR_VERSION
        or envelope.get("digest") != request.digest
        or envelope.get("root_id") != request.root_id
    ):
        raise ProbeProtocolError("probe result association is invalid")
    coverage = envelope["coverage"]
    if not isinstance(coverage, Mapping) or set(coverage) != {
        "complete",
        "diagnostics",
    }:
        raise ProbeProtocolError("probe coverage is invalid")
    diagnostics = coverage["diagnostics"]
    if type(diagnostics) is list:
        diagnostics = tuple(diagnostics)
    if type(coverage["complete"]) is not bool:
        raise ProbeProtocolError("probe coverage is invalid")
    from dryml.environments import EnvironmentRequirement
    from dryml.worlds import WorldRequirement

    return ProbeWireResult(
        coverage["complete"],
        _safe_diagnostics(diagnostics),
        _result_from_data(envelope["environment"], EnvironmentRequirement),
        _result_from_data(envelope["world"], WorldRequirement),
    )


__all__ = [
    "ProbeProtocolError",
    "ProbeRequest",
    "ProbeWireResult",
    "build_request",
    "decode_result",
    "execute_probe",
]
