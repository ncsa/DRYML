"""Immutable, redacted public values for the persistent session facade."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from dryml.environments import EnvironmentRequirement
from dryml.formats import deep_freeze_json, json_ready, make_envelope, semantic_id, validate_envelope
from dryml.worlds import LocalResourceInventory, ProcessAllocation, ResourceSpec

_SECRET_NAME = re.compile(r"(?:password|passwd|secret|token|api[_-]?key|credential)", re.I)
_SECRET_ASSIGNMENT = re.compile(r"(?i)(password|passwd|secret|token|api[_-]?key|credential)\s*=\s*[^\s,&]+")
_WINDOWS_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")
_PATH_NAME = re.compile(r"(?:^|[_-])(?:path|directory|folder|cwd|filename)(?:$|[_-])", re.I)
_CONFIGURATION_SCHEMA = "dryml.session_configuration.v1.1"
_CONFIGURATION_KIND = "session_configuration"
_CONFIGURATION_PREFIX = "sessioncfg"
_CONFIGURATION_BOUNDS = {"max_depth": 8, "max_nodes": 1_024, "max_entries": 64}
_CONFIGURATION_FIELDS = frozenset({"mode", "resources", "allocation", "environment", "requirement_axes", "controls"})


def default_requirement_axes(mode: str) -> Mapping[str, bool]:
    """Return the default compatibility axes for one public mode.

    Args:
        mode: One of the public session modes.

    Returns:
        A detached immutable exact-boolean axis mapping.

    Raises:
        ValueError: If ``mode`` is outside the public session vocabulary.
    """

    if mode not in {"python", "managed", "orchestrator"}:
        raise ValueError("session mode must be python, managed, or orchestrator")
    return MappingProxyType({"environment": mode != "python", "world": mode != "python", "runtime": mode != "python"})


def freeze_requirement_axes(value: Mapping[str, bool]) -> Mapping[str, bool]:
    """Validate and freeze the three identity-bearing compatibility axes.

    Args:
        value: Exact mapping of ``environment``, ``world``, and ``runtime``.

    Returns:
        Immutable normalized axis values.

    Raises:
        TypeError: If the mapping is not closed or values are not booleans.
    """

    expected = {"environment", "world", "runtime"}
    if not isinstance(value, Mapping) or set(value) != expected or any(type(value[name]) is not bool for name in expected):
        raise TypeError("session requirement_axes must contain exact environment, world, and runtime booleans")
    return MappingProxyType({name: value[name] for name in sorted(expected)})


@dataclass(frozen=True, slots=True)
class SelectedWorldAllocation:
    """One role-qualified exact process selected from a world allocation.

    Args:
        role: Role that owns ``process``.
        process: Immutable exact current-process assignment.
    """

    role: str
    process: ProcessAllocation

    def __post_init__(self) -> None:
        """Validate the role-qualified process selection.

        Raises:
            TypeError: If the role or process does not meet the exact-allocation
                contract.
        """

        if not isinstance(self.role, str) or not self.role or not isinstance(self.process, ProcessAllocation):
            raise TypeError("selected allocation requires a non-empty role and ProcessAllocation")

    def to_data(self) -> dict[str, Any]:
        """Return a detached canonical selection projection.

        Returns:
            A JSON-ready role and exact process payload.
        """

        return {"role": self.role, "process": self.process.to_payload()}


@dataclass(frozen=True, slots=True)
class SessionConfiguration:
    """Complete immutable session configuration independent of publication.

    ``restage_retries`` is intentionally absent: it controls one publication
    attempt and is never persistent session identity.

    Args:
        mode: Public ``python``, ``managed``, or ``orchestrator`` mode.
        resources: Optional concise managed resource request.
        allocation: Optional role-qualified exact managed process.
        environment: Explicit current-process software requirements.
        requirement_axes: Exact environment/world/runtime compatibility mask.
        controls: Bounded derived process-control diagnostics.
    """

    mode: str
    resources: ResourceSpec | None = None
    allocation: SelectedWorldAllocation | None = None
    environment: EnvironmentRequirement = field(default_factory=EnvironmentRequirement)
    requirement_axes: Mapping[str, bool] = field(default_factory=lambda: default_requirement_axes("python"))
    controls: Mapping[str, str] = field(default_factory=dict)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        """Freeze projections and calculate the canonical configuration ID.

        Raises:
            TypeError: If fields violate the pure configuration contract.
        """

        if self.mode not in {"python", "managed", "orchestrator"}:
            raise TypeError("session mode must be python, managed, or orchestrator")
        if self.mode != "managed" and (self.resources is not None or self.allocation is not None):
            raise TypeError("only managed session configuration may hold resources or allocation")
        if self.resources is not None and not isinstance(self.resources, ResourceSpec):
            raise TypeError("session resources must be a ResourceSpec")
        if self.allocation is not None and not isinstance(self.allocation, SelectedWorldAllocation):
            raise TypeError("session allocation must be SelectedWorldAllocation")
        if not isinstance(self.environment, EnvironmentRequirement):
            raise TypeError("session environment must be an EnvironmentRequirement")
        object.__setattr__(self, "requirement_axes", freeze_requirement_axes(self.requirement_axes))
        controls = deep_freeze_json(self.controls)
        if not isinstance(controls, Mapping):
            raise TypeError("session controls must be a JSON mapping")
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "fingerprint", semantic_id(_CONFIGURATION_PREFIX, _CONFIGURATION_SCHEMA, _CONFIGURATION_KIND, self._identifying_payload(), **_CONFIGURATION_BOUNDS))

    @property
    def id(self) -> str:
        """Return the canonical semantic identifier for this configuration."""

        return self.fingerprint

    def to_payload(self) -> dict[str, Any]:
        """Return the closed complete configuration payload.

        Returns:
            JSON-ready semantic state excluding envelope diagnostics.
        """

        return {
            "mode": self.mode,
            "resources": None if self.resources is None else self.resources.to_data(),
            "allocation": None if self.allocation is None else self.allocation.to_data(),
            "environment": self.environment.to_data(),
            "requirement_axes": dict(self.requirement_axes),
            "controls": json_ready(self.controls),
        }

    def _identifying_payload(self) -> dict[str, Any]:
        """Return semantic data without nested diagnostic metadata."""

        return _identifying_configuration_payload(self.to_payload())

    @classmethod
    def from_payload(cls, data: Mapping[str, Any]) -> "SessionConfiguration":
        """Decode one closed session-configuration payload.

        Args:
            data: Exact payload emitted by :meth:`to_payload`.

        Returns:
            An immutable typed configuration.

        Raises:
            TypeError: If a field is absent, unknown, or incompatible.
        """

        if not isinstance(data, Mapping) or set(data) != _CONFIGURATION_FIELDS:
            raise TypeError("session configuration payload fields are closed")
        resources_data = data["resources"]
        allocation_data = data["allocation"]
        if resources_data is not None and not isinstance(resources_data, Mapping):
            raise TypeError("session configuration resources must be a mapping or null")
        if allocation_data is not None and not isinstance(allocation_data, Mapping):
            raise TypeError("session configuration allocation must be a mapping or null")
        if allocation_data is not None and set(allocation_data) != {"role", "process"}:
            raise TypeError("session configuration allocation fields are closed")
        return cls(
            data["mode"],
            None if resources_data is None else ResourceSpec.from_data(resources_data),
            None if allocation_data is None else SelectedWorldAllocation(allocation_data["role"], ProcessAllocation.from_payload(allocation_data["process"])),
            EnvironmentRequirement.from_data(data["environment"]),
            data["requirement_axes"],
            data["controls"],
        )

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "SessionConfiguration":
        """Decode a self-validating closed v1.1 configuration envelope.

        Args:
            data: Canonical ``dryml.session_configuration.v1.1`` envelope.

        Returns:
            The configuration represented by the validated envelope.

        Raises:
            TypeError: If the envelope, semantic ID, or payload is malformed.
        """

        raw = dict(data)
        envelope = validate_envelope(raw, schema=_CONFIGURATION_SCHEMA, kind=_CONFIGURATION_KIND, prefix=_CONFIGURATION_PREFIX, identifying_payload=_identifying_configuration_payload(raw.get("payload", {})), **_CONFIGURATION_BOUNDS)
        value = cls.from_payload(envelope["payload"])
        if "id" in envelope and envelope["id"] != value.fingerprint:
            raise TypeError("session configuration attached ID does not match payload")
        return value

    def to_data(self) -> dict[str, Any]:
        """Return this configuration as a closed self-validating v1.1 envelope.

        Returns:
            Detached canonical data with the attached semantic ID.
        """

        return make_envelope(schema=_CONFIGURATION_SCHEMA, kind=_CONFIGURATION_KIND, prefix=_CONFIGURATION_PREFIX, payload=self.to_payload(), semantic_id=self.fingerprint, identifying_payload=self._identifying_payload(), **_CONFIGURATION_BOUNDS)


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    """Detached immutable public projection of one published generation.

    Attributes:
        mode: Public ``python``, ``managed``, or ``orchestrator`` mode.
        resources: Current managed resource declaration, if any.
        allocation: Role-qualified exact current-process assignment, if any.
        environment: Explicit software requirements with redacted diagnostics.
        requirement_axes: Effective environment/world/runtime compatibility mask.
        controls: Redacted immutable session-control diagnostics.
        statuses: Independent immutable runtime/framework control outcomes.
        runtime: Redacted low-level runtime projection.
        generation: Monotonic authoritative publication generation.
        health: ``healthy`` or terminal ``failed`` publication state.
        inventory: Optional detached retained resource observation.
    """

    mode: str
    resources: ResourceSpec | None
    allocation: SelectedWorldAllocation | None
    environment: EnvironmentRequirement
    requirement_axes: Mapping[str, bool]
    controls: Mapping[str, Any]
    statuses: Mapping[str, str]
    runtime: Any
    generation: int
    health: str
    inventory: LocalResourceInventory | None = None

    def __post_init__(self) -> None:
        """Freeze all public mapping projections.

        Raises:
            TypeError: If the snapshot has invalid axes or generation data.
        """

        object.__setattr__(self, "requirement_axes", freeze_requirement_axes(self.requirement_axes))
        object.__setattr__(self, "controls", deep_freeze_json(_redact(json_ready(self.controls))))
        object.__setattr__(self, "statuses", deep_freeze_json(self.statuses))
        if self.allocation is not None:
            object.__setattr__(self, "allocation", SelectedWorldAllocation(self.allocation.role, _redacted_process(self.allocation.process)))
        if self.inventory is not None:
            object.__setattr__(self, "inventory", LocalResourceInventory(self.inventory.cpus, self.inventory.accelerators, self.inventory.memory, self.inventory.accelerator_memory, _redact(json_ready(self.inventory.metadata))))
        environment_data = self.environment.to_data()
        payload = environment_data.get("payload")
        if isinstance(payload, Mapping) and "details" in payload:
            payload["details"] = _redact(payload["details"])
        if "metadata" in environment_data:
            environment_data["metadata"] = _redact(environment_data["metadata"])
        object.__setattr__(self, "environment", EnvironmentRequirement.from_data(environment_data))
        object.__setattr__(self, "runtime", _redacted_runtime(self.runtime))
        if isinstance(self.generation, bool) or not isinstance(self.generation, int) or self.generation < 0:
            raise TypeError("session generation must be a non-negative integer")

    def to_data(self) -> dict[str, Any]:
        """Return a bounded, detached, deeply redacted display projection.

        Returns:
            JSON-ready data that omits environment values, recognizable secrets,
            local paths, and URI credentials/query/fragment components.
        """

        return _redact({
            "mode": self.mode,
            "resources": None if self.resources is None else self.resources.to_data(),
            "allocation": None if self.allocation is None else self.allocation.to_data(),
            "environment": self.environment.to_data(),
            "requirement_axes": dict(self.requirement_axes),
            "controls": json_ready(self.controls),
            "statuses": json_ready(self.statuses),
            "generation": self.generation,
            "health": self.health,
            "inventory": None if self.inventory is None else self.inventory.summary(),
        })


def _redact(value: Any, *, key: str | None = None) -> Any:
    """Detach a public value while removing bounded diagnostic secrets."""

    if key is not None and _PATH_NAME.search(key) and not (isinstance(value, str) and value.startswith("file:")):
        return "<local-path>"
    if key is not None and (_SECRET_NAME.search(key) or key in {"env", "environment"}):
        if key == "environment" and not isinstance(value, Mapping):
            return "<redacted>"
        if key == "env" and isinstance(value, Mapping):
            return {str(name): "<redacted>" for name in value}
        if _SECRET_NAME.search(key):
            return "<redacted>"
    if isinstance(value, Mapping):
        return {str(name): _redact(item, key=str(name)) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_redact(item) for item in value]
    if not isinstance(value, str):
        return value
    parsed = urlsplit(value)
    if parsed.scheme and (parsed.netloc or parsed.scheme == "file"):
        if parsed.scheme == "file":
            return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
        host = parsed.hostname or ""
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    if _SECRET_ASSIGNMENT.search(value):
        return _SECRET_ASSIGNMENT.sub(r"\1=<redacted>", value)
    if value.startswith(("/", "~/", "./", "../")) or _WINDOWS_PATH.match(value):
        return "<local-path>"
    return value


def _redacted_process(process: ProcessAllocation) -> ProcessAllocation:
    """Return an exact process projection with all environment values masked."""

    return ProcessAllocation(
        process.replica,
        process.rank,
        process.local_rank,
        process.cpus,
        process.memory,
        process.accelerators,
        process.accelerator_memory,
        _redact(json_ready(process.devices)),
        _redact(json_ready(process.named)),
        _redact(process.environment),
        {name: "<redacted>" for name in process.env},
        _redact(json_ready(process.metadata)),
    )


def _redacted_runtime(runtime: Any) -> Any:
    """Preserve a RuntimeState's shape while masking its public projections."""

    try:
        from dryml.runtime import RuntimeAllocationView, RuntimeState
    except ImportError:
        return runtime
    if not isinstance(runtime, RuntimeState):
        return runtime
    allocation = runtime.allocation
    if isinstance(allocation, RuntimeAllocationView):
        allocation = RuntimeAllocationView(
            allocation.role,
            allocation.replica,
            allocation.rank,
            allocation.local_rank,
            allocation.cpus,
            allocation.memory,
            allocation.accelerators,
            allocation.accelerator_memory,
            {name: "<redacted>" for name in allocation.env},
            allocation.world_allocation_id,
            _redact(json_ready(allocation.metadata)),
        )
    return RuntimeState(runtime.mode, allocation, runtime.spec, _redact(json_ready(runtime.controls)))


def _identifying_configuration_payload(value: Any) -> dict[str, Any]:
    """Strip nested non-identifying diagnostics before envelope ID validation."""

    if not isinstance(value, Mapping):
        return {}
    payload = dict(value)
    environment = payload.get("environment")
    if isinstance(environment, Mapping):
        environment = dict(environment)
        environment.pop("metadata", None)
        environment_payload = environment.get("payload")
        if isinstance(environment_payload, Mapping):
            environment_payload = dict(environment_payload)
            environment_payload.pop("details", None)
            environment["payload"] = environment_payload
        payload["environment"] = environment
    allocation = payload.get("allocation")
    if isinstance(allocation, Mapping):
        allocation = dict(allocation)
        process = allocation.get("process")
        if isinstance(process, Mapping):
            process = dict(process)
            process.pop("metadata", None)
            allocation["process"] = process
        payload["allocation"] = allocation
    return payload


__all__ = ["SelectedWorldAllocation", "SessionConfiguration", "SessionSnapshot", "default_requirement_axes", "freeze_requirement_axes"]
