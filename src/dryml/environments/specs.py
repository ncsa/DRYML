"""Closed v1.1 environment selectors and lock references."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from dryml.formats import deep_freeze_json, json_ready, make_envelope, semantic_id, validate_envelope

from .errors import EnvironmentSpecError

ProbeWorkerCommand = ("-m", "dryml.environments.probe_worker", "--json")
_SPEC_SCHEMA, _SPEC_KIND, _SPEC_PREFIX = "dryml.environment_spec.v1.1", "environment_spec", "envspec"
_LOCK_SCHEMA, _LOCK_KIND, _LOCK_PREFIX = "dryml.environment_lock.v1.1", "environment_lock", "envlock"


def _check_fields(data: Mapping[str, Any], expected: set[str]) -> None:
    unknown, missing = set(data) - expected, expected - set(data)
    if unknown or missing:
        raise EnvironmentSpecError("environment spec fields are closed", context={"unknown": sorted(unknown), "missing": sorted(missing)})


class _SpecBase:
    kind: str

    @property
    def semantic_id(self) -> str:
        """Return the stable ID over every closed selector payload field."""

        return semantic_id(_SPEC_PREFIX, _SPEC_SCHEMA, _SPEC_KIND, self._payload())

    @property
    def id(self) -> str:
        """Alias for the selector's v1.1 semantic identifier."""

        return self.semantic_id

    def to_data(self) -> dict[str, Any]:
        """Return the complete v1.1 selector envelope."""

        return make_envelope(schema=_SPEC_SCHEMA, kind=_SPEC_KIND, prefix=_SPEC_PREFIX, payload=self._payload(), semantic_id=self.semantic_id, identifying_payload=self._payload(), metadata=self.metadata)


@dataclass(frozen=True, slots=True)
class CurrentEnvironmentSpec(_SpecBase):
    """Selector for the current process, without inspection or activation."""

    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)
    kind: Literal["current"] = "current"

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", deep_freeze_json(self.metadata))
        deep_freeze_json(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {"kind": self.kind}


@dataclass(frozen=True, slots=True)
class PythonExecutableSpec(_SpecBase):
    """Selector for an explicit Python executable and launch declarations."""

    executable: str
    env: Mapping[str, str] = field(default_factory=dict)
    pythonpath_policy: str = "none"
    extra_pythonpath: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)
    kind: Literal["python"] = "python"

    def __post_init__(self) -> None:
        object.__setattr__(self, "env", deep_freeze_json(self.env))
        object.__setattr__(self, "extra_pythonpath", tuple(self.extra_pythonpath))
        object.__setattr__(self, "metadata", deep_freeze_json(self.metadata))
        deep_freeze_json(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {"kind": self.kind, "executable": self.executable, "env": json_ready(self.env), "pythonpath_policy": self.pythonpath_policy, "extra_pythonpath": list(self.extra_pythonpath)}

    def probe_command(self) -> list[str]:
        """Return the explicit probe-worker command; it is never run implicitly."""

        return [self.executable, *ProbeWorkerCommand]


@dataclass(frozen=True, slots=True)
class CondaEnvironmentSpec(_SpecBase):
    """Selector for exactly one named or prefixed Conda environment."""

    prefix: str | None = None
    name: str | None = None
    conda_executable: str = "conda"
    launch_mode: Literal["direct", "conda-run"] = "direct"
    env: Mapping[str, str] = field(default_factory=dict)
    pythonpath_policy: str = "none"
    extra_pythonpath: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)
    kind: Literal["conda"] = "conda"

    def __post_init__(self) -> None:
        if bool(self.prefix) == bool(self.name):
            raise EnvironmentSpecError("CondaEnvironmentSpec requires exactly one of prefix or name")
        if self.launch_mode not in {"direct", "conda-run"}:
            raise EnvironmentSpecError("unsupported Conda launch mode", context={"launch_mode": self.launch_mode})
        object.__setattr__(self, "env", deep_freeze_json(self.env))
        object.__setattr__(self, "extra_pythonpath", tuple(self.extra_pythonpath))
        object.__setattr__(self, "metadata", deep_freeze_json(self.metadata))
        deep_freeze_json(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {"kind": self.kind, "prefix": self.prefix, "name": self.name, "conda_executable": self.conda_executable, "launch_mode": self.launch_mode, "env": json_ready(self.env), "pythonpath_policy": self.pythonpath_policy, "extra_pythonpath": list(self.extra_pythonpath)}

    def direct_python_executable(self, *, os_name: str | None = None) -> str:
        """Return the direct interpreter path for a prefix selector."""

        if not self.prefix:
            raise EnvironmentSpecError("Conda direct launch requires a prefix")
        return os.path.join(self.prefix, "python.exe" if (os_name or os.name) == "nt" else "bin/python")

    def probe_command(self, *, os_name: str | None = None) -> list[str]:
        """Return the explicit probe-worker command for this selector."""

        if self.launch_mode == "direct":
            return [self.direct_python_executable(os_name=os_name), *ProbeWorkerCommand]
        command = [self.conda_executable, "run", "-p" if self.prefix else "-n", self.prefix or self.name or "", "--no-capture-output", "--", "python", *ProbeWorkerCommand]
        return command


@dataclass(frozen=True, slots=True)
class ContainerEnvironmentSpec(_SpecBase):
    """Structural container selector; container execution remains unsupported."""

    image: str
    runtime: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)
    kind: Literal["container"] = "container"

    def _payload(self) -> dict[str, Any]:
        return {"kind": self.kind, "image": self.image, "runtime": self.runtime}

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", deep_freeze_json(self.metadata))
        deep_freeze_json(self._payload())


@dataclass(frozen=True, slots=True)
class EnvironmentLockRef:
    """Reference to an exact external environment lock in a v1.1 envelope."""

    kind: str
    uri: str
    digest: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", deep_freeze_json(self.metadata))
        deep_freeze_json(self._payload())

    @property
    def semantic_id(self) -> str:
        """Return the stable ID over every lock payload field."""

        return semantic_id(_LOCK_PREFIX, _LOCK_SCHEMA, _LOCK_KIND, self._payload())

    @property
    def id(self) -> str:
        """Alias for the lock's semantic identifier."""

        return self.semantic_id

    def _payload(self) -> dict[str, Any]:
        return {"kind": self.kind, "uri": self.uri, "digest": self.digest}

    def to_data(self) -> dict[str, Any]:
        """Return a complete v1.1 environment-lock envelope."""

        return make_envelope(schema=_LOCK_SCHEMA, kind=_LOCK_KIND, prefix=_LOCK_PREFIX, payload=self._payload(), semantic_id=self.semantic_id, identifying_payload=self._payload(), metadata=self.metadata)

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "EnvironmentLockRef":
        """Decode a closed v1.1 environment-lock envelope."""

        return _decode(data, _LOCK_SCHEMA, _LOCK_KIND, _LOCK_PREFIX, {"kind", "uri", "digest"}, cls)


EnvironmentSpec = CurrentEnvironmentSpec | PythonExecutableSpec | CondaEnvironmentSpec | ContainerEnvironmentSpec


def _decode(data: Mapping[str, Any], schema: str, envelope_kind: str, prefix: str, fields: set[str], constructor: Any) -> Any:
    raw = dict(data)
    attached = raw.pop("id", None)
    envelope = validate_envelope(raw, schema=schema, kind=envelope_kind, prefix=prefix, identifying_payload=raw.get("payload", {}))
    payload = envelope["payload"]
    _check_fields(payload, fields)
    value = constructor(**dict(payload), metadata=envelope.get("metadata", {}))
    if attached is not None and attached != value.semantic_id:
        raise EnvironmentSpecError("attached semantic ID does not match payload", context={"expected": value.semantic_id, "observed": attached})
    return value


def spec_from_data(data: Mapping[str, Any]) -> EnvironmentSpec:
    """Decode a closed tagged v1.1 environment-spec envelope."""

    payload = data.get("payload") if isinstance(data, Mapping) else None
    if not isinstance(payload, Mapping):
        raise EnvironmentSpecError("environment spec envelope payload must be a mapping")
    kind = payload.get("kind")
    constructors = {"current": (CurrentEnvironmentSpec, {"kind"}), "python": (PythonExecutableSpec, {"kind", "executable", "env", "pythonpath_policy", "extra_pythonpath"}), "conda": (CondaEnvironmentSpec, {"kind", "prefix", "name", "conda_executable", "launch_mode", "env", "pythonpath_policy", "extra_pythonpath"}), "container": (ContainerEnvironmentSpec, {"kind", "image", "runtime"})}
    if kind not in constructors:
        raise EnvironmentSpecError("unsupported environment spec kind", context={"kind": kind})
    constructor, fields = constructors[kind]
    return _decode(data, _SPEC_SCHEMA, _SPEC_KIND, _SPEC_PREFIX, fields, constructor)


for _spec_type in (CurrentEnvironmentSpec, PythonExecutableSpec, CondaEnvironmentSpec, ContainerEnvironmentSpec):
    _spec_type.from_data = classmethod(lambda cls, data, _cls=_spec_type: spec_from_data(data))


__all__ = ["CondaEnvironmentSpec", "ContainerEnvironmentSpec", "CurrentEnvironmentSpec", "EnvironmentLockRef", "EnvironmentSpec", "PythonExecutableSpec", "spec_from_data"]
