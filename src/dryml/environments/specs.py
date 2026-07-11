"""Environment selection specs and lock references."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from .errors import EnvironmentSpecError
from .ids import content_id
from .schema import ENVIRONMENT_LOCK_REF_SCHEMA_VERSION, ENVIRONMENT_SPEC_SCHEMA_VERSION
from .serialization import deep_freeze_json, json_ready
from .utils import coerce_tuple


ProbeWorkerCommand = ("-m", "dryml.environments.probe_worker", "--json")
_PYTHONPATH_POLICIES = frozenset({"none", "explicit", "inherit", "dryml-source"})


def _normalize_pythonpath_policy(value: Any) -> str:
    """Return the canonical spelling for a Python-path probe policy."""

    if not isinstance(value, str):
        raise EnvironmentSpecError("Python path probe policy must be a string", context={"pythonpath_policy": value})
    return value.strip().lower().replace("_", "-")


def _validate_pythonpath_policy(value: Any) -> None:
    """Reject invalid probe policy values before a worker can be launched."""

    if _normalize_pythonpath_policy(value) not in _PYTHONPATH_POLICIES:
        raise EnvironmentSpecError(
            f"unknown Python path probe policy {value!r}",
            context={"pythonpath_policy": value},
        )


@dataclass(frozen=True, slots=True)
class CurrentEnvironmentSpec:
    """Spec selecting the current Python process environment."""

    kind: Literal["current"] = "current"
    schema_version: int = ENVIRONMENT_SPEC_SCHEMA_VERSION

    @property
    def id(self) -> str:
        """Stable content ID for this spec."""

        return content_id("envspec", self.schema_version, self.to_data())

    def to_data(self) -> dict[str, Any]:
        """Return JSON-compatible spec data."""

        return {"schema_version": self.schema_version, "kind": self.kind}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "CurrentEnvironmentSpec":
        """Build a current-environment spec from serialized data."""

        return cls(schema_version=data.get("schema_version", ENVIRONMENT_SPEC_SCHEMA_VERSION))


@dataclass(frozen=True, slots=True)
class PythonExecutableSpec:
    """Spec selecting an environment by Python executable path."""

    executable: str
    env: Mapping[str, str] = field(default_factory=dict)
    pythonpath_policy: str = "none"
    extra_pythonpath: tuple[str, ...] = ()
    kind: Literal["python"] = "python"
    schema_version: int = ENVIRONMENT_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_pythonpath_policy(self.pythonpath_policy)
        object.__setattr__(self, "env", deep_freeze_json(self.env))
        object.__setattr__(self, "pythonpath_policy", _normalize_pythonpath_policy(self.pythonpath_policy))
        object.__setattr__(self, "extra_pythonpath", tuple(str(path) for path in coerce_tuple(self.extra_pythonpath)))

    @property
    def id(self) -> str:
        """Stable content ID for this spec."""

        return content_id("envspec", self.schema_version, self.to_data())

    def probe_command(self) -> list[str]:
        """Return the command used to run the environment probe worker."""

        return [self.executable, *ProbeWorkerCommand]

    def to_data(self) -> dict[str, Any]:
        """Return JSON-compatible spec data."""

        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "executable": self.executable,
            "env": json_ready(self.env),
            "pythonpath_policy": self.pythonpath_policy,
            "extra_pythonpath": list(self.extra_pythonpath),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "PythonExecutableSpec":
        """Build a Python executable spec from serialized data."""

        return cls(
            executable=data["executable"],
            env=data.get("env", {}),
            pythonpath_policy=data.get("pythonpath_policy", "none"),
            extra_pythonpath=tuple(data.get("extra_pythonpath", ())),
            schema_version=data.get("schema_version", ENVIRONMENT_SPEC_SCHEMA_VERSION),
        )


@dataclass(frozen=True, slots=True)
class CondaEnvironmentSpec:
    """Spec selecting a Conda environment by prefix or name."""

    prefix: str | None = None
    name: str | None = None
    conda_executable: str = "conda"
    launch_mode: Literal["direct", "conda-run"] = "direct"
    env: Mapping[str, str] = field(default_factory=dict)
    pythonpath_policy: str = "none"
    extra_pythonpath: tuple[str, ...] = ()
    kind: Literal["conda"] = "conda"
    schema_version: int = ENVIRONMENT_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.prefix and self.name:
            raise EnvironmentSpecError("CondaEnvironmentSpec accepts prefix or name, not both")
        if self.launch_mode not in {"direct", "conda-run"}:
            raise EnvironmentSpecError(
                f"unsupported Conda launch mode {self.launch_mode!r}",
                context={"launch_mode": self.launch_mode},
            )
        _validate_pythonpath_policy(self.pythonpath_policy)
        object.__setattr__(self, "env", deep_freeze_json(self.env))
        object.__setattr__(self, "pythonpath_policy", _normalize_pythonpath_policy(self.pythonpath_policy))
        object.__setattr__(self, "extra_pythonpath", tuple(str(path) for path in coerce_tuple(self.extra_pythonpath)))

    @property
    def id(self) -> str:
        """Stable content ID for this spec."""

        return content_id("envspec", self.schema_version, self.to_data())

    def direct_python_executable(self, *, os_name: str | None = None) -> str:
        """Return the direct Python executable path for a prefix-based Conda spec."""

        if not self.prefix:
            raise EnvironmentSpecError("Conda direct launch requires a prefix")
        if (os_name or os.name) == "nt":
            return os.path.join(self.prefix, "python.exe")
        return os.path.join(self.prefix, "bin", "python")

    def probe_command(self, *, os_name: str | None = None) -> list[str]:
        """Return the probe command for direct or ``conda run`` launch mode."""

        if self.launch_mode == "direct":
            return [self.direct_python_executable(os_name=os_name), *ProbeWorkerCommand]
        if not self.prefix and not self.name:
            raise EnvironmentSpecError("conda-run launch requires prefix or name")
        cmd = [self.conda_executable, "run"]
        if self.prefix:
            cmd.extend(["-p", self.prefix])
        else:
            cmd.extend(["-n", self.name or ""])
        cmd.extend(["--no-capture-output", "--", "python", *ProbeWorkerCommand])
        return cmd

    def to_data(self) -> dict[str, Any]:
        """Return JSON-compatible spec data."""

        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "prefix": self.prefix,
            "name": self.name,
            "conda_executable": self.conda_executable,
            "launch_mode": self.launch_mode,
            "env": json_ready(self.env),
            "pythonpath_policy": self.pythonpath_policy,
            "extra_pythonpath": list(self.extra_pythonpath),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "CondaEnvironmentSpec":
        """Build a Conda environment spec from serialized data."""

        return cls(
            prefix=data.get("prefix"),
            name=data.get("name"),
            conda_executable=data.get("conda_executable", "conda"),
            launch_mode=data.get("launch_mode", "direct"),
            env=data.get("env", {}),
            pythonpath_policy=data.get("pythonpath_policy", "none"),
            extra_pythonpath=tuple(data.get("extra_pythonpath", ())),
            schema_version=data.get("schema_version", ENVIRONMENT_SPEC_SCHEMA_VERSION),
        )


@dataclass(frozen=True, slots=True)
class ContainerEnvironmentSpec:
    """Structural container environment reference.

    Container execution/probing is intentionally deferred; this spec is a
    serializable placeholder for future dispatch and record sprints.
    """

    image: str
    runtime: str | None = None
    kind: Literal["container"] = "container"
    schema_version: int = ENVIRONMENT_SPEC_SCHEMA_VERSION

    @property
    def id(self) -> str:
        """Stable content ID for this spec."""

        return content_id("envspec", self.schema_version, self.to_data())

    def to_data(self) -> dict[str, Any]:
        """Return JSON-compatible spec data."""

        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "image": self.image,
            "runtime": self.runtime,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ContainerEnvironmentSpec":
        """Build a container spec from serialized data."""

        return cls(
            image=data["image"],
            runtime=data.get("runtime"),
            schema_version=data.get("schema_version", ENVIRONMENT_SPEC_SCHEMA_VERSION),
        )


@dataclass(frozen=True, slots=True)
class EnvironmentLockRef:
    """Reference to an exact external environment lock or recipe."""

    kind: str
    uri: str
    digest: str | None = None
    schema_version: int = ENVIRONMENT_LOCK_REF_SCHEMA_VERSION

    @property
    def id(self) -> str:
        """Stable content ID for this lock reference."""

        return content_id("envlock", self.schema_version, self.to_data())

    def to_data(self) -> dict[str, Any]:
        """Return JSON-compatible lock-reference data."""

        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "uri": self.uri,
            "digest": self.digest,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "EnvironmentLockRef":
        """Build a lock reference from serialized data."""

        return cls(
            kind=data["kind"],
            uri=data["uri"],
            digest=data.get("digest"),
            schema_version=data.get("schema_version", ENVIRONMENT_LOCK_REF_SCHEMA_VERSION),
        )


EnvironmentSpec = CurrentEnvironmentSpec | PythonExecutableSpec | CondaEnvironmentSpec | ContainerEnvironmentSpec


def spec_from_data(data: Mapping[str, Any]) -> EnvironmentSpec:
    """Deserialize a tagged environment spec."""

    kind = data.get("kind")
    if kind == "current":
        return CurrentEnvironmentSpec.from_data(data)
    if kind == "python":
        return PythonExecutableSpec.from_data(data)
    if kind == "conda":
        return CondaEnvironmentSpec.from_data(data)
    if kind == "container":
        return ContainerEnvironmentSpec.from_data(data)
    raise EnvironmentSpecError(f"unsupported environment spec kind {kind!r}")


__all__ = [
    "CurrentEnvironmentSpec",
    "PythonExecutableSpec",
    "CondaEnvironmentSpec",
    "ContainerEnvironmentSpec",
    "EnvironmentLockRef",
    "EnvironmentSpec",
    "spec_from_data",
]
