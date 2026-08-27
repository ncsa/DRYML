"""Requirement fragments and decorator sugar for classes/providers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from .errors import EnvironmentRequirementError
from .requirements import EnvironmentRequirement
from .schema import ENVIRONMENT_FRAGMENT_SCHEMA_VERSION
from .serialization import freeze_mapping
from .utils import coerce_tuple, normalize_requirement_string

FRAGMENT_ATTR = "__dryml_environment_fragments__"


@dataclass(frozen=True, slots=True)
class RequirementFragment:
    """Composable piece of an environment requirement."""

    requirements: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    python: str | None = None
    dryml_protocol: str | None = None
    schema_versions: Mapping[str, str] = field(default_factory=dict)
    source: str | None = None
    mode: Literal["base", "add", "override"] = "add"
    schema_version: str = ENVIRONMENT_FRAGMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "requirements", tuple(normalize_requirement_string(req) for req in coerce_tuple(self.requirements)))
        object.__setattr__(self, "excludes", tuple(str(item) for item in coerce_tuple(self.excludes)))
        object.__setattr__(self, "capabilities", tuple(str(item) for item in coerce_tuple(self.capabilities)))
        object.__setattr__(self, "tags", tuple(str(item) for item in coerce_tuple(self.tags)))
        object.__setattr__(self, "schema_versions", freeze_mapping(self.schema_versions))
        if self.mode not in {"base", "add", "override"}:
            raise EnvironmentRequirementError(f"unknown requirement fragment mode {self.mode!r}")

    def to_data(self) -> dict[str, Any]:
        """Return JSON-compatible fragment data."""

        return {
            "schema_version": self.schema_version,
            "requirements": list(self.requirements),
            "excludes": list(self.excludes),
            "capabilities": list(self.capabilities),
            "tags": list(self.tags),
            "python": self.python,
            "dryml_protocol": self.dryml_protocol,
            "schema_versions": dict(self.schema_versions),
            "source": self.source,
            "mode": self.mode,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "RequirementFragment":
        """Build a requirement fragment from exact v1.1 serialized data."""

        expected = {
            "schema_version", "requirements", "excludes", "capabilities",
            "tags", "python", "dryml_protocol", "schema_versions", "source", "mode",
        }
        unknown, missing = set(data) - expected, expected - set(data)
        if unknown or missing:
            raise EnvironmentRequirementError(
                "environment requirement fragment fields are closed",
                context={"unknown": sorted(unknown), "missing": sorted(missing)},
            )
        if data["schema_version"] != ENVIRONMENT_FRAGMENT_SCHEMA_VERSION:
            raise EnvironmentRequirementError(
                "unsupported environment requirement fragment version",
                context={"observed_version": data["schema_version"], "supported_version": ENVIRONMENT_FRAGMENT_SCHEMA_VERSION},
            )

        return cls(
            requirements=tuple(data.get("requirements", ())),
            excludes=tuple(data.get("excludes", ())),
            capabilities=tuple(data.get("capabilities", ())),
            tags=tuple(data.get("tags", ())),
            python=data.get("python"),
            dryml_protocol=data.get("dryml_protocol"),
            schema_versions=data.get("schema_versions", {}),
            source=data.get("source"),
            mode=data.get("mode", "add"),
            schema_version=data["schema_version"],
        )


def _fragment_source(target: type, mode: str) -> str:
    return f"{target.__module__}.{target.__qualname__}:{mode}"


def _attach_fragment(cls: type, fragment: RequirementFragment) -> type:
    own = tuple(cls.__dict__.get(FRAGMENT_ATTR, ()))
    setattr(cls, FRAGMENT_ATTR, own + (fragment,))
    return cls


def req(**kwargs: Any):
    """Decorator adding a base environment requirement fragment to a class."""

    def decorate(cls: type) -> type:
        params = dict(kwargs)
        fragment = RequirementFragment(mode="base", source=params.pop("source", None) or _fragment_source(cls, "base"), **params)
        return _attach_fragment(cls, fragment)

    return decorate


def add_req(**kwargs: Any):
    """Decorator adding an additive environment requirement fragment to a class."""

    def decorate(cls: type) -> type:
        params = dict(kwargs)
        fragment = RequirementFragment(mode="add", source=params.pop("source", None) or _fragment_source(cls, "add"), **params)
        return _attach_fragment(cls, fragment)

    return decorate


def override_req(**kwargs: Any):
    """Decorator adding an explicit override environment requirement fragment."""

    def decorate(cls: type) -> type:
        params = dict(kwargs)
        fragment = RequirementFragment(mode="override", source=params.pop("source", None) or _fragment_source(cls, "override"), **params)
        return _attach_fragment(cls, fragment)

    return decorate


def fragments_for_class(cls: type) -> tuple[RequirementFragment, ...]:
    """Return class fragments in deterministic base-to-subclass MRO order."""

    fragments: list[RequirementFragment] = []
    for base in reversed(cls.__mro__):
        if base is object:
            continue
        fragments.extend(base.__dict__.get(FRAGMENT_ATTR, ()))
    return tuple(fragments)


def compose_fragments(fragments: Iterable[RequirementFragment]) -> EnvironmentRequirement:
    """Compose fragments into one deterministic environment requirement."""

    requirements: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    python: str | None = None
    dryml_protocol: str | None = None
    schema_versions: dict[str, str] = {}
    sources: list[str] = []

    for fragment in fragments:
        if fragment.source:
            sources.append(fragment.source)
        if fragment.mode == "base":
            requirements = requirements + fragment.requirements
            excludes = excludes + fragment.excludes
            capabilities = capabilities + fragment.capabilities
            tags = tags + fragment.tags
            if fragment.python is not None:
                python = fragment.python
            if fragment.dryml_protocol is not None:
                dryml_protocol = fragment.dryml_protocol
            schema_versions.update(fragment.schema_versions)
            continue
        if fragment.mode == "override":
            if fragment.requirements:
                requirements = fragment.requirements
            if fragment.excludes:
                excludes = fragment.excludes
            if fragment.capabilities:
                capabilities = fragment.capabilities
            if fragment.tags:
                tags = fragment.tags
            if fragment.python is not None:
                python = fragment.python
            if fragment.dryml_protocol is not None:
                dryml_protocol = fragment.dryml_protocol
            if fragment.schema_versions:
                schema_versions.update(fragment.schema_versions)
            continue
        requirements = requirements + fragment.requirements
        excludes = excludes + fragment.excludes
        capabilities = capabilities + fragment.capabilities
        tags = tags + fragment.tags
        if fragment.python is not None:
            if python is not None and python != fragment.python:
                raise EnvironmentRequirementError(
                    "conflicting additive Python requirement fragments",
                    context={"current": python, "new": fragment.python},
                )
            python = fragment.python
        if fragment.dryml_protocol is not None:
            if dryml_protocol is not None and dryml_protocol != fragment.dryml_protocol:
                raise EnvironmentRequirementError(
                    "conflicting additive DRYML protocol requirement fragments",
                    context={"current": dryml_protocol, "new": fragment.dryml_protocol},
                )
            dryml_protocol = fragment.dryml_protocol
        for key, value in fragment.schema_versions.items():
            if key in schema_versions and schema_versions[key] != value:
                raise EnvironmentRequirementError(
                    "conflicting additive schema requirement fragments",
                    context={"schema": key, "current": schema_versions[key], "new": value},
                )
            schema_versions[key] = value

    return EnvironmentRequirement(
        python=python,
        requirements=requirements,
        excludes=excludes,
        capabilities=capabilities,
        tags=tags,
        dryml_protocol=dryml_protocol,
        schema_versions=schema_versions,
        details={"sources": tuple(sources)} if sources else {},
    )


def requirements_for_class(cls: type) -> EnvironmentRequirement:
    """Compose environment requirements declared on a class hierarchy."""

    return compose_fragments(fragments_for_class(cls))


__all__ = [
    "RequirementFragment",
    "FRAGMENT_ATTR",
    "req",
    "add_req",
    "override_req",
    "fragments_for_class",
    "requirements_for_class",
    "compose_fragments",
]
