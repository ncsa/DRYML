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
    schema_version: int = ENVIRONMENT_FRAGMENT_SCHEMA_VERSION

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
        """Build a requirement fragment from serialized data."""

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
            schema_version=data.get("schema_version", ENVIRONMENT_FRAGMENT_SCHEMA_VERSION),
        )


def _fragment_source(target: type, mode: str) -> str:
    return f"{target.__module__}.{target.__qualname__}:{mode}"


def _attach_fragment(cls: type, fragment: RequirementFragment) -> type:
    own = tuple(cls.__dict__.get(FRAGMENT_ATTR, ()))
    setattr(cls, FRAGMENT_ATTR, own + (fragment,))
    _attach_generic_fragment(cls, fragment)
    return cls


def _attach_generic_fragment(cls: type, fragment: RequirementFragment) -> None:
    from dryml.annotations.decorators import attach_fragment
    from dryml.annotations.env import normalize_environment_requirement_fragment
    from dryml.annotations.model import AnnotationFragment, AnnotationTarget, SourceTrace
    from dryml.annotations.namespaces import ENVIRONMENT

    source = SourceTrace(
        "decorator",
        AnnotationTarget("class", cls.__module__, cls.__qualname__),
        label=fragment.source,
        namespace=ENVIRONMENT,
        metadata={"legacy_environment_fragment_mode": fragment.mode},
    )
    annotation = AnnotationFragment(
        ENVIRONMENT,
        "requirement",
        normalize_environment_requirement_fragment(
            python=fragment.python,
            requirements=fragment.requirements,
            excludes=fragment.excludes,
            capabilities=fragment.capabilities,
            tags=fragment.tags,
            dryml_protocol=fragment.dryml_protocol,
            schema_versions=fragment.schema_versions,
        ),
        source,
        merge_policy=fragment.mode,
    )
    attach_fragment(cls, annotation)


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

    result = EnvironmentRequirement()
    sources: list[str] = []

    for fragment in fragments:
        if fragment.source:
            sources.append(fragment.source)
        current = EnvironmentRequirement(
            requirements=fragment.requirements,
            excludes=fragment.excludes,
            capabilities=fragment.capabilities,
            tags=fragment.tags,
            python=fragment.python,
            dryml_protocol=fragment.dryml_protocol,
            schema_versions=fragment.schema_versions,
        )
        if fragment.mode == "base":
            result = result.merge(current)
            continue
        if fragment.mode == "override":
            result = EnvironmentRequirement(
                requirements=current.requirements or result.requirements,
                excludes=current.excludes or result.excludes,
                capabilities=current.capabilities or result.capabilities,
                tags=current.tags or result.tags,
                python=current.python if current.python is not None else result.python,
                dryml_protocol=current.dryml_protocol if current.dryml_protocol is not None else result.dryml_protocol,
                schema_versions={**result.schema_versions, **current.schema_versions},
            )
            continue
        result = result.merge(current)

    return EnvironmentRequirement(**{**result.to_data(), "details": {"sources": tuple(sources)} if sources else {}})


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
