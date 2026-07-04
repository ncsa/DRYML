"""Canonical model objects for DRYML planning annotations."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from dryml.formats import CanonicalJSONError, deep_freeze_json, json_ready

from .errors import AnnotationValidationError

FragmentKind = Literal["requirement", "default"]

_NAMESPACE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_TARGET_KINDS = {"function", "method", "class", "provider", "operation_kind", "synthetic"}
_SOURCE_KINDS = {"decorator", "provider", "override", "stored_record", "cached_probe", "synthetic"}
_FRAGMENT_KINDS = {"requirement", "default"}


@dataclass(frozen=True, slots=True)
class AnnotationTarget:
    """Stable JSON-ready description of the Python target carrying metadata."""

    kind: str
    module: str | None
    qualname: str | None
    owner_module: str | None = None
    owner_qualname: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in _TARGET_KINDS:
            raise AnnotationValidationError("unknown annotation target kind", context={"kind": self.kind})
        object.__setattr__(self, "metadata", _freeze_json_mapping(self.metadata, "target.metadata"))

    def to_data(self) -> dict[str, Any]:
        """Return canonical JSON-ready target data."""

        return {
            "kind": self.kind,
            "module": self.module,
            "qualname": self.qualname,
            "owner_module": self.owner_module,
            "owner_qualname": self.owner_qualname,
            "metadata": json_ready(self.metadata),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "AnnotationTarget":
        """Build a target from JSON-ready data."""

        if not isinstance(data, Mapping):
            raise AnnotationValidationError("annotation target must be a mapping", context={"type": type(data).__name__})
        unknown = set(data) - {"kind", "module", "qualname", "owner_module", "owner_qualname", "metadata"}
        if unknown:
            raise AnnotationValidationError("annotation target has unknown fields", context={"fields": sorted(unknown)})
        return cls(
            kind=data.get("kind"),
            module=data.get("module"),
            qualname=data.get("qualname"),
            owner_module=data.get("owner_module"),
            owner_qualname=data.get("owner_qualname"),
            metadata=data.get("metadata") or {},
        )


@dataclass(frozen=True, slots=True)
class SourceTrace:
    """Machine-readable source of an annotation fragment or override."""

    kind: str
    target: AnnotationTarget | None = None
    label: str | None = None
    namespace: str | None = None
    path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in _SOURCE_KINDS:
            raise AnnotationValidationError("unknown annotation source kind", context={"kind": self.kind})
        if self.namespace is not None:
            validate_namespace(self.namespace)
        object.__setattr__(self, "metadata", _freeze_json_mapping(self.metadata, "source.metadata"))

    def to_data(self) -> dict[str, Any]:
        """Return canonical JSON-ready source data."""

        return {
            "kind": self.kind,
            "target": self.target.to_data() if self.target is not None else None,
            "label": self.label,
            "namespace": self.namespace,
            "path": self.path,
            "metadata": json_ready(self.metadata),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "SourceTrace":
        """Build a source trace from JSON-ready data."""

        if not isinstance(data, Mapping):
            raise AnnotationValidationError("annotation source must be a mapping", context={"type": type(data).__name__})
        unknown = set(data) - {"kind", "target", "label", "namespace", "path", "metadata"}
        if unknown:
            raise AnnotationValidationError("annotation source has unknown fields", context={"fields": sorted(unknown)})
        target = data.get("target")
        return cls(
            kind=data.get("kind"),
            target=AnnotationTarget.from_data(target) if target is not None else None,
            label=data.get("label"),
            namespace=data.get("namespace"),
            path=data.get("path"),
            metadata=data.get("metadata") or {},
        )


@dataclass(frozen=True, slots=True)
class AnnotationFragment:
    """One mergeable sidecar requirement or default fragment."""

    namespace: str
    kind: FragmentKind
    fragment: Mapping[str, Any]
    source: SourceTrace
    priority: int = 0
    merge_policy: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        validate_namespace(self.namespace)
        if self.kind not in _FRAGMENT_KINDS:
            raise AnnotationValidationError("unknown annotation fragment kind", context={"kind": self.kind})
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise AnnotationValidationError("annotation priority must be an integer", context={"priority": self.priority})
        if self.schema_version != 1:
            raise AnnotationValidationError("unsupported annotation schema version", context={"schema_version": self.schema_version})
        object.__setattr__(self, "fragment", _freeze_json_mapping(self.fragment, "fragment"))

    def to_data(self) -> dict[str, Any]:
        """Return canonical JSON-ready fragment data."""

        return {
            "namespace": self.namespace,
            "kind": self.kind,
            "fragment": json_ready(self.fragment),
            "source": self.source.to_data(),
            "priority": self.priority,
            "merge_policy": self.merge_policy,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "AnnotationFragment":
        """Build an annotation fragment from JSON-ready data."""

        if not isinstance(data, Mapping):
            raise AnnotationValidationError("annotation fragment must be a mapping", context={"type": type(data).__name__})
        unknown = set(data) - {"namespace", "kind", "fragment", "source", "priority", "merge_policy", "schema_version"}
        if unknown:
            raise AnnotationValidationError("annotation fragment has unknown fields", context={"fields": sorted(unknown)})
        return cls(
            namespace=data.get("namespace"),
            kind=data.get("kind"),
            fragment=data.get("fragment") or {},
            source=SourceTrace.from_data(data.get("source") or {"kind": "synthetic"}),
            priority=data.get("priority", 0),
            merge_policy=data.get("merge_policy"),
            schema_version=data.get("schema_version", 1),
        )


def validate_namespace(namespace: str) -> str:
    """Validate and return an annotation namespace string."""

    if not isinstance(namespace, str) or _NAMESPACE_RE.fullmatch(namespace) is None:
        raise AnnotationValidationError("invalid annotation namespace", context={"namespace": namespace})
    return namespace


def source_from_target(target: Any, *, namespace: str | None = None, kind: str = "decorator", label: str | None = None) -> SourceTrace:
    """Create a source trace for a decorated class/function/method target."""

    if isinstance(target, type):
        annotation_target = AnnotationTarget("class", getattr(target, "__module__", None), getattr(target, "__qualname__", None))
    else:
        qualname = getattr(target, "__qualname__", None)
        owner_qualname = qualname.rsplit(".", 1)[0] if isinstance(qualname, str) and "." in qualname else None
        target_kind = "method" if owner_qualname else "function"
        annotation_target = AnnotationTarget(
            target_kind,
            getattr(target, "__module__", None),
            qualname,
            getattr(target, "__module__", None) if owner_qualname else None,
            owner_qualname,
        )
    return SourceTrace(kind=kind, target=annotation_target, label=label, namespace=namespace)


def _freeze_json_mapping(value: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AnnotationValidationError("annotation JSON payload must be a mapping", context={"path": path, "type": type(value).__name__})
    try:
        frozen = deep_freeze_json(value)
    except CanonicalJSONError as exc:
        raise AnnotationValidationError("annotation payload is not JSON-ready", context={"path": path, **exc.context}) from exc
    assert isinstance(frozen, Mapping)
    return frozen


__all__ = [
    "AnnotationFragment",
    "AnnotationTarget",
    "FragmentKind",
    "SourceTrace",
    "source_from_target",
    "validate_namespace",
]
