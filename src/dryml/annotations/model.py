"""Immutable closed v1.1 models for declaration-only annotations."""

from __future__ import annotations

import re
import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from dryml.formats import deep_freeze_json, json_ready, make_envelope, semantic_id, validate_envelope
from dryml.formats.errors import CanonicalJSONError, ContentIDError, EnvelopeError

from .errors import AnnotationValidationError

_SCHEMA = "dryml.annotation.v1.1"
_KIND = "annotation_fragment"
_PREFIX = "annotation"
_BOUNDS = {"max_depth": 8, "max_nodes": 1024, "max_entries": 64}
_NAMESPACE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_TARGET_KINDS = frozenset({"function", "method", "class", "descriptor", "synthetic"})
_SOURCE_KINDS = frozenset({"decorator", "override", "synthetic"})


def validate_namespace(namespace: str) -> str:
    """Validate and return one bounded annotation namespace.

    Args:
        namespace: Namespace such as ``"environment"``.

    Returns:
        The validated namespace.

    Raises:
        AnnotationValidationError: If the namespace is not a bounded identifier.
    """

    if not isinstance(namespace, str) or _NAMESPACE.fullmatch(namespace) is None:
        raise AnnotationValidationError("invalid annotation namespace", context={"namespace": namespace})
    return namespace


def _frozen_mapping(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AnnotationValidationError(f"{field_name} must be a mapping")
    try:
        frozen = deep_freeze_json(value, **_BOUNDS)
    except CanonicalJSONError as error:
        raise AnnotationValidationError(f"{field_name} must be bounded JSON", context=error.context) from error
    assert isinstance(frozen, Mapping)
    return frozen


@dataclass(frozen=True, slots=True)
class AnnotationTarget:
    """Bounded identifying description of the live target carrying a fragment.

    Args:
        kind: Closed target category.
        module: Defining module name, if known.
        qualname: Defining qualified name, if known.
        owner_module: Owning class module for members, if known.
        owner_qualname: Owning class qualified name for members, if known.
        member_name: Member name for a descriptor or method, if known.
        descriptor_kind: Static descriptor category, if applicable.
        metadata: Additional bounded identifying JSON data.
    """

    kind: str
    module: str | None
    qualname: str | None
    owner_module: str | None = None
    owner_qualname: str | None = None
    member_name: str | None = None
    descriptor_kind: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in _TARGET_KINDS:
            raise AnnotationValidationError("unknown annotation target kind", context={"kind": self.kind})
        if any(value is not None and not isinstance(value, str) for value in (self.module, self.qualname, self.owner_module, self.owner_qualname, self.member_name, self.descriptor_kind)):
            raise AnnotationValidationError("annotation target text fields must be strings or null")
        _bounded_text(self.module, self.qualname, self.owner_module, self.owner_qualname, self.member_name, self.descriptor_kind)
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata, "target.metadata"))

    def to_data(self) -> dict[str, Any]:
        """Return the closed JSON-ready identifying target payload."""

        return {"kind": self.kind, "module": self.module, "qualname": self.qualname, "owner_module": self.owner_module, "owner_qualname": self.owner_qualname, "member_name": self.member_name, "descriptor_kind": self.descriptor_kind, "metadata": json_ready(self.metadata, **_BOUNDS)}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "AnnotationTarget":
        """Decode one closed target payload without importing its module."""

        fields = {"kind", "module", "qualname", "owner_module", "owner_qualname", "member_name", "descriptor_kind", "metadata"}
        if not isinstance(data, Mapping) or set(data) != fields:
            raise AnnotationValidationError("annotation target fields are closed")
        return cls(**dict(data))


@dataclass(frozen=True, slots=True)
class SourceTrace:
    """Identifying source location and explanation for an annotation fragment.

    Args:
        kind: Closed source category.
        target: Declaring target, if available.
        label: Human-readable declaration label.
        namespace: Declared namespace, if available.
        path: Source path retained as identifying data.
        line: One-based source line, if known.
        column: Zero-based source column, if known.
        metadata: Additional bounded identifying JSON data.
    """

    kind: str
    target: AnnotationTarget | None = None
    label: str | None = None
    namespace: str | None = None
    path: str | None = None
    line: int | None = None
    column: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in _SOURCE_KINDS:
            raise AnnotationValidationError("unknown annotation source kind", context={"kind": self.kind})
        if self.namespace is not None:
            validate_namespace(self.namespace)
        if any(value is not None and not isinstance(value, str) for value in (self.label, self.path)):
            raise AnnotationValidationError("source label and path must be strings or null")
        _bounded_text(self.label, self.path)
        if self.line is not None and (isinstance(self.line, bool) or not isinstance(self.line, int) or self.line <= 0):
            raise AnnotationValidationError("source line must be a positive integer or null")
        if self.column is not None and (isinstance(self.column, bool) or not isinstance(self.column, int) or self.column < 0):
            raise AnnotationValidationError("source line and column must be non-negative integers or null")
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata, "source.metadata"))

    def to_data(self) -> dict[str, Any]:
        """Return the closed JSON-ready identifying source payload."""

        return {"kind": self.kind, "target": None if self.target is None else self.target.to_data(), "label": self.label, "namespace": self.namespace, "path": self.path, "line": self.line, "column": self.column, "metadata": json_ready(self.metadata, **_BOUNDS)}

    def to_display_data(self) -> dict[str, Any]:
        """Return redacted source diagnostics without changing semantic identity."""

        data = self.to_data()
        if data["path"] is not None:
            data["path"] = "<local-path>"
        data["metadata"] = _redact_diagnostics(data["metadata"])
        return data

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "SourceTrace":
        """Decode one closed source payload without following its target."""

        fields = {"kind", "target", "label", "namespace", "path", "line", "column", "metadata"}
        if not isinstance(data, Mapping) or set(data) != fields:
            raise AnnotationValidationError("annotation source fields are closed")
        values = dict(data)
        target = values.pop("target")
        return cls(target=AnnotationTarget.from_data(target) if target is not None else None, **values)


@dataclass(frozen=True, slots=True)
class AnnotationFragment:
    """One closed, immutable, identifying requirement or default declaration.

    Every supplied field participates in ``semantic_id``.  It is metadata on a
    live target only and has no CDef, Store, query, session, or call behavior.

    Args:
        target: Bound declaration target.
        namespace: Namespace whose values are combined.
        kind: Either ``"requirement"`` or ``"default"``.
        fragment: Bounded JSON payload or typed family envelope.
        source: Identifying declaration source.
        priority: Higher values apply later during resolution.
        merge_policy: Closed policy or ``None`` for semantic merge.
    """

    target: AnnotationTarget
    namespace: str
    kind: Literal["requirement", "default"]
    fragment: Mapping[str, Any]
    source: SourceTrace
    priority: int = 0
    merge_policy: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target, AnnotationTarget) or not isinstance(self.source, SourceTrace):
            raise AnnotationValidationError("annotation target and source must be typed values")
        validate_namespace(self.namespace)
        if self.kind not in {"requirement", "default"}:
            raise AnnotationValidationError("unknown annotation fragment kind", context={"kind": self.kind})
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise AnnotationValidationError("annotation priority must be an integer")
        if self.merge_policy is not None and not isinstance(self.merge_policy, str):
            raise AnnotationValidationError("annotation merge policy must be a string or null")
        object.__setattr__(self, "fragment", _frozen_mapping(self.fragment, "fragment"))

    @property
    def semantic_id(self) -> str:
        """Return the complete ``annotation-v1.1`` identity for this fragment."""

        return semantic_id(_PREFIX, _SCHEMA, _KIND, self.to_payload(), **_BOUNDS)

    @property
    def id(self) -> str:
        """Return the semantic-ID alias used by other v1.1 families."""

        return self.semantic_id

    def to_payload(self) -> dict[str, Any]:
        """Return the complete closed identifying fragment payload."""

        return {"target": self.target.to_data(), "namespace": self.namespace, "kind": self.kind, "fragment": json_ready(self.fragment, **_BOUNDS), "source": self.source.to_data(), "priority": self.priority, "merge_policy": self.merge_policy}

    def to_data(self) -> dict[str, Any]:
        """Return a closed, self-validating ``dryml.annotation.v1.1`` envelope."""

        return make_envelope(schema=_SCHEMA, kind=_KIND, prefix=_PREFIX, payload=self.to_payload(), semantic_id=self.semantic_id, identifying_payload=self.to_payload(), **_BOUNDS)

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "AnnotationFragment":
        """Validate and decode a complete annotation envelope and attached ID."""

        try:
            envelope = validate_envelope(data, schema=_SCHEMA, kind=_KIND, prefix=_PREFIX, identifying_payload=data.get("payload", {}) if isinstance(data, Mapping) else {}, **_BOUNDS)
        except (CanonicalJSONError, ContentIDError, EnvelopeError) as error:
            raise AnnotationValidationError(str(error), context=getattr(error, "context", {})) from error
        payload = envelope["payload"]
        fields = {"target", "namespace", "kind", "fragment", "source", "priority", "merge_policy"}
        if set(payload) != fields:
            raise AnnotationValidationError("annotation fragment payload fields are closed")
        value = cls(AnnotationTarget.from_data(payload["target"]), payload["namespace"], payload["kind"], payload["fragment"], SourceTrace.from_data(payload["source"]), payload["priority"], payload["merge_policy"])
        if envelope.get("id") is not None and value.semantic_id != envelope["id"]:
            raise AnnotationValidationError("annotation attached ID does not match payload")
        return value


@dataclass(frozen=True, slots=True)
class UnresolvedAnnotationResult:
    """Import-free outcome for an unresolved Definition or CDef target.

    Args:
        reason: Stable explanation of why no live class was inspected.
        method_name: Requested method name, when collecting a method.
    """

    reason: str
    method_name: str | None = None


def source_from_target(target: Any, *, namespace: str, label: str | None = None) -> SourceTrace:
    """Create an identifying decorator source without invoking ``target``.

    Args:
        target: Supplied live function, descriptor, or class.
        namespace: Declared namespace.
        label: Optional user-facing source label.

    Returns:
        A source trace describing only already-available target attributes.
    """

    return SourceTrace("decorator", target=target_from_live(target), label=label, namespace=namespace)


def target_from_live(target: Any) -> AnnotationTarget:
    """Describe one supplied live target without binding or executing it.

    Args:
        target: Function, class, static/class method, or extensible descriptor.

    Returns:
        A bounded identifying target description.
    """

    if isinstance(target, type):
        return AnnotationTarget("class", getattr(target, "__module__", None), getattr(target, "__qualname__", None))
    if isinstance(target, (staticmethod, classmethod)):
        function = target.__func__
        return AnnotationTarget("descriptor", getattr(function, "__module__", None), getattr(function, "__qualname__", None), descriptor_kind=type(target).__name__)
    if inspect.isfunction(target):
        qualname = getattr(target, "__qualname__", None)
        return AnnotationTarget("function" if isinstance(qualname, str) and "." not in qualname else "method", getattr(target, "__module__", None), qualname)
    return AnnotationTarget("descriptor", getattr(type(target), "__module__", None), getattr(type(target), "__qualname__", None), descriptor_kind=type(target).__name__)


def _bounded_text(*values: str | None) -> None:
    """Validate public target/source text against the common JSON bounds."""

    try:
        deep_freeze_json({str(index): value for index, value in enumerate(values) if value is not None}, **_BOUNDS)
    except CanonicalJSONError as error:
        raise AnnotationValidationError("annotation text exceeds bounds", context=error.context) from error


def _redact_diagnostics(value: Any, *, key: str | None = None) -> Any:
    """Redact recognizable secrets and direct paths from display-only data."""

    if key is not None and re.search(r"password|passwd|secret|token|api[_-]?key|credential", key, re.I):
        return "<redacted>"
    if key is not None and re.search(r"(?:^|[_-])(?:path|directory|folder|cwd|filename)(?:$|[_-])", key, re.I):
        return "<local-path>"
    if isinstance(value, Mapping):
        return {str(name): _redact_diagnostics(item, key=str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [_redact_diagnostics(item) for item in value]
    if isinstance(value, str) and (value.startswith(("/", "~/", "./", "../", "\\\\")) or re.match(r"^[A-Za-z]:[\\/]", value)):
        return "<local-path>"
    return value


__all__ = ["AnnotationFragment", "AnnotationTarget", "SourceTrace", "UnresolvedAnnotationResult", "source_from_target", "target_from_live", "validate_namespace"]
