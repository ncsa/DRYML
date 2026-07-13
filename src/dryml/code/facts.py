from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


JSON_PRIMITIVE = str | int | float | bool | None


def json_compatible(value: Any) -> JSON_PRIMITIVE | dict[str, Any] | list[Any]:
    """Return a JSON-compatible representation of *value*.

    Args:
        value: Arbitrary Python value collected by a code analyzer.

    Returns:
        A value composed only of JSON primitive, mapping, and list types.
    """

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_data") and callable(value.to_data):
        return json_compatible(value.to_data())
    if isinstance(value, Mapping):
        return {str(k): json_compatible(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_compatible(v) for v in value]
    return repr(value)


def _mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(json_compatible(value or {}))


@dataclass(frozen=True, slots=True)
class CodeFact:
    """Serializable fact produced by a DRYML code analyzer.

    Args:
        kind: Stable fact-kind string such as ``"callable"`` or ``"source"``.
        source: JSON-compatible metadata describing where the fact came from.
        data: JSON-compatible fact payload.
    """

    kind: str
    source: Mapping[str, Any] = field(default_factory=dict)
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _mapping(self.source))
        object.__setattr__(self, "data", _mapping(self.data))

    def to_data(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this fact."""

        return {
            "kind": self.kind,
            "source": json_compatible(self.source),
            "data": json_compatible(self.data),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "CodeFact":
        """Build a fact from JSON-compatible data.

        Args:
            data: Mapping created by :meth:`to_data`.

        Returns:
            A typed fact where the kind is known, otherwise a generic fact.
        """

        kind = data.get("kind")
        if kind == "diagnostic":
            return DiagnosticFact.from_data(data)
        if kind == "requirement":
            return RequirementFact.from_data(data)
        type_map = {
            "callable": CallableFact,
            "source": SourceFact,
            "symbol": SymbolFact,
            "annotation": AnnotationFact,
            "method_contract": MethodContractFact,
            "shape": ShapeFact,
            "ast_access": ASTAccessFact,
            "call_site": CallSiteFact,
            "static_call": StaticCallFact,
        }
        fact_type = type_map.get(kind, CodeFact)
        return fact_type(
            kind=kind,
            source=data.get("source") or {},
            data=data.get("data") or {},
        )


@dataclass(frozen=True, slots=True)
class CallableFact(CodeFact):
    """Fact describing callable identity, shape, and importability."""

    kind: str = "callable"


@dataclass(frozen=True, slots=True)
class SourceFact(CodeFact):
    """Fact containing source text and source-location metadata."""

    kind: str = "source"


@dataclass(frozen=True, slots=True)
class SymbolFact(CodeFact):
    """Fact describing an import or source-backed symbol reference."""

    kind: str = "symbol"


@dataclass(frozen=True, slots=True)
class AnnotationFact(CodeFact):
    """Fact containing a raw DRYML annotation fragment."""

    kind: str = "annotation"


@dataclass(frozen=True, slots=True)
class MethodContractFact(CodeFact):
    """Fact describing available DRYML Method contract metadata."""

    kind: str = "method_contract"


@dataclass(frozen=True, slots=True)
class ShapeFact(CodeFact):
    """Fact describing code-derived shape information."""

    kind: str = "shape"


@dataclass(frozen=True, slots=True)
class ASTAccessFact(CodeFact):
    """Fact describing syntactic, non-authoritative AST access hints."""

    kind: str = "ast_access"


@dataclass(frozen=True, slots=True)
class CallSiteFact(CodeFact):
    """Fact describing one syntactic call-like site without semantic resolution."""

    kind: str = "call_site"


@dataclass(frozen=True, slots=True)
class StaticCallFact(CodeFact):
    """Conservative static possibility for one source-level call expression.

    ``status`` and ``confidence`` live in ``data``. A resolved fact identifies a
    defensible static target; it never claims the call executes at runtime.
    """

    kind: str = "static_call"

    def __post_init__(self) -> None:
        """Validate the bounded, serializable static-call fact contract."""

        if self.kind != "static_call":
            raise ValueError("StaticCallFact kind must be 'static_call'")
        required = {
            "status", "confidence", "syntax", "display", "receiver",
            "method_name", "target", "reason", "relative_line",
            "absolute_line", "col_offset",
        }
        if not isinstance(self.source, Mapping) or not isinstance(self.data, Mapping):
            raise ValueError("StaticCallFact source and data must be mappings")
        missing = required.difference(self.data)
        if missing:
            raise ValueError(f"StaticCallFact data is missing required fields: {sorted(missing)!r}")
        if set(self.data) != required:
            raise ValueError("StaticCallFact data must use the fixed static-call schema")
        status = self.data["status"]
        confidence = self.data["confidence"]
        if status not in {"resolved", "unresolved", "ambiguous", "unsupported"}:
            raise ValueError(f"unsupported StaticCallFact status {status!r}")
        if confidence not in {"exact_static", "conservative_hint"}:
            raise ValueError(f"unsupported StaticCallFact confidence {confidence!r}")
        if not isinstance(self.data["syntax"], str) or len(self.data["syntax"]) > 4_096:
            raise ValueError("StaticCallFact syntax must be a bounded string")
        if self.data["syntax"] not in {"direct_name", "annotated_receiver_method", "attribute_chain", "other"}:
            raise ValueError(f"unsupported StaticCallFact syntax {self.data['syntax']!r}")
        if set(self.source).difference({"analyzer", "target_kind", "filename"}):
            raise ValueError("StaticCallFact source must use the fixed static-call schema")
        if self.source.get("analyzer") != "static_calls" or "target_kind" not in self.source:
            raise ValueError("StaticCallFact source must identify the static_calls analyzer and target kind")
        for field_name in ("target_kind", "filename"):
            value = self.source.get(field_name)
            if field_name == "target_kind" and (not isinstance(value, str) or not value):
                raise ValueError("StaticCallFact source target_kind must be a non-empty string")
            if value is not None and (not isinstance(value, str) or len(value) > 4_096):
                raise ValueError(f"StaticCallFact source {field_name} must be a bounded string or null")
        for field_name in ("display", "receiver", "method_name", "reason"):
            value = self.data[field_name]
            if value is not None and (not isinstance(value, str) or len(value) > 4_096):
                raise ValueError(f"StaticCallFact {field_name} must be a bounded string or null")
        if not isinstance(self.data["display"], str) or not self.data["display"]:
            raise ValueError("StaticCallFact display must be a non-empty bounded string")
        target = self.data["target"]
        if status == "resolved":
            if confidence != "exact_static" or self.data["reason"] is not None:
                raise ValueError("resolved StaticCallFact must be exact_static without a reason")
            if not isinstance(target, Mapping) or set(target) != {"kind", "import_path", "method_name", "subject_ref"}:
                raise ValueError("resolved StaticCallFact target must use the fixed target-reference schema")
            for value in target.values():
                if value is not None and (not isinstance(value, str) or len(value) > 4_096):
                    raise ValueError("StaticCallFact target values must be bounded strings or null")
            if not isinstance(target["kind"], str) or not target["kind"]:
                raise ValueError("resolved StaticCallFact target kind must be a non-empty string")
        else:
            if confidence != "conservative_hint" or target is not None:
                raise ValueError("non-resolved StaticCallFact must be a conservative hint without a target")
            if not isinstance(self.data["reason"], str) or not self.data["reason"] or len(self.data["reason"]) > 4_096:
                raise ValueError("non-resolved StaticCallFact requires a non-empty bounded reason")
        for field_name in ("relative_line", "absolute_line", "col_offset"):
            value = self.data[field_name]
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise ValueError(f"StaticCallFact {field_name} must be an integer or null")
        for field_name in ("relative_line", "absolute_line"):
            value = self.data[field_name]
            if value is not None and value < 1:
                raise ValueError(f"StaticCallFact {field_name} must be positive when present")
        if self.data["col_offset"] is not None and self.data["col_offset"] < 0:
            raise ValueError("StaticCallFact col_offset must be non-negative when present")
        # The fixed schema above proves conversion cannot recursively copy
        # arbitrary metadata before JSON normalization.
        CodeFact.__post_init__(self)


@dataclass(frozen=True, slots=True)
class DiagnosticFact(CodeFact):
    """Structured diagnostic emitted during code analysis.

    Severity values are ``debug``, ``info``, ``warning``, and ``error``.
    """

    kind: str = "diagnostic"
    severity: str = "info"
    code: str = "dryml.code.diagnostic"
    message: str = ""

    def to_data(self) -> dict[str, Any]:
        data = CodeFact.to_data(self)
        data.update({
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        })
        return data

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "DiagnosticFact":
        """Build a diagnostic fact from JSON-compatible data."""

        return cls(
            severity=data.get("severity", "info"),
            code=data.get("code", "dryml.code.diagnostic"),
            message=data.get("message", ""),
            source=data.get("source") or {},
            data=data.get("data") or {},
        )


@dataclass(frozen=True, slots=True)
class RequirementFact(CodeFact):
    """Fact representing one raw annotation requirement/default fragment."""

    kind: str = "requirement"
    namespace: str = "environment"
    requirement_kind: str = "requirement"
    fragment: Mapping[str, Any] = field(default_factory=dict)
    priority: int = 0
    merge_policy: str | None = None

    def __post_init__(self) -> None:
        CodeFact.__post_init__(self)
        object.__setattr__(self, "fragment", _mapping(self.fragment))

    def to_data(self) -> dict[str, Any]:
        data = CodeFact.to_data(self)
        data.update({
            "namespace": self.namespace,
            "requirement_kind": self.requirement_kind,
            "fragment": json_compatible(self.fragment),
            "priority": self.priority,
            "merge_policy": self.merge_policy,
        })
        return data

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "RequirementFact":
        """Build a requirement fact from JSON-compatible data."""

        return cls(
            namespace=data.get("namespace", "environment"),
            requirement_kind=data.get("requirement_kind", data.get("kind", "requirement")),
            fragment=data.get("fragment") or {},
            priority=data.get("priority", 0),
            merge_policy=data.get("merge_policy"),
            source=data.get("source") or {},
            data=data.get("data") or {},
        )


__all__ = [
    "ASTAccessFact",
    "AnnotationFact",
    "CallSiteFact",
    "CallableFact",
    "CodeFact",
    "DiagnosticFact",
    "MethodContractFact",
    "RequirementFact",
    "ShapeFact",
    "SourceFact",
    "StaticCallFact",
    "SymbolFact",
    "json_compatible",
]
