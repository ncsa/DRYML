from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
import re
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

        if not isinstance(data, Mapping):
            raise TypeError("CodeFact data must be a mapping")
        kind = data.get("kind")
        if kind == "dynamic_call":
            if type(data) is not dict:
                raise TypeError("DynamicCallFact wire data must be an exact dict")
            if set(data) != {"kind", "source", "data"}:
                raise ValueError("DynamicCallFact must use the fixed top-level schema")
            return DynamicCallFact(
                kind=kind,
                source=data.get("source"),
                data=data.get("data"),
            )
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


_DYNAMIC_REFERENCE_RE = re.compile(r"^[0-9a-f]{16,}$")
_DYNAMIC_CDEF_REFERENCE_RE = re.compile(r"^cdef-v[1-9][0-9]*-[0-9a-f]{16,}$")
_ANNOTATION_NAMESPACE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_ANNOTATION_SOURCE_KINDS = {
    "decorator", "provider", "override", "stored_record", "cached_probe", "synthetic",
}
_ANNOTATION_TARGET_KINDS = {"function", "method", "class", "provider", "operation_kind", "synthetic"}
_REQUIREMENT_TRACE_FIELDS = {
    "namespace", "kind", "label", "target_label", "module", "qualname",
    "priority", "merge_policy", "fragment_index", "data",
}
_REQUIREMENT_RESOLUTION_FIELDS = {
    "environment_requirement", "world_requirement", "runtime_requirement",
    "environment_default", "world_default", "runtime_default", "fragments",
    "source_traces", "diagnostics", "merge_report",
}
_METHOD_CONTRACT_FIELDS = {
    "method_contract_detected", "class_module", "class_qualname", "trait_impls",
    "has_user_call",
}
_SHAPE_FACT_FIELDS = {"input_handles", "output_handles"}
_BACKEND_VALUES = {"numpy", "tf", "torch", "jax"}
_BATCH_MODE_VALUES = {"batched", "element"}


@dataclass(frozen=True, slots=True)
class DynamicCallFact(CodeFact):
    """One bounded method call directly observed by ``dryml.code.trace``.

    Receiver references are hash-derived observation keys, not equality,
    authorization, merge, or dispatch keys. Exact equality requires a live
    structural comparison or a separately equality-verified registry.
    ``method_facts`` contains only serialized facts from the current annotation
    and method-contract APIs; this object never retains live trace values.
    """

    kind: str = "dynamic_call"

    def __post_init__(self) -> None:
        """Validate the complete fixed dynamic-call wire schema."""

        if self.kind != "dynamic_call":
            raise ValueError("DynamicCallFact kind must be 'dynamic_call'")
        if type(self.source) is not dict or set(self.source) != {"analyzer", "target_kind"}:
            raise ValueError("DynamicCallFact source must use the fixed schema")
        if self.source.get("analyzer") != "dynamic_trace":
            raise ValueError("DynamicCallFact source analyzer must be 'dynamic_trace'")
        target_kind = self.source.get("target_kind")
        if not isinstance(target_kind, str) or not target_kind or len(target_kind) > 4_096:
            raise ValueError("DynamicCallFact target_kind must be a bounded non-empty string")

        required = {
            "sequence", "receiver_kind", "receiver_ref", "receiver_class",
            "method_name", "args", "kwargs", "method_facts",
        }
        if type(self.data) is not dict or set(self.data) != required:
            raise ValueError("DynamicCallFact data must use the fixed schema")
        sequence = self.data["sequence"]
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("DynamicCallFact sequence must be a non-negative integer")
        receiver_kind = self.data["receiver_kind"]
        if receiver_kind not in {"definition", "concrete_definition"}:
            raise ValueError("DynamicCallFact receiver_kind is unsupported")
        _validate_dynamic_reference(receiver_kind, self.data["receiver_ref"], field="receiver_ref")
        receiver_class = self.data["receiver_class"]
        if receiver_class is not None and not _valid_bounded_import_path(receiver_class):
            raise ValueError("DynamicCallFact receiver_class must be a bounded import path or null")
        method_name = self.data["method_name"]
        if (
            not isinstance(method_name, str)
            or not method_name.isidentifier()
            or method_name.startswith("__")
            or method_name.endswith("__")
            or len(method_name) > 512
        ):
            raise ValueError("DynamicCallFact method_name must be a bounded non-dunder identifier")

        args = self.data["args"]
        kwargs = self.data["kwargs"]
        if not isinstance(args, list):
            raise ValueError("DynamicCallFact args must be a JSON array")
        if not isinstance(kwargs, dict) or any(type(key) is not str for key in kwargs):
            raise ValueError("DynamicCallFact kwargs must be a string-key JSON object")
        counter = [0]
        _validate_dynamic_value(args, depth=0, active=set(), counter=counter)
        _validate_dynamic_value(kwargs, depth=0, active=set(), counter=counter)

        method_facts = self.data["method_facts"]
        if not isinstance(method_facts, list) or len(method_facts) > 256:
            raise ValueError("DynamicCallFact method_facts must be a bounded JSON array")
        normalized_method_facts = []
        for fact_data in method_facts:
            _validate_dynamic_method_fact_wire(fact_data)
            normalized_method_facts.append(CodeFact.from_data(fact_data).to_data())

        normalized_data = dict(self.data)
        normalized_data["method_facts"] = normalized_method_facts
        object.__setattr__(self, "data", normalized_data)
        CodeFact.__post_init__(self)
        encoded = json.dumps(self.to_data(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(encoded) > 1_048_576:
            raise ValueError("DynamicCallFact exceeds the serialized byte limit")


def _validate_dynamic_reference(kind: str, value: Any, *, field: str) -> None:
    if not isinstance(value, str) or not value or len(value) > 4_096:
        raise ValueError(f"DynamicCallFact {field} must be a bounded reference")
    pattern = _DYNAMIC_REFERENCE_RE if kind == "definition" else _DYNAMIC_CDEF_REFERENCE_RE
    if pattern.fullmatch(value) is None:
        raise ValueError(f"DynamicCallFact {field} does not match receiver_kind")


def _valid_bounded_import_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or len(value) > 4_096 or value.count(":") != 1:
        return False
    module, qualname = value.split(":", 1)
    return bool(module and qualname and all(part.isidentifier() for part in module.split(".")) and all(part.isidentifier() for part in qualname.split(".")))


def _validate_dynamic_value(value: Any, *, depth: int, active: set[int], counter: list[int]) -> None:
    if depth > 32:
        raise ValueError("DynamicCallFact value depth exceeds 32")
    if value is None or type(value) is bool:
        return
    if type(value) is int:
        if value.bit_length() > 4_096:
            raise ValueError("DynamicCallFact integer exceeds 4096 bits")
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("DynamicCallFact floats must be finite")
        return
    if type(value) is str:
        if len(value) > 4_096:
            raise ValueError("DynamicCallFact strings must be bounded")
        return
    if type(value) not in {list, dict}:
        raise ValueError("DynamicCallFact values must use the trace JSON grammar")
    oid = id(value)
    if oid in active:
        raise ValueError("DynamicCallFact values must be acyclic")
    active.add(oid)
    try:
        counter[0] += len(value)
        if counter[0] > 10_000:
            raise ValueError("DynamicCallFact value entries exceed 10000")
        if type(value) is list:
            for child in value:
                _validate_dynamic_value(child, depth=depth + 1, active=active, counter=counter)
            return
        if set(value) == {"definition_kind", "definition_ref"}:
            kind = value["definition_kind"]
            if kind not in {"definition", "concrete_definition"}:
                raise ValueError("DynamicCallFact nested definition_kind is unsupported")
            _validate_dynamic_reference(kind, value["definition_ref"], field="definition_ref")
            return
        if any(type(key) is not str or len(key) > 4_096 for key in value):
            raise ValueError("DynamicCallFact mapping keys must be bounded strings")
        for child in value.values():
            _validate_dynamic_value(child, depth=depth + 1, active=active, counter=counter)
    finally:
        active.remove(oid)


def _validate_plain_json(value: Any, *, depth: int, active: set[int], counter: list[int]) -> None:
    """Validate nested method-fact input before generic JSON conversion."""

    if depth > 32:
        raise ValueError("method fact JSON depth exceeds 32")
    if value is None or type(value) is bool:
        return
    if type(value) is int:
        if value.bit_length() > 4_096:
            raise ValueError("method fact integer exceeds 4096 bits")
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("method fact float must be finite")
        return
    if type(value) is str:
        if len(value) > 4_096:
            raise ValueError("method fact string is too long")
        return
    if type(value) not in {list, dict}:
        raise ValueError("method fact contains a non-JSON value")
    oid = id(value)
    if oid in active:
        raise ValueError("method fact JSON must be acyclic")
    active.add(oid)
    try:
        counter[0] += len(value)
        if counter[0] > 10_000:
            raise ValueError("method fact JSON contains too many entries")
        children = value
        if type(value) is dict:
            if any(type(key) is not str or len(key) > 4_096 for key in value):
                raise ValueError("method fact JSON keys must be bounded strings")
            children = value.values()
        for child in children:
            _validate_plain_json(child, depth=depth + 1, active=active, counter=counter)
    finally:
        active.remove(oid)


def _validate_dynamic_method_fact_wire(value: Any) -> None:
    """Validate one nested method fact before typed restoration.

    ``DynamicCallFact`` accepts only the standard wire forms emitted by the
    annotation and method-contract analyzers.  Validate their exact top-level
    schemas before ``CodeFact.from_data`` can apply its permissive legacy
    defaults or recursively normalize arbitrary mappings.
    """

    _validate_plain_json(value, depth=0, active=set(), counter=[0])
    if type(value) is not dict:
        raise ValueError("DynamicCallFact method_facts entries must be exact dicts")

    kind = value.get("kind")
    expected = {"kind", "source", "data"}
    if kind == "requirement":
        expected.update({
            "namespace",
            "requirement_kind",
            "fragment",
            "priority",
            "merge_policy",
        })
    elif kind not in {"annotation", "method_contract", "shape"}:
        raise ValueError("DynamicCallFact method_facts contains an unsupported fact")
    if set(value) != expected:
        raise ValueError("DynamicCallFact method fact must use its fixed wire schema")
    if type(value["source"]) is not dict or type(value["data"]) is not dict:
        raise ValueError("DynamicCallFact method fact source and data must be exact dicts")

    if kind in {"annotation", "requirement"}:
        _validate_annotation_method_fact_wire(value)
    elif kind == "method_contract":
        _validate_method_contract_fact_wire(value)
    else:
        _validate_shape_fact_wire(value)


def _validate_annotation_namespace(value: Any) -> None:
    if type(value) is not str or _ANNOTATION_NAMESPACE_RE.fullmatch(value) is None:
        raise ValueError("DynamicCallFact annotation namespace is invalid")


def _validate_annotation_fragment_wire(value: Any) -> None:
    """Validate the exact AnnotationFragment.to_data() form used in method facts."""

    required = {"namespace", "kind", "fragment", "source", "priority", "merge_policy", "schema_version"}
    if type(value) is not dict or set(value) != required:
        raise ValueError("DynamicCallFact annotation fragment must use the standard wire schema")
    _validate_annotation_namespace(value["namespace"])
    if value["kind"] not in {"requirement", "default"}:
        raise ValueError("DynamicCallFact annotation fragment kind is invalid")
    if type(value["fragment"]) is not dict:
        raise ValueError("DynamicCallFact annotation fragment payload must be an exact dict")
    if isinstance(value["priority"], bool) or not isinstance(value["priority"], int):
        raise ValueError("DynamicCallFact annotation fragment priority is invalid")
    if value["merge_policy"] is not None and type(value["merge_policy"]) is not str:
        raise ValueError("DynamicCallFact annotation fragment merge_policy is invalid")
    if value["schema_version"] != 1:
        raise ValueError("DynamicCallFact annotation fragment schema version is invalid")
    _validate_annotation_source_trace_wire(value["source"])
    if value["source"]["namespace"] not in {None, value["namespace"]}:
        raise ValueError("DynamicCallFact annotation source namespace is inconsistent")


def _validate_annotation_source_trace_wire(value: Any) -> None:
    required = {"kind", "target", "label", "namespace", "path", "metadata"}
    if type(value) is not dict or set(value) != required:
        raise ValueError("DynamicCallFact annotation source must use the standard wire schema")
    if value["kind"] not in _ANNOTATION_SOURCE_KINDS:
        raise ValueError("DynamicCallFact annotation source kind is invalid")
    for field in ("label", "path"):
        if value[field] is not None and type(value[field]) is not str:
            raise ValueError(f"DynamicCallFact annotation source {field} is invalid")
    if value["namespace"] is not None:
        _validate_annotation_namespace(value["namespace"])
    if type(value["metadata"]) is not dict:
        raise ValueError("DynamicCallFact annotation source metadata must be an exact dict")
    target = value["target"]
    if target is None:
        return
    target_required = {"kind", "module", "qualname", "owner_module", "owner_qualname", "metadata"}
    if type(target) is not dict or set(target) != target_required:
        raise ValueError("DynamicCallFact annotation target must use the standard wire schema")
    if target["kind"] not in _ANNOTATION_TARGET_KINDS:
        raise ValueError("DynamicCallFact annotation target kind is invalid")
    if any(target[field] is not None and type(target[field]) is not str for field in ("module", "qualname", "owner_module", "owner_qualname")):
        raise ValueError("DynamicCallFact annotation target fields are invalid")
    if type(target["metadata"]) is not dict:
        raise ValueError("DynamicCallFact annotation target metadata must be an exact dict")


def _validate_direct_annotation_source(value: Any, annotation_source: Any) -> None:
    if type(value) is not dict or set(value) != {"analyzer", "target_kind", "annotation_source"}:
        raise ValueError("DynamicCallFact annotation method fact source is invalid")
    if value["analyzer"] != "direct_annotations":
        raise ValueError("DynamicCallFact annotation method fact analyzer is invalid")
    if type(value["target_kind"]) is not str or not value["target_kind"]:
        raise ValueError("DynamicCallFact annotation method fact target kind is invalid")
    _validate_annotation_source_trace_wire(value["annotation_source"])
    if value["annotation_source"] != annotation_source:
        raise ValueError("DynamicCallFact annotation method fact source does not match its fragment")


def _validate_annotation_method_fact_wire(value: dict[str, Any]) -> None:
    if value["kind"] == "annotation":
        _validate_annotation_fragment_wire(value["data"])
        _validate_direct_annotation_source(value["source"], value["data"]["source"])
        return

    _validate_annotation_namespace(value["namespace"])
    if value["requirement_kind"] not in {"requirement", "default"}:
        raise ValueError("DynamicCallFact requirement method fact kind is invalid")
    if type(value["fragment"]) is not dict:
        raise ValueError("DynamicCallFact requirement method fact fragment must be an exact dict")
    if isinstance(value["priority"], bool) or not isinstance(value["priority"], int):
        raise ValueError("DynamicCallFact requirement method fact priority is invalid")
    if value["merge_policy"] is not None and type(value["merge_policy"]) is not str:
        raise ValueError("DynamicCallFact requirement method fact merge_policy is invalid")
    required_data = {"annotation", "source_trace", "resolution"}
    if set(value["data"]) != required_data:
        raise ValueError("DynamicCallFact requirement method fact data is invalid")
    annotation = value["data"]["annotation"]
    _validate_annotation_fragment_wire(annotation)
    if (
        annotation["namespace"] != value["namespace"]
        or annotation["kind"] != value["requirement_kind"]
        or annotation["fragment"] != value["fragment"]
        or annotation["priority"] != value["priority"]
        or annotation["merge_policy"] != value["merge_policy"]
    ):
        raise ValueError("DynamicCallFact requirement method fact does not match its annotation")
    _validate_direct_annotation_source(value["source"], annotation["source"])
    source_trace = value["data"]["source_trace"]
    resolution = value["data"]["resolution"]
    _validate_requirement_source_trace_wire(source_trace, annotation=annotation)
    _validate_requirement_resolution_wire(
        resolution,
        annotation=annotation,
        source_trace=source_trace,
    )


def _validate_requirement_source_trace_wire(value: Any, *, annotation: dict[str, Any]) -> None:
    """Validate one exact ``RequirementSourceTrace.to_data`` mapping."""

    if type(value) is not dict or set(value) != _REQUIREMENT_TRACE_FIELDS:
        raise ValueError("DynamicCallFact requirement source trace is invalid")
    _validate_annotation_namespace(value["namespace"])
    if value["kind"] not in {"requirement", "default"}:
        raise ValueError("DynamicCallFact requirement source trace kind is invalid")
    if type(value["label"]) is not str:
        raise ValueError("DynamicCallFact requirement source trace label is invalid")
    for field in ("target_label", "module", "qualname", "merge_policy"):
        if value[field] is not None and type(value[field]) is not str:
            raise ValueError(f"DynamicCallFact requirement source trace {field} is invalid")
    if isinstance(value["priority"], bool) or not isinstance(value["priority"], int):
        raise ValueError("DynamicCallFact requirement source trace priority is invalid")
    if (
        isinstance(value["fragment_index"], bool)
        or not isinstance(value["fragment_index"], int)
        or value["fragment_index"] < 0
    ):
        raise ValueError("DynamicCallFact requirement source trace fragment index is invalid")
    data = value["data"]
    if type(data) is not dict or set(data) not in ({"source"}, {"source", "resolution_source"}):
        raise ValueError("DynamicCallFact requirement source trace data is invalid")
    _validate_annotation_source_trace_wire(data["source"])
    if "resolution_source" in data and type(data["resolution_source"]) is not str:
        raise ValueError("DynamicCallFact requirement source trace resolution source is invalid")
    if (
        value["namespace"] != annotation["namespace"]
        or value["kind"] != annotation["kind"]
        or value["priority"] != annotation["priority"]
        or value["merge_policy"] != annotation["merge_policy"]
        or data["source"] != annotation["source"]
    ):
        raise ValueError("DynamicCallFact requirement source trace does not match its annotation")


def _validate_requirement_resolution_wire(
    value: Any,
    *,
    annotation: dict[str, Any],
    source_trace: dict[str, Any],
) -> None:
    """Validate the serialized ``RequirementResolution`` used by one fact."""

    if type(value) is not dict or set(value) != _REQUIREMENT_RESOLUTION_FIELDS:
        raise ValueError("DynamicCallFact requirement resolution is invalid")
    fragments = value["fragments"]
    traces = value["source_traces"]
    diagnostics = value["diagnostics"]
    if type(fragments) is not list or type(traces) is not list or len(fragments) != len(traces):
        raise ValueError("DynamicCallFact requirement resolution fragments are invalid")
    if type(diagnostics) is not list:
        raise ValueError("DynamicCallFact requirement resolution diagnostics are invalid")
    for fragment in fragments:
        _validate_annotation_fragment_wire(fragment)
    for trace, fragment in zip(traces, fragments, strict=True):
        _validate_requirement_source_trace_wire(trace, annotation=fragment)
    index = source_trace["fragment_index"]
    if index >= len(fragments) or fragments[index] != annotation or traces[index] != source_trace:
        raise ValueError("DynamicCallFact requirement resolution does not match its requirement")
    for diagnostic in diagnostics:
        _validate_requirement_diagnostic_wire(diagnostic)
    _validate_requirement_merge_report_wire(value["merge_report"])


def _validate_requirement_diagnostic_wire(value: Any) -> None:
    required = {"level", "code", "message", "target_label", "data"}
    if type(value) is not dict or set(value) != required:
        raise ValueError("DynamicCallFact requirement resolution diagnostic is invalid")
    if value["level"] not in {"debug", "info", "warning", "error"}:
        raise ValueError("DynamicCallFact requirement resolution diagnostic level is invalid")
    if type(value["code"]) is not str or type(value["message"]) is not str:
        raise ValueError("DynamicCallFact requirement resolution diagnostic fields are invalid")
    if value["target_label"] is not None and type(value["target_label"]) is not str:
        raise ValueError("DynamicCallFact requirement resolution diagnostic target is invalid")
    if type(value["data"]) is not dict:
        raise ValueError("DynamicCallFact requirement resolution diagnostic data is invalid")


def _validate_requirement_merge_report_wire(value: Any) -> None:
    if value is None:
        return
    if type(value) is not dict or set(value) != {"ok", "issues"}:
        raise ValueError("DynamicCallFact requirement merge report is invalid")
    if type(value["ok"]) is not bool or type(value["issues"]) is not list:
        raise ValueError("DynamicCallFact requirement merge report fields are invalid")
    required_issue = {"severity", "namespace", "path", "message", "expected", "actual", "sources"}
    for issue in value["issues"]:
        if type(issue) is not dict or set(issue) != required_issue:
            raise ValueError("DynamicCallFact requirement merge issue is invalid")
        if issue["severity"] not in {"debug", "info", "warning", "error"}:
            raise ValueError("DynamicCallFact requirement merge issue severity is invalid")
        _validate_annotation_namespace(issue["namespace"])
        for field in ("path", "message"):
            if issue[field] is not None and type(issue[field]) is not str:
                raise ValueError("DynamicCallFact requirement merge issue fields are invalid")
        if type(issue["sources"]) is not list:
            raise ValueError("DynamicCallFact requirement merge issue sources are invalid")
        for source in issue["sources"]:
            _validate_annotation_source_trace_wire(source)


def _validate_method_contract_source(value: Any) -> None:
    if type(value) is not dict or set(value) != {"analyzer", "target_kind"}:
        raise ValueError("DynamicCallFact method-contract source is invalid")
    if value["analyzer"] != "method_contracts":
        raise ValueError("DynamicCallFact method-contract analyzer is invalid")
    if type(value["target_kind"]) is not str or not value["target_kind"]:
        raise ValueError("DynamicCallFact method-contract target kind is invalid")


def _validate_method_contract_fact_wire(value: dict[str, Any]) -> None:
    _validate_method_contract_source(value["source"])
    if set(value["data"]) != _METHOD_CONTRACT_FIELDS:
        raise ValueError("DynamicCallFact method-contract data is invalid")
    if value["data"]["method_contract_detected"] is not True or type(value["data"]["has_user_call"]) is not bool:
        raise ValueError("DynamicCallFact method-contract flags are invalid")
    if any(value["data"][field] is not None and type(value["data"][field]) is not str for field in ("class_module", "class_qualname")):
        raise ValueError("DynamicCallFact method-contract class fields are invalid")
    if type(value["data"]["trait_impls"]) is not list:
        raise ValueError("DynamicCallFact method-contract trait implementations are invalid")
    for implementation in value["data"]["trait_impls"]:
        if type(implementation) is not dict or set(implementation) != {"name", "traits"} or not _valid_bounded_nonempty_string(implementation["name"]):
            raise ValueError("DynamicCallFact method-contract trait implementation is invalid")
        traits = implementation["traits"]
        if type(traits) is not dict or set(traits) != {"backend", "batch_mode"}:
            raise ValueError("DynamicCallFact method-contract traits are invalid")
        if traits["backend"] is not None and traits["backend"] not in _BACKEND_VALUES:
            raise ValueError("DynamicCallFact method-contract backend is invalid")
        if traits["batch_mode"] is not None and traits["batch_mode"] not in _BATCH_MODE_VALUES:
            raise ValueError("DynamicCallFact method-contract trait values are invalid")


def _validate_shape_fact_wire(value: dict[str, Any]) -> None:
    """Validate the narrow method-contract ShapeFact wire form used by traces."""

    _validate_method_contract_source(value["source"])
    if type(value["data"]) is not dict or set(value["data"]) != _SHAPE_FACT_FIELDS:
        raise ValueError("DynamicCallFact shape data is invalid")
    if type(value["data"]["input_handles"]) is not list or type(value["data"]["output_handles"]) is not list:
        raise ValueError("DynamicCallFact shape handles are invalid")


def _valid_bounded_nonempty_string(value: Any) -> bool:
    return type(value) is str and bool(value) and len(value) <= 4_096


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
    "DynamicCallFact",
    "MethodContractFact",
    "RequirementFact",
    "ShapeFact",
    "SourceFact",
    "StaticCallFact",
    "SymbolFact",
    "json_compatible",
]
