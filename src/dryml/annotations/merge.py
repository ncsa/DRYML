"""Pure deterministic resolution for explicit annotation fragments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from dryml.environments import EnvironmentRequirement
from dryml.environments.specs import spec_from_data
from dryml.formats import deep_freeze_json, json_ready
from dryml.worlds import WorldRequirement, WorldSpec

from .collect import collect_fragments
from .model import AnnotationFragment, SourceTrace, UnresolvedAnnotationResult
from .namespaces import ENVIRONMENT, RUNTIME, WORLD


@dataclass(frozen=True, slots=True)
class RequirementDiagnostic:
    """Immutable diagnostic emitted during declaration-only resolution.

    Args:
        level: Diagnostic severity.
        code: Stable machine-oriented category.
        message: Human-readable explanation.
        sources: Relevant declaration sources in application order.
    """

    level: Literal["error", "warning", "info"]
    code: str
    message: str
    sources: tuple[SourceTrace, ...] = ()

    def __post_init__(self) -> None:
        """Validate severity and freeze the ordered source tuple."""

        if self.level not in {"error", "warning", "info"}:
            raise ValueError("requirement diagnostic level is invalid")
        object.__setattr__(self, "sources", tuple(self.sources))


@dataclass(frozen=True, slots=True)
class RequirementResolution:
    """Inspectable immutable result of explicit annotation resolution.

    The result only records declarations and diagnostics. It does not activate a
    runtime, inspect an environment, probe frameworks, or mutate session state.

    Args:
        environment_requirement: Merged hard environment requirement.
        world_requirement: Merged hard world requirement.
        runtime_requirement: Merged hard runtime mapping.
        environment_default: Merged environment selector default.
        world_default: Merged world default mapping.
        runtime_default: Merged runtime default mapping.
        fragments: Ordered raw declarations, including caller overrides.
        source_traces: Ordered source traces for the declarations.
        diagnostics: Resolution diagnostics.
    """

    environment_requirement: EnvironmentRequirement | None = None
    world_requirement: WorldRequirement | None = None
    runtime_requirement: Mapping[str, Any] | None = None
    environment_default: Any | None = None
    world_default: WorldSpec | None = None
    runtime_default: Mapping[str, Any] | None = None
    fragments: tuple[AnnotationFragment, ...] = ()
    source_traces: tuple[SourceTrace, ...] = ()
    diagnostics: tuple[RequirementDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fragments", tuple(self.fragments))
        object.__setattr__(self, "source_traces", tuple(self.source_traces))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        for name in ("runtime_requirement", "runtime_default"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, deep_freeze_json(value))

    @property
    def usable(self) -> bool:
        """Return false whenever a declaration produced an error diagnostic."""

        return all(diagnostic.level != "error" for diagnostic in self.diagnostics)

    def to_data(self) -> dict[str, Any]:
        """Return detached JSON-ready inspection data without applying it."""

        return {"environment_requirement": _data(self.environment_requirement), "world_requirement": _data(self.world_requirement), "runtime_requirement": _data(self.runtime_requirement), "environment_default": _data(self.environment_default), "world_default": _data(self.world_default), "runtime_default": _data(self.runtime_default), "fragments": [fragment.to_data() for fragment in self.fragments], "source_traces": [source.to_display_data() for source in self.source_traces], "diagnostics": [{"level": item.level, "code": item.code, "message": item.message, "sources": [source.to_display_data() for source in item.sources]} for item in self.diagnostics], "usable": self.usable}


def resolve_fragments(fragments: Iterable[AnnotationFragment], *, namespace: str | None = None) -> RequirementResolution:
    """Resolve ordered declaration fragments with stable priority semantics.

    Args:
        fragments: Explicit fragments in collection order. Sets and frozensets
            are normalized by semantic ID because their input order is absent.
        namespace: Optional namespace filter.

    Returns:
        Immutable merged values, source traces, and diagnostics.
    """

    values = tuple(fragments)
    if isinstance(fragments, (set, frozenset)):
        values = tuple(sorted(values, key=lambda item: item.semantic_id))
    diagnostics: list[RequirementDiagnostic] = []
    valid = []
    for fragment in values:
        if not isinstance(fragment, AnnotationFragment):
            diagnostics.append(RequirementDiagnostic("error", "invalid_fragment", "annotation resolution requires AnnotationFragment values"))
        elif namespace is None or fragment.namespace == namespace:
            valid.append(fragment)
    ordered = tuple(sorted(valid, key=lambda item: item.priority))
    states: dict[tuple[str, str], tuple[Any, SourceTrace | None]] = {}
    for fragment in ordered:
        group = (fragment.namespace, fragment.kind)
        policy = fragment.merge_policy or "merge"
        current, current_source = states.get(group, (None, None))
        if policy == "clear":
            if dict(fragment.fragment):
                diagnostics.append(_issue("clear_requires_empty", "clear policy requires an empty fragment", fragment.source))
            else:
                states[group] = (None, fragment.source)
            continue
        try:
            value = _decode(fragment)
        except Exception as error:
            diagnostics.append(_issue("invalid_fragment", str(error), fragment.source))
            continue
        try:
            states[group] = (_apply(current, value, fragment, policy), fragment.source)
        except ValueError as error:
            sources = tuple(source for source in (current_source, fragment.source) if source is not None)
            diagnostics.append(RequirementDiagnostic("error", "merge_conflict", str(error), sources))
    def get(namespace_name: str, kind: str) -> Any:
        return states.get((namespace_name, kind), (None, None))[0]
    return RequirementResolution(
        environment_requirement=get(ENVIRONMENT, "requirement"),
        world_requirement=get(WORLD, "requirement"),
        runtime_requirement=get(RUNTIME, "requirement"),
        environment_default=get(ENVIRONMENT, "default"),
        world_default=get(WORLD, "default"),
        runtime_default=get(RUNTIME, "default"),
        fragments=ordered,
        source_traces=tuple(fragment.source for fragment in ordered),
        diagnostics=tuple(diagnostics),
    )


def resolve_target_requirements(target: Any, *, overrides: Iterable[AnnotationFragment] = ()) -> RequirementResolution:
    """Collect explicit declarations and append caller overrides before resolve.

    Args:
        target: Supplied live target, Definition, or ConcreteDefinition.
        overrides: Caller declarations appended after collected fragments.

    Returns:
        An immutable, data-only requirement resolution.
    """

    collected = collect_fragments(target)
    if isinstance(collected, UnresolvedAnnotationResult):
        return RequirementResolution(diagnostics=(RequirementDiagnostic("error", "unresolved_target", collected.reason),))
    ordered_overrides = tuple(sorted(overrides, key=lambda item: item.semantic_id)) if isinstance(overrides, (set, frozenset)) else tuple(overrides)
    return resolve_fragments((*collected, *ordered_overrides))


def _decode(fragment: AnnotationFragment) -> Any:
    payload = json_ready(fragment.fragment)
    if fragment.namespace == ENVIRONMENT and fragment.kind == "requirement":
        return EnvironmentRequirement.from_data(payload)
    if fragment.namespace == WORLD and fragment.kind == "requirement":
        return WorldRequirement.from_data(payload)
    if fragment.namespace == ENVIRONMENT and fragment.kind == "default":
        return spec_from_data(payload)
    if fragment.namespace == WORLD and fragment.kind == "default":
        return WorldSpec.from_data(payload)
    if not isinstance(payload, Mapping):
        raise ValueError("annotation fragment payload must be a mapping")
    return payload


def _apply(current: Any, value: Any, fragment: AnnotationFragment, policy: str) -> Any:
    if policy not in {"merge", "replace", "append", "error_on_conflict"}:
        raise ValueError(f"unknown annotation merge policy {policy!r}")
    if current is None:
        if policy == "append" and not _appendable(value):
            raise ValueError("append policy requires compatible sequence values")
        return value
    if policy == "replace":
        return value
    if policy == "error_on_conflict":
        if _empty(current) or current == value:
            return value if _empty(current) else current
        raise ValueError("error_on_conflict declarations differ")
    if policy == "append":
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            return _mapping_append(current, value)
        if type(current) is not type(value) or not _sequence_value(current):
            raise ValueError("append policy requires compatible sequence values")
        return current + value
    if isinstance(current, EnvironmentRequirement) and isinstance(value, EnvironmentRequirement):
        return current.merge(value)
    if isinstance(current, WorldRequirement) and isinstance(value, WorldRequirement):
        return current.merge(value)
    if isinstance(current, Mapping) and isinstance(value, Mapping):
        return _mapping_merge(current, value)
    if current == value:
        return current
    raise ValueError("merge declarations disagree on a scalar value")


def _mapping_merge(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(left)
    for key, value in right.items():
        if key not in result:
            result[key] = value
        elif isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = _mapping_merge(result[key], value)
        elif result[key] != value:
            raise ValueError(f"merge declarations disagree at {key!r}")
    return result


def _mapping_append(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Append only matching sequence leaves while preserving mapping shape."""

    result = dict(left)
    for key, value in right.items():
        if key not in result:
            result[key] = value
        elif isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = _mapping_append(result[key], value)
        elif type(result[key]) is type(value) and _sequence_value(result[key]):
            result[key] = result[key] + value
        else:
            raise ValueError(f"append policy requires compatible sequence values at {key!r}")
    return result


def _sequence_value(value: Any) -> bool:
    return isinstance(value, (tuple, list)) and not isinstance(value, (str, bytes, bytearray))


def _appendable(value: Any) -> bool:
    """Return whether every leaf in an initial append value is a sequence."""

    if isinstance(value, Mapping):
        return bool(value) and all(_appendable(item) for item in value.values())
    return _sequence_value(value)


def _empty(value: Any) -> bool:
    return value is None or (isinstance(value, Mapping) and not value) or (_sequence_value(value) and not value)


def _issue(code: str, message: str, source: SourceTrace) -> RequirementDiagnostic:
    return RequirementDiagnostic("error", code, message, (source,))


def _data(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_data"):
        return value.to_data()
    return json_ready(value)


__all__ = ["RequirementDiagnostic", "RequirementResolution", "resolve_fragments", "resolve_target_requirements"]
