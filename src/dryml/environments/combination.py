"""Diagnostic-first combination of passive environment declarations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import re

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

from dryml.requirements import RequirementDeclaration, RequirementIssue, RequirementReport, RequirementResult, RequirementSource, combine_requirements
from dryml.requirements.collection import collect_declarations

from .declarations import ENVIRONMENT_REQUIREMENT_KEY
from .errors import EnvironmentRequirementError
from .requirements import (
    EnvironmentRequirement,
    _intersect_specifiers,
    _markers_proven_disjoint,
    _merge_package_requirements,
    _specifier_has_obvious_conflict,
)
from .utils import normalize_distribution_name

_MAX_DECLARATIONS = 64
_MAX_PATHS = 1024
_MAX_OCCURRENCES = 4096
_MAX_BYTES = 1024 * 1024
_SENSITIVE_SEGMENT = re.compile(r"(?i)(password|passwd|secret|token|api[_-]?key|credential)")


def _hard_package_valid(requirement: str) -> bool:
    """Return whether one normalized package requirement is verifiable from records."""

    parsed = Requirement(requirement)
    return not parsed.extras and not parsed.url and not (parsed.marker is not None and "extra" in str(parsed.marker))


def _paths(value: EnvironmentRequirement) -> tuple[str, ...]:
    """Return one value's canonical semantic paths after strict validation."""

    if type(value) is not EnvironmentRequirement:
        raise EnvironmentRequirementError("environment declaration value is invalid")
    fields = (value.requirements, value.excludes, value.capabilities, value.tags)
    if any(len(field) > _MAX_DECLARATIONS for field in fields) or len(value.schema_versions) > _MAX_DECLARATIONS:
        raise EnvironmentRequirementError("environment declaration exceeds entry limit")
    paths: list[str] = []
    if value.python:
        paths.append("python")
    for text in value.requirements:
        if type(text) is not str or len(text) > 4096 or not _hard_package_valid(text):
            raise EnvironmentRequirementError("environment declaration package requirement is unsupported")
        paths.append(f"requirements.{normalize_distribution_name(Requirement(text).name)}")
    for field, entries in (("excludes", value.excludes), ("capabilities", value.capabilities), ("tags", value.tags)):
        for entry in entries:
            if type(entry) is not str or len(entry) > 4096:
                raise EnvironmentRequirementError("environment declaration text is invalid")
            paths.append(f"{field}.{entry}")
    if value.dryml_protocol:
        paths.append("dryml_protocol")
    for key, specifier in value.schema_versions.items():
        if type(key) is not str or type(specifier) is not str or len(key) > 4096 or len(specifier) > 4096:
            raise EnvironmentRequirementError("environment declaration schema version is invalid")
        paths.append(f"schema_versions.{key}")
    return tuple(paths)


def _preflight(declarations: tuple[RequirementDeclaration[EnvironmentRequirement], ...]) -> None:
    """Validate full diagnostic work before semantic combination can begin."""

    if len(declarations) > _MAX_DECLARATIONS:
        raise EnvironmentRequirementError("environment declaration limit exceeded")
    paths = tuple(path for declaration in declarations for path in _paths(declaration.value))
    if len(set(paths)) > _MAX_PATHS or len(paths) > _MAX_OCCURRENCES:
        raise EnvironmentRequirementError("environment requirement combination exceeds path capacity")
    text_work = sum(len(path.encode("utf-8")) for path in paths)
    for declaration in declarations:
        value = declaration.value
        text_work += sum(
            len(text.encode("utf-8"))
            for text in (
                *(item for item in (value.python, value.dryml_protocol) if item),
                *value.requirements,
                *value.excludes,
                *value.capabilities,
                *value.tags,
                *(item for pair in value.schema_versions.items() for item in pair),
                declaration.source.label,
                *(item for item in (declaration.source.module, declaration.source.qualname) if item),
            )
        )
    if text_work > _MAX_BYTES:
        raise EnvironmentRequirementError("environment requirement combination exceeds byte capacity")


def _sources_for(declarations: tuple[RequirementDeclaration[EnvironmentRequirement], ...], path: str) -> tuple:
    """Return every ordinalized declaration source contributing to one path."""

    field, _, name = path.partition(".")
    paths = {path}
    if field == "requirements" and name:
        paths.add(f"excludes.{name}")
    return tuple(declaration.source for declaration in declarations if paths & set(_paths(declaration.value)))


def _safe_path(path: str, ordinal: int) -> str:
    """Project caller-derived path segments without collapsing sensitive names."""

    field, _, segment = path.partition(".")
    if not segment or not (_SENSITIVE_SEGMENT.search(segment) or "/" in segment or "\\" in segment):
        return path
    return f"{field}.<redacted-{ordinal}>"


def _conflicts(values: tuple[EnvironmentRequirement, ...]) -> tuple[str, ...]:
    """Return every raw canonical path whose hard constraints conflict."""

    conflicts: set[str] = set()
    for name, constraints in (("python", tuple(value.python for value in values if value.python)), ("dryml_protocol", tuple(value.dryml_protocol for value in values if value.dryml_protocol))):
        if constraints and _specifier_has_obvious_conflict(SpecifierSet(",".join(constraints))):
            conflicts.add(name)

    schema_names = sorted({name for value in values for name in value.schema_versions})
    for name in schema_names:
        constraints = tuple(value.schema_versions[name] for value in values if name in value.schema_versions)
        if _specifier_has_obvious_conflict(SpecifierSet(",".join(constraints))):
            conflicts.add(f"schema_versions.{name}")

    requirements_by_name: dict[str, list[Requirement]] = defaultdict(list)
    excludes: set[str] = set()
    for value in values:
        excludes.update(value.excludes)
        for text in value.requirements:
            parsed = Requirement(text)
            requirements_by_name[normalize_distribution_name(parsed.name)].append(parsed)
    for name, requirements in requirements_by_name.items():
        if name in excludes:
            conflicts.add(f"requirements.{name}")
            continue
        groups: dict[str | None, list[Requirement]] = defaultdict(list)
        for requirement in requirements:
            groups[None if requirement.marker is None else str(requirement.marker)].append(requirement)
        combined = {
            marker: SpecifierSet(",".join(str(item.specifier) for item in group if str(item.specifier)))
            for marker, group in groups.items()
        }
        if any(_specifier_has_obvious_conflict(specifier) for specifier in combined.values()):
            conflicts.add(f"requirements.{name}")
        marker_items = tuple(sorted(combined.items(), key=lambda item: "" if item[0] is None else item[0]))
        for index, (marker, specifier) in enumerate(marker_items):
            for other_marker, other_specifier in marker_items[index + 1:]:
                if _markers_proven_disjoint(marker, other_marker):
                    continue
                if _specifier_has_obvious_conflict(SpecifierSet(",".join(filter(None, (str(specifier), str(other_specifier)))))):
                    conflicts.add(f"requirements.{name}")
    return tuple(sorted(conflicts))


def _combined_value(values: tuple[EnvironmentRequirement, ...], *, details: dict | None = None) -> EnvironmentRequirement:
    """Build one already-proven-compatible environment requirement intersection."""

    python = None
    protocol = None
    schema: dict[str, str] = {}
    requirements: tuple[str, ...] = ()
    for value in values:
        python = _intersect_specifiers(python, value.python, path="python")
        protocol = _intersect_specifiers(protocol, value.dryml_protocol, path="dryml_protocol")
        requirements = _merge_package_requirements(requirements, value.requirements)
        for name, specifier in value.schema_versions.items():
            schema[name] = _intersect_specifiers(schema.get(name), specifier, path=f"schema_versions.{name}") or ""
    return EnvironmentRequirement(
        python=python,
        requirements=requirements,
        excludes=tuple(sorted({item for value in values for item in value.excludes})),
        capabilities=tuple(sorted({item for value in values for item in value.capabilities})),
        tags=tuple(sorted({item for value in values for item in value.tags})),
        dryml_protocol=protocol,
        schema_versions=schema,
        details={} if details is None else details,
    )


class _EnvironmentCombiner:
    """Domain-owned shared-combiner adapter for environment requirements."""

    def combine(self, declarations: tuple[RequirementDeclaration[EnvironmentRequirement], ...]) -> RequirementResult[EnvironmentRequirement]:
        """Return a complete conflict report or one compatible requirement value."""

        _preflight(declarations)
        values = tuple(declaration.value for declaration in declarations)
        conflicts = _conflicts(values)
        if conflicts:
            issues = tuple(
                RequirementIssue(
                    "dryml.environments.requirement_conflict",
                    "conflicting environment requirement constraints",
                    path=_safe_path(path, ordinal),
                    sources=_sources_for(declarations, path),
                )
                for ordinal, path in enumerate(conflicts, start=1)
            )
            return RequirementResult(report=RequirementReport(issues))
        return RequirementResult(_combined_value(values, details={"sources": tuple(item.source.label for item in declarations)}))


def requirements_for(target: object) -> RequirementResult[EnvironmentRequirement]:
    """Combine passive environment declarations directly attached to a target.

    Args:
        target: A class, function, or supported descriptor inspected statically.

    Returns:
        An empty, valued, or conflict result for only environment declarations.

    Raises:
        EnvironmentRequirementError: If attached metadata or declaration values
            are malformed or exceed environment combination bounds.

    Side Effects:
        None. The target is not bound, invoked, or otherwise modified.
    """

    try:
        declarations = collect_declarations(target, key=ENVIRONMENT_REQUIREMENT_KEY, value_type=EnvironmentRequirement)
        return combine_requirements(declarations, combiner=_EnvironmentCombiner())
    except Exception:
        raise EnvironmentRequirementError("environment requirement collection or combination failed") from None


def requirements_for_method(owner: type | object, method_name: str) -> RequirementResult[EnvironmentRequirement]:
    """Combine a class's declarations with one statically selected method.

    Args:
        owner: A class or instance whose exact class supplies the selected method.
        method_name: Exact method name resolved through static MRO inspection.

    Returns:
        An empty, valued, or conflict environment requirement result.

    Raises:
        EnvironmentRequirementError: If static selection, metadata, or values are
            malformed or exceed combination bounds.

    Side Effects:
        None. Instance state and dynamic attribute hooks are never consulted.
    """

    try:
        declarations = collect_declarations(owner, key=ENVIRONMENT_REQUIREMENT_KEY, value_type=EnvironmentRequirement, method_name=method_name)
        return combine_requirements(declarations, combiner=_EnvironmentCombiner())
    except Exception:
        raise EnvironmentRequirementError("environment method requirement collection or combination failed") from None


def merge_environment_requirements(left: EnvironmentRequirement, right: EnvironmentRequirement, *, sources: tuple[str, ...] = ()) -> EnvironmentRequirement:
    """Merge two environment values while retaining the legacy exception surface."""

    try:
        _preflight((RequirementDeclaration(left, source=RequirementSource("left")), RequirementDeclaration(right, source=RequirementSource("right"))))
        conflicts = _conflicts((left, right))
        if conflicts:
            raise EnvironmentRequirementError("conflicting environment requirement constraints", context={"path": conflicts[0]})
        details_sources = tuple(left.details.get("sources", ())) + tuple(right.details.get("sources", ())) + tuple(sources)
        return _combined_value((left, right), details={"sources": details_sources} if details_sources else {})
    except EnvironmentRequirementError:
        raise
    except Exception:
        raise EnvironmentRequirementError("environment requirement merge failed") from None


__all__ = ["requirements_for", "requirements_for_method"]
