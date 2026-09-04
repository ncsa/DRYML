"""Passive hard environment requirement declarations for live targets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Callable, TypeVar

from packaging.requirements import Requirement

from dryml.requirements import RequirementDeclaration, RequirementSource
from dryml.requirements.collection import attach_declaration

from .errors import EnvironmentRequirementError
from .requirements import EnvironmentRequirement

T = TypeVar("T")
ENVIRONMENT_REQUIREMENT_KEY = "dryml.environments.requirement"
_MAX_FIELD_ENTRIES = 64
_MAX_TEXT = 4096


def _bounded_strings(values: Iterable[str], *, field: str) -> tuple[str, ...]:
    """Consume one string iterable once while enforcing its declaration bound."""

    try:
        iterator = iter(values)
    except Exception:
        raise EnvironmentRequirementError("environment requirement field must be iterable") from None
    result: list[str] = []
    try:
        for _ in range(_MAX_FIELD_ENTRIES + 1):
            try:
                value = next(iterator)
            except StopIteration:
                break
            if type(value) is not str or len(value) > _MAX_TEXT:
                raise EnvironmentRequirementError("environment requirement text is invalid")
            result.append(value)
    except EnvironmentRequirementError:
        raise
    except Exception:
        raise EnvironmentRequirementError("environment requirement field is invalid") from None
    if len(result) > _MAX_FIELD_ENTRIES:
        raise EnvironmentRequirementError("environment requirement field exceeds entry limit", context={"field": field, "limit": _MAX_FIELD_ENTRIES})
    return tuple(result)


def _bounded_mapping(values: Mapping[str, str] | None, *, field: str) -> dict[str, str]:
    """Copy one string mapping after validating its finite declaration boundary."""

    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise EnvironmentRequirementError("environment requirement mapping is invalid")
    if len(values) > _MAX_FIELD_ENTRIES:
        raise EnvironmentRequirementError("environment requirement mapping exceeds entry limit", context={"field": field, "limit": _MAX_FIELD_ENTRIES})
    result: dict[str, str] = {}
    for key, value in values.items():
        if type(key) is not str or type(value) is not str or len(key) > _MAX_TEXT or len(value) > _MAX_TEXT:
            raise EnvironmentRequirementError("environment requirement mapping text is invalid")
        result[key] = value
    return result


def _validate_hard_package(requirement: str) -> None:
    """Reject package forms that an observed environment record cannot prove."""

    try:
        parsed = Requirement(requirement)
    except Exception:
        raise EnvironmentRequirementError("invalid package requirement") from None
    if parsed.extras or parsed.url or parsed.marker is not None and "extra" in str(parsed.marker):
        raise EnvironmentRequirementError("environment hard requirements do not support extras, URLs, or extra markers")


def _source(value: RequirementSource | str | None) -> RequirementSource:
    """Normalize one explicit source without deriving identity from a target."""

    if value is None:
        return RequirementSource("@dryml.environments.req")
    if type(value) is RequirementSource:
        return value
    if type(value) is str:
        return RequirementSource(value)
    raise EnvironmentRequirementError("environment requirement source is invalid")


def req(
    *,
    python: str | None = None,
    requirements: Iterable[str] = (),
    excludes: Iterable[str] = (),
    capabilities: Iterable[str] = (),
    tags: Iterable[str] = (),
    dryml_protocol: str | None = None,
    schema_versions: Mapping[str, str] | None = None,
    source: RequirementSource | str | None = None,
) -> Callable[[T], T]:
    """Create a passive hard environment-requirement decorator.

    Args:
        python: Optional Python version specifier.
        requirements: Bounded PEP 508 package requirements without extras, URLs,
            or ``extra`` markers.
        excludes: Bounded normalized distribution names that must be absent.
        capabilities: Bounded required environment capability names.
        tags: Bounded required environment tag names.
        dryml_protocol: Optional DRYML protocol version specifier.
        schema_versions: Bounded mapping of schema names to version specifiers.
        source: Optional explicit shared source, label, or the fixed decorator
            label when omitted.

    Returns:
        A decorator returning the exact supplied supported target unchanged.

    Raises:
        EnvironmentRequirementError: If declaration input is malformed, exceeds a
            bound, or cannot be attached through passive annotations.

    Side Effects:
        Applying the returned decorator appends one process-local annotation. It
        never wraps, invokes, or binds the target and does not inspect a host.
    """

    if python is not None and (type(python) is not str or len(python) > _MAX_TEXT):
        raise EnvironmentRequirementError("environment Python constraint is invalid")
    if dryml_protocol is not None and (type(dryml_protocol) is not str or len(dryml_protocol) > _MAX_TEXT):
        raise EnvironmentRequirementError("environment DRYML protocol constraint is invalid")
    normalized_requirements = _bounded_strings(requirements, field="requirements")
    for requirement in normalized_requirements:
        _validate_hard_package(requirement)
    value = EnvironmentRequirement(
        python=python,
        requirements=normalized_requirements,
        excludes=_bounded_strings(excludes, field="excludes"),
        capabilities=_bounded_strings(capabilities, field="capabilities"),
        tags=_bounded_strings(tags, field="tags"),
        dryml_protocol=dryml_protocol,
        schema_versions=_bounded_mapping(schema_versions, field="schema_versions"),
    )
    declaration = RequirementDeclaration(value, source=_source(source))

    def decorate(target: T) -> T:
        """Attach this prevalidated declaration while preserving target identity."""

        try:
            return attach_declaration(target, key=ENVIRONMENT_REQUIREMENT_KEY, declaration=declaration)
        except Exception:
            raise EnvironmentRequirementError("environment requirement annotation attachment failed") from None

    return decorate


__all__ = ["ENVIRONMENT_REQUIREMENT_KEY", "req"]
