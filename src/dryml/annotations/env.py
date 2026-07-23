"""Environment requirement annotation namespace helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dryml.environments.requirements import EnvironmentRequirement
from dryml.environments.specs import spec_from_data
from dryml.environments.utils import coerce_tuple, normalize_requirement_string

from .decorators import default as default_fragment
from .decorators import require
from .namespaces import ENVIRONMENT


def normalize_environment_requirement_fragment(
    *,
    python: str | None = None,
    packages: Mapping[str, str | None] | None = None,
    requirements=(),
    excludes=(),
    capabilities=(),
    tags=(),
    dryml_protocol: str | None = None,
    schema_versions: Mapping[str, str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Normalize decorator kwargs into an EnvironmentRequirement payload."""

    if extra:
        raise TypeError(f"unknown environment requirement fields: {', '.join(sorted(extra))}")
    reqs: list[str] = []
    for name, spec in (packages or {}).items():
        reqs.append(normalize_requirement_string(str(name) if spec in (None, "") else f"{name}{spec}"))
    reqs.extend(normalize_requirement_string(req) for req in coerce_tuple(requirements))
    requirement = EnvironmentRequirement(
        python=python,
        requirements=tuple(reqs),
        excludes=tuple(str(item) for item in coerce_tuple(excludes)),
        capabilities=tuple(str(item) for item in coerce_tuple(capabilities)),
        tags=tuple(str(item) for item in coerce_tuple(tags)),
        dryml_protocol=dryml_protocol,
        schema_versions=schema_versions or {},
    )
    return requirement.to_data()


def req(**kwargs: Any):
    """Decorate a target with a hard software-environment requirement.

    Args:
        **kwargs: Environment requirement fields plus annotation metadata fields
            ``source``, ``priority``, and ``merge_policy``. Metadata fields are
            stored on the annotation fragment, not in the requirement payload.

    Returns:
        A decorator that attaches the requirement fragment without wrapping the
        target.
    """

    source = kwargs.pop("source", None)
    priority = kwargs.pop("priority", 0)
    merge_policy = kwargs.pop("merge_policy", None)
    return require(namespace=ENVIRONMENT, fragment=normalize_environment_requirement_fragment(**kwargs), source=source, priority=priority, merge_policy=merge_policy)


def default(spec: Mapping[str, Any] | Any | None = None, **kwargs: Any):
    """Decorate a target with an overrideable environment selection default.

    ``spec`` may be an environment spec object or JSON-ready environment spec
    mapping. Explicit dispatch environments still take precedence over this
    annotation default.
    """

    source = kwargs.pop("source", None)
    priority = kwargs.pop("priority", 0)
    merge_policy = kwargs.pop("merge_policy", None)
    if spec is not None and kwargs:
        raise TypeError("environment default accepts either spec or spec fields, not both")
    value = spec if spec is not None else kwargs
    if hasattr(value, "to_data"):
        value = value.to_data()
    if not isinstance(value, Mapping):
        raise TypeError("environment default spec must be a mapping or environment spec object")
    return default_fragment(
        namespace=ENVIRONMENT,
        fragment=spec_from_data(value).to_data(),
        source=source,
        priority=priority,
        merge_policy=merge_policy,
    )


__all__ = ["default", "normalize_environment_requirement_fragment", "req"]
