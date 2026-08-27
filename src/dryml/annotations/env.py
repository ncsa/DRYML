"""Environment annotation sugar that remains passive and import-light."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dryml.environments import EnvironmentRequirement
from dryml.environments.specs import CondaEnvironmentSpec, ContainerEnvironmentSpec, CurrentEnvironmentSpec, PythonExecutableSpec, spec_from_data

from .decorators import default as declare_default
from .decorators import require
from .namespaces import ENVIRONMENT


def req(**kwargs: Any):
    """Declare a hard environment requirement without wrapping a target.

    Args:
        **kwargs: ``EnvironmentRequirement`` fields plus optional ``source``,
            ``priority``, and ``merge_policy`` annotation metadata.

    Returns:
        An identity-preserving declaration decorator.
    """

    source, priority, merge_policy = _metadata(kwargs)
    return require(namespace=ENVIRONMENT, fragment=EnvironmentRequirement(**kwargs).to_data(), source=source, priority=priority, merge_policy=merge_policy)


def default(spec: Any | None = None, **kwargs: Any):
    """Declare an environment selector default without selecting or probing it.

    Args:
        spec: Typed environment spec or complete spec envelope.
        **kwargs: Spec fields when ``spec`` is absent, plus annotation metadata.

    Returns:
        An identity-preserving declaration decorator.
    """

    source, priority, merge_policy = _metadata(kwargs)
    if spec is not None and kwargs:
        raise TypeError("environment default accepts either spec or spec fields")
    value = spec if spec is not None else _spec_from_fields(kwargs)
    if hasattr(value, "to_data"):
        value = value.to_data()
    if not isinstance(value, Mapping):
        raise TypeError("environment default requires a typed spec or mapping")
    return declare_default(namespace=ENVIRONMENT, fragment=spec_from_data(value).to_data(), source=source, priority=priority, merge_policy=merge_policy)


def _metadata(kwargs: dict[str, Any]) -> tuple[Any, int, str | None]:
    return kwargs.pop("source", None), kwargs.pop("priority", 0), kwargs.pop("merge_policy", None)


def _spec_from_fields(fields: dict[str, Any]) -> Any:
    """Construct one typed selector from the closed concise sugar spelling."""

    kind = fields.pop("kind", "current")
    constructors = {"current": CurrentEnvironmentSpec, "python": PythonExecutableSpec, "conda": CondaEnvironmentSpec, "container": ContainerEnvironmentSpec}
    try:
        return constructors[kind](**fields)
    except KeyError as error:
        raise TypeError(f"unknown environment spec kind {kind!r}") from error


__all__ = ["default", "req"]
