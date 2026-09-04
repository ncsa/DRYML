"""Passive hard world requirement declarations for live targets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, TypeVar

from dryml.requirements import RequirementDeclaration, RequirementSource
from dryml.requirements.collection import attach_declaration

from .errors import WorldRequirementError
from .resources import CountConstraint, ResourceRequirement
from .specs import RoleRequirement, WorldRequirement

T = TypeVar("T")
WORLD_REQUIREMENT_KEY = "dryml.worlds.requirement"
_MAX_TEXT = 4096


def _source(value: RequirementSource | str | None) -> RequirementSource:
    """Normalize an explicit source without deriving it from a target."""

    if value is None:
        return RequirementSource("@dryml.worlds.req")
    if type(value) is RequirementSource:
        return value
    if type(value) is str:
        return RequirementSource(value)
    raise WorldRequirementError("world requirement source is invalid")


def _constraint(value: Any) -> CountConstraint:
    """Normalize one flattened count constraint through resource values."""

    return ResourceRequirement.from_data({"cpus": value}).cpus


def _validate_text(value: Any) -> None:
    """Reject oversized text in a normalized world payload before attachment."""

    if type(value) is str:
        if len(value) > _MAX_TEXT:
            raise WorldRequirementError("world requirement text exceeds declaration limit")
        return
    if value is None or type(value) in (bool, int, float):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_text(key)
            _validate_text(item)
        return
    if type(value) in (tuple, list):
        for item in value:
            _validate_text(item)
        return
    raise WorldRequirementError("world requirement contains unsupported normalized data")


def req(
    *,
    role: str = "main",
    roles: Mapping[str, RoleRequirement | Mapping[str, Any]] | None = None,
    replicas: CountConstraint | int | Mapping[str, int | None] | None = None,
    cpus: CountConstraint | int | Mapping[str, int | None] | None = None,
    memory: CountConstraint | int | str | Mapping[str, int | str | None] | None = None,
    accelerators: Mapping[str, CountConstraint | int | Mapping[str, int | None]] | None = None,
    accelerator_memory: Mapping[str, CountConstraint | int | str | Mapping[str, int | str | None]] | None = None,
    devices: Mapping[str, CountConstraint | int | Mapping[str, int | None]] | None = None,
    named: Mapping[str, CountConstraint | int | Mapping[str, int | None]] | None = None,
    topology: Mapping[str, Any] | None = None,
    source: RequirementSource | str | None = None,
) -> Callable[[T], T]:
    """Create a passive hard world-requirement decorator.

    Args:
        role: Single-role name for flattened fields; ``"main"`` when omitted.
        roles: Complete multi-role requirements. This grammar is exclusive with
            flattened resource and topology fields.
        replicas: Optional inclusive replica count constraint.
        cpus: Optional per-replica CPU count constraint.
        memory: Optional per-replica memory byte constraint.
        accelerators: Optional accelerator-kind count constraints.
        accelerator_memory: Optional accelerator-kind memory constraints.
        devices: Optional device-kind count constraints.
        named: Optional named-resource count constraints.
        topology: Optional representable role topology requirements.
        source: Explicit shared source, label, or the fixed decorator label.

    Returns:
        A decorator that returns the exact supported target after adding one
        process-local hard world annotation.

    Raises:
        WorldRequirementError: If grammar input, constraints, source, or passive
            annotation attachment is malformed.

    Side Effects:
        Applying the returned decorator attaches one annotation. It never wraps,
        invokes, binds, probes, allocates, or otherwise activates the target.
    """

    flattened = (replicas, cpus, memory, accelerators, accelerator_memory, devices, named, topology)
    try:
        if roles is not None:
            if role != "main" or any(value is not None for value in flattened):
                raise WorldRequirementError("complete world roles cannot be combined with flattened fields")
            if not isinstance(roles, Mapping):
                raise WorldRequirementError("world roles must be a mapping")
            value = WorldRequirement(roles)
        else:
            resources = ResourceRequirement.from_data(
                {
                    "cpus": cpus,
                    "memory": memory,
                    "accelerators": {} if accelerators is None else accelerators,
                    "accelerator_memory": {} if accelerator_memory is None else accelerator_memory,
                    "devices": {} if devices is None else devices,
                    "named": {} if named is None else named,
                }
            )
            value = WorldRequirement(
                {
                    role: RoleRequirement(
                        replicas=_constraint(replicas),
                        resources=resources,
                        topology={} if topology is None else topology,
                    )
                }
            )
        _validate_text(value.to_payload())
        declaration = RequirementDeclaration(value, source=_source(source))
    except WorldRequirementError:
        raise
    except Exception:
        raise WorldRequirementError("world requirement declaration is invalid") from None

    def decorate(target: T) -> T:
        """Attach this prevalidated declaration while preserving target identity."""

        try:
            return attach_declaration(target, key=WORLD_REQUIREMENT_KEY, declaration=declaration)
        except Exception:
            raise WorldRequirementError("world requirement annotation attachment failed") from None

    return decorate


__all__ = ["WORLD_REQUIREMENT_KEY", "req"]
