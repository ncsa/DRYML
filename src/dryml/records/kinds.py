"""Record kinds and spec-family metadata for Sprint 1 sidecars."""

from __future__ import annotations

from dataclasses import dataclass


RECORD_KINDS = frozenset(
    {
        "stored_state",
        "data",
        "execution",
        "adapter",
        "program",
        "probe_report",
        "compatibility_report",
        "lowering_report",
    }
)


@dataclass(frozen=True, slots=True)
class SpecFamily:
    """Static metadata for a known spec family."""

    family: str
    schema: str | None
    prefix: str
    dir_name: str
    schema_version: int = 1


SPEC_FAMILIES: dict[str, SpecFamily] = {
    "representation": SpecFamily("representation", "dryml.representation.v1", "repr", "representation"),
    "operation": SpecFamily("operation", "dryml.operation.v1", "op", "operation"),
    "dispatch": SpecFamily("dispatch", "dryml.dispatch.v1", "dispatch", "dispatch"),
    "execution_recipe": SpecFamily("execution_recipe", "dryml.execution_recipe.v1", "recipe", "execution_recipe"),
    "environment_record": SpecFamily("environment_record", "dryml.environments.record.v1", "envrec", "environment_record"),
    "environment_requirement": SpecFamily("environment_requirement", "dryml.environments.requirement.v1", "envreq", "environment_requirement"),
    "environment_spec": SpecFamily("environment_spec", "dryml.environments.spec.v1", "envspec", "environment_spec"),
    "environment_lock": SpecFamily("environment_lock", "dryml.environments.lock.v1", "envlock", "environment_lock"),
    "world": SpecFamily("world", "dryml.world.v1", "world", "world"),
    "world_requirement": SpecFamily("world_requirement", "dryml.world_requirement.v1", "worldreq", "world_requirement"),
    "world_allocation": SpecFamily("world_allocation", "dryml.world_allocation.v1", "worldalloc", "world_allocation"),
    "runtime": SpecFamily("runtime", "dryml.runtime.v1", "runtime", "runtime"),
    "annotation": SpecFamily("annotation", "dryml.annotation.v1", "annotation", "annotation"),
    "generic": SpecFamily("generic", None, "spec", "generic"),
}


SPEC_FAMILY_BY_PREFIX = {family.prefix: name for name, family in SPEC_FAMILIES.items()}
SPEC_FAMILY_BY_SCHEMA = {family.schema: name for name, family in SPEC_FAMILIES.items() if family.schema is not None}


__all__ = [
    "RECORD_KINDS",
    "SPEC_FAMILIES",
    "SPEC_FAMILY_BY_PREFIX",
    "SPEC_FAMILY_BY_SCHEMA",
    "SpecFamily",
]
