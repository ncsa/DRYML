"""Topology and resource specs for DRYML execution worlds.

``dryml.worlds`` owns hard resource requirements, requested/default world
shapes, actual backend allocations, and compatibility checks. It intentionally
does not activate process-local framework state; use :mod:`dryml.runtime` for
runtime mode, visibility, bootstrap, and guard APIs.
"""

from dryml.worlds.allocation import (
    ProcessAllocation,
    RuntimeResourceView,
    WorldAllocation,
    attach_world_allocation_id,
    compute_world_allocation_id,
    make_world_allocation_spec,
    validate_world_allocation_spec,
)
from dryml.worlds.compatibility import (
    CompatibilityIssue,
    CompatibilityReport,
    check_allocation_satisfies_requirement,
    check_world_spec_satisfies_requirement,
)
from dryml.worlds.current import current, discover_current, reset_current, set_current, use
from dryml.worlds.resources import ByteSize, CountConstraint, ResourceRequirement, ResourceSpec, parse_byte_size
from dryml.worlds.specs import (
    ProcessSpec,
    RoleRequirement,
    RoleSpec,
    WorldRequirement,
    WorldSpec,
    attach_world_id,
    attach_world_requirement_id,
    compute_world_id,
    compute_world_requirement_id,
    make_world_requirement_spec,
    make_world_spec,
    validate_world_requirement_spec,
    validate_world_spec,
)

__all__ = [
    "ByteSize",
    "CompatibilityIssue",
    "CompatibilityReport",
    "CountConstraint",
    "ProcessAllocation",
    "ProcessSpec",
    "ResourceRequirement",
    "ResourceSpec",
    "RoleRequirement",
    "RoleSpec",
    "RuntimeResourceView",
    "WorldAllocation",
    "WorldRequirement",
    "WorldSpec",
    "attach_world_allocation_id",
    "attach_world_id",
    "attach_world_requirement_id",
    "check_allocation_satisfies_requirement",
    "check_world_spec_satisfies_requirement",
    "compute_world_allocation_id",
    "compute_world_id",
    "compute_world_requirement_id",
    "current",
    "discover_current",
    "make_world_allocation_spec",
    "make_world_requirement_spec",
    "make_world_spec",
    "parse_byte_size",
    "reset_current",
    "set_current",
    "use",
    "validate_world_allocation_spec",
    "validate_world_requirement_spec",
    "validate_world_spec",
]
