"""Explicit, import-light declarations and deterministic local world planning.

This package only represents requirements, requested launch shapes, exact local
assignments, and lightweight capacity evidence. It does not launch processes,
reserve resources, probe frameworks, or activate a runtime.
"""

from .allocation import ProcessAllocation, WorldAllocation
from .compatibility import WorldCompatibilityIssue, WorldCompatibilityReport, check_allocation_satisfies_requirement, check_world_spec_satisfies_requirement
from .combination import requirements_for, requirements_for_method
from .current import current, reset_current, set_current, use
from .declarations import WORLD_REQUIREMENT_KEY, req
from .errors import ResourceValidationError, WorldCompatibilityError, WorldError, WorldRequirementError, WorldSpecValidationError
from .inventory import LocalResourceInventory, local_inventory
from .local_allocation import assign_local_world
from .resources import CountConstraint, ResourceRequirement, ResourceSpec, canonical_byte_size, parse_byte_size
from .specs import ProcessSpec, RoleRequirement, RoleSpec, WorldRequirement, WorldSpec
from .synthesis import WorldSynthesisDiagnostic, WorldSynthesisResult, synthesize

__all__ = [
    "CountConstraint", "LocalResourceInventory", "ProcessAllocation", "ProcessSpec", "ResourceRequirement", "ResourceSpec", "ResourceValidationError", "RoleRequirement", "RoleSpec", "WorldAllocation", "WorldCompatibilityError", "WorldCompatibilityIssue", "WorldCompatibilityReport", "WorldError", "WorldRequirement", "WorldRequirementError", "WorldSpec", "WorldSpecValidationError", "WorldSynthesisDiagnostic", "WorldSynthesisResult", "assign_local_world", "canonical_byte_size", "check_allocation_satisfies_requirement", "check_world_spec_satisfies_requirement", "current", "local_inventory", "parse_byte_size", "req", "requirements_for", "requirements_for_method", "reset_current", "set_current", "synthesize", "use",
]
