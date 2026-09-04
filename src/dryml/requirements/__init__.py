"""Dependency-light contracts for explicit domain-owned hard requirements."""

from .barrier import AdmissionReport, require_admission
from .combination import RequirementCombiner, combine_requirements
from .errors import RequirementBarrierError, RequirementCombinationError, RequirementError
from .model import RequirementDeclaration, RequirementIssue, RequirementReport, RequirementResult, RequirementSource

__all__ = [
    "AdmissionReport",
    "RequirementBarrierError",
    "RequirementCombinationError",
    "RequirementCombiner",
    "RequirementDeclaration",
    "RequirementError",
    "RequirementIssue",
    "RequirementReport",
    "RequirementResult",
    "RequirementSource",
    "combine_requirements",
    "require_admission",
]
