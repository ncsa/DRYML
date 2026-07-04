"""Generic sidecar annotations for requirements, defaults, and source traces."""

from .arg_roles import ARG_ROLE_HELP, ArgRole, MaterializeArg, RefCDef, RefCDefArg, SelectorArg, ValueArg
from .collect import collect_fragments, fragments_for, fragments_for_callable, fragments_for_class
from .decorators import FRAGMENT_ATTR, default, require
from .errors import AnnotationConflictError, AnnotationError, AnnotationMergeError, AnnotationResolutionError, AnnotationValidationError
from .merge import ResolvedDefaults, ResolvedRequirements, ResolutionResult, resolve, resolve_defaults, resolve_environment_requirement, resolve_requirements, resolve_runtime_default, resolve_world_default, resolve_world_requirement
from .model import AnnotationFragment, AnnotationTarget, SourceTrace, source_from_target, validate_namespace
from .report import AnnotationIssue, AnnotationReport, format_report
from .storage import annotation_payload_for_id, attach_annotation_id, compute_annotation_id, make_annotation_spec, validate_annotation_spec

__all__ = [
    "ARG_ROLE_HELP",
    "FRAGMENT_ATTR",
    "AnnotationConflictError",
    "AnnotationError",
    "AnnotationFragment",
    "AnnotationIssue",
    "AnnotationMergeError",
    "AnnotationReport",
    "AnnotationResolutionError",
    "AnnotationTarget",
    "AnnotationValidationError",
    "ArgRole",
    "MaterializeArg",
    "RefCDef",
    "RefCDefArg",
    "ResolvedDefaults",
    "ResolvedRequirements",
    "ResolutionResult",
    "SelectorArg",
    "SourceTrace",
    "ValueArg",
    "annotation_payload_for_id",
    "attach_annotation_id",
    "collect_fragments",
    "compute_annotation_id",
    "default",
    "format_report",
    "fragments_for",
    "fragments_for_callable",
    "fragments_for_class",
    "make_annotation_spec",
    "require",
    "resolve",
    "resolve_defaults",
    "resolve_environment_requirement",
    "resolve_requirements",
    "resolve_runtime_default",
    "resolve_world_default",
    "resolve_world_requirement",
    "source_from_target",
    "validate_annotation_spec",
    "validate_namespace",
]
