"""Direct, immutable, declaration-only requirement and default annotations."""

from .collect import collect_fragments, fragments_for_class, fragments_for_definition_method, fragments_for_method
from .decorators import default, require
from .errors import AnnotationError, AnnotationMergeError, AnnotationValidationError, UnsupportedAnnotationTargetError
from .merge import RequirementDiagnostic, RequirementResolution, resolve_fragments, resolve_target_requirements
from .model import AnnotationFragment, AnnotationTarget, SourceTrace, UnresolvedAnnotationResult, source_from_target, target_from_live, validate_namespace
from .storage import FRAGMENT_ATTR, attach_fragment, own_fragments

__all__ = ["AnnotationError", "AnnotationFragment", "AnnotationMergeError", "AnnotationTarget", "AnnotationValidationError", "FRAGMENT_ATTR", "RequirementDiagnostic", "RequirementResolution", "SourceTrace", "UnresolvedAnnotationResult", "UnsupportedAnnotationTargetError", "attach_fragment", "collect_fragments", "default", "fragments_for_class", "fragments_for_definition_method", "fragments_for_method", "own_fragments", "require", "resolve_fragments", "resolve_target_requirements", "source_from_target", "target_from_live", "validate_namespace"]
