"""Tests for the annotation kernel's import and public-surface boundary."""

import subprocess
import sys

import pytest

import dryml
import dryml.annotations as annotations


_PUBLIC = {
    "Annotation",
    "AnnotatedMember",
    "ANNOTATION_ATTR",
    "attach_annotation",
    "own_annotations",
    "collect_annotations",
    "annotations_for_class",
    "annotations_for_members",
    "annotations_for_method",
    "AnnotationError",
    "AnnotationValidationError",
    "UnsupportedAnnotationTargetError",
}
_RETIRED = {
    "AnnotationFragment",
    "AnnotationMergeError",
    "AnnotationTarget",
    "FRAGMENT_ATTR",
    "RequirementDiagnostic",
    "RequirementResolution",
    "SourceTrace",
    "UnresolvedAnnotationResult",
    "attach_fragment",
    "collect_fragments",
    "default",
    "fragments_for_class",
    "fragments_for_definition_method",
    "fragments_for_method",
    "own_fragments",
    "require",
    "resolve_fragments",
    "resolve_target_requirements",
    "source_from_target",
    "target_from_live",
    "validate_namespace",
}


def test_kernel_exports_are_exact_and_retired_submodules_are_absent():
    """The clean break exposes only the passive kernel and no compatibility shim."""

    assert set(annotations.__all__) == _PUBLIC
    assert not {name for name in _RETIRED if hasattr(annotations, name)}
    for module in ("decorators", "env", "world", "runtime", "merge", "namespaces", "storage"):
        with pytest.raises(ModuleNotFoundError):
            __import__(f"dryml.annotations.{module}")


def test_root_aliases_restore_domain_views_without_annotation_facades():
    """Root conveniences are plural-owner identities, not annotation facades."""

    assert dryml.env is dryml.environments
    assert dryml.world is dryml.worlds
    assert {"env", "world", "requirements"} <= set(dryml.__all__)
    for owner in (dryml.env, dryml.world):
        assert {"current", "set_current", "reset_current", "use"} <= set(owner.__all__)
        assert not {"default", "default_for", "set_default", "reset_default", "use_default"} & set(owner.__all__)
    import dryml.runtime as runtime

    assert "default" not in runtime.__all__
    assert not hasattr(runtime, "default")


def test_fresh_annotation_import_has_no_consumer_or_runtime_side_effect():
    """Importing the kernel loads no consumer, session, worker, or backend module."""

    script = """
import json
import sys
import dryml.annotations
forbidden = (
    'dryml.artifacts', 'dryml.context', 'dryml.core', 'dryml.data',
    'dryml.environments', 'dryml.execute', 'dryml.formats', 'dryml.models',
    'dryml.runtime', 'dryml.session', 'dryml.worlds', 'tensorflow', 'torch',
    'jax', 'jaxlib', 'ray',
)
print(json.dumps(sorted(name for name in sys.modules if name in forbidden)))
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"
