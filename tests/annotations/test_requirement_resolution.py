"""Requirement-resolution data-only and CDef non-persistence coverage."""

from dryml.annotations import require, resolve_fragments, resolve_target_requirements
from dryml.core import Definition, Object


def test_unknown_policy_is_reported_and_caller_override_follows_target_fragments():
    @require(namespace="runtime", fragment={"limits": {"threads": 1}})
    class Subject(Object):
        pass
    override = Subject.__dryml_annotation_fragments__[0].__class__(
        Subject.__dryml_annotation_fragments__[0].target,
        "runtime", "requirement", {"limits": {"threads": 2}},
        Subject.__dryml_annotation_fragments__[0].source, 1, "unknown",
    )
    result = resolve_target_requirements(Subject, overrides=(override,))
    assert not result.usable
    assert result.fragments[-1] is override


def test_annotation_sidecars_do_not_enter_cdef_identity_or_pickle_state():
    class Subject(Object):
        pass
    before = Definition(Subject).concretize()
    require(namespace="runtime", fragment={"limits": {"threads": 1}})(Subject)
    after = Definition(Subject).concretize()
    assert before == after
    assert before.stable_hash() == after.stable_hash()
    assert set(before.__getstate__()) == {"identity_version", "cls", "parameters", "stable_hash_cache"}
