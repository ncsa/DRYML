from __future__ import annotations

import dryml.code as code


def _requirements(result, namespace=None):
    facts = result.facts_of_kind("requirement")
    if namespace is not None:
        facts = tuple(fact for fact in facts if fact.namespace == namespace)
    return facts


def test_decorated_function_and_class_emit_requirement_facts(requirement_targets):
    function = code.analyze(requirement_targets.run_training, algorithms=("direct_annotations",))
    cls = code.analyze(requirement_targets.LightningModel, algorithms=("direct_annotations",))

    assert _requirements(function, "environment")[0].fragment["requirements"] == ["pandas>=2"]
    assert [fact.fragment["requirements"] for fact in _requirements(cls, "environment")] == [["torch>=2"], ["lightning>=2"]]


def test_method_and_undecorated_function_behavior(requirement_targets):
    method = code.analyze(requirement_targets.LightningModel().train, algorithms=("direct_annotations",))
    undecorated = code.analyze(requirement_targets.plain_importable_function, algorithms=("direct_annotations",))

    assert _requirements(method, "world")[0].fragment["roles"]["main"]["resources"]["accelerators"] == {"gpu": {"min": 1}}
    assert undecorated.facts == ()


def test_classmethod_staticmethod_and_metadata_preservation(requirement_targets):
    classmethod_result = code.analyze(requirement_targets.ClassMethodTargets.inner_decorated, algorithms=("direct_annotations",))
    staticmethod_result = code.analyze(requirement_targets.StaticMethodTargets.inner_decorated, algorithms=("direct_annotations",))

    cls_fact = _requirements(classmethod_result, "environment")[0]
    static_fact = _requirements(staticmethod_result, "environment")[0]

    assert cls_fact.requirement_kind == "requirement"
    assert cls_fact.priority == 0
    assert cls_fact.merge_policy is None
    assert cls_fact.fragment["requirements"] == ["inner-classmethod>=1"]
    assert static_fact.fragment["requirements"] == ["inner-staticmethod>=1"]


def test_no_final_requirement_merging(requirement_targets):
    result = code.analyze(requirement_targets.LightningModel, algorithms=("direct_annotations",))

    assert len(_requirements(result, "environment")) == 2


def test_descriptor_target_collects_outer_decorated_classmethod_and_staticmethod(requirement_targets):
    classmethod_target = code.target_from_class_attribute(requirement_targets.ClassMethodTargets, "outer_decorated")
    staticmethod_target = code.target_from_class_attribute(requirement_targets.StaticMethodTargets, "outer_decorated")

    classmethod_result = code.analyze(classmethod_target, algorithms=("direct_annotations",))
    staticmethod_result = code.analyze(staticmethod_target, algorithms=("direct_annotations",))

    assert _requirements(classmethod_result, "environment")[0].fragment["requirements"] == ["outer-classmethod>=1"]
    assert _requirements(staticmethod_result, "environment")[0].fragment["requirements"] == ["outer-staticmethod>=1"]
