from __future__ import annotations

import pytest

import dryml
import dryml.annotations as ann


def _requirements(fragments):
    values: list[str] = []
    for fragment in fragments:
        values.extend(fragment.fragment.get("requirements", ()))
    return tuple(values)


@dryml.env.req(requirements=("base-class>=1",))
class BaseModel:
    @dryml.env.req(requirements=("base-method>=1",))
    def train(self):
        return "base"


@dryml.env.req(requirements=("child-class>=1",))
class InheritedModel(BaseModel):
    pass


@dryml.env.req(requirements=("override-class>=1",))
class OverrideModel(BaseModel):
    @dryml.env.req(requirements=("override-method>=1",))
    def train(self):
        return "override"


def test_fragments_for_method_includes_class_then_method_fragments():
    fragments = ann.fragments_for_method(BaseModel, "train", namespace="environment")

    assert _requirements(fragments) == ("base-class>=1", "base-method>=1")


def test_fragments_for_method_applies_namespace_and_kind_filters():
    class Filtered:
        @ann.default(namespace="runtime", fragment={"frameworks": {"plain": {}}})
        @dryml.env.req(requirements=("filtered>=1",))
        def train(self):
            return None

    env_requirements = ann.fragments_for_method(Filtered, "train", namespace="environment", kind="requirement")
    runtime_defaults = ann.fragments_for_method(Filtered, "train", namespace="runtime", kind="default")

    assert _requirements(env_requirements) == ("filtered>=1",)
    assert len(runtime_defaults) == 1
    assert runtime_defaults[0].fragment == {"frameworks": {"plain": {}}}


def test_resolve_method_requirements_merges_class_and_method_fragments():
    resolution = ann.resolve_method_requirements(BaseModel, "train")

    assert tuple(resolution.environment_requirement.requirements) == ("base-class>=1", "base-method>=1")


def test_inherited_method_includes_inherited_implementation_fragments():
    fragments = ann.fragments_for_method(InheritedModel, "train", namespace="environment")

    assert _requirements(fragments) == ("base-class>=1", "child-class>=1", "base-method>=1")


def test_inherited_method_source_trace_points_to_inherited_implementation():
    resolution = ann.resolve_method_requirements(InheritedModel, "train")
    trace_by_requirement = {
        tuple(fragment.fragment.get("requirements", ())): trace
        for fragment, trace in zip(resolution.fragments, resolution.source_traces, strict=True)
    }

    assert trace_by_requirement[("base-method>=1",)].qualname == "BaseModel.train"


def test_overridden_method_excludes_base_method_fragments_by_default():
    fragments = ann.fragments_for_method(OverrideModel, "train", namespace="environment")

    assert _requirements(fragments) == ("base-class>=1", "override-class>=1", "override-method>=1")
    assert "base-method>=1" not in _requirements(fragments)


def test_overridden_method_source_trace_points_to_overriding_implementation():
    resolution = ann.resolve_method_requirements(OverrideModel, "train")
    trace_by_requirement = {
        tuple(fragment.fragment.get("requirements", ())): trace
        for fragment, trace in zip(resolution.fragments, resolution.source_traces, strict=True)
    }

    assert ("base-method>=1",) not in trace_by_requirement
    assert trace_by_requirement[("override-method>=1",)].qualname == "OverrideModel.train"


def test_bound_method_and_unbound_method_resolve_with_owner_class():
    bound = ann.resolve_target_requirements(OverrideModel().train)
    unbound = ann.resolve_target_requirements(OverrideModel.train)

    assert tuple(bound.environment_requirement.requirements) == ("base-class>=1", "override-class>=1", "override-method>=1")
    assert tuple(unbound.environment_requirement.requirements) == tuple(bound.environment_requirement.requirements)


def test_callable_instance_is_not_treated_as_bound_method():
    class CallableThing:
        @dryml.env.req(requirements=("call-method>=1",))
        def __call__(self):
            return None

    instance = dryml.env.req(requirements=("instance>=1",))(CallableThing())

    assert _requirements(ann.fragments_for(instance, namespace="environment")) == ("instance>=1",)


def test_fragments_for_method_missing_method_raises_attribute_error():
    with pytest.raises(AttributeError):
        ann.fragments_for_method(BaseModel, "missing")


def test_fragments_for_method_validates_inputs():
    with pytest.raises(TypeError):
        ann.fragments_for_method(BaseModel(), "train")
    with pytest.raises(TypeError):
        ann.fragments_for_method(BaseModel, 1)
