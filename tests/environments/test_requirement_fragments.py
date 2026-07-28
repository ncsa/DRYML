import pytest

import dryml.environments as envs


def test_req_and_add_req_decorators_compose_without_mutating_parent():
    @envs.req(requirements=("torch>=2",), tags=("torch",))
    class Base:
        pass

    @envs.add_req(requirements=("transformers>=4",), tags=("nlp",))
    class Child(Base):
        pass

    base_req = envs.requirements_for_class(Base)
    child_req = envs.requirements_for_class(Child)
    assert base_req.requirements == ("torch>=2",)
    assert child_req.requirements == ("torch>=2", "transformers>=4")
    assert child_req.tags == ("nlp", "torch")
    assert len(envs.fragments_for_class(Base)) == 1
    assert len(envs.fragments_for_class(Child)) == 2


def test_override_req_replaces_specific_fields():
    @envs.req(requirements=("torch>=2",), tags=("torch",), python=">=3.10")
    class Base:
        pass

    @envs.override_req(requirements=("torch>=2.6",), python=">=3.11")
    class Child(Base):
        pass

    req = envs.requirements_for_class(Child)
    assert req.requirements == ("torch>=2.6",)
    assert req.tags == ("torch",)
    assert req.python == ">=3.11"


def test_multiple_inheritance_fragment_order_is_deterministic():
    @envs.req(requirements=("a",), tags=("a",))
    class A:
        pass

    @envs.req(requirements=("b",), tags=("b",))
    class B:
        pass

    class C(A, B):
        pass

    assert [fragment.source.rsplit(":", 1)[-1] for fragment in envs.fragments_for_class(C)] == ["base", "base"]
    assert envs.requirements_for_class(C).requirements == ("a", "b")


def test_requirements_for_class_no_fragments_empty_requirement():
    class Plain:
        pass

    req = envs.requirements_for_class(Plain)
    assert req.requirements == ()
    assert req.explain_sources() == "Environment requirement has no recorded fragment sources."


def test_compose_fragments_conflict_detection_and_sources():
    fragments = (
        envs.RequirementFragment(python=">=3.10", source="base"),
        envs.RequirementFragment(python=">=3.11", source="child"),
    )
    assert envs.compose_fragments(fragments).python == ">=3.10,>=3.11"
    composed = envs.compose_fragments((envs.RequirementFragment(requirements=("dryml",), source="one"),))
    assert "one" in composed.explain_sources()


def test_fragment_roundtrip_and_invalid_requirement():
    fragment = envs.RequirementFragment(requirements=("dryml>=0.3",), source="test", mode="add")
    assert envs.RequirementFragment.from_data(fragment.to_data()).to_data() == fragment.to_data()
    with pytest.raises(envs.EnvironmentRequirementError):
        envs.RequirementFragment(requirements=("not valid !!!",))
