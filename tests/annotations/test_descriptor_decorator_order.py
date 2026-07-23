from __future__ import annotations

import dryml
import dryml.annotations as ann
from dryml.annotations.decorators import attach_fragment


def _requirements(fragments):
    values: list[str] = []
    for fragment in fragments:
        values.extend(fragment.fragment.get("requirements", ()))
    return tuple(values)


@dryml.env.req(requirements=("descriptor-class>=1",))
class DescriptorTargets:
    @classmethod
    @dryml.env.req(requirements=("inner-classmethod>=1",))
    def inner_classmethod(cls):
        return cls.__name__

    @dryml.env.req(requirements=("outer-classmethod>=1",))
    @classmethod
    def outer_classmethod(cls):
        return cls.__name__

    @staticmethod
    @dryml.env.req(requirements=("inner-staticmethod>=1",))
    def inner_staticmethod():
        return "inner"

    @dryml.env.req(requirements=("outer-staticmethod>=1",))
    @staticmethod
    def outer_staticmethod():
        return "outer"


def test_classmethod_inner_and_outer_decorator_orders_work():
    inner = ann.fragments_for_method(DescriptorTargets, "inner_classmethod", namespace="environment")
    outer = ann.fragments_for_method(DescriptorTargets, "outer_classmethod", namespace="environment")

    assert DescriptorTargets.inner_classmethod() == "DescriptorTargets"
    assert _requirements(inner) == ("descriptor-class>=1", "inner-classmethod>=1")
    assert _requirements(outer) == ("descriptor-class>=1", "outer-classmethod>=1")


def test_staticmethod_inner_and_outer_decorator_orders_work():
    inner = ann.fragments_for_method(DescriptorTargets, "inner_staticmethod", namespace="environment")
    outer = ann.fragments_for_method(DescriptorTargets, "outer_staticmethod", namespace="environment")

    assert DescriptorTargets.inner_staticmethod() == "inner"
    assert DescriptorTargets.outer_staticmethod() == "outer"
    assert _requirements(inner) == ("descriptor-class>=1", "inner-staticmethod>=1")
    assert _requirements(outer) == ("descriptor-class>=1", "outer-staticmethod>=1")


def test_descriptor_collection_deduplicates_shared_fragment_objects():
    raw = DescriptorTargets.__dict__["inner_classmethod"]
    fragment = ann.own_fragments(raw.__func__)[0]
    attach_fragment(raw, fragment)

    fragments = ann.fragments_for_method(DescriptorTargets, "inner_classmethod", namespace="environment")

    assert _requirements(fragments).count("inner-classmethod>=1") == 1
