"""Passive annotation composition for managed declarations."""

from __future__ import annotations

from dryml.annotations import Annotation, annotations_for_method, attach_annotation, own_annotations
from dryml.managed import managed_operation


def test_managed_decorator_preserves_annotations_in_both_orders_and_inheritance():
    """Keep annotation ownership passive regardless of decorator ordering or MRO."""

    before = Annotation("managed.test.before", "before")
    after = Annotation("managed.test.after", "after")

    def attach_before(target):
        """Attach metadata before the managed descriptor is constructed."""

        return attach_annotation(target, before)

    def attach_after(target):
        """Attach metadata after the managed descriptor is constructed."""

        return attach_annotation(target, after)

    class Base:
        @managed_operation()
        @attach_before
        def inherited(self, *, managed):
            """Declare an inherited managed method with preexisting metadata."""

    class Child(Base):
        pass

    class Ordered:
        @attach_after
        @managed_operation()
        def local(self, *, managed):
            """Declare a managed method with descriptor-owned metadata."""

    assert own_annotations(Base.inherited) == (before,)
    assert own_annotations(Ordered.local) == (after,)
    assert Child.inherited.member == "inherited"
    assert annotations_for_method(Child, "inherited") == (before,)
    assert annotations_for_method(Ordered, "local") == (after,)
