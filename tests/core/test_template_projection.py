"""Focused tests for deliberately loose template selector projection."""

from __future__ import annotations

import pytest

from dryml.core import Definition, F, Generator, Object, Repo, Serializable, UniformFromSet
from dryml.core.store.dir import DirStore
from dryml.core.template import Par


class ProjectionModel:
    """Constructor declaration used to test preserved projection positions."""

    def __init__(self, values, *, activation, flag=False):
        self.values = values
        self.activation = activation
        self.flag = flag


class ProjectionChild(Object):
    """Fixed child definition used as an exact selector anchor."""

    def __init__(self, value):
        self.value = value


class ProjectionOwner(Serializable):
    """Own a fixed child and a variable width without constructing either."""

    def __init__(self, child, width):
        self.child = child
        self.width = width

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        pass


def test_loose_projection_preserves_known_values_and_sequence_positions():
    """Unknown values become local wildcards while fixed facts still constrain."""
    definition = Definition(
        ProjectionModel,
        (Par("width"), "fixed", Par("depth")),
        activation="relu",
        flag=False,
    )
    selector = definition.loose_selector()

    assert selector.matches(Definition(ProjectionModel, (999, "fixed", 0), activation="relu", flag=False))
    assert not selector.matches(Definition(ProjectionModel, (999, "wrong", 0), activation="relu", flag=False))
    assert not selector.matches(Definition(ProjectionModel, (999, "fixed", 0), activation="relu", flag=True))


def test_loose_projection_keeps_partial_factory_call_shape_without_building():
    """Projected factories retain target, arity, known values, and keyword presence."""
    definition = Definition(
        ProjectionModel,
        (F("builtins:tuple", Par("width"), "fixed", flag=Par("flag")),),
        activation="relu",
    )
    selector = definition.loose_selector()

    assert selector.matches(Definition(
        ProjectionModel,
        (F("builtins:tuple", 64, "fixed", flag=False),),
        activation="relu",
    ))
    assert not selector.matches(Definition(
        ProjectionModel,
        (F("builtins:tuple", 64, "other", flag=False),),
        activation="relu",
    ))


@pytest.mark.parametrize("query_index", ("memory", "sqlite"))
def test_loose_projection_preserves_fixed_cdef_anchors_in_queries(tmp_path, query_index):
    """Mapping-like CDefs stay exact in loose selectors and Generator prefilters."""

    child = Definition(ProjectionChild, value=7).concretize()
    other_child = Definition(ProjectionChild, value=8).concretize()
    authored = Definition(ProjectionOwner, child=child, width=Par("width"))
    candidate = authored.sub(width=32).concretize()
    other = Definition(ProjectionOwner, child=other_child, width=32).concretize()
    loose = authored.loose_selector()
    support = Generator(authored, {"width": UniformFromSet((32, 64))}).support_selector()

    assert loose.root.child is child
    assert support.prefilter.root.child is child
    assert loose.matches(candidate)
    assert authored.loose_selector(strict=True).matches(candidate)
    assert support.matches(candidate)
    assert not loose.matches(other)
    assert not support.matches(other)

    repo = Repo(DirStore(tmp_path / "store", query_index=query_index))
    repo.declare_object(candidate)
    repo.declare_object(other)

    assert repo.query().sel(loose).cdefs().one() == candidate
    assert repo.query().sel(support).cdefs().one() == candidate
