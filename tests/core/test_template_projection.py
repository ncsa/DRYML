"""Focused tests for deliberately loose template selector projection."""

from __future__ import annotations

from dryml.core import Definition, F
from dryml.core.template import Par, Template


class ProjectionModel:
    """Constructor declaration used to test preserved projection positions."""

    def __init__(self, values, *, activation, flag=False):
        self.values = values
        self.activation = activation
        self.flag = flag


def test_loose_projection_preserves_known_values_and_sequence_positions():
    """Unknown values become local wildcards while fixed facts still constrain."""
    template = Template(
        ProjectionModel,
        (Par("width"), "fixed", Par("depth")),
        activation="relu",
        flag=False,
    )
    selector = template.as_selector()

    assert selector.matches(Definition(ProjectionModel, (999, "fixed", 0), activation="relu", flag=False))
    assert not selector.matches(Definition(ProjectionModel, (999, "wrong", 0), activation="relu", flag=False))
    assert not selector.matches(Definition(ProjectionModel, (999, "fixed", 0), activation="relu", flag=True))


def test_loose_projection_keeps_partial_factory_call_shape_without_building():
    """Projected factories retain target, arity, known values, and keyword presence."""
    template = Template(
        ProjectionModel,
        (F("builtins:tuple", Par("width"), "fixed", flag=Par("flag")),),
        activation="relu",
    )
    selector = template.as_selector()

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
