"""Tests for capture-wide raw annotation attachment bounds."""

import pytest

from dryml.annotations import (
    Annotation,
    attach_annotation,
    collect_annotations,
)
from dryml.code import capture_inspection
from dryml.environments import EnvironmentRequirementsKernel


def test_capture_bound_precedes_carrier_validation() -> None:
    """Capture rejects 4,097 entries; public collection remains unbounded."""

    def target() -> None:
        return None

    for _ in range(4097):
        attach_annotation(target, Annotation("test.unrelated", object()))

    assert len(collect_annotations(target)) == 4097
    capture = capture_inspection(target)
    with pytest.raises(Exception, match="raw attachment limit"):
        EnvironmentRequirementsKernel._from_capture(capture)


def test_capture_bound_precedes_malformed_carrier_validation() -> None:
    """An over-limit tuple fails on its raw count before a later bad entry."""

    def target() -> None:
        return None

    for _ in range(4096):
        attach_annotation(target, Annotation("test.unrelated", object()))
    target.__dryml_annotations__ += ("malformed",)

    with pytest.raises(Exception, match="raw attachment limit"):
        EnvironmentRequirementsKernel._from_capture(capture_inspection(target))
