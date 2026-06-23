import pytest

import core2_objects as objects
from dryml.core2 import Definition, SKIP_ARGS
from dryml.core2.definition import selector_match
from dryml.core2.query.fingerprint import (
    requirements_satisfied,
    selector_requirements,
    target_fingerprint,
)


def assert_no_fingerprint_false_negative(selector, target, *, class_match="selector"):
    assert selector_match(selector, target, strict=False)
    requirements = selector_requirements(selector, class_match=class_match)
    fingerprint = target_fingerprint(target)
    assert requirements_satisfied(fingerprint, requirements)


def test_fingerprint_never_rejects_container_compatible_target():
    target = objects.TestNest([1, 2]).definition
    selector = Definition(objects.TestNest, (1, 2))

    assert_no_fingerprint_false_negative(selector, target)


@pytest.mark.parametrize(
    "selector,target_factory",
    [
        (
            Definition(objects.TestClass1, 10),
            lambda: objects.TestClass1(10, test="a").definition,
        ),
        (
            Definition(objects.TestClass1, SKIP_ARGS, test="a"),
            lambda: objects.TestClass1(10, test="a").definition,
        ),
        (
            Definition(objects.TestClass1, lambda x: x == 10, test=lambda x: x == "a"),
            lambda: objects.TestClass1(10, test="a").definition,
        ),
        (
            Definition(objects.TestNest3, SKIP_ARGS, child=Definition(objects.TestNest2, SKIP_ARGS)),
            lambda: objects.TestNest3(child=objects.TestNest2("x")).definition,
        ),
        (
            Definition(objects.TestNest, {"key": (1, 2)}),
            lambda: objects.TestNest({"key": (1, 2)}).definition,
        ),
    ],
)
def test_fingerprint_no_false_negative_property_matrix(selector, target_factory):
    assert_no_fingerprint_false_negative(selector, target_factory())


def test_exact_class_fingerprint_requirements_are_sound_for_exact_match():
    target = objects.TestNest3(child=objects.TestNest2("x")).definition
    selector = Definition(objects.TestNest3, SKIP_ARGS, child=Definition(objects.TestNest2, SKIP_ARGS))

    assert_no_fingerprint_false_negative(selector, target, class_match="exact")
