import pytest

import core2_objects as objects
from dryml.core2 import Definition, SKIP_ARGS
from dryml.core2.definition import selector_match
from dryml.core2.freeze import FrozenList
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


def test_frozen_list_selector_is_not_filtered_out():
    selector = Definition(objects.TestNest, FrozenList([1, 2]))
    target = objects.TestNest([1, 2]).definition

    assert_no_fingerprint_false_negative(selector, target)


def test_frozen_list_selector_matches_list_and_tuple_compatible_targets():
    selector = Definition(objects.TestNest, FrozenList([1, 2]))

    assert_no_fingerprint_false_negative(selector, objects.TestNest([1, 2]).definition)
    assert_no_fingerprint_false_negative(selector, objects.TestNest((1, 2)).definition)


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


def selector_corpus():
    exact_child = objects.TestNest3(a=1)
    return [
        Definition(objects.TestClass1, 10),
        Definition(objects.TestClass1, SKIP_ARGS),
        Definition(objects.TestClass1, SKIP_ARGS, test=lambda x: x in {"a", "b"}),
        Definition(objects.TestNest, (1, 2)),
        Definition(objects.TestNest, [1, 2]),
        Definition(objects.TestNest, {"key": (1, 2)}),
        Definition(objects.TestNest, {objects.TestBase}),
        Definition(objects.TestNest3, SKIP_ARGS, child=Definition(objects.TestNest2, SKIP_ARGS)),
        Definition(objects.TestNest3, SKIP_ARGS, members={exact_child.definition}),
    ]


def target_corpus():
    exact_child = objects.TestNest3(a=1)
    return [
        objects.TestClass1(10, test="a").definition,
        objects.TestClass1(20, test="b").definition,
        objects.TestNest([1, 2]).definition,
        objects.TestNest((1, 2)).definition,
        objects.TestNest({"key": (1, 2)}).definition,
        objects.TestNest({objects.TestClassA}).definition,
        objects.TestNest3(child=objects.TestNest2("x")).definition,
        objects.TestNest3(members={exact_child}).definition,
        objects.TestNest3(members={objects.TestNest3(a=1, b=2)}).definition,
    ]


@pytest.mark.parametrize("selector", selector_corpus())
@pytest.mark.parametrize("target", target_corpus())
def test_fingerprint_filter_never_rejects_structural_match_matrix(selector, target):
    if selector_match(selector, target, strict=False):
        requirements = selector_requirements(selector)
        fingerprint = target_fingerprint(target)
        assert requirements_satisfied(fingerprint, requirements)
