import pytest

import core2_objects as objects
from dryml.core2 import Definition, SKIP_ARGS, Satisfies, Selector
from dryml.core2.definition import selector_match
from dryml.core2.freeze import FrozenList
from dryml.core2.query.fingerprint import (
    legacy_requirements_satisfied,
    legacy_selector_requirements,
    legacy_target_fingerprint,
    target_local_fingerprint,
)
from dryml.core2.utils.stable_hash import stable_hash_function


def assert_no_fingerprint_false_negative(selector, target, *, class_match="selector"):
    assert Selector(selector, cls_policy=class_match).matches(target)
    requirements = legacy_selector_requirements(selector, class_match=class_match)
    fingerprint = legacy_target_fingerprint(target)
    assert legacy_requirements_satisfied(fingerprint, requirements)


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
            Definition(objects.TestClass1, Satisfies(lambda x: x == 10, name="is-10"), test=Satisfies(lambda x: x == "a", name="is-a")),
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
        Definition(objects.TestClass1, SKIP_ARGS, test=Satisfies(lambda x: x in {"a", "b"}, name="is-a-or-b")),
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
        requirements = legacy_selector_requirements(selector)
        fingerprint = legacy_target_fingerprint(target)
        assert legacy_requirements_satisfied(fingerprint, requirements)


def test_local_fingerprint_stops_at_nested_cdef_boundary():
    child = objects.TestNest2("needle")
    parent = objects.TestNest3(child=child).definition

    fingerprint = target_local_fingerprint(parent)
    child_scalar = stable_hash_function("needle")

    assert any(token.kind == "CDEF_EDGE_AT_PATH" and str(token.path) == "$.child" for token in fingerprint.counts)
    assert all(
        not (token.kind == "SCALAR_VALUE" and token.payload == child_scalar)
        for token in fingerprint.counts
    )


def test_child_local_fingerprint_contains_child_interior():
    child = objects.TestNest2("needle").definition

    fingerprint = target_local_fingerprint(child)
    child_scalar = stable_hash_function("needle")

    assert any(token.kind == "SCALAR_VALUE" and token.payload == child_scalar for token in fingerprint.counts)
