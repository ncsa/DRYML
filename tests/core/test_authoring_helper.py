"""Policy-neutral authoring helper dispatch and failure-boundary coverage."""

from __future__ import annotations

from inspect import signature
from concurrent.futures import ThreadPoolExecutor

import pytest

import dryml
from dryml.core import Definition, F, Object, Par, Ref, authoring_helper


class AuthoredValue(Object):
    """Inert construction target whose counter detects accidental materialization."""

    constructions = 0

    def __init__(self, value):
        type(self).constructions += 1
        self.value = value


def test_authoring_helper_binds_defaults_and_parameter_kinds_once_without_target_effects():
    """Policies receive named inputs; authoring never calls the concrete target."""

    events = []

    def choose(arguments):
        events.append(("choose", dict(arguments)))
        with pytest.raises(TypeError):
            arguments["value"] = 0
        return isinstance(arguments["value"], Par)

    def validate(arguments):
        events.append(("validate", dict(arguments)))

    def build(arguments):
        events.append(("build", dict(arguments)))
        return Definition(AuthoredValue, value=arguments["value"])

    @authoring_helper(
        author_definition=build,
        should_author=choose,
        validate_known_arguments=validate,
    )
    def helper(value, /, *extras, scale=2, **options):
        """Return the unchanged concrete inputs for call-spelling verification."""
        events.append(("concrete", value, extras, scale, options))
        return value, extras, scale, options

    AuthoredValue.constructions = 0
    result = helper(Par("value"), 7, label="fixed")
    expected = {"value": Par("value"), "extras": (7,), "scale": 2, "options": {"label": "fixed"}}
    assert isinstance(result, Definition)
    assert [event[0] for event in events] == ["choose", "validate", "build"]
    assert all(event[1] == expected for event in events)
    assert AuthoredValue.constructions == 0
    assert helper.__name__ == "helper"
    assert helper.__doc__ == helper.__wrapped__.__doc__
    assert signature(helper) == signature(helper.__wrapped__)

    events.clear()
    assert helper(3, 7, scale=4, should_author="workload") == (3, (7,), 4, {"should_author": "workload"})
    assert [event[0] for event in events] == ["choose", "concrete"]


def test_authoring_helper_default_detection_respects_nested_inputs_and_quotes():
    """Expressions and Definitions author; quoted and concrete values do not."""

    concrete_calls = []

    @authoring_helper(author_definition=lambda arguments: Definition(AuthoredValue, value=arguments["value"]))
    def helper(value):
        concrete_calls.append(value)
        return value

    nested = {"factories": (F("builtins:tuple", Par("width") * 2),)}
    assert isinstance(helper(nested), Definition)
    soft = Definition(AuthoredValue, value=7)
    assert isinstance(helper(soft), Definition)
    for value in (soft.concretize(), Ref(soft), soft.quote()):
        assert helper(value) is value
    assert len(concrete_calls) == 3
    assert dryml.authoring_helper is authoring_helper


def test_authoring_helper_default_definition_is_detected_after_default_binding():
    """Omitted defaults participate in the same input policy as explicit values."""

    default = Definition(AuthoredValue, value=7)

    @authoring_helper(author_definition=lambda arguments: Definition(AuthoredValue, value=arguments["value"]))
    def helper(value=default):
        raise AssertionError("the concrete target must not run")

    assert helper().value is default


def test_authoring_helper_symbolic_branch_bypasses_concrete_signature_normalization():
    """Only concrete calls activate Ref admission, never placeholder authoring."""

    calls = []

    @authoring_helper(
        author_definition=lambda arguments: Definition(AuthoredValue, value=arguments["value"]),
        normalize_concrete=True,
    )
    def helper(value: Ref[Definition]) -> Ref[Definition]:
        calls.append(value)
        return value

    symbolic = helper(Par("value"))
    assert isinstance(symbolic, Definition)
    assert calls == []
    concrete = Definition(AuthoredValue, value=7)
    assert isinstance(helper(concrete), Definition)
    assert calls == []
    assert helper(Ref(concrete)) is concrete
    assert calls == [concrete]


def test_authoring_helper_policy_and_builder_failures_never_fall_back():
    """Known validation errors and malformed policy results cannot invoke the target."""

    calls = []

    def target(value):
        calls.append(value)

    def fail(arguments):
        raise ValueError("known argument failure")

    helper = authoring_helper(
        author_definition=lambda arguments: calls.append("builder"),
        validate_known_arguments=fail,
    )(target)
    with pytest.raises(ValueError, match="known argument failure"):
        helper(Par("value"))
    assert calls == []

    helper = authoring_helper(
        author_definition=lambda arguments: Definition(AuthoredValue, value=arguments["value"]),
        should_author=lambda arguments: 1,
    )(target)
    with pytest.raises(TypeError, match="exact bool"):
        helper(7)

    helper = authoring_helper(author_definition=lambda arguments: 7)(target)
    with pytest.raises(TypeError, match="return a Definition"):
        helper(Par("value"))

    helper = authoring_helper(author_definition=fail)(target)
    with pytest.raises(ValueError, match="known argument failure"):
        helper(Par("value"))
    assert calls == []


def test_authoring_helper_invalid_calls_fail_before_policies():
    """Python signature binding guards missing, duplicate, and unknown arguments."""

    calls = []

    @authoring_helper(
        author_definition=lambda arguments: calls.append("builder"),
        should_author=lambda arguments: calls.append("predicate"),
    )
    def helper(value, *, scale=2):
        calls.append("concrete")

    for args, kwargs in (((), {}), ((1,), {"value": 2}), ((1,), {"unknown": 2})):
        with pytest.raises(TypeError):
            helper(*args, **kwargs)
    assert calls == []


def test_authoring_helper_concurrent_calls_keep_bound_arguments_separate():
    """One decorated helper retains no bound mapping shared between callers."""

    @authoring_helper(author_definition=lambda arguments: Definition(AuthoredValue, value=arguments["value"]))
    def helper(value):
        raise AssertionError("the concrete target must not run")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(helper, (Par("left"), Par("right"))))

    assert tuple(result.value.name for result in results) == ("left", "right")


@pytest.mark.parametrize("options", (
    {"author_definition": None},
    {"author_definition": lambda arguments: None, "should_author": False},
    {"author_definition": lambda arguments: None, "validate_known_arguments": 7},
    {"author_definition": lambda arguments: None, "normalize_concrete": 1},
))
def test_authoring_helper_rejects_invalid_configuration(options):
    """Invalid decorator policies fail before decoration or helper invocation."""

    with pytest.raises(TypeError):
        authoring_helper(**options)
