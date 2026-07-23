from __future__ import annotations

import inspect

import dryml.code as code
from dryml.code.callable_info import analyze_callable


class CallableInstance:
    def __call__(self, value, scale=1):
        return value * scale


class HostileCallable:
    def __getattribute__(self, name):
        raise AssertionError("callable analysis must not dynamically inspect targets")

    def __call__(self):
        raise AssertionError("callable analysis must not invoke targets")


class HostileMeta(type):
    def __getattribute__(cls, name):
        raise AssertionError("callable analysis must not invoke metaclass hooks")


class HostileMetaCallable(metaclass=HostileMeta):
    def __call__(self):
        raise AssertionError("callable analysis must not invoke targets")


class HostileWrapped:
    def __getattribute__(self, name):
        raise AssertionError("callable analysis must not follow wrapped metadata")


class HostileDefault:
    def __repr__(self):
        raise AssertionError("callable analysis must not render hostile defaults")


class HostileSignature(inspect.Signature):
    @property
    def parameters(self):
        raise AssertionError("callable analysis must not inspect custom signatures")


class HostileParameter(inspect.Parameter):
    hooks_active = False

    @property
    def default(self):
        if self.hooks_active:
            raise AssertionError("callable analysis must not inspect custom parameters")
        return inspect.Parameter.default.__get__(self, type(self))


def test_callable_algorithm_module_function(requirement_targets):
    result = code.analyze(requirement_targets.plain_importable_function, algorithms=("callables",))
    fact = result.facts_of_kind("callable")[0]

    assert fact.data["qualname"] == "plain_importable_function"
    assert fact.data["importable"] is True
    assert fact.data["signature"] == "(value=1)"


def test_callable_algorithm_lambda_and_local_not_importable(requirement_targets):
    local = requirement_targets.make_local_training_function()

    lambda_result = code.analyze(requirement_targets.local_lambda_with_annotation, algorithms=("callables",))
    local_result = code.analyze(local, algorithms=("callables",))

    assert lambda_result.facts_of_kind("callable")[0].data["is_lambda"] is True
    assert lambda_result.diagnostics_of_code("dryml.code.not_importable")
    assert local_result.diagnostics_of_code("dryml.code.not_importable")


def test_callable_algorithm_methods_and_callable_instance(requirement_targets):
    bound = code.analyze(requirement_targets.LightningModel().train, algorithms=("callables",))
    instance = code.analyze(CallableInstance(), algorithms=("callables",))

    assert bound.facts_of_kind("callable")[0].data["is_bound_method"] is True
    assert instance.facts_of_kind("callable")[0].data["is_callable_instance"] is True


def test_old_analyze_callable_compatibility(requirement_targets):
    info = analyze_callable(requirement_targets.plain_importable_function)

    assert info.is_function is True
    assert info.qualname == "plain_importable_function"


def test_callable_algorithm_avoids_hostile_callable_and_metaclass_metadata():
    hostile_result = code.analyze(HostileCallable(), algorithms=("callables",))
    metaclass_result = code.analyze(HostileMetaCallable(), algorithms=("callables",))

    assert hostile_result.facts_of_kind("callable")
    assert metaclass_result.facts_of_kind("callable")
    assert not hostile_result.diagnostics_of_code("dryml.code.algorithm_failed")
    assert not metaclass_result.diagnostics_of_code("dryml.code.algorithm_failed")


def test_callable_algorithm_does_not_follow_hostile_wrapped_metadata():
    def target(value=1):
        return value

    target.__wrapped__ = HostileWrapped()
    result = code.analyze(target, algorithms=("callables",))

    assert result.ok
    assert result.facts_of_kind("callable")[0].data["signature"] == "(value=1)"


def test_callable_algorithm_does_not_render_hostile_defaults():
    hostile_default = HostileDefault()

    def target(value=hostile_default):
        return value

    result = code.analyze(target, algorithms=("callables",))

    assert result.ok
    assert result.facts_of_kind("callable")[0].data["signature"] == "(value=...)"


def test_callable_algorithm_preserves_safely_renderable_signature_metadata():
    def target(value: int = ()) -> str:
        return str(value)

    result = code.analyze(target, algorithms=("callables",))

    assert result.ok
    assert result.facts_of_kind("callable")[0].data["signature"] == "(value: 'int' = ()) -> 'str'"


def test_callable_algorithm_does_not_inspect_custom_signature_types():
    def target(value=1):
        return value

    target.__signature__ = HostileSignature()
    result = code.analyze(target, algorithms=("callables",))

    assert result.ok
    assert result.facts_of_kind("callable")[0].data["signature"] is None
    assert result.diagnostics_of_code("dryml.code.signature_unavailable")


def test_callable_algorithm_does_not_inspect_custom_parameter_types():
    def target(value=1):
        return value

    parameter = HostileParameter("value", inspect.Parameter.POSITIONAL_OR_KEYWORD)
    target.__signature__ = inspect.Signature((parameter,))
    HostileParameter.hooks_active = True
    try:
        result = code.analyze(target, algorithms=("callables",))
    finally:
        HostileParameter.hooks_active = False

    assert result.ok
    assert result.facts_of_kind("callable")[0].data["signature"] is None
    assert result.diagnostics_of_code("dryml.code.signature_unavailable")
