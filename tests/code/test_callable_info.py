"""Tests for closed, non-invoking callable inspection."""

from __future__ import annotations

import inspect
import types

import pytest

from dryml.code import InvalidTargetError, analyze_callable


def plain(value: int, *, flag: str = "x") -> int:
    """Return a value without being invoked by these tests."""

    return value


class MethodOwner:
    """Provide a normal Python instance method."""

    def method(self, value: int) -> int:
        """Return a value without side effects."""

        return value


class CallableInstance:
    """Provide a supported raw Python ``__call__`` method."""

    def __call__(self, value: int) -> int:
        """Return a value without side effects."""

        return value


def test_analyze_callable_preserves_supported_callable_metadata() -> None:
    """Normal forms retain request-local handles and standard signatures."""

    function = analyze_callable(plain)
    method = analyze_callable(MethodOwner().method)
    instance = CallableInstance()
    callable_instance = analyze_callable(instance)

    assert function.func is plain
    assert function.signature == inspect.signature(plain)
    assert function.is_function
    assert method.bound_self.__class__ is MethodOwner
    assert method.is_bound_method
    assert callable_instance.func is CallableInstance.__call__
    assert callable_instance.bound_self is instance
    assert callable_instance.is_callable_instance


def test_analyze_callable_does_not_follow_custom_wrappers_or_signatures() -> None:
    """User-controlled signature and wrapper protocols are rejected statically."""

    def wrapped() -> None:
        """Supply a local function for metadata mutation."""

    wrapped.__wrapped__ = object()  # type: ignore[attr-defined]
    with pytest.raises(InvalidTargetError, match="unsupported callable") as wrapped_error:
        analyze_callable(wrapped)

    def signed() -> None:
        """Supply a local function for metadata mutation."""

    signed.__signature__ = inspect.Signature()  # type: ignore[attr-defined]
    with pytest.raises(InvalidTargetError, match="unsupported callable") as signature_error:
        analyze_callable(signed)

    assert wrapped_error.value.code == "target.invalid"
    assert signature_error.value.code == "target.invalid"


def test_analyze_callable_rejects_dynamic_lookup_without_invoking_it() -> None:
    """Callable-looking objects cannot opt into analysis through dynamic hooks."""

    class Dynamic:
        def __getattribute__(self, name: str) -> object:
            raise AssertionError(f"dynamic lookup invoked for {name}")

    with pytest.raises(InvalidTargetError, match="unsupported callable"):
        analyze_callable(Dynamic())


@pytest.mark.skipif("__annotate__" not in types.FunctionType.__dict__, reason="deferred annotation hook requires Python 3.14+")
def test_analyze_callable_rejects_deferred_annotations_without_evaluation() -> None:
    """Python 3.14 deferred annotation hooks remain uninvoked."""

    evaluated: list[bool] = []

    def marker() -> type[int]:
        evaluated.append(True)
        raise RuntimeError("/private/path annotation-secret")

    namespace: dict[str, object] = {"marker": marker}
    code = compile(
        "def subject(value: marker()):\n    return value\n",
        "<deferred-annotations>",
        "exec",
        dont_inherit=True,
    )
    exec(code, namespace)

    with pytest.raises(InvalidTargetError, match="unsupported callable"):
        analyze_callable(namespace["subject"])  # type: ignore[arg-type]
    assert evaluated == []


@pytest.mark.skipif("__annotate__" not in types.FunctionType.__dict__, reason="deferred annotation hook requires Python 3.14+")
def test_analyze_callable_accepts_stringized_deferred_annotations() -> None:
    """Compiler-stringized annotations remain inspectable without evaluation."""

    evaluated: list[bool] = []

    def marker() -> type[int]:
        evaluated.append(True)
        raise RuntimeError("annotation evaluated")

    namespace: dict[str, object] = {"marker": marker}
    exec(
        "from __future__ import annotations\ndef subject(value: marker()) -> int:\n    return value\n",
        namespace,
    )

    info = analyze_callable(namespace["subject"])  # type: ignore[arg-type]

    assert info.signature.parameters["value"].annotation == "marker()"
    assert info.signature.return_annotation == "int"
    assert evaluated == []
