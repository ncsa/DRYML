"""Closed, non-invoking metadata inspection for supported Python callables."""

from __future__ import annotations

import inspect
import types
from dataclasses import dataclass
from typing import Any, Callable

from .errors import InvalidTargetError


def _function_metadata(func: types.FunctionType) -> tuple[str | None, str | None]:
    """Read non-evaluating Python-function provenance after hook rejection."""

    if "__signature__" in func.__dict__ or "__wrapped__" in func.__dict__:
        raise InvalidTargetError("unsupported callable")
    annotations = func.__annotations__
    if type(annotations) is not dict:
        raise InvalidTargetError("unsupported callable")
    module = func.__module__ if type(func.__module__) is str else None
    qualname = func.__qualname__ if type(func.__qualname__) is str else None
    return qualname, module


def _raw_call_descriptor(cls: type) -> object | None:
    """Find ``__call__`` in class dictionaries without binding a descriptor."""

    for base in type.__getattribute__(cls, "__mro__"):
        namespace = type.__getattribute__(base, "__dict__")
        if "__call__" in namespace:
            return namespace["__call__"]
    return None


def _function_from_descriptor(descriptor: object) -> types.FunctionType | None:
    """Return an admitted raw function from a built-in descriptor wrapper."""

    if type(descriptor) is types.FunctionType:
        return descriptor
    if type(descriptor) in (staticmethod, classmethod):
        function = descriptor.__func__
        return function if type(function) is types.FunctionType else None
    return None


@dataclass(frozen=True, slots=True)
class CallableInfo:
    """Request-local metadata for one admitted Python callable.

    Args:
        original: Original caller-provided callable handle.
        func: Raw Python function implementing the callable behavior.
        bound_self: Bound receiver or callable instance when applicable.
        signature: Non-unwrapped signature of ``func`` with annotations unevaluated.
        qualname: Raw function qualified name when available.
        module: Raw function module name when available.
        is_bound_method: Whether this is a bound Python method.
        is_function: Whether this is a direct Python function.
        is_callable_instance: Whether a raw class ``__call__`` implements it.

    Side Effects:
        None. This value retains request-local live handles and must not be
        copied into framework-created provenance.
    """

    original: object
    func: Callable[..., Any]
    bound_self: object | None
    signature: inspect.Signature
    qualname: str | None
    module: str | None
    is_bound_method: bool
    is_function: bool
    is_callable_instance: bool


def analyze_callable(obj: Callable[..., Any]) -> CallableInfo:
    """Classify an admitted callable without invoking dynamic protocols.

    Args:
        obj: Direct Python function, bound Python method, or callable instance
            whose raw class ``__call__`` is an admitted Python function.

    Returns:
        Immutable request-local callable metadata and a standard signature.

    Raises:
        InvalidTargetError: If the callable relies on a custom wrapper,
            signature, annotation container, descriptor, built-in, or dynamic
            lookup protocol.

    Side Effects:
        None. Target bodies, descriptors, and caller-controlled reflection hooks
        are never invoked.
    """

    if type(obj) is types.FunctionType:
        qualname, module = _function_metadata(obj)
        return CallableInfo(
            original=obj,
            func=obj,
            bound_self=None,
            signature=inspect.signature(obj, follow_wrapped=False, eval_str=False),
            qualname=qualname,
            module=module,
            is_bound_method=False,
            is_function=True,
            is_callable_instance=False,
        )

    if type(obj) is types.MethodType and type(obj.__func__) is types.FunctionType:
        func = obj.__func__
        qualname, module = _function_metadata(func)
        return CallableInfo(
            original=obj,
            func=func,
            bound_self=obj.__self__,
            signature=inspect.signature(func, follow_wrapped=False, eval_str=False),
            qualname=qualname,
            module=module,
            is_bound_method=True,
            is_function=False,
            is_callable_instance=False,
        )

    cls = type(obj)
    descriptor = _raw_call_descriptor(cls)
    func = _function_from_descriptor(descriptor) if descriptor is not None else None
    if func is None:
        raise InvalidTargetError("unsupported callable")
    qualname, module = _function_metadata(func)
    return CallableInfo(
        original=obj,
        func=func,
        bound_self=obj,
        signature=inspect.signature(func, follow_wrapped=False, eval_str=False),
        qualname=qualname,
        module=module,
        is_bound_method=False,
        is_function=False,
        is_callable_instance=True,
    )


__all__ = ["CallableInfo", "analyze_callable"]
