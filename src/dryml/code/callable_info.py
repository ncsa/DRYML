"""Closed, non-invoking metadata inspection for supported Python callables."""

from __future__ import annotations

import inspect
import types
from dataclasses import dataclass
from typing import Any, Callable

from .errors import InvalidTargetError


def _function_slot(func: types.FunctionType, name: str) -> object:
    """Read one exact function slot without invoking deferred annotations."""

    return types.FunctionType.__dict__[name].__get__(func, types.FunctionType)


def _type_slot(cls: type, name: str) -> object:
    """Read one built-in type slot without metaclass descriptor lookup."""

    return type.__dict__[name].__get__(cls, type(cls))


def _module_namespace(module: types.ModuleType) -> dict[str, object]:
    """Return a module's storage without invoking subclass descriptors."""

    return types.ModuleType.__dict__["__dict__"].__get__(module, type(module))


def _function_metadata(func: types.FunctionType) -> tuple[str | None, str | None]:
    """
    Read native provenance without consulting wrapper or annotation hooks.
    """
    raw_module = _function_slot(func, "__module__")
    raw_qualname = _function_slot(func, "__qualname__")
    module = raw_module if type(raw_module) is str else None
    qualname = raw_qualname if type(raw_qualname) is str else None
    return qualname, module


def _native_signature(func: types.FunctionType) -> inspect.Signature:
    """Derive a signature from native code/default slots without annotations.

    ``inspect.signature`` consults user-controlled ``__signature__`` even when
    wrapper following is disabled. Static admission instead needs only native
    parameter/default evidence and intentionally omits annotations.
    """

    code = _function_slot(func, "__code__")
    defaults = _function_slot(func, "__defaults__")
    keyword_defaults = _function_slot(func, "__kwdefaults__")
    if (
        type(code) is not types.CodeType
        or defaults is not None and type(defaults) is not tuple
        or keyword_defaults is not None and type(keyword_defaults) is not dict
    ):
        raise InvalidTargetError("unsupported callable")
    positional = code.co_argcount
    positional_only = getattr(code, "co_posonlyargcount", 0)
    names = code.co_varnames
    values = defaults if type(defaults) is tuple else ()
    parameters: list[inspect.Parameter] = []
    required = positional - len(values)
    for index in range(positional):
        kind = (inspect.Parameter.POSITIONAL_ONLY if index < positional_only
                else inspect.Parameter.POSITIONAL_OR_KEYWORD)
        default = inspect.Parameter.empty if index < required else values[
            index - required]
        parameters.append(
            inspect.Parameter(names[index], kind, default=default))
    keyword_offset = positional
    keyword = keyword_defaults if type(keyword_defaults) is dict else {}
    if code.co_flags & inspect.CO_VARARGS:
        parameters.append(
            inspect.Parameter(
                names[keyword_offset + code.co_kwonlyargcount],
                inspect.Parameter.VAR_POSITIONAL,
            ))
    for index in range(code.co_kwonlyargcount):
        name = names[keyword_offset + index]
        parameters.append(
            inspect.Parameter(name,
                              inspect.Parameter.KEYWORD_ONLY,
                              default=keyword.get(name,
                                                  inspect.Parameter.empty)))
    offset = keyword_offset + code.co_kwonlyargcount
    if code.co_flags & inspect.CO_VARARGS:
        offset += 1
    if code.co_flags & inspect.CO_VARKEYWORDS:
        parameters.append(
            inspect.Parameter(names[offset], inspect.Parameter.VAR_KEYWORD))
    return inspect.Signature(parameters)


def _raw_call_descriptor(cls: type) -> object | None:
    """Find ``__call__`` in class dictionaries without binding a descriptor."""

    for base in _type_slot(cls, "__mro__"):  # type: ignore[union-attr]
        namespace = _type_slot(base, "__dict__")
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
            signature=_native_signature(obj),
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
            signature=_native_signature(func),
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
        signature=_native_signature(func),
        qualname=qualname,
        module=module,
        is_bound_method=False,
        is_function=False,
        is_callable_instance=True,
    )


__all__ = ["CallableInfo", "analyze_callable"]
