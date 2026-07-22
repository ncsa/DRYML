"""Semantic model objects and descriptor-safe method protocols."""

from __future__ import annotations

import types
from typing import Any

from dryml.core.methods.compiler_info import CompilerInfo
from dryml.core.methods.method import Method, traits
from dryml.core.methods.traits import BatchMode, Traits


_MISSING = object()


def _static_attribute(value: Any, name: str, default: Any = None) -> Any:
    """Read an instance or class attribute without invoking Python hooks."""

    value_type = type(value)
    try:
        value_type_mro = type.__getattribute__(value_type, "__mro__")
    except (AttributeError, TypeError):
        return default

    if type in value_type_mro:
        return _static_type_attribute(value, name, default)

    namespace_descriptor = _static_type_attribute(
        value_type, "__dict__", _MISSING
    )
    if type(namespace_descriptor) is types.GetSetDescriptorType:
        try:
            namespace = object.__getattribute__(value, "__dict__")
        except (AttributeError, TypeError):
            namespace = None
        if type(namespace) is dict and name in namespace:
            return namespace[name]

    return _static_type_attribute(value_type, name, default)


def _static_type_attribute(cls: type, name: str, default: Any) -> Any:
    """Read a class MRO directly through ``type``, bypassing metaclasses."""

    try:
        mro = type.__getattribute__(cls, "__mro__")
    except (AttributeError, TypeError):
        return default
    if type(mro) is not tuple:
        return default

    for base in mro:
        try:
            namespace = type.__getattribute__(base, "__dict__")
        except (AttributeError, TypeError):
            return default
        if type(namespace) is not types.MappingProxyType:
            return default
        candidate = namespace.get(name, _MISSING)
        if candidate is not _MISSING:
            return candidate
    return default


def descriptor_function(value: Any) -> Any:
    """Return a descriptor's plain function without invoking descriptor binding.

    The protocol intentionally depends only on a static ``__func__`` attribute,
    allowing higher-level descriptors to participate without a core import of
    their policy package.
    """

    if type(value) in {staticmethod, classmethod}:
        return object.__getattribute__(value, "__func__")
    if type(value) is types.MethodType:
        return object.__getattribute__(value, "__func__")
    candidate = _static_attribute(value, "__func__")
    return candidate if type(candidate) is types.FunctionType else value


def bound_method_parts(value: Any) -> tuple[Any, Any] | None:
    """Return ``(self, func)`` for builtin or protocol-compatible bindings."""

    if type(value) is types.MethodType:
        return (
            object.__getattribute__(value, "__self__"),
            object.__getattribute__(value, "__func__"),
        )
    marker = _static_type_attribute(
        type(value), "__dryml_bound_method__", False
    )
    if marker is not True:
        return None
    receiver = _static_attribute(value, "__self__", _MISSING)
    func = _static_attribute(value, "__func__", _MISSING)
    if receiver is _MISSING:
        return None
    if type(func) is not types.FunctionType:
        return None
    return receiver, func


__all__ = [
    "BatchMode",
    "CompilerInfo",
    "Method",
    "Traits",
    "bound_method_parts",
    "descriptor_function",
    "traits",
]
