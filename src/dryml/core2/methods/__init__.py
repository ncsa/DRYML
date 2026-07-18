"""Semantic model objects and descriptor-safe method protocols."""

from __future__ import annotations

import inspect
import types
from typing import Any

from dryml.core2.methods.compiler_info import CompilerInfo
from dryml.core2.methods.method import Method, traits
from dryml.core2.methods.traits import BatchMode, Traits


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
    candidate = inspect.getattr_static(value, "__func__", None)
    return candidate if type(candidate) is types.FunctionType else value


def bound_method_parts(value: Any) -> tuple[Any, Any] | None:
    """Return ``(self, func)`` for builtin or protocol-compatible bindings."""

    if type(value) is types.MethodType:
        return (
            object.__getattribute__(value, "__self__"),
            object.__getattribute__(value, "__func__"),
        )
    marker = inspect.getattr_static(type(value), "__dryml_bound_method__", False)
    if marker is not True:
        return None
    receiver = object.__getattribute__(value, "__self__")
    func = object.__getattribute__(value, "__func__")
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
