"""Passive descriptions of callable relationships owned by core invocation.

The values describe only relationships core itself establishes for its single
invocation boundary. They deliberately do not implement generic wrapper
analysis, inspect arbitrary decorator attributes, or activate descriptors.
"""

from __future__ import annotations

import inspect
import types
from dataclasses import dataclass
from typing import Any, Callable, Literal


CallableOwner = Literal["ordinary", "function", "method", "managed"]
NativeModality = Literal[
    "sync", "coroutine", "generator", "async_generator", "unknown",
]
_TYPE_MRO = type.__dict__["__mro__"]
_TYPE_DICT = type.__dict__["__dict__"]


class _FunctionInvocationOwner:
    """Bind a ``function`` wrapper to its target without transport policy.

    Args:
        target: The callable passed directly to :func:`dryml.core.function`.

    The wrapper is bound after construction.  A copied attribute retains the
    original wrapper identity and therefore cannot authorize another wrapper.
    """

    __slots__ = ("target", "wrapper")

    def __init__(self, target: Callable[..., Any]) -> None:
        """Store the trusted target before wrapper construction."""

        object.__setattr__(self, "target", target)
        object.__setattr__(self, "wrapper", None)

    def __setattr__(self, name: str, value: object) -> None:
        """Keep the owner relationship immutable once its wrapper is bound."""

        if name == "target" and hasattr(self, "target"):
            raise AttributeError("function invocation target is immutable")
        if name == "wrapper" and self.wrapper is not None:
            raise AttributeError("function invocation wrapper is immutable")
        object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class CallableDescription:
    """Passive core-owned facts for one submitted callable.

    Args:
        original: Exact callable supplied by the caller and retained for direct
            invocation semantics.
        raw_target: Exact target used by core Execute for a recognized owner;
            otherwise ``original``.
        owner: The established core invocation owner.
        declaration_carriers: Ordered callable layers carrying declarations;
            duplicate identities are retained only once.
        native_modality: Established native root modality, or ``"unknown"``.

    The description owns no callable, descriptor, receiver, or declaration
    activation policy beyond these borrowed references. Unknown wrappers remain
    ordinary roots rather than being unwrapped through ``__wrapped__`` or
    copied owner fields.
    """

    original: Callable[..., Any]
    raw_target: Callable[..., Any]
    owner: CallableOwner
    declaration_carriers: tuple[object, ...]
    native_modality: NativeModality


def describe_callable(target: Callable[..., Any]) -> CallableDescription:
    """Describe a callable using only established core invocation ownership.

    Args:
        target: Callable submitted to core Execute or retained for direct use.

    Returns:
        A passive description whose raw target agrees with core transport's
        invocation owner.  Native functions and bound methods expose their code
        flags; arbitrary callable objects expose ``"unknown"`` without invoking
        ``__call__`` or descriptor machinery.

    Raises:
        TypeError: If ``target`` is not callable.

    Side Effects:
        None. The function never calls ``target``, imports code analysis,
        selects Method implementations, invokes constructors, follows wrapped
        or reads arbitrary decorator metadata.
    """

    if not callable(target):
        raise TypeError("callable inspection requires a callable")
    function, receiver = _function_and_receiver(target)
    if function is not None:
        description = _function_owner_description(target, function, receiver)
        if description is not None:
            return description
        return CallableDescription(
            target, target, "ordinary", (target,), _native_modality(target),
        )
    owner = _known_instance_owner(target)
    if owner is not None:
        return CallableDescription(
            target,
            target,
            owner,
            _owner_carriers(target, owner),
            _owner_modality(target),
        )
    return _ordinary(target)


def _function_and_receiver(
        target: Callable[..., Any],
) -> tuple[types.FunctionType | None, object | None]:
    """Return a native function and optional receiver without lookup."""

    if type(target) is types.FunctionType:
        return target, None
    if type(target) is types.MethodType and type(
            target.__func__) is types.FunctionType:
        return target.__func__, target.__self__
    return None, None


def _function_owner_description(
        original: Callable[..., Any], function: types.FunctionType,
        receiver: object | None,
) -> CallableDescription | None:
    """Follow only identity- and closure-proven function owners."""

    carriers: list[object] = []
    seen: set[int] = set()
    current = function
    current_receiver = receiver
    while True:
        if id(current) in seen:
            return None
        seen.add(id(current))
        owner = current.__dict__.get("__dryml_function_invocation_owner__")
        if type(
                owner
        ) is not _FunctionInvocationOwner or owner.wrapper is not current:
            if not carriers:
                return None
            raw: Callable[..., Any] = (
                types.MethodType(current, current_receiver)
                if current_receiver is not None else current
            )
            carriers.append(current)
            return CallableDescription(
                original, raw, "function", _unique_carriers(*carriers),
                _native_modality(raw),
            )
        raw = owner.target
        if not callable(raw) or _closure_target(current) is not raw:
            return None
        carriers.append(current)
        next_function, next_receiver = _function_and_receiver(raw)
        if next_function is None:
            carriers.append(raw)
            return CallableDescription(
                original, raw, "function", _unique_carriers(*carriers),
                _native_modality(raw),
            )
        current = next_function
        # A wrapper bound from a class method carries its receiver through
        # every unwrapped native function layer.
        current_receiver = (current_receiver if next_receiver is None
                            else next_receiver)


def _closure_target(function: types.FunctionType) -> object | None:
    """
    Read the function decorator's closed-over target without dynamic lookup.
    """

    closure = function.__closure__
    if closure is None:
        return None
    for name, cell in zip(function.__code__.co_freevars, closure):
        if name == "target":
            try:
                return cell.cell_contents
            except ValueError:
                return None
    return None


def _known_instance_owner(target: Callable[..., Any]) -> CallableOwner | None:
    """Recognize only the concrete core owners already used by transport."""

    classes = _native_mro(type(target))
    from dryml.managed.descriptor import _BoundComposite, _BoundOperation
    from dryml.methods.method import Method

    if Method in classes:
        return "method"
    if _BoundOperation in classes or _BoundComposite in classes:
        return "managed"
    return None


def _owner_carriers(
        target: Callable[..., Any], owner: CallableOwner,
) -> tuple[object, ...]:
    """Return exact declaration carriers for a known invocation owner.

    Managed bound views are temporary call objects, while their descriptor owns
    the passive declarations that must remain visible to Dispatch inspection.
    """

    if owner != "managed":
        return (target,)
    descriptor = object.__getattribute__(target, "_descriptor")
    from dryml.managed.descriptor import ManagedOperation

    if type(descriptor) is ManagedOperation:
        carriers = [target, descriptor, descriptor._target, descriptor._executable]
        composite = getattr(target, "_composite", None)
        outer = getattr(composite, "_outer", None)
        if type(outer) is types.FunctionType:
            carriers.append(outer)
        return _unique_carriers(*carriers)
    return (target,)


def _ordinary(target: Callable[..., Any]) -> CallableDescription:
    """Describe an unsupported composition as its actual ordinary root."""

    return CallableDescription(
        target, target, "ordinary", (target,), _native_modality(target),
    )


def _unique_carriers(*values: object) -> tuple[object, ...]:
    """Preserve declaration-carrier order while deduplicating only identity."""

    result: list[object] = []
    seen: set[int] = set()
    for value in values:
        if id(value) not in seen:
            seen.add(id(value))
            result.append(value)
    return tuple(result)


def _native_modality(target: object) -> NativeModality:
    """Read native code flags without following wrapper/signature metadata."""

    function, _ = _function_and_receiver(target) if callable(target) else (
        None, None)
    if function is None and callable(target):
        for cls in _native_mro(type(target)):
            candidate = _native_dict(cls).get("__call__")
            if candidate is not None:
                function = candidate if type(
                    candidate) is types.FunctionType else None
                break
    if function is None:
        return "unknown"
    return _native_function_modality(function)


def _owner_modality(target: object) -> NativeModality:
    """Read a known owner's established invocation seam without binding it."""

    for cls in _native_mro(type(target)):
        candidate = _native_dict(cls).get("__dryml_execute_invoke__")
        if candidate is not None:
            return _native_function_modality(candidate)
    return "unknown"


def _native_function_modality(value: object) -> NativeModality:
    """Classify an exact native function without descriptor activation."""

    if type(value) is not types.FunctionType:
        return "unknown"
    flags = value.__code__.co_flags
    if flags & inspect.CO_ASYNC_GENERATOR:
        return "async_generator"
    if flags & inspect.CO_COROUTINE:
        return "coroutine"
    if flags & inspect.CO_GENERATOR:
        return "generator"
    return "sync"


def _native_mro(cls: type) -> tuple[type, ...]:
    """Read a class MRO from the native type slot without metaclass lookup."""

    return _TYPE_MRO.__get__(cls, type(cls))


def _native_dict(cls: type) -> dict[str, object]:
    """
    Read a class namespace from the native type slot without metaclass lookup.
    """

    return _TYPE_DICT.__get__(cls, type(cls))


__all__ = [
    "CallableDescription",
    "CallableOwner",
    "NativeModality",
    "describe_callable",
]
