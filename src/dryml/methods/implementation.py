"""Inspectable callable carriers for authored Method implementations."""

from __future__ import annotations

import types
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable
from .errors import ImplementationDeclarationError, ImplementationSelectionError
from .signature import MethodCallNode, runtime_node_for_constraint, satisfies
from .traits import Traits


_ACTIVE_DIRECT_RECEIVERS: ContextVar[tuple[int, ...]] = ContextVar(
    "dryml_method_active_direct_receivers",
    default=(),
)


def direct_invocation_active(receiver: object) -> bool:
    """Return whether this execution context is inside the receiver's direct target."""

    return id(receiver) in _ACTIVE_DIRECT_RECEIVERS.get()


def invoke_direct_descriptor(
    descriptor: object,
    receiver: object,
    receiver_type: type,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    *,
    name: str,
) -> object:
    """Invoke a captured direct target while marking cooperative-super context."""

    active = _ACTIVE_DIRECT_RECEIVERS.get()
    token = _ACTIVE_DIRECT_RECEIVERS.set((*active, id(receiver)))
    try:
        return invoke_descriptor(
            descriptor,
            receiver,
            receiver_type,
            args,
            kwargs,
            name=name,
        )
    finally:
        _ACTIVE_DIRECT_RECEIVERS.reset(token)


def _descriptor_kind(descriptor: object) -> type | None:
    """Return the supported native binding owner for one raw declaration."""

    descriptor_type = type(descriptor)
    if descriptor_type is types.FunctionType:
        return types.FunctionType
    if issubclass(descriptor_type, staticmethod):
        return staticmethod
    if issubclass(descriptor_type, classmethod):
        return classmethod
    return None


def ensure_supported_descriptor(descriptor: object, *, name: str) -> None:
    """Reject a declaration that cannot be bound through native descriptor rules.

    Args:
        descriptor: Raw class-namespace declaration retained by a Method.
        name: Bounded declaration name used only for diagnostics.

    Raises:
        ImplementationDeclarationError: If the declaration is not an instance,
            static, or class method supported by the Method catalog.
    """

    if _descriptor_kind(descriptor) is None:
        raise ImplementationDeclarationError(
            f"Method implementation {name!r} uses an unsupported descriptor."
        )


def invoke_descriptor(
    descriptor: object,
    receiver: object,
    receiver_type: type,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    *,
    name: str,
) -> object:
    """Bind one retained raw descriptor only at invocation time and call it.

    Args:
        descriptor: Raw declaration previously retained in a catalog carrier.
        receiver: Method instance receiving the invocation.
        receiver_type: Runtime type used for ordinary classmethod binding.
        args: Logical positional Method arguments.
        kwargs: Logical keyword Method arguments.
        name: Bounded declaration name for an unsupported-descriptor diagnostic.

    Returns:
        The raw target's return value.

    Raises:
        ImplementationDeclarationError: If the retained declaration has an
            unsupported descriptor form. No target is invoked in that case.

    Side Effects:
        Performs normal Python descriptor binding immediately before user code
        runs. Catalog construction itself does not bind descriptors.
    """

    kind = _descriptor_kind(descriptor)
    if kind is None:
        ensure_supported_descriptor(descriptor, name=name)
        raise AssertionError("unsupported descriptors always raise")
    bound = kind.__get__(descriptor, receiver, receiver_type)
    return bound(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class MethodImplementation:
    """One immutable, inspectable authored implementation and its invocation carrier.

    Args:
        name: Stable class declaration name for this implementation.
        target: Exact raw descriptor stored in the authoring class namespace.
        traits: Complete closed trait set supplied by the author.

    Calling the carrier validates its retained first-input constraint and then
    binds its private raw descriptor to its private Method receiver using ordinary
    Python rules. A Method-local composite may instead retain a private selected
    invoker; the raw ``target`` remains inspectable and catalog inspection never
    invokes it.
    """

    name: str
    target: object
    traits: Traits
    _descriptor: object | None = field(default=None, repr=False, compare=False)
    _receiver: object | None = field(default=None, repr=False, compare=False)
    _receiver_type: type | None = field(default=None, repr=False, compare=False)
    _input_spec: MethodCallNode | None = field(default=None, repr=False, compare=False)
    _direct: bool = field(default=False, repr=False, compare=False)
    _invoker: Callable[..., object] | None = field(default=None, repr=False, compare=False)

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Bind and invoke this authored target with unchanged logical arguments.

        Args:
            *args: Logical positional Method arguments.
            **kwargs: Logical keyword Method arguments.

        Returns:
            The authored target's return value.

        Raises:
            ImplementationDeclarationError: If the retained descriptor is not a
                supported instance, static, or class method declaration, or if
                a manually constructed carrier has no Method binding.
            ImplementationSelectionError: If this carrier was returned by
                ``find_implementation`` and its first runtime argument is
                missing, unsupported, or conflicts with its retained input
                specification. The target is not invoked on this failure.
        """

        if self._descriptor is None or self._receiver is None or self._receiver_type is None:
            raise ImplementationDeclarationError(
                f"Method implementation {self.name!r} is not bound to a Method instance."
            )
        if self._input_spec is not None:
            if not args:
                raise ImplementationSelectionError("conflict")
            try:
                valid = satisfies(
                    self._input_spec,
                    runtime_node_for_constraint(args[0], self._input_spec),
                )
            except TypeError as error:
                raise ImplementationSelectionError("conflict") from error
            if not valid:
                raise ImplementationSelectionError("conflict")
        if self._invoker is not None:
            return self._invoker(*args, **kwargs)
        if not self._direct:
            return invoke_descriptor(
                self._descriptor,
                self._receiver,
                self._receiver_type,
                args,
                kwargs,
                name=self.name,
            )
        return invoke_direct_descriptor(
            self._descriptor,
            self._receiver,
            self._receiver_type,
            args,
            kwargs,
            name=self.name,
        )


__all__ = ["MethodImplementation"]
