"""Inspectable callable carriers for authored Method implementations."""

from __future__ import annotations

import types
import weakref
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Callable

from .errors import ImplementationDeclarationError, ImplementationSelectionError, MethodError
from .signature import MethodCallNode, runtime_node_for_constraint, satisfies
from .traits import Traits


_ACTIVE_DIRECT_RECEIVERS: ContextVar[tuple[tuple[int, int], ...]] = ContextVar(
    "dryml_method_active_direct_receivers",
    default=(),
)


def direct_invocation_active(receiver: object, descriptor: object) -> bool:
    """Return whether a different direct target is active for cooperative ``super``.

    Args:
        receiver: Method instance receiving the prospective direct call.
        descriptor: Raw direct descriptor captured by the gateway owner.

    Returns:
        ``True`` only when a different direct descriptor for ``receiver`` is
        active in this logical Method boundary.

    Independent recursion through the active descriptor intentionally returns
    ``False`` so it establishes its own normalization boundary.
    """

    active = _ACTIVE_DIRECT_RECEIVERS.get()
    receiver_id, descriptor_id = id(receiver), id(descriptor)
    return any(active_receiver == receiver_id for active_receiver, _ in active) and (
        receiver_id,
        descriptor_id,
    ) not in active


def invoke_direct_bound(
    target: Callable[..., object],
    receiver: object,
    descriptor: object,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    """Invoke a bound direct target within its cooperative-super context.

    Args:
        target: Selected callable to invoke without descriptor lookup.
        receiver: Method instance defining the logical direct-call boundary.
        descriptor: Raw descriptor identity distinguishing recursion from ``super``.
        args: Already-normalized positional arguments.
        kwargs: Already-normalized keyword arguments.

    Returns:
        The target's raw result before return normalization.

    Side Effects:
        Installs a ContextVar marker for the dynamic extent of target execution.
    """

    active = _ACTIVE_DIRECT_RECEIVERS.get()
    token = _ACTIVE_DIRECT_RECEIVERS.set((*active, (id(receiver), id(descriptor))))
    try:
        return target(*args, **kwargs)
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


def _bound_descriptor(
    descriptor: object,
    receiver: object,
    receiver_type: type,
    *,
    name: str,
) -> Callable[..., object]:
    """Bind one supported descriptor through normal Python descriptor semantics."""

    kind = _descriptor_kind(descriptor)
    if kind is None:
        ensure_supported_descriptor(descriptor, name=name)
        raise AssertionError("unsupported descriptors always raise")
    bound = kind.__get__(descriptor, receiver, receiver_type)
    if not callable(bound):
        raise ImplementationDeclarationError(
            f"Method implementation {name!r} did not bind to a callable target."
        )
    return bound


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
    """

    return _bound_descriptor(descriptor, receiver, receiver_type, name=name)(*args, **kwargs)


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

    return invoke_direct_bound(
        _bound_descriptor(descriptor, receiver, receiver_type, name=name),
        receiver,
        descriptor,
        args,
        kwargs,
    )


@dataclass(slots=True)
class SelectedDescriptorAdapter:
    """Run one selected descriptor through its core signature plan.

    The selected target is bound once, to a weak receiver proxy, when compiling
    its signature. The adapter retains no strong Method reference and uses its
    raw descriptor for invocation, so the existing weak preparation cache remains
    collectable while carrier, eager, learning, and cached calls share one path.
    """

    name: str
    descriptor: object | None
    receiver_ref: weakref.ReferenceType[object]
    receiver_type: type
    direct: bool
    invoker: Callable[..., object] | None
    plan: object

    @classmethod
    def create(
        cls,
        name: str,
        descriptor: object | None,
        receiver: object,
        receiver_type: type,
        direct: bool,
        invoker: Callable[..., object] | None,
    ) -> "SelectedDescriptorAdapter":
        """Bind and compile a selected target with a lazy core-signature import.

        Args:
            name: Bounded declaration name for diagnostics.
            descriptor: Selected raw descriptor, unless ``invoker`` is supplied.
            receiver: Method instance selected for this invocation.
            receiver_type: Runtime Method type for descriptor binding.
            direct: Whether target invocation establishes a cooperative boundary.
            invoker: Optional Method-local selected callable.

        Returns:
            A reusable adapter retaining the selected target's compiled signature.

        Raises:
            ImplementationDeclarationError: If descriptor binding or weak receiver
                ownership is unavailable.
            SignatureError: If lazy core signature compilation rejects annotations.
        """

        try:
            receiver_ref = weakref.ref(receiver)
        except TypeError as error:
            raise ImplementationDeclarationError(
                f"Method implementation {name!r} requires a weak-referenceable receiver."
            ) from error
        if invoker is not None:
            target = invoker
        elif descriptor is not None:
            target = _bound_descriptor(descriptor, weakref.proxy(receiver), receiver_type, name=name)
        else:
            raise ImplementationDeclarationError(f"Method implementation {name!r} is not bindable.")
        from dryml.core.signatures import compile_signature

        plan = compile_signature(target)
        # Ordinary BoundaryPlan delivery never calls SignaturePlan.target. Replacing
        # the bound target prevents cached plans from retaining their Method key.
        return cls(name, descriptor, receiver_ref, receiver_type, direct, invoker, replace(plan, target=lambda: None))

    def prepare(
        self,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> tuple[tuple[object, ...], dict[str, object]]:
        """Normalize one fresh input boundary through the retained selected plan."""

        return self.plan.prepare_args(args, kwargs).deliver_args()

    def invoke(self, args: tuple[object, ...], kwargs: dict[str, object]) -> object:
        """Normalize inputs, invoke the selected target, and normalize its return."""

        call_args, call_kwargs = self.prepare(args, kwargs)
        return self.invoke_prepared(call_args, call_kwargs)

    def invoke_prepared(self, args: tuple[object, ...], kwargs: dict[str, object]) -> object:
        """Invoke already-normalized inputs and freshly normalize the result."""

        receiver = self.receiver_ref()
        if receiver is None:
            raise MethodError("The selected Method receiver is no longer live.")
        if self.invoker is not None:
            result = self.invoker(*args, **kwargs)
        elif self.descriptor is not None:
            invoke = lambda *call_args, **call_kwargs: self._invoke_raw(
                receiver,
                call_args,
                call_kwargs,
            )
            result = (
                invoke_direct_bound(invoke, receiver, self.descriptor, args, kwargs)
                if self.direct
                else invoke(*args, **kwargs)
            )
        else:
            raise ImplementationDeclarationError(f"Method implementation {self.name!r} is not bindable.")
        return self.plan.prepare_return(result).deliver_return()

    def _invoke_raw(
        self,
        receiver: object,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        """Call a selected raw descriptor without rebinding its Method receiver."""

        assert self.descriptor is not None
        kind = _descriptor_kind(self.descriptor)
        if kind is types.FunctionType:
            return self.descriptor(receiver, *args, **kwargs)
        if kind is staticmethod:
            return self.descriptor.__func__(*args, **kwargs)
        if kind is classmethod:
            return self.descriptor.__func__(self.receiver_type, *args, **kwargs)
        ensure_supported_descriptor(self.descriptor, name=self.name)
        raise AssertionError("supported descriptors always have a binding kind")


@dataclass(frozen=True, slots=True)
class MethodImplementation:
    """One immutable, inspectable authored implementation and its invocation carrier.

    Args:
        name: Stable class declaration name for this implementation.
        target: Exact raw descriptor stored in the authoring class namespace.
        traits: Complete closed trait set supplied by the author.

    Calling the carrier validates its retained first-input constraint, then uses a
    selected-descriptor adapter to apply the bound target's core signature before
    invocation and on every fresh return.
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

    def selected_adapter(self) -> SelectedDescriptorAdapter:
        """Create a reusable signature adapter for this already-selected carrier.

        Returns:
            An adapter with one bound target-signature plan and a weak receiver.

        Raises:
            ImplementationDeclarationError: If this carrier lacks its Method
                binding or has an unsupported descriptor.
            SignatureError: If selected target signature activation fails.
        """

        if self._descriptor is None or self._receiver is None or self._receiver_type is None:
            raise ImplementationDeclarationError(
                f"Method implementation {self.name!r} is not bound to a Method instance."
            )
        return SelectedDescriptorAdapter.create(
            self.name,
            self._descriptor,
            self._receiver,
            self._receiver_type,
            self._direct,
            self._invoker,
        )

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Validate and invoke this target through its selected signature adapter.

        Args:
            *args: Logical positional Method arguments.
            **kwargs: Logical keyword Method arguments.

        Returns:
            The selected target's freshly normalized return value.

        Raises:
            ImplementationDeclarationError: If this manually constructed carrier
                has no Method binding or an unsupported descriptor.
            ImplementationSelectionError: If retained first-input validation fails
                before target signature normalization or invocation.
            SignatureError: If selected argument or return normalization fails.
        """

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
        return self.selected_adapter().invoke(args, kwargs)


__all__ = ["MethodImplementation"]
