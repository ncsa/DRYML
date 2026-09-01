"""Logical Method objects, static catalogs, and the simple authored-call gateway."""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import ClassVar

from dryml.annotations import AnnotatedMember, collect_annotations
from dryml.core.object import Object
from dryml.core.tensor_spec import SpecTree

from .errors import ImplementationDeclarationError, MethodError
from .implementation import MethodImplementation, ensure_supported_descriptor, invoke_descriptor
from .traits import METHOD_TRAITS_KEY, Traits

_DIRECT_CALL_ATTR = "__dryml_method_direct_call__"


@dataclass(frozen=True, slots=True)
class _CapturedDirectCall:
    """Retain one raw direct-call declaration after its gateway replacement."""

    descriptor: object


class _MethodGatewayDescriptor:
    """Bind the shared gateway while retaining the class that authored a call."""

    __slots__ = ("owner",)

    def __init__(self, owner: type) -> None:
        self.owner = owner

    def __get__(self, instance: object, owner: type | None = None):
        if instance is None:
            return self

        def invoke(*args: object, **kwargs: object) -> object:
            return Method._call_gateway(instance, self.owner, args, kwargs)

        return invoke


def _is_method_descriptor(descriptor: object) -> bool:
    """Return whether one declaration can carry Method annotation evidence."""

    descriptor_type = type(descriptor)
    if descriptor_type is types.FunctionType:
        return True
    if issubclass(descriptor_type, (staticmethod, classmethod)):
        return True
    if isinstance(descriptor, property):
        return False
    return any("__get__" in base.__dict__ for base in descriptor_type.__mro__)


def _method_members(cls: type) -> tuple[AnnotatedMember, ...]:
    """Collect passive Method evidence below Object without binding descriptors.

    ``annotations_for_members`` intentionally scans a complete class MRO and
    rejects unsupported members. ``Object`` has ordinary unsupported properties,
    so Method interprets the same U2 annotation primitives only for declarations
    from ``Method`` through the supplied subclass. This keeps core APIs outside
    the Method-owned catalog while retaining U2's descriptor annotation rules.
    """

    matching_names: set[str] = set()
    members: list[AnnotatedMember] = []
    reached_method = False
    for owner in reversed(cls.__mro__):
        if owner is Method:
            reached_method = True
            continue
        if not reached_method:
            continue
        namespace = owner.__dict__
        for name, descriptor in namespace.items():
            annotations = () if type(descriptor) is _MethodGatewayDescriptor else (
                collect_annotations(descriptor, key=METHOD_TRAITS_KEY)
                if _is_method_descriptor(descriptor)
                else ()
            )
            if annotations:
                matching_names.add(name)
                members.append(AnnotatedMember(owner, name, descriptor, annotations))
            elif name in matching_names:
                members.append(AnnotatedMember(owner, name, descriptor, ()))
    return tuple(members)


class Method(Object):
    """A CDef-backed logical callable with inspectable authored implementations.

    Subclasses author either one ordinary ``__call__`` implementation or named
    trait-decorated alternatives. A direct ``__call__`` is captured during class
    creation and reached through one owner-aware gateway, preserving cooperative
    ``super().__call__`` routing. U3 supports direct-call forwarding and static
    catalog inspection; trait compatibility, selection, and preparation are
    intentionally introduced by U4.
    """

    _method_gateway_marker: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Capture direct calls and reject local mixed or duplicate trait declarations.

        Args:
            **kwargs: Class-creation keywords forwarded to the Object hierarchy.

        Raises:
            ImplementationDeclarationError: If one class declares both a direct
                call and traits, or attaches multiple Method trait annotations to
                one completed declaration.

        Side Effects:
            Replaces a subclass-authored ``__call__`` descriptor with an
            owner-aware gateway while retaining the raw descriptor privately on
            the declaring class.
        """

        super().__init_subclass__(**kwargs)
        namespace = cls.__dict__
        direct = namespace.get("__call__")
        local_members = tuple(
            member
            for member in _method_members(cls)
            if member.owner is cls and member.annotations
        )
        for member in local_members:
            if len(member.annotations) != 1:
                raise ImplementationDeclarationError(
                    f"Method implementation {member.name!r} has multiple trait annotations."
                )
        if direct is not None and local_members:
            raise ImplementationDeclarationError(
                "A Method class cannot declare both a direct __call__ and trait alternatives."
            )
        if direct is not None:
            type.__setattr__(cls, _DIRECT_CALL_ATTR, _CapturedDirectCall(direct))
            type.__setattr__(cls, "__call__", _MethodGatewayDescriptor(cls))

    @staticmethod
    def _call_gateway(
        receiver: object,
        owner: type,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        """Route one ordinary call to the direct target captured by ``owner``.

        A gateway reached through ``super()`` carries the base descriptor's
        authoring owner, so this function invokes that next raw target exactly
        once instead of restarting lookup from the runtime subclass.
        """

        captured = owner.__dict__.get(_DIRECT_CALL_ATTR)
        if type(captured) is _CapturedDirectCall:
            return invoke_descriptor(
                captured.descriptor,
                receiver,
                type(receiver),
                args,
                kwargs,
                name="__call__",
            )
        raise MethodError(
            "This Method has implementation alternatives; alternative invocation is not available until U4."
        )

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Run the central gateway for subclasses without a captured direct call.

        Args:
            *args: Logical positional Method arguments.
            **kwargs: Logical keyword Method arguments.

        Returns:
            The captured direct implementation's return value.

        Raises:
            MethodError: If no direct target is available. Trait alternative
                selection is deliberately deferred to U4.
        """

        return self._call_gateway(self, Method, args, kwargs)

    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        """Infer a normalized output specification without executing an implementation.

        Args:
            input_spec: Normalized specification for the first logical input.

        Returns:
            The subclass-defined normalized output specification.

        Raises:
            NotImplementedError: Always in U3 unless a subclass supplies the
                pure inference contract.
        """

        raise NotImplementedError(f"{type(self).__name__}.infer_output_spec is not implemented.")

    def implementations(self) -> tuple[MethodImplementation, ...]:
        """Return the deterministic authored catalog without binding or invoking targets.

        Returns:
            Immutable implementation carriers in base-to-subclass declaration
            order, with annotated overrides replacing inherited name slots.

        Raises:
            ImplementationDeclarationError: If visible declaration evidence is
                malformed, ambiguous, shadowed without traits, or uses an
                unsupported descriptor. No target is bound or invoked first.
        """

        evidence = {
            (id(member.owner), member.name): member
            for member in _method_members(type(self))
        }
        slots: dict[str, tuple[type, MethodImplementation]] = {}
        order: list[str] = []
        for owner in reversed(type(self).__mro__):
            if owner is object:
                continue
            namespace = owner.__dict__
            captured = namespace.get(_DIRECT_CALL_ATTR)
            for name, descriptor in namespace.items():
                if name == "__call__" and type(captured) is _CapturedDirectCall:
                    implementation = self._implementation_for(
                        name,
                        captured.descriptor,
                        Traits(),
                    )
                    self._place_catalog_slot(slots, order, owner, implementation)
                member = evidence.get((id(owner), name))
                if member is None:
                    continue
                if not member.annotations:
                    raise ImplementationDeclarationError(
                        f"Method implementation {name!r} has an unannotated shadow."
                    )
                if len(member.annotations) != 1:
                    raise ImplementationDeclarationError(
                        f"Method implementation {name!r} has multiple trait annotations."
                    )
                declared_traits = member.annotations[0].value
                if type(declared_traits) is not Traits:
                    raise ImplementationDeclarationError(
                        f"Method implementation {name!r} must carry a Traits value."
                    )
                implementation = self._implementation_for(name, descriptor, declared_traits)
                self._place_catalog_slot(slots, order, owner, implementation)
        return tuple(slots[name][1] for name in order)

    def _implementation_for(
        self,
        name: str,
        descriptor: object,
        declared_traits: Traits,
    ) -> MethodImplementation:
        """Build one carrier after descriptor validation without binding it."""

        ensure_supported_descriptor(descriptor, name=name)
        return MethodImplementation(
            name=name,
            target=descriptor,
            traits=declared_traits,
            _descriptor=descriptor,
            _receiver=self,
            _receiver_type=type(self),
        )

    @staticmethod
    def _place_catalog_slot(
        slots: dict[str, tuple[type, MethodImplementation]],
        order: list[str],
        owner: type,
        implementation: MethodImplementation,
    ) -> None:
        """Replace an inherited slot or reject unrelated declaration conflicts."""

        previous = slots.get(implementation.name)
        if previous is None:
            slots[implementation.name] = (owner, implementation)
            order.append(implementation.name)
            return
        previous_owner, _ = previous
        if previous_owner not in owner.__mro__[1:]:
            raise ImplementationDeclarationError(
                f"Method implementation {implementation.name!r} has an inherited name conflict."
            )
        slots[implementation.name] = (owner, implementation)


__all__ = ["Method"]
