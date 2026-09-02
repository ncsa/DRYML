"""Logical Method objects, static catalogs, and the simple authored-call gateway."""

from __future__ import annotations

import os
import weakref
from dataclasses import dataclass, replace
from threading import Lock

from dryml.annotations import AnnotatedMember, annotations_for_members
from dryml.core.backend import Backend
from dryml.core.object import Object
from dryml.core.tensor_spec import BatchMode, SpecTree

from .errors import ImplementationDeclarationError, ImplementationSelectionError, MethodError, PreparedCallMismatchError
from .implementation import (
    MethodImplementation,
    direct_invocation_active,
    ensure_supported_descriptor,
    invoke_direct_descriptor,
    invoke_descriptor,
)
from .signature import (
    MethodCallMode,
    MethodCallNode,
    MethodCallSignature,
    call_signature,
    complete_backend_constraint,
    node_facts,
    runtime_facts,
    spec_from_runtime_node,
    spec_node,
)
from .traits import METHOD_TRAITS_KEY, Traits

_DIRECT_CALL_ATTR = "__dryml_method_direct_call__"


@dataclass(frozen=True, slots=True)
class _CapturedDirectCall:
    """Retain one raw direct-call declaration after its gateway replacement."""

    descriptor: object


@dataclass(slots=True)
class _CachedInvocation:
    """An unbound descriptor invocation record that cannot retain its Method key."""

    name: str
    descriptor: object
    receiver_ref: weakref.ReferenceType[object]
    receiver_type: type
    direct: bool = False
    invoker: object | None = None

    def invoke(self, args: tuple[object, ...], kwargs: dict[str, object]) -> object:
        """Bind the retained descriptor to its still-live receiver and invoke it."""

        receiver = self.receiver_ref()
        if receiver is None:
            raise MethodError("The cached Method receiver is no longer live.")
        if callable(self.invoker):
            return self.invoker(*args, **kwargs)
        invocation = invoke_direct_descriptor if self.direct else invoke_descriptor
        return invocation(self.descriptor, receiver, self.receiver_type, args, kwargs, name=self.name)


@dataclass(slots=True)
class _PreparationState:
    """The process-local state associated with exactly one weak Method identity."""

    receiver_ref: weakref.ReferenceType[object]
    default_batched: bool | None = None
    mode: MethodCallMode = "eager"
    signature: MethodCallSignature | None = None
    cached: _CachedInvocation | None = None


_STATE_LOCK = Lock()
_STATES: dict[int, _PreparationState] = {}


def _state_for(receiver: object) -> _PreparationState:
    """Return an identity-keyed weak side-table state, creating it under the package lock."""

    key = id(receiver)

    def cleanup(dead_ref: weakref.ReferenceType[object], *, state_key: int = key) -> None:
        with _STATE_LOCK:
            state = _STATES.get(state_key)
            if state is not None and state.receiver_ref is dead_ref:
                _STATES.pop(state_key, None)

    with _STATE_LOCK:
        state = _STATES.get(key)
        if state is not None and state.receiver_ref() is receiver:
            return state
        try:
            receiver_ref = weakref.ref(receiver, cleanup)
        except TypeError as error:
            raise MethodError("Method instances must support weak references.") from error
        state = _PreparationState(receiver_ref)
        _STATES[key] = state
        return state


def _fork_child_reset() -> None:
    """Replace inherited synchronization objects in a forked child without locking them."""

    global _STATE_LOCK, _STATES
    _STATE_LOCK = Lock()
    _STATES = {}


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_fork_child_reset)


class _MethodGatewayDescriptor:
    """Bind the shared gateway while retaining the class that authored a call."""

    __slots__ = ("owner", "__dict__")

    def __init__(self, owner: type) -> None:
        self.owner = owner

    def __get__(self, instance: object, owner: type | None = None):
        if instance is None:
            return self

        def invoke(*args: object, **kwargs: object) -> object:
            return Method._call_gateway(instance, self.owner, args, kwargs)

        return invoke


def _method_members(cls: type) -> tuple[AnnotatedMember, ...]:
    """Collect annotation-owned evidence below the Method authoring boundary."""

    return annotations_for_members(cls, key=METHOD_TRAITS_KEY, after=Method)


class Method(Object):
    """A CDef-backed logical callable with inspectable authored implementations.

    Subclasses author either one ordinary ``__call__`` implementation or named
    trait-decorated alternatives. A direct ``__call__`` is captured during class
    creation and reached through one owner-aware gateway, preserving cooperative
    ``super().__call__`` routing. Alternative-backed calls select a local target
    eagerly, learn one exact call signature, or reuse that cached target. The
    selection/default/cache state is process-local and never enters Object or
    CDef state; concurrent mode/default transitions on the same instance require
    caller coordination.
    """

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
        if type(captured) is _CapturedDirectCall and direct_invocation_active(receiver):
            return invoke_descriptor(
                captured.descriptor,
                receiver,
                type(receiver),
                args,
                kwargs,
                name="__call__",
            )
        return Method._alternative_call(receiver, args, kwargs)

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Run the central gateway for subclasses without a captured direct call.

        Args:
            *args: Logical positional Method arguments.
            **kwargs: Logical keyword Method arguments.

        Returns:
            The captured direct implementation's return value.

        Raises:
            MethodError: If implementation declaration, selection, signature
                normalization, or cached-call validation fails before a target.
        """

        return self._call_gateway(self, Method, args, kwargs)

    @property
    def default_batched(self) -> bool | None:
        """Return this instance's eager-only local default batch preference.

        Returns:
            ``True``, ``False``, or ``None``. This process-local value is used
            only when eager/learning runtime values expose no batch fact.

        Side Effects:
            Creates an otherwise empty weak side-table entry for this live Method
            instance; it never changes CDef, Object, or serialized state.
        """

        state = _state_for(self)
        with _STATE_LOCK:
            return state.default_batched

    @default_batched.setter
    def default_batched(self, value: bool | None) -> None:
        """Set the exact eager-only local batch default.

        Args:
            value: Exact ``True`` for batched intent, exact ``False`` for element
                intent, or ``None`` to leave intent unknown.

        Raises:
            TypeError: If ``value`` is not an exact bool or ``None``.
            RuntimeError: If this Method is learning or cached. State is unchanged.

        Side Effects:
            Updates only process-local weak side-table state while eager.
        """

        if value is not None and type(value) is not bool:
            raise TypeError("Method default_batched must be an exact bool or None.")
        state = _state_for(self)
        with _STATE_LOCK:
            if state.mode != "eager":
                raise RuntimeError("Method default_batched may be changed only while eager.")
            state.default_batched = value

    @property
    def call_mode(self) -> MethodCallMode:
        """Return whether this instance is eager, learning, or cached.

        Returns:
            The process-local invocation mode for this exact live Method.

        Side Effects:
            Creates an empty weak side-table entry when first observed.
        """

        state = _state_for(self)
        with _STATE_LOCK:
            return state.mode

    @property
    def cached_signature(self) -> MethodCallSignature | None:
        """Return the immutable learned exact signature, if one is cached.

        Returns:
            A recursively immutable diagnostic signature in cached mode, otherwise
            ``None``. Mutating caller containers cannot affect this value.

        Side Effects:
            Creates an empty weak side-table entry when first observed.
        """

        state = _state_for(self)
        with _STATE_LOCK:
            return state.signature

    def learn(self) -> None:
        """Clear a prior cache and make the next alternative call learn exactly once.

        Selection, target invocation, backend import, persistence, and output
        inference do not occur until the next call supplies real arguments.

        Side Effects:
            Changes only this live instance's weak process-local mode and clears
            any cached signature/target while preserving ``default_batched``.
        """

        state = _state_for(self)
        with _STATE_LOCK:
            state.mode = "learning"
            state.signature = None
            state.cached = None

    def eager(self) -> None:
        """Clear learning/cached state and restore eager selection.

        Side Effects:
            Clears only this live instance's process-local learned signature and
            target. Its explicitly configured ``default_batched`` is preserved.
        """

        state = _state_for(self)
        with _STATE_LOCK:
            state.mode = "eager"
            state.signature = None
            state.cached = None

    def compatible_implementations(
        self,
        input_spec: SpecTree | None = None,
        *,
        backend: Backend | str | None = None,
        batch_mode: BatchMode | str | None = None,
    ) -> tuple[MethodImplementation, ...]:
        """Return every catalog candidate compatible with known constraints.

        Args:
            input_spec: Optional normalized constraint for exactly the first
                logical argument.
            backend: Optional required backend value or closed string spelling.
            batch_mode: Optional required element/batched value or string spelling.

        Returns:
            Compatible authored candidates in deterministic catalog order.

        Raises:
            ImplementationSelectionError: If supplied constraints are malformed or
                contradict each other. No target is invoked or selected first.

        Side Effects:
            Inspects the authored catalog only. It never reads preparation state,
            ranks candidates, invokes targets, or accesses a cache.
        """

        input_node, required_backend, required_batch = self._selection_constraints(
            input_spec, backend, batch_mode
        )
        del input_node
        return self._compatible(required_backend, required_batch)

    def find_implementation(
        self,
        input_spec: SpecTree | None = None,
        *,
        backend: Backend | str | None = None,
        batch_mode: BatchMode | str | None = None,
    ) -> MethodImplementation:
        """Select one safe most-specific callable implementation.

        Args:
            input_spec: Optional normalized first-argument constraint retained by
                the returned callable for directional runtime validation.
            backend: Optional required backend value or closed string spelling.
            batch_mode: Optional required element/batched value or string spelling.

        Returns:
            One callable carrier retaining its raw authored target and traits.

        Raises:
            ImplementationSelectionError: With ``no_candidate``, ``ambiguous``,
                ``unknown_traits``, or ``conflict`` before target invocation.

        Side Effects:
            Inspects and binds a local callable only. It neither reads nor mutates
            this Method's eager/learning/cached preparation state.
        """

        return self._find_implementation(
            input_spec,
            backend=backend,
            batch_mode=batch_mode,
            derive_spec_batch=True,
        )

    def _find_implementation(
        self,
        input_spec: SpecTree | None,
        *,
        backend: Backend | str | None,
        batch_mode: BatchMode | str | None,
        derive_spec_batch: bool,
    ) -> MethodImplementation:
        """Construct one selected callable with explicit spec-fact handling."""

        input_node, required_backend, required_batch = self._selection_constraints(
            input_spec,
            backend,
            batch_mode,
            derive_spec_batch=derive_spec_batch,
        )
        implementation = self._select(required_backend, required_batch)
        return replace(
            implementation,
            _input_spec=complete_backend_constraint(input_node, required_backend),
        )

    def _prepare_implementation(
        self,
        input_spec: SpecTree | None,
        *,
        backend: Backend | None,
        batch_mode: BatchMode | None,
    ) -> MethodImplementation:
        """Select one learning-time callable without guessing batch from shape."""

        return self._find_implementation(
            input_spec,
            backend=backend,
            batch_mode=batch_mode,
            derive_spec_batch=False,
        )

    def _selection_constraints(
        self,
        input_spec: SpecTree | None,
        backend: Backend | str | None,
        batch_mode: BatchMode | str | None,
        *,
        derive_spec_batch: bool = True,
    ) -> tuple[MethodCallNode | None, Backend | None, BatchMode | None]:
        """Normalize API constraints and reject contradictory known facts."""

        try:
            required_backend = None if backend is None else Backend(backend)
            required_batch = None if batch_mode is None else BatchMode(batch_mode)
            input_node = None if input_spec is None else spec_node(input_spec)
            spec_backend, spec_batch = (None, None) if input_node is None else node_facts(input_node)
            if not derive_spec_batch:
                spec_batch = None
        except (TypeError, ValueError) as error:
            raise ImplementationSelectionError("conflict") from error
        if required_backend is not None and spec_backend is not None and required_backend != spec_backend:
            raise ImplementationSelectionError("conflict")
        if required_batch is not None and spec_batch is not None and required_batch != spec_batch:
            raise ImplementationSelectionError("conflict")
        return input_node, required_backend or spec_backend, required_batch or spec_batch

    def _compatible(
        self,
        backend: Backend | None,
        batch_mode: BatchMode | None,
    ) -> tuple[MethodImplementation, ...]:
        """Return ordered catalog alternatives whose supplied traits do not conflict."""

        return tuple(
            candidate
            for candidate in self.implementations()
            if (candidate.traits.backend is None or backend is None or candidate.traits.backend == backend)
            and (candidate.traits.batch_mode is None or batch_mode is None or candidate.traits.batch_mode == batch_mode)
        )

    def _select(self, backend: Backend | None, batch_mode: BatchMode | None) -> MethodImplementation:
        """Choose one direct-safe candidate or raise a typed bounded diagnostic."""

        candidates = self._compatible(backend, batch_mode)
        if not candidates:
            raise ImplementationSelectionError("no_candidate")
        unknown = tuple(
            name
            for name, value in (("backend", backend), ("batch_mode", batch_mode))
            if value is None and any(getattr(candidate.traits, name) is not None for candidate in candidates)
        )
        safe = tuple(
            candidate
            for candidate in candidates
            if (backend is not None or candidate.traits.backend is None)
            and (batch_mode is not None or candidate.traits.batch_mode is None)
        )
        if not safe:
            raise ImplementationSelectionError("unknown_traits", unknown)
        specificity = lambda candidate: int(candidate.traits.backend is not None) + int(candidate.traits.batch_mode is not None)
        best = max(specificity(candidate) for candidate in safe)
        winners = tuple(candidate for candidate in safe if specificity(candidate) == best)
        if len(winners) != 1:
            raise ImplementationSelectionError("ambiguous")
        return winners[0]

    @staticmethod
    def _alternative_call(
        receiver: object,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        """Run the eager, learning, or cached alternative-backed call gateway."""

        if not isinstance(receiver, Method):
            raise MethodError("Method alternative receiver is invalid.")
        state = _state_for(receiver)
        with _STATE_LOCK:
            mode = state.mode
            expected = state.signature
            cached = state.cached
            default_batched = None if mode == "cached" else state.default_batched
        if mode == "cached":
            if expected is None or cached is None:
                raise MethodError("Method cached state is incomplete.")
            try:
                observed = call_signature(args, kwargs)
            except (TypeError, ValueError) as error:
                raise PreparedCallMismatchError(expected, expected) from error
            if observed.batch_mode is None:
                observed = replace(observed, batch_mode=expected.batch_mode)
            if observed != expected:
                raise PreparedCallMismatchError(expected, observed)
            return cached.invoke(args, kwargs)
        try:
            backend, batch_mode = runtime_facts(args, kwargs)
        except ValueError as error:
            raise ImplementationSelectionError("conflict") from error
        effective_batch = batch_mode
        if effective_batch is None and default_batched is not None:
            effective_batch = BatchMode.batched if default_batched else BatchMode.element
        if mode == "learning":
            try:
                signature = call_signature(args, kwargs)
            except TypeError as error:
                raise MethodError("Method learning requires supported tensor-like call values.") from error
            backend = signature.backend
            effective_batch = signature.batch_mode
            if effective_batch is None and default_batched is not None:
                effective_batch = BatchMode.batched if default_batched else BatchMode.element
            try:
                input_spec = (
                    None
                    if not signature.args
                    else spec_from_runtime_node(signature.args[0], effective_batch)
                )
                implementation = receiver._prepare_implementation(
                    input_spec,
                    backend=backend,
                    batch_mode=effective_batch,
                )
            except TypeError as error:
                raise MethodError("Method learning could not normalize its first input.") from error
            try:
                cached = _CachedInvocation(
                    implementation.name,
                    implementation._descriptor,
                    weakref.ref(receiver),
                    implementation._receiver_type,
                    implementation._direct,
                    implementation._invoker,
                )
            except TypeError as error:
                raise MethodError("Method instances must support weak references.") from error
            if cached.descriptor is None or cached.receiver_type is None:
                raise ImplementationDeclarationError("Selected Method implementation is not bindable.")
            signature = replace(signature, batch_mode=effective_batch)
            with _STATE_LOCK:
                # Same-instance transition races are unsupported; a successful
                # learning call publishes atomically before user target code.
                state.mode = "cached"
                state.signature = signature
                state.cached = cached
            return cached.invoke(args, kwargs)
        implementation = receiver._select(backend, effective_batch)
        return implementation(*args, **kwargs)

    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        """Infer a normalized output specification without executing an implementation.

        Args:
            input_spec: Normalized specification for the first logical input.

        Returns:
            The subclass-defined normalized output specification.

        Raises:
            NotImplementedError: If the subclass does not supply the pure
                inference contract.
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
        catalog = tuple(slots[name][1] for name in order)
        if any(candidate._direct for candidate in catalog) and any(
            not candidate._direct for candidate in catalog
        ):
            raise ImplementationDeclarationError(
                "A Method hierarchy cannot combine a direct __call__ with trait alternatives."
            )
        return catalog

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
            _direct=name == "__call__",
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
