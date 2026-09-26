"""Logical Method objects, static catalogs, and the simple authored-call gateway."""

from __future__ import annotations

import os
import weakref
from dataclasses import dataclass, replace
from threading import Lock

from dryml.annotations import AnnotatedMember, annotations_for_members
from dryml.core.backend import Backend
from dryml.core.object import Object, _register_class_transformer
from dryml.core.tensor_spec import BatchMode, SpecTree

from .errors import ImplementationDeclarationError, ImplementationSelectionError, MethodError, PreparedCallMismatchError
from .implementation import (
    MethodImplementation,
    SelectedDescriptorAdapter,
    direct_invocation_active,
    ensure_supported_descriptor,
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
    spec_nodes,
)
from .traits import METHOD_TRAITS_KEY, Traits

_DIRECT_CALL_ATTR = "__dryml_method_direct_call__"
_UNDECLARED = object()
_ITERATION_INDEPENDENCE: weakref.WeakKeyDictionary[type, object] = weakref.WeakKeyDictionary()


@dataclass(frozen=True, slots=True)
class _CapturedDirectCall:
    """Retain one raw direct-call declaration after its gateway replacement."""

    descriptor: object


@dataclass(slots=True)
class _PreparationState:
    """The process-local state associated with exactly one weak Method identity."""

    receiver_ref: weakref.ReferenceType[object]
    default_batched: bool | None = None
    mode: MethodCallMode = "eager"
    signature: MethodCallSignature | None = None
    cached: SelectedDescriptorAdapter | None = None


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

    def __init__(self, owner: type, *, is_abstract: bool = False) -> None:
        self.owner = owner
        self.__isabstractmethod__ = is_abstract

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
        declared_independence = namespace.get("iteration_independent", _UNDECLARED)
        if declared_independence is not _UNDECLARED:
            # Properties are valid capability declarations but not Method catalog
            # targets. Preserve the declaration privately before catalog scans.
            type.__delattr__(cls, "iteration_independent")
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
            type.__setattr__(
                cls,
                "__call__",
                _MethodGatewayDescriptor(
                    cls,
                    is_abstract=bool(getattr(direct, "__isabstractmethod__", False)),
                ),
            )
        # A changed subclass must opt in again instead of inheriting its parent's
        # assertion about omitted calls. The base property reads this exact-class
        # declaration, so catalog inspection never treats a property as a target.
        _ITERATION_INDEPENDENCE[cls] = (
            False if declared_independence is _UNDECLARED else declared_independence
        )

    @staticmethod
    def _call_gateway(
        receiver: object,
        owner: type,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        on_raw_result=None,
    ) -> object:
        """Route one ordinary call to the direct target captured by ``owner``.

        A gateway reached through ``super()`` carries the base descriptor's
        authoring owner, so this function invokes that next raw target exactly
        once instead of restarting lookup from the runtime subclass.
        """

        captured = owner.__dict__.get(_DIRECT_CALL_ATTR)
        if type(captured) is _CapturedDirectCall and direct_invocation_active(receiver, captured.descriptor):
            result = invoke_descriptor(
                captured.descriptor,
                receiver,
                type(receiver),
                args,
                kwargs,
                name="__call__",
            )
            return on_raw_result(result) if on_raw_result is not None else result
        return Method._alternative_call(receiver, args, kwargs, on_raw_result=on_raw_result)

    __dryml_execute_owner__ = "method"

    def __dryml_execute_invoke__(self, args, kwargs, *, repo, on_raw_result):
        """Run one worker-local Method boundary and expose its raw selected result.

        The method remains responsible for selection, argument delivery, and final
        return normalization. Execute supplies only the worker Repo and result
        interception callback.
        """
        from dryml.core.signatures import signature_context

        with signature_context(repo=repo, reuse_live="never"):
            return self._call_gateway(self, type(self), tuple(args), dict(kwargs), on_raw_result)

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
    def iteration_independent(self) -> bool:
        """Return whether discarded iteration calls may be omitted safely.

        Returns:
            ``False`` unless this exact concrete Method class explicitly declares
            that skipped calls cannot affect later results or required effects.

        Side Effects:
            This declaration never selects, prepares, or invokes an
            implementation and never changes Method-local runtime state.
        """

        declaration = _ITERATION_INDEPENDENCE.get(type(self), False)
        if isinstance(declaration, property):
            return declaration.__get__(self, type(self)) is True
        return declaration is True

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
        *additional_input_specs: SpecTree,
        backend: Backend | str | None = None,
        batch_mode: BatchMode | str | None = None,
    ) -> tuple[MethodImplementation, ...]:
        """Return every catalog candidate compatible with known constraints.

        Args:
            input_spec: Optional normalized constraint for the first logical
                argument. Additional constraints require this argument.
            *additional_input_specs: Normalized constraints for later positional
                logical arguments. They validate calls but never rank candidates.
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

        input_nodes, required_backend, required_batch = self._selection_constraints(
            input_spec, additional_input_specs, backend, batch_mode
        )
        del input_nodes
        return self._compatible(required_backend, required_batch)

    def find_implementation(
        self,
        input_spec: SpecTree | None = None,
        *additional_input_specs: SpecTree,
        backend: Backend | str | None = None,
        batch_mode: BatchMode | str | None = None,
        output_spec: SpecTree | None = None,
    ) -> MethodImplementation:
        """Select one safe most-specific callable implementation.

        Args:
            input_spec: Optional normalized first-argument constraint retained by
                the returned callable for directional runtime validation.
            *additional_input_specs: Normalized later positional constraints. They
                require ``input_spec`` and do not influence candidate ranking.
            backend: Optional required backend value or closed string spelling.
            batch_mode: Optional required element/batched value or string spelling.
            output_spec: Optional normalized raw-result constraint retained by the
                returned callable. It validates after target execution but before
                a raw-result callback or successful return.

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
            additional_input_specs,
            backend=backend,
            batch_mode=batch_mode,
            output_spec=output_spec,
            derive_spec_batch=True,
        )

    def _find_implementation(
        self,
        input_spec: SpecTree | None,
        additional_input_specs: tuple[SpecTree, ...] = (),
        *,
        backend: Backend | str | None,
        batch_mode: BatchMode | str | None,
        output_spec: SpecTree | None = None,
        derive_spec_batch: bool,
    ) -> MethodImplementation:
        """Construct one selected callable with explicit spec-fact handling."""

        input_nodes, required_backend, required_batch = self._selection_constraints(
            input_spec,
            additional_input_specs,
            backend,
            batch_mode,
            derive_spec_batch=derive_spec_batch,
        )
        try:
            output_node = None if output_spec is None else spec_node(output_spec)
        except TypeError as error:
            raise ImplementationSelectionError("conflict") from error
        implementation = self._select(required_backend, required_batch)
        return replace(
            implementation,
            _input_specs=tuple(
                complete_backend_constraint(input_node, required_backend)
                for input_node in input_nodes
            ),
            _output_spec=output_node,
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
            (),
            backend=backend,
            batch_mode=batch_mode,
            output_spec=None,
            derive_spec_batch=False,
        )

    def _selection_constraints(
        self,
        input_spec: SpecTree | None,
        additional_input_specs: tuple[SpecTree, ...],
        backend: Backend | str | None,
        batch_mode: BatchMode | str | None,
        *,
        derive_spec_batch: bool = True,
    ) -> tuple[tuple[MethodCallNode, ...], Backend | None, BatchMode | None]:
        """Normalize API constraints and reject contradictory known facts."""

        try:
            required_backend = None if backend is None else Backend(backend)
            required_batch = None if batch_mode is None else BatchMode(batch_mode)
            input_nodes = spec_nodes(input_spec, additional_input_specs)
            spec_backend, spec_batch = (None, None) if not input_nodes else node_facts(input_nodes[0])
            if not derive_spec_batch:
                spec_batch = None
        except (TypeError, ValueError) as error:
            raise ImplementationSelectionError("conflict") from error
        if required_backend is not None and spec_backend is not None and required_backend != spec_backend:
            raise ImplementationSelectionError("conflict")
        if required_batch is not None and spec_batch is not None and required_batch != spec_batch:
            raise ImplementationSelectionError("conflict")
        selected_backend = required_backend or spec_backend
        try:
            for input_node in input_nodes[1:]:
                additional_backend, _ = node_facts(input_node)
                if (
                    selected_backend is not None
                    and additional_backend is not None
                    and additional_backend != selected_backend
                ):
                    raise ImplementationSelectionError("conflict")
        except ValueError as error:
            raise ImplementationSelectionError("conflict") from error
        return input_nodes, selected_backend, required_batch or spec_batch

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

    def _runtime_selection_facts(
        self,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> tuple[Backend | None, BatchMode | None]:
        """Return eager trait facts without changing selected-call validation.

        Generic Methods retain legacy behavior by observing every logical call
        argument. Multi-input subclasses may narrow this hook when only one
        argument has selection semantics.
        """

        return runtime_facts(args, kwargs)

    @staticmethod
    def _alternative_call(
        receiver: object,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        *,
        on_raw_result=None,
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
            return cached.invoke(args, kwargs, on_raw_result=on_raw_result)
        try:
            backend, batch_mode = receiver._runtime_selection_facts(args, kwargs)
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
            adapter = implementation.selected_adapter()
            call_args, call_kwargs = adapter.prepare(args, kwargs)
            signature = replace(signature, batch_mode=effective_batch)
            with _STATE_LOCK:
                # Same-instance transition races are unsupported; a successful
                # learning call publishes atomically before user target code.
                state.mode = "cached"
                state.signature = signature
                state.cached = adapter
            return adapter.invoke_prepared(call_args, call_kwargs, on_raw_result=on_raw_result)
        implementation = receiver._select(backend, effective_batch)
        return (
            implementation.invoke_with_raw_result(args, kwargs, on_raw_result)
            if on_raw_result is not None else implementation(*args, **kwargs)
        )

    def infer_output_spec(self, input_spec: SpecTree, *additional_input_specs: SpecTree) -> SpecTree:
        """Infer a normalized output specification without executing an implementation.

        Args:
            input_spec: Normalized specification for the first logical input.
            *additional_input_specs: Normalized specifications for later logical
                positional inputs when a subclass supports a multi-input Method.

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

        return _catalog_for_class(type(self), receiver=self)

def _catalog_for_class(
    cls: type,
    *,
    receiver: Method | None,
) -> tuple[MethodImplementation, ...]:
    """Collect concrete Method targets without binding or invoking them.

    ``receiver`` is optional so class finalization can validate a trait catalog
    before any Object allocation. Abstract declarations are deliberately checked
    but omitted from executable candidates.
    """

    evidence = {
        (id(member.owner), member.name): member
        for member in _method_members(cls)
    }
    slots: dict[str, tuple[type, MethodImplementation]] = {}
    order: list[str] = []
    for owner in reversed(cls.__mro__):
        if owner is object:
            continue
        namespace = owner.__dict__
        captured = namespace.get(_DIRECT_CALL_ATTR)
        for name, descriptor in namespace.items():
            if name == "__call__" and type(captured) is _CapturedDirectCall:
                ensure_supported_descriptor(captured.descriptor, name=name)
                if getattr(captured.descriptor, "__isabstractmethod__", False):
                    _remove_catalog_slot(slots, order, name)
                else:
                    _place_catalog_slot(
                        slots,
                        order,
                        owner,
                        _catalog_implementation(
                            name, captured.descriptor, Traits(), receiver, cls,
                        ),
                    )
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
            ensure_supported_descriptor(descriptor, name=name)
            if getattr(descriptor, "__isabstractmethod__", False):
                _remove_catalog_slot(slots, order, name)
                continue
            _place_catalog_slot(
                slots,
                order,
                owner,
                _catalog_implementation(name, descriptor, declared_traits, receiver, cls),
            )
    catalog = tuple(slots[name][1] for name in order)
    if any(candidate._direct for candidate in catalog) and any(
        not candidate._direct for candidate in catalog
    ):
        raise ImplementationDeclarationError(
            "A Method hierarchy cannot combine a direct __call__ with trait alternatives."
        )
    return catalog


def _catalog_implementation(
    name: str,
    descriptor: object,
    declared_traits: Traits,
    receiver: Method | None,
    receiver_type: type,
) -> MethodImplementation:
    """Build an inspected catalog carrier without binding its target."""

    return MethodImplementation(
        name=name,
        target=descriptor,
        traits=declared_traits,
        _descriptor=descriptor,
        _receiver=receiver,
        _receiver_type=receiver_type if receiver is not None else None,
        _direct=name == "__call__",
    )


def _place_catalog_slot(
    slots: dict[str, tuple[type, MethodImplementation]],
    order: list[str],
    owner: type,
    implementation: MethodImplementation,
) -> None:
    """Replace an inherited catalog slot or reject an unrelated conflict."""

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


def _remove_catalog_slot(
    slots: dict[str, tuple[type, MethodImplementation]],
    order: list[str],
    name: str,
) -> None:
    """Remove an inherited implementation hidden by one abstract declaration."""

    previous = slots.get(name)
    if previous is None:
        return
    del slots[name]
    order.remove(name)


def _finalize_method_abstractness(cls: type) -> None:
    """Discharge an abstract logical call only when concrete traits prove it."""

    if "__call__" not in cls.__abstractmethods__:
        return
    catalog = _catalog_for_class(cls, receiver=None)
    if not catalog:
        return
    type.__setattr__(cls, "__call__", _MethodGatewayDescriptor(cls))


_register_class_transformer(Method, _finalize_method_abstractness)


__all__ = ["Method"]
