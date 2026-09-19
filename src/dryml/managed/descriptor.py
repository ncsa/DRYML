"""Checked managed-method descriptors and synchronous bound lifecycle views."""

from __future__ import annotations

import functools
import inspect
import types
from dataclasses import replace

from dryml.annotations import attach_annotation, own_annotations

from .config import ManagedConfig
from .errors import ManagedConfigError, ManagedDeclarationError


def managed_operation(*, resumable: bool = False):
    """Create a checked descriptor for one synchronous managed instance method.

    Args:
        resumable: Exact bool declaring whether a later lifecycle may resume from
            an associated Object-state checkpoint.

    Returns:
        A decorator that retains the checked native function as a descriptor.

    Raises:
        ManagedDeclarationError: If ``resumable`` is not an exact bool or the
            decorated target is not a supported native instance function.

    Side Effects:
        Decoration performs static signature validation only. It does not bind or
        invoke the authored method and creates no lifecycle authority.
    """

    if type(resumable) is not bool:
        raise ManagedDeclarationError(message="resumable must be an exact bool")

    def decorate(target: object) -> "ManagedOperation":
        """Validate and retain one exact native method function."""

        return ManagedOperation(target, resumable=resumable)

    return decorate


class ManagedOperation:
    """Inert managed declaration that binds a lightweight caller-facing view.

    Args:
        target: Exact synchronous Python function with a required keyword-only
            ``managed`` parameter.
        resumable: Exact declaration flag retained for future resume policy.

    Raises:
        ManagedDeclarationError: If the target has forged inspection metadata,
            unsupported descriptor form, async/generator behavior, or an invalid
            native signature.

    Side Effects:
        Copies passive direct annotations from ``target``. Class binding records a
        stable member name but starts no lifecycle and never invokes ``target``.
    """

    from dryml.core.signatures import _FUNCTION_DECLARATION_PARTICIPANT

    __dryml_function_participant__ = _FUNCTION_DECLARATION_PARTICIPANT

    def __init__(self, target: object, *, resumable: bool):
        """Create a statically checked declaration without evaluating annotations."""

        authored, executable, handoff_owner, annotation_sources = _managed_target(
            target
        )
        if "__signature__" in authored.__dict__ or "__wrapped__" in authored.__dict__:
            raise ManagedDeclarationError(
                message="managed operations reject forged authored signature metadata"
            )
        if inspect.iscoroutinefunction(authored) or inspect.isasyncgenfunction(authored) or inspect.isgeneratorfunction(authored):
            raise ManagedDeclarationError(message="managed operations require a synchronous non-generator function")
        self._target = authored
        self._executable = executable
        self._function_handoff_owner = handoff_owner
        self._resumable = resumable
        self._author_signature = _native_signature(authored)
        parameters = tuple(self._author_signature.parameters.values())
        if not parameters or parameters[0].kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            raise ManagedDeclarationError(message="managed operations require a positional instance parameter")
        managed_parameter = self._author_signature.parameters.get("managed")
        if (
            managed_parameter is None
            or managed_parameter.kind is not inspect.Parameter.KEYWORD_ONLY
            or managed_parameter.default is not inspect.Parameter.empty
        ):
            raise ManagedDeclarationError(message="managed operations require a keyword-only managed parameter")
        self._instance_parameter = parameters[0].name
        self._signature_plan = None
        self._member: str | None = None
        self._owner: type | None = None
        # Do not copy the function dictionary: it can already contain passive
        # annotations, which are copied below through their owning API.
        functools.update_wrapper(self, authored, updated=())
        self.__signature__ = self._author_signature
        seen_annotations: set[int] = set()
        for source in annotation_sources:
            for annotation in own_annotations(source):
                if id(annotation) not in seen_annotations:
                    seen_annotations.add(id(annotation))
                    attach_annotation(self, annotation)

    @property
    def resumable(self) -> bool:
        """Return the author-declared exact resume capability."""

        return self._resumable

    @property
    def member(self) -> str | None:
        """Return the stable class member name after declaration binding."""

        return self._member

    @property
    def author_signature(self) -> inspect.Signature:
        """Return the trusted native author signature including context injection."""

        return self._author_signature

    @property
    def instance_parameter(self) -> str:
        """Return the native first positional parameter name used only for binding."""

        return self._instance_parameter

    def bind_arguments(self, instance: object, args: tuple[object, ...], kwargs: dict[str, object]):
        """Bind ordinary caller values once while excluding lifecycle controls.

        Args:
            instance: Bound managed receiver for the native first parameter.
            args: Caller positional ordinary arguments.
            kwargs: Caller keyword ordinary arguments without ``managed``.

        Returns:
            A core ``BoundArguments`` record in authored parameter order, with
            defaults applied and receiver/injected context removed.

        Raises:
            ManagedConfigError: If the native author signature cannot bind the
                supplied ordinary call. No lifecycle authority is created.

        Side Effects:
            None. It only performs native signature binding.
        """

        from dryml.core.bound_args import BoundArguments

        try:
            bound = self._author_signature.bind(instance, *args, managed=None, **kwargs)
            bound.apply_defaults()
        except TypeError as error:
            raise ManagedConfigError(message="managed arguments do not bind") from error
        return BoundArguments(
            (name, value) for name, value in bound.arguments.items()
            if name not in {self._instance_parameter, "managed"}
        )

    def signature_plan(self):
        """Return the cached core plan for ordinary authored managed slots.

        Returns:
            A ``SignaturePlan`` excluding the receiver and injected ``managed``
            control parameter while retaining all ordinary authored annotations.

        Raises:
            SignatureError: If activation of an ordinary authored annotation fails.

        Side Effects:
            Compiles and caches the immutable core plan on first lifecycle use;
            no binding, Store access, selection, or materialization occurs.
        """

        if self._signature_plan is None:
            from dryml.core.signatures import compile_signature

            parameters = tuple(
                parameter for parameter in self._author_signature.parameters.values()
                if parameter.name not in {self._instance_parameter, "managed"}
            )

            def boundary_target(*args, **kwargs):
                """Carry the ordinary managed call surface for core compilation."""

            boundary_target.__signature__ = self._author_signature.replace(parameters=parameters)
            boundary_target.__annotations__ = {
                name: annotation for name, annotation in self._target.__annotations__.items()
                if name not in {self._instance_parameter, "managed"}
            }
            plan = compile_signature(boundary_target, annotation_namespace=self._target.__globals__)
            self._signature_plan = replace(plan, target=lambda: None)
        return self._signature_plan

    def __set_name__(self, owner: type, name: str) -> None:
        """Record one stable member name and reject multi-name descriptor reuse."""

        if self._member is None:
            self._member = name
            self._owner = owner
        elif self._member != name:
            raise ManagedDeclarationError(message="a managed descriptor cannot use multiple member names")

    def __call__(self, instance: object, *args: object,
                 managed: ManagedConfig | None = None, **kwargs: object) -> object:
        """Forward an unbound wrapper call to this descriptor's bound lifecycle.

        This private descriptor protocol supports a statically validated ordinary
        wrapper around a managed declaration.  Direct callers should use normal
        attribute binding rather than invoke a declaration object themselves.
        """

        return _BoundOperation(self, instance)(*args, managed=managed, **kwargs)

    def __dryml_finalize_hidden_member__(self, owner: type, name: str,
                                          outer: types.FunctionType) -> object:
        """Build the private bound surface for one validated outer wrapper."""

        _validate_hidden_member(self, owner, name, outer)
        self.__set_name__(owner, name)
        return _ManagedComposite(self, outer)

    def __get__(self, instance: object | None, owner: type | None = None) -> object:
        """Return this inert declaration on classes or a bound view on instances."""

        if instance is None:
            return self
        return _BoundOperation(self, instance)


class _ManagedComposite:
    """Private descriptor retaining an executable wrapper around one declaration."""

    def __init__(self, descriptor: ManagedOperation,
                 outer: types.FunctionType) -> None:
        """Retain already-validated member and executable wrapper evidence."""

        self._descriptor = descriptor
        self._outer = outer
        functools.update_wrapper(self, outer, updated=())
        seen_annotations: set[int] = set()
        for source in (descriptor, outer):
            for annotation in own_annotations(source):
                if id(annotation) not in seen_annotations:
                    seen_annotations.add(id(annotation))
                    attach_annotation(self, annotation)

    def __get__(self, instance: object | None, owner: type | None = None) -> object:
        """Return controls and invocation behavior before the first call."""

        if instance is None:
            return self
        return _BoundComposite(self, instance)


class _BoundComposite:
    """Bound managed controls that execute one retained outer wrapper."""

    __dryml_execute_owner__ = "managed"

    def __init__(self, composite: _ManagedComposite, instance: object) -> None:
        """Bind the composite while retaining the public managed call surface."""

        self._composite = composite
        self._descriptor = composite._descriptor
        self._instance = instance
        self.__name__ = composite.__name__
        self.__qualname__ = composite.__qualname__
        self.__doc__ = composite.__doc__
        self.__wrapped__ = composite._outer
        self.__signature__ = _bound_signature(self._descriptor.author_signature)

    def __call__(self, *args: object, managed: ManagedConfig | None = None,
                 **kwargs: object) -> object:
        """Execute the retained outer wrapper and its one inner lifecycle."""

        if managed is not None and type(managed) is not ManagedConfig:
            raise ManagedConfigError(message="managed must be a ManagedConfig or None")
        return self._composite._outer(
            self._instance, *args, managed=managed, **kwargs,
        )

    def __dryml_execute_invoke__(self, args, kwargs, *, repo, on_raw_result):
        """Execute the outer wrapper before Execute publishes its delivered value."""

        return on_raw_result(self(*tuple(args), **dict(kwargs)))

    def status(self, *, state_repo: object = None, control_store: object = None) -> object:
        """Delegate selected-authority inspection to the enclosed declaration."""

        return _BoundOperation(self._descriptor, self._instance).status(
            state_repo=state_repo, control_store=control_store,
        )

    def request_interrupt(self, *, state_repo: object = None,
                          control_store: object = None,
                          expected_attempt_id: str | None = None) -> object:
        """Delegate cooperative interruption to the enclosed declaration."""

        return _BoundOperation(self._descriptor, self._instance).request_interrupt(
            state_repo=state_repo, control_store=control_store,
            expected_attempt_id=expected_attempt_id,
        )


class _BoundOperation:
    """Short-lived callable view that validates managed calls before execution."""

    def __init__(self, descriptor: ManagedOperation, instance: object):
        """Bind the declaration and advertise the caller-facing signature."""

        self._descriptor = descriptor
        self._instance = instance
        self.__name__ = descriptor.__name__
        self.__qualname__ = descriptor.__qualname__
        self.__doc__ = descriptor.__doc__
        self.__wrapped__ = descriptor._target
        self.__signature__ = _bound_signature(descriptor.author_signature)

    def __call__(self, *args: object, managed: ManagedConfig | None = None, **kwargs: object) -> object:
        """Validate and synchronously execute one selected managed lifecycle.

        Args:
            *args: Ordinary positional method arguments.
            managed: Optional caller-owned ManagedConfig selecting state/control
                authority, rerun policy, and checkpoint callbacks.
            **kwargs: Ordinary keyword method arguments.

        Raises:
            ManagedConfigError: If config or normalized ordinary arguments are
                unsupported.
            ManagedError: If selected lifecycle authority cannot be used safely.

        Side Effects:
            Acquires retained state ownership, then invokes the raw method only
            after a running control snapshot is committed.

        Returns:
            The wrapped method's ordinary return value after final publication
            and control completion association succeed.
        """

        if managed is not None and type(managed) is not ManagedConfig:
            raise ManagedConfigError(message="managed must be a ManagedConfig or None")
        from .runtime import invoke

        return invoke(self._descriptor, self._instance, args, managed, kwargs)

    __dryml_execute_owner__ = "managed"

    def __dryml_execute_invoke__(self, args, kwargs, *, repo, on_raw_result):
        """Run the managed lifecycle once against invocation-time worker defaults.

        Execute does not inject state or control authority. The managed owner reads
        the current Repo and private worker control default at invocation time, then
        performs its own admission, checkpoint, final-state publication, cleanup,
        and one return-normalization boundary.
        """
        from .runtime import invoke

        return invoke(
            self._descriptor, self._instance, tuple(args),
            None,
            dict(kwargs), on_raw_result=on_raw_result,
        )

    def status(self, *, state_repo: object = None, control_store: object = None) -> object:
        """Read only the selected lifecycle authority without invoking workload.

        Args:
            state_repo: Explicit selected state Repo or Store override.
            control_store: Explicit selected control Store override.

        Raises:
            ManagedError: If selected authority or its retained state is invalid.

        Returns:
            The immutable ManagedStatus projected from the selected control and
            exact state authority.

        Side Effects:
            Reads selected Stores only; it does not invoke hooks, bootstrap
            control state, change routing, or acquire workload ownership.
        """

        from .runtime import status

        return status(self._descriptor, self._instance, state_repo=state_repo, control_store=control_store)

    def request_interrupt(self, *, state_repo: object = None, control_store: object = None, expected_attempt_id: str | None = None) -> object:
        """Publish a cooperative interruption request to selected running authority.

        Args:
            state_repo: Explicit selected state Repo or Store override.
            control_store: Explicit selected control Store override.
            expected_attempt_id: Optional stale-attempt precondition.

        Raises:
            ManagedError: If selected authority cannot safely accept a request.

        Returns:
            An immutable InterruptRequestResult describing request admission.

        Side Effects:
            May publish only an interruption request under the short control
            lock. It never runs workload code or guarantees another checkpoint.
        """

        from .runtime import request_interrupt

        return request_interrupt(
            self._descriptor, self._instance, state_repo=state_repo,
            control_store=control_store, expected_attempt_id=expected_attempt_id,
        )


def _managed_target(target: object) -> tuple[
    types.FunctionType, types.FunctionType, object | None, tuple[object, ...],
]:
    """Validate native declaration and executable-wrapper evidence separately."""

    if type(target) is not types.FunctionType:
        raise ManagedDeclarationError(
            message="managed operations require a native instance function"
        )
    from dryml.core._callable_inspection import _FunctionInvocationOwner, describe_callable

    executable = target
    current = target
    sources: list[object] = []
    seen: set[int] = set()
    while True:
        if id(current) in seen:
            raise ManagedDeclarationError(
                message="managed operations reject cyclic wrapper evidence"
            )
        seen.add(id(current))
        if not _is_synchronous_function(current):
            raise ManagedDeclarationError(
                message="managed operations require synchronous non-generator wrappers"
            )
        sources.append(current)
        description = describe_callable(current)
        if description.owner == "function":
            owner = current.__dict__.get("__dryml_function_invocation_owner__")
            if (
                type(owner) is not _FunctionInvocationOwner
                or owner.wrapper is not current
                or type(description.raw_target) is not types.FunctionType
            ):
                raise ManagedDeclarationError(
                    message="managed operations require an identity-proven function wrapper"
                )
            return (
                description.raw_target, executable, owner,
                _unique_sources(*sources, *description.declaration_carriers),
            )
        wrapped = current.__dict__.get("__wrapped__")
        if wrapped is None:
            if "__signature__" in current.__dict__:
                raise ManagedDeclarationError(
                    message="managed operations reject forged signature metadata"
                )
            return current, executable, None, _unique_sources(*sources)
        if type(wrapped) is not types.FunctionType or not _retains_target(current, wrapped):
            raise ManagedDeclarationError(
                message="managed operations require an executable signature-preserving wrapper"
            )
        if not _same_signature(current, wrapped):
            raise ManagedDeclarationError(
                message="managed operations reject inconsistent wrapper signatures"
            )
        current = wrapped


def _retains_target(wrapper: types.FunctionType, target: object) -> bool:
    """Require a wrapper closure or default to retain its exact forwarded target."""

    if wrapper.__closure__ is not None:
        for cell in wrapper.__closure__:
            try:
                if cell.cell_contents is target:
                    return True
            except ValueError:
                continue
    return any(value is target for value in (wrapper.__defaults__ or ())) or any(
        value is target for value in (wrapper.__kwdefaults__ or {}).values()
    )


def _is_synchronous_function(function: types.FunctionType) -> bool:
    """Reject coroutine and generator wrappers without unwrapping their metadata."""

    flags = function.__code__.co_flags
    return not flags & (inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR | inspect.CO_GENERATOR)


def _validate_hidden_member(descriptor: ManagedOperation, owner: type,
                            name: str, outer: types.FunctionType) -> None:
    """Prove one class-installed ordinary wrapper forwards one declaration."""

    if (
        not isinstance(owner, type)
        or not isinstance(name, str)
        or owner.__dict__.get(name) is not outer
        or type(outer) is not types.FunctionType
    ):
        raise ManagedDeclarationError(
            message="managed hidden declaration has no bound owner/member evidence"
        )
    current = outer
    seen: set[int] = set()
    while True:
        if id(current) in seen:
            raise ManagedDeclarationError(
                message="managed hidden declaration has cyclic wrapper evidence"
            )
        seen.add(id(current))
        if not _is_synchronous_function(current):
            raise ManagedDeclarationError(
                message="managed hidden declaration requires a synchronous wrapper"
            )
        if not _compatible_wrapper_signature(current, descriptor):
            raise ManagedDeclarationError(
                message="managed hidden declaration has inconsistent wrapper signature"
            )
        wrapped = current.__dict__.get("__wrapped__")
        if wrapped is descriptor:
            if not _retains_target(current, descriptor):
                raise ManagedDeclarationError(
                    message="managed hidden declaration requires closure/default evidence"
                )
            return
        if type(wrapped) is not types.FunctionType or not _retains_target(current, wrapped):
            raise ManagedDeclarationError(
                message="managed hidden declaration requires closure/default evidence"
            )
        current = wrapped


def _unique_sources(*values: object) -> tuple[object, ...]:
    """Return declaration sources once while preserving wrapper order."""

    result: list[object] = []
    seen: set[int] = set()
    for value in values:
        if id(value) not in seen:
            seen.add(id(value))
            result.append(value)
    return tuple(result)


def _same_signature(left: types.FunctionType, right: types.FunctionType) -> bool:
    """Compare wrapper declaration evidence without invoking either callable."""

    try:
        return inspect.signature(left) == inspect.signature(right)
    except (TypeError, ValueError):
        return False


def _compatible_wrapper_signature(wrapper: types.FunctionType,
                                  target: object) -> bool:
    """Reject concrete wrapper parameters that cannot forward the declaration."""

    try:
        # ``functools.wraps`` copies ``__signature__`` through ``__dict__``.  Read
        # native code/defaults so copied presentation metadata cannot hide an
        # extra concrete forwarding parameter.
        direct = _native_signature(wrapper)
        declared = (
            target.author_signature if type(target) is ManagedOperation
            else _native_signature(target)
        )
    except (TypeError, ValueError):
        return False
    target_parameters = declared.parameters
    for parameter in direct.parameters.values():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        # A default retaining the target is transportable forwarding evidence, not
        # a public declaration parameter.
        if parameter.default is target:
            continue
        expected = target_parameters.get(parameter.name)
        if expected is None or expected.kind is not parameter.kind:
            return False
    return True


def _invoke_target(descriptor: ManagedOperation, instance: object,
                   args: tuple[object, ...], context: object,
                   kwargs: dict[str, object]) -> object:
    """Invoke the retained executable chain with any exact one-shot handoff."""

    if descriptor._function_handoff_owner is None:
        return descriptor._executable(instance, *args, managed=context, **kwargs)
    from dryml.core.signatures import function_normalization_handoff

    with function_normalization_handoff(descriptor._function_handoff_owner):
        return descriptor._executable(instance, *args, managed=context, **kwargs)


def _native_signature(target: types.FunctionType) -> inspect.Signature:
    """Construct a trusted signature directly from native code and defaults."""

    code = target.__code__
    positional_count = code.co_argcount
    positional_only_count = code.co_posonlyargcount
    positional_names = code.co_varnames[:positional_count]
    keyword_only_names = code.co_varnames[positional_count:positional_count + code.co_kwonlyargcount]
    defaults = target.__defaults__ or ()
    positional_defaults = dict(zip(positional_names[len(positional_names) - len(defaults):], defaults, strict=True))
    keyword_defaults = target.__kwdefaults__ or {}
    parameters: list[inspect.Parameter] = []
    for index, name in enumerate(positional_names):
        kind = inspect.Parameter.POSITIONAL_ONLY if index < positional_only_count else inspect.Parameter.POSITIONAL_OR_KEYWORD
        parameters.append(inspect.Parameter(
            name, kind, default=positional_defaults.get(name, inspect.Parameter.empty),
            annotation=target.__annotations__.get(name, inspect.Parameter.empty),
        ))
    position = positional_count + code.co_kwonlyargcount
    if code.co_flags & inspect.CO_VARARGS:
        name = code.co_varnames[position]
        parameters.append(inspect.Parameter(
            name, inspect.Parameter.VAR_POSITIONAL,
            annotation=target.__annotations__.get(name, inspect.Parameter.empty),
        ))
        position += 1
    for name in keyword_only_names:
        parameters.append(inspect.Parameter(
            name, inspect.Parameter.KEYWORD_ONLY,
            default=keyword_defaults.get(name, inspect.Parameter.empty),
            annotation=target.__annotations__.get(name, inspect.Parameter.empty),
        ))
    if code.co_flags & inspect.CO_VARKEYWORDS:
        name = code.co_varnames[position]
        parameters.append(inspect.Parameter(
            name, inspect.Parameter.VAR_KEYWORD,
            annotation=target.__annotations__.get(name, inspect.Parameter.empty),
        ))
    return inspect.Signature(
        parameters,
        return_annotation=target.__annotations__.get("return", inspect.Signature.empty),
    )


def _bound_signature(author_signature: inspect.Signature) -> inspect.Signature:
    """Project a caller signature that replaces injected context with config policy."""

    parameters = list(author_signature.parameters.values())[1:]
    projected = []
    for parameter in parameters:
        if parameter.name == "managed":
            parameter = parameter.replace(annotation=ManagedConfig | None, default=None)
        projected.append(parameter)
    return author_signature.replace(parameters=projected)


__all__ = ["ManagedOperation", "managed_operation"]
