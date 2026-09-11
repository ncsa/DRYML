"""Checked managed-method descriptors and synchronous bound lifecycle views."""

from __future__ import annotations

import functools
import inspect
import types

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

    def __init__(self, target: object, *, resumable: bool):
        """Create a statically checked declaration without evaluating annotations."""

        if type(target) is not types.FunctionType:
            raise ManagedDeclarationError(message="managed operations require a native instance function")
        if "__signature__" in target.__dict__ or "__wrapped__" in target.__dict__:
            raise ManagedDeclarationError(message="managed operations reject forged signature or wrapping metadata")
        if inspect.iscoroutinefunction(target) or inspect.isasyncgenfunction(target) or inspect.isgeneratorfunction(target):
            raise ManagedDeclarationError(message="managed operations require a synchronous non-generator function")
        self._target = target
        self._resumable = resumable
        self._author_signature = _native_signature(target)
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
        self._member: str | None = None
        # Do not copy the function dictionary: it can already contain passive
        # annotations, which are copied below through their owning API.
        functools.update_wrapper(self, target, updated=())
        for annotation in own_annotations(target):
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

    def __set_name__(self, owner: type, name: str) -> None:
        """Record one stable member name and reject multi-name descriptor reuse."""

        if self._member is None:
            self._member = name
        elif self._member != name:
            raise ManagedDeclarationError(message="a managed descriptor cannot use multiple member names")

    def __get__(self, instance: object | None, owner: type | None = None) -> object:
        """Return this inert declaration on classes or a bound view on instances."""

        if instance is None:
            return self
        return _BoundOperation(self, instance)


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
        parameters.append(inspect.Parameter(name, kind, default=positional_defaults.get(name, inspect.Parameter.empty)))
    position = positional_count + code.co_kwonlyargcount
    if code.co_flags & inspect.CO_VARARGS:
        parameters.append(inspect.Parameter(code.co_varnames[position], inspect.Parameter.VAR_POSITIONAL))
        position += 1
    for name in keyword_only_names:
        parameters.append(inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=keyword_defaults.get(name, inspect.Parameter.empty)))
    if code.co_flags & inspect.CO_VARKEYWORDS:
        parameters.append(inspect.Parameter(code.co_varnames[position], inspect.Parameter.VAR_KEYWORD))
    return inspect.Signature(parameters)


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
