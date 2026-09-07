"""Checked managed-method descriptors with inert U3 bound invocation views."""

from __future__ import annotations

import functools
import inspect
import types

from dryml.annotations import attach_annotation, own_annotations

from .config import ManagedConfig
from .errors import ManagedConfigError, ManagedControlError, ManagedDeclarationError
from .identity import argument_digest


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
    """Short-lived callable view that validates U3 calls without executing them."""

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
        """Validate config and ordinary arguments, then defer lifecycle execution.

        Args:
            *args: Ordinary positional method arguments.
            managed: Optional caller-owned U3 configuration.
            **kwargs: Ordinary keyword method arguments.

        Raises:
            ManagedConfigError: If config or normalized ordinary arguments are
                unsupported.
            ManagedControlError: Always in U3, because U6 owns lifecycle entry.

        Side Effects:
            None. The raw method and callbacks are never invoked by this unit.
        """

        if managed is not None and type(managed) is not ManagedConfig:
            raise ManagedConfigError(message="managed must be a ManagedConfig or None")
        config = ManagedConfig() if managed is None else managed
        config.snapshot()
        supplied = dict(kwargs)
        supplied["managed"] = managed
        argument_digest(self._descriptor, self._instance, args, supplied)
        raise ManagedControlError("lifecycle_unavailable", "managed lifecycle is unavailable until U6")

    def status(self, *, state_store: object = None, control_store: object = None) -> object:
        """Reserve the future inert status API without reading authority in U3.

        Args:
            state_store: Future explicit state Store override.
            control_store: Future explicit control Store override.

        Raises:
            ManagedControlError: Always because U6 owns status inspection.
        """

        raise ManagedControlError("lifecycle_unavailable", "managed status is unavailable until U6")

    def request_interrupt(self, *, state_store: object = None, control_store: object = None, expected_attempt_id: str | None = None) -> object:
        """Reserve the future inert interruption API without writing authority in U3.

        Args:
            state_store: Future explicit state Store override.
            control_store: Future explicit control Store override.
            expected_attempt_id: Future stale-attempt precondition.

        Raises:
            ManagedControlError: Always because U6 owns request publication.
        """

        raise ManagedControlError("lifecycle_unavailable", "managed interruption is unavailable until U6")


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
