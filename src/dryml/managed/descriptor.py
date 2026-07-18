"""Callable descriptor implementation for managed Object methods."""

from __future__ import annotations

import inspect
from functools import update_wrapper
from types import MappingProxyType
from typing import Any, Callable

from dryml.core2.definition import ConcreteDefinition, Definition

from .declarations import (
    DelegatedOutputs,
    ManagedMethodDeclaration,
    ManagedOutputs,
    _as_cdef,
    normalize_outputs,
)
from .errors import ManagedLifecycleUnavailableError, UnknownOutputError
from .refs import ManagedOutputRef


class BoundManagedMethod:
    """Callable binding of a managed descriptor to one producer Object."""

    __dryml_bound_method__ = True

    def __init__(self, descriptor: "ManagedMethod", owner: Any):
        self._descriptor = descriptor
        self.__self__ = owner
        self.__func__ = descriptor.__func__
        update_wrapper(self, descriptor.__func__, updated=())
        signature = inspect.signature(descriptor.__func__)
        parameters = tuple(signature.parameters.values())
        if parameters:
            signature = signature.replace(parameters=parameters[1:])
        self.__signature__ = signature
        declarations = descriptor.output_declarations(owner)
        refs = descriptor.output_refs(owner, declarations=declarations)
        self.outputs = MappingProxyType(refs)
        self.result = self.outputs[declarations.primary.slot]

    def __call__(self, *args, **kwargs):
        """Invoke the underlying method with its normal Python call semantics."""

        return self.__func__(self.__self__, *args, **kwargs)

    def output(self, slot: str) -> ManagedOutputRef:
        """Return the stable logical reference for a declared output slot."""

        try:
            return self.outputs[slot]
        except KeyError as exc:
            raise UnknownOutputError(slot) from exc

    def status(self):
        """Return lifecycle status once a managed runtime is installed.

        U1 deliberately provides declarations and logical references only.
        Realization-backed status is added by the managed runtime layer.
        """

        raise ManagedLifecycleUnavailableError(
            f"Managed lifecycle status is unavailable for {self._descriptor.method_name!r}."
        )


class ManagedMethod:
    """Descriptor declaring one callable managed Object method."""

    def __init__(self, func: Callable[..., Any], declaration: ManagedMethodDeclaration):
        if not callable(func):
            raise TypeError("@managed can decorate only callables.")
        self.__func__ = func
        self.declaration = declaration
        self.outputs = declaration.outputs
        self.method_name = getattr(func, "__name__", None)
        self.owner = None
        update_wrapper(self, func)

    def __set_name__(self, owner: type, name: str) -> None:
        self.owner = owner
        self.method_name = name

    def __get__(self, instance: Any, owner: type | None = None):
        if instance is None:
            return self
        return BoundManagedMethod(self, instance)

    def __call__(self, *args, **kwargs):
        """Support ordinary unbound method calls through class access."""

        return self.__func__(*args, **kwargs)

    def output_declarations(
        self,
        producer: ConcreteDefinition | Definition | Any | None = None,
    ) -> ManagedOutputs:
        """Return statically declared outputs for an optional producer."""

        return self.declaration.output_declarations(producer)

    def output_refs(
        self,
        producer: ConcreteDefinition | Definition | Any,
        *,
        declarations: ManagedOutputs | None = None,
    ) -> dict[str, ManagedOutputRef]:
        """Build logical output refs from a producer definition only."""

        cdef = _producer_cdef(producer)
        if declarations is None:
            declarations = self.output_declarations(cdef)
        return {
            output.slot: ManagedOutputRef(
                producer=cdef,
                method=self.method_name,
                slot=output.slot,
            )
            for output in declarations
        }

    def output_ref(
        self,
        producer: ConcreteDefinition | Definition | Any,
        slot: str,
    ) -> ManagedOutputRef:
        """Build the logical output ref for one validated slot."""

        refs = self.output_refs(producer)
        try:
            return refs[slot]
        except KeyError as exc:
            raise UnknownOutputError(slot) from exc


def managed(
    func: Callable[..., Any] | None = None,
    *,
    outputs: ManagedOutputs | DelegatedOutputs | Any | None = None,
    delegate: tuple[str | int, ...] | str | None = None,
    resumable: bool = False,
    checkpoint_schema: str | None = None,
    early_completion: bool = False,
):
    """Declare an Object method managed while preserving normal call behavior.

    Args:
        func: Function supplied by decorator application.
        outputs: Static output declarations or :class:`DelegatedOutputs`.
        delegate: Convenience path for ``DelegatedOutputs``; mutually exclusive
            with ``outputs``.
        resumable: Definition-level resume capability declaration.
        checkpoint_schema: Optional stable checkpoint schema name.
        early_completion: Whether operation-specific early completion is valid.

    Returns:
        A :class:`ManagedMethod` descriptor or decorator producing one.
    """

    if outputs is not None and delegate is not None:
        raise TypeError("@managed accepts either outputs or delegate, not both.")
    if delegate is not None:
        outputs = DelegatedOutputs(delegate)
    if outputs is None:
        raise TypeError("@managed requires deterministic output declarations.")
    declaration = ManagedMethodDeclaration(
        outputs=normalize_outputs(outputs),
        resumable=resumable,
        checkpoint_schema=checkpoint_schema,
        early_completion=early_completion,
    )

    def decorate(target: Callable[..., Any]) -> ManagedMethod:
        return ManagedMethod(target, declaration)

    return decorate(func) if func is not None else decorate


def _producer_cdef(value: ConcreteDefinition | Definition | Any) -> ConcreteDefinition:
    try:
        return _as_cdef(value)
    except TypeError as exc:
        raise TypeError(
            "Managed output refs require an Object, Definition, or ConcreteDefinition producer."
        ) from exc


__all__ = ["BoundManagedMethod", "ManagedMethod", "managed"]
