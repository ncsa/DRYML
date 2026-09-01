"""Passive, closed implementation traits owned by :mod:`dryml.methods`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from dryml.annotations import Annotation, attach_annotation
from dryml.core.backend import Backend
from dryml.core.tensor_spec import BatchMode

from .errors import ImplementationDeclarationError

F = TypeVar("F", bound=Callable[..., object])
METHOD_TRAITS_KEY = "dryml.methods.traits"


@dataclass(frozen=True, slots=True)
class Traits:
    """The complete closed trait vocabulary for one authored implementation.

    Args:
        backend: Optional supported execution backend. Exact ``Backend`` values
            and their string forms are normalized to :class:`Backend`.
        batch_mode: Optional element or batched behavior. Exact ``BatchMode``
            values and their string forms are normalized to :class:`BatchMode`.

    Omitted dimensions remain unspecified; this carrier does not choose a
    backend or infer batching behavior.

    Raises:
        ValueError: If a string is not a member of the closed backend or batch
            mode vocabulary.
        TypeError: If a value cannot be normalized by the corresponding enum.
    """

    backend: Backend | str | None = None
    batch_mode: BatchMode | str | None = None

    def __post_init__(self) -> None:
        """Normalize accepted string forms without adding trait dimensions."""

        if self.backend is not None and not isinstance(self.backend, Backend):
            object.__setattr__(self, "backend", Backend(self.backend))
        if self.batch_mode is not None and not isinstance(self.batch_mode, BatchMode):
            object.__setattr__(self, "batch_mode", BatchMode(self.batch_mode))


def traits(
    *,
    backend: Backend | str | None = None,
    batch_mode: BatchMode | str | None = None,
) -> Callable[[F], F]:
    """Attach exactly one passive Method trait annotation to a target.

    Args:
        backend: Optional backend declaration normalized into :class:`Traits`.
        batch_mode: Optional element or batched declaration normalized into
            :class:`Traits`.

    Returns:
        A decorator that returns the exact supplied function or supported
        descriptor object without wrapping or binding it.

    Raises:
        ImplementationDeclarationError: If trait values are invalid or the
            target cannot carry passive annotation metadata.

    Side Effects:
        Appends one process-local annotation to the supplied target. Repeated
        decoration is retained as evidence and rejected by Method catalog
        validation rather than silently merged.
    """

    try:
        declared_traits = Traits(backend=backend, batch_mode=batch_mode)
    except (TypeError, ValueError) as error:
        raise ImplementationDeclarationError("Method trait values are invalid.") from error

    def decorate(target: F) -> F:
        """Attach this declaration's immutable traits and preserve target identity."""

        try:
            return attach_annotation(target, Annotation(METHOD_TRAITS_KEY, declared_traits))
        except Exception as error:
            raise ImplementationDeclarationError(
                "Method traits require a statically annotatable target."
            ) from error

    return decorate


__all__ = ["BatchMode", "Backend", "METHOD_TRAITS_KEY", "Traits", "traits"]
