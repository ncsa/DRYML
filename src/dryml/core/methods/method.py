from __future__ import annotations

from typing import Callable, ClassVar

from dryml.core.backend import Backend, discover_backends
from dryml.core.object import Object
from dryml.core.tensor_spec import TensorSpec
from dryml.core.utils.recurse import iter_leaves
from dryml.core.methods.traits import BatchMode, Traits


def traits(backend: Backend | str | None = Backend.numpy, batch_mode: BatchMode | str | None = None, traits: Traits | None = None):
    """Mark a method implementation for a specific backend/batch trait.

    Args:
        backend: Backend selector, backend name, or ``None`` as wildcard.
        batch_mode: Batch-mode selector, batch-mode name, or ``None`` as wildcard.
        traits: Optional prebuilt :class:`Traits` object to attach directly.

    Returns:
        Decorator that stores the selected traits on the function.
    """

    if traits is None:
        traits = Traits(backend=backend, batch_mode=batch_mode)

    def deco(f):
        f.__dryml_traits__ = traits
        return f
    return deco


class Method(Object):
    """Base DRYML method semantic model.

    Two supported styles:

    1. Simple method: subclass defines ``__call__`` directly.
    2. Multi-backend method: subclass defines backend implementations such as
       ``call_tf``/``call_torch`` and does not override ``__call__``. The base
       class auto-dispatches to the best matching implementation.
    """

    __trait_impls__: ClassVar[tuple[tuple[Traits, str], ...]] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        impls = []
        for name, obj in cls.__dict__.items():
            impl_traits = getattr(obj, "__dryml_traits__", None)
            if impl_traits is not None:
                impls.append((impl_traits, name))

        parent_impls = list(getattr(cls, "__trait_impls__", ()))
        parent_impls.extend(impls)
        cls.__trait_impls__ = tuple(parent_impls)

        has_user_call = "__call__" in cls.__dict__
        if impls and not has_user_call:
            cls.__call__ = Method._dispatch_call

    def _dispatch_call(self, *args, _hint_batched: bool | None = None, **kwargs):
        """Dispatch this call to the best matching trait implementation."""

        impl = self.resolve_impl_for(*args, _hint_batched=_hint_batched, **kwargs)
        return impl(*args, **kwargs)

    def bind_first(self, first_value, *, input_spec=None):
        """Resolve this method once, apply it to *first_value*, and return both."""

        if getattr(type(self), "__trait_impls__", ()):
            impl = self.resolve_impl_for(first_value, input_spec=input_spec)
        else:
            impl = self
        return impl, impl(first_value)

    def resolve_impl(self, requested_traits: Traits) -> Callable:
        """Return the best implementation matching *requested_traits*.

        Args:
            requested_traits: Desired backend/batch selector.

        Raises:
            NotImplementedError: No implementation matches.
            ValueError: Multiple equally specific implementations match.
        """

        matches = []
        for impl_traits, name in type(self).__trait_impls__:
            if impl_traits.match(requested_traits):
                matches.append((impl_traits.specificity, name, impl_traits))

        if not matches:
            raise NotImplementedError(
                f"{type(self).__name__} has no implementation matching {requested_traits!r}."
            )

        matches.sort(key=lambda item: item[0], reverse=True)
        if len(matches) > 1 and matches[0][0] == matches[1][0]:
            raise ValueError(
                f"Multiple implementations of {type(self).__name__} match {requested_traits!r}: "
                f"{matches[0][2]!r} and {matches[1][2]!r}."
            )

        return getattr(self, matches[0][1])

    def resolve_impl_for(self, *args, input_spec=None, _hint_batched: bool | None = None, **kwargs) -> Callable:
        """Infer traits from call arguments and return the best implementation."""

        backends = discover_backends(*args, **kwargs)
        if len(backends) > 1:
            raise ValueError(f"Multiple backends detected {backends}, can't dispatch properly.")
        backend = next(iter(backends)) if backends else None

        batch_mode = self._batch_mode_from_hint(input_spec, _hint_batched)
        return self.resolve_impl(Traits(backend=backend, batch_mode=batch_mode))

    @staticmethod
    def _batch_mode_from_hint(input_spec=None, hint: bool | None = None) -> BatchMode | None:
        """Infer batch mode from an explicit hint or TensorSpec leaves."""

        if hint is not None:
            return BatchMode.batched if hint else BatchMode.element

        if input_spec is None:
            return None

        leaves = list(iter_leaves(input_spec, pred=lambda x: isinstance(x, TensorSpec)))
        if not leaves:
            return None
        batched = {leaf.batched for leaf in leaves}
        if len(batched) != 1:
            return None
        return BatchMode.batched if batched.pop() else BatchMode.element

    def get_impl(self, backend: Backend | str | None, batchmode: BatchMode | str | None = None) -> Callable | None:
        """Return a matching bound implementation or ``None``."""

        try:
            return self.resolve_impl(Traits(backend=backend, batch_mode=batchmode))
        except NotImplementedError:
            return None

    @classmethod
    def get_impl_func(cls, backend: Backend | str | None, batchmode: BatchMode | str | None = None) -> Callable | None:
        """Return the raw class function for compiler use, or ``None``."""

        requested = Traits(backend=backend, batch_mode=batchmode)
        matches = []
        for impl_traits, name in cls.__trait_impls__:
            if impl_traits.match(requested):
                matches.append((impl_traits.specificity, name))
        if not matches:
            return None
        matches.sort(key=lambda item: item[0], reverse=True)
        return getattr(cls, matches[0][1])

    def infer_output_spec(self, *args, **kwargs):
        """Infer output spec for this method.

        Subclasses override this when static output-spec inference is available.
        """

        raise NotImplementedError(
            f"{type(self).__name__}.infer_output_spec is not implemented."
        )


__all__ = ["Method", "traits"]
