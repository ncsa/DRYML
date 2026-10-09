"""Explicit authoring/concrete dispatch for trusted graph-producing helpers.

The decorator shares call binding and branch selection, not domain graph recipes
or signature admission policy. It imports no product package or optional backend.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import wraps
from inspect import signature
from types import MappingProxyType
from typing import Any, TYPE_CHECKING

from .template import _contains_template_value

if TYPE_CHECKING:
    from .definition import Definition


def authoring_helper(
    *,
    author_definition: Callable[[Mapping[str, Any]], Definition],
    should_author: Callable[[Mapping[str, Any]], bool] | None = None,
    validate_known_arguments: Callable[[Mapping[str, Any]], None] | None = None,
    normalize_concrete: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a helper with explicit inert authoring and concrete call paths.

    Args:
        author_definition: Callback building one Definition from a read-only
            named argument mapping, including defaults and variadic collections.
            It must not invoke or materialize the graph being authored.
        should_author: Optional callback returning an exact bool from that same
            mapping. By default, soft Definitions and Expr values trigger
            authoring through supported containers, factories, and Mat links;
            Ref/quotation boundaries remain opaque. Exact references alone do
            not trigger authoring. A callback can declare a broader input policy.
        validate_known_arguments: Optional authoring-only validator, called
            before the builder. It should reject known literal errors and defer
            checks depending on unresolved values to normal graph binding.
        normalize_concrete: Whether the concrete path uses core ``function``
            argument/return normalization rather than calling the target directly.

    Returns:
        A decorator preserving the target's signature, name, documentation, and
        wrapped relationship. The decorated helper returns the authored Definition
        or the concrete target's result without interpreting workload keywords as
        decorator controls. Concrete calls retain their original call spelling.

    Raises:
        TypeError: If a policy is invalid, call binding fails, the predicate does
            not return an exact bool, or the builder does not return a Definition.
        ValueError: If the target's Python signature is unavailable.
        Exception: Propagates validator, builder, concrete target, and optional
            signature-normalization failures without fallback to another branch.

    Side Effects:
        Decoration inspects the Python signature and optionally compiles the
        concrete core signature boundary. Authoring bypasses that boundary and
        the concrete target entirely. Policies are trusted callbacks, not a
        sandbox; their own effects remain their responsibility. Concrete calls
        may materialize inputs when normalization is explicitly enabled.
    """

    for name, callback in (
        ("author_definition", author_definition),
        ("should_author", should_author),
        ("validate_known_arguments", validate_known_arguments),
    ):
        if not callable(callback) and (name == "author_definition" or callback is not None):
            raise TypeError(f"{name} must be callable")
    if type(normalize_concrete) is not bool:
        raise TypeError("normalize_concrete must be an exact bool")

    def decorate(target: Callable[..., Any]) -> Callable[..., Any]:
        call_signature = signature(target)
        concrete = target
        if normalize_concrete:
            from .signatures import function

            concrete = function(target)

        @wraps(target)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            bound = call_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = MappingProxyType(bound.arguments)
            author = (
                _contains_template_value(arguments)
                if should_author is None else should_author(arguments)
            )
            if type(author) is not bool:
                raise TypeError("should_author must return an exact bool")
            if not author:
                return concrete(*args, **kwargs)
            if validate_known_arguments is not None:
                validate_known_arguments(arguments)
            result = author_definition(arguments)
            from .definition import Definition

            if not isinstance(result, Definition):
                raise TypeError("author_definition must return a Definition")
            return result

        return wrapped

    return decorate


__all__ = ["authoring_helper"]
