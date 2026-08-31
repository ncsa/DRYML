"""Immutable semantic constructor records for V2 CDef binding and projection.

The helpers in this module bind live classes only when creating or
materializing a CDef. Persisted V2 records are decoded and inspected from their
name/value data without resolving the referenced class.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Any

from .freeze import FrozenDict


@dataclass(frozen=True, slots=True, init=False)
class BoundArguments(Mapping[str, Any]):
    """Immutable semantic constructor-name/value record.

    The record deliberately stores only parameter names and values. Constructor
    parameter kinds remain a property of a live signature and are consulted
    only while binding a new call or projecting a persisted record for runtime
    materialization.

    Args:
        values: A mapping or ordered iterable of ``(name, value)`` pairs.

    Raises:
        TypeError: If a name is not a string or an item is not a pair.
        ValueError: If a name occurs more than once.
    """

    _values: FrozenDict[str, Any]

    def __init__(self, values: Mapping[str, Any] | Iterable[tuple[str, Any]] = ()):
        pairs = list(values.items()) if isinstance(values, Mapping) else list(values)
        names: set[str] = set()
        for item in pairs:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("Bound argument records must contain (name, value) pairs.")
            name, _ = item
            if not isinstance(name, str):
                raise TypeError(f"Bound argument names must be strings; got {type(name).__name__}.")
            if name in names:
                raise ValueError(f"Bound argument record contains duplicate name {name!r}.")
            names.add(name)
        object.__setattr__(self, "_values", FrozenDict(pairs))

    def __getitem__(self, name: str) -> Any:
        """Return the immutable value recorded for a semantic parameter.

        Args:
            name: Persisted constructor parameter name.

        Returns:
            The recorded canonical value.

        Raises:
            KeyError: If ``name`` is not recorded.
        """

        return self._values[name]

    def __iter__(self):
        """Iterate parameter names in deterministic binding order.

        Returns:
            An iterator over string parameter names.
        """

        return iter(self._values)

    def __len__(self) -> int:
        """Return the number of recorded semantic parameters.

        Returns:
            The number of name/value pairs in this immutable record.
        """

        return len(self._values)

    def items(self):
        """Return immutable semantic parameter name/value pairs.

        Returns:
            A dynamic mapping-items view over the immutable record.
        """

        return self._values.items()

    def as_frozen_dict(self) -> FrozenDict[str, Any]:
        """Return the immutable persisted name-to-value representation.

        Returns:
            The ``FrozenDict`` stored by this record; mutating it is unsupported.
        """

        return self._values


def bind_complete_arguments(cls: type, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> BoundArguments:
    """Bind a complete constructor call and capture all declared defaults.

    Args:
        cls: Class whose ``__init__`` signature defines the call surface.
        args: Positional arguments after any preparation hook has run.
        kwargs: Keyword arguments after any preparation hook has run.

    Returns:
        A semantic record containing every effective declared parameter,
        including variadic buckets and omitted defaults.

    Raises:
        TypeError: If the prepared call does not fully bind the constructor.
    """

    sig = _constructor_signature(cls)
    try:
        bound = sig.bind(*args, **kwargs)
    except TypeError as error:
        raise _binding_error(error) from error
    bound.apply_defaults()
    return BoundArguments(bound.arguments)


def bind_partial_arguments(cls: type, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> BoundArguments:
    """Bind only supplied constructor fields without applying defaults.

    Args:
        cls: Class whose ``__init__`` signature defines the call surface.
        args: Supplied positional arguments.
        kwargs: Supplied keyword arguments.

    Returns:
        A semantic record for supplied fields only.

    Raises:
        TypeError: If the supplied call cannot partially bind the constructor.
    """

    sig = _constructor_signature(cls)
    try:
        bound = sig.bind_partial(*args, **kwargs)
    except TypeError as error:
        raise _binding_error(error) from error
    return BoundArguments(bound.arguments)


def project_bound_arguments(cls: type, bound_args: BoundArguments) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Project a semantic record onto the current constructor call surface.

    Parameter kinds are intentionally read from the current resolved class.
    Positional-or-keyword fields are projected positionally before the variadic
    bucket so a defaulted field cannot bind twice when ``*args`` is non-empty.

    Args:
        cls: Resolved runtime class whose current signature is authoritative.
        bound_args: Persisted semantic parameter record.

    Returns:
        ``(args, kwargs)`` suitable for invoking ``cls``.

    Raises:
        TypeError: If the current signature cannot accept the persisted names
            and values.
    """

    sig = _constructor_signature(cls)
    values = dict(bound_args.items())
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    var_keyword: Parameter | None = None
    parameters = tuple(sig.parameters.values())
    var_positional = next(
        (parameter for parameter in parameters if parameter.kind is Parameter.VAR_POSITIONAL),
        None,
    )
    positional_tail = values.get(var_positional.name, ()) if var_positional is not None else ()
    has_positional_tail = bool(positional_tail)
    last_supplied_positional_only = max(
        (
            index
            for index, parameter in enumerate(parameters)
            if parameter.kind is Parameter.POSITIONAL_ONLY and parameter.name in values
        ),
        default=-1,
    )

    for index, parameter in enumerate(parameters):
        name = parameter.name
        if parameter.kind is Parameter.POSITIONAL_ONLY:
            if name in values:
                args.append(values.pop(name))
            elif has_positional_tail or index < last_supplied_positional_only:
                if parameter.default is Parameter.empty:
                    raise TypeError(f"Invalid constructor arguments at {name}: missing a required argument")
                args.append(parameter.default)
        elif parameter.kind is Parameter.POSITIONAL_OR_KEYWORD:
            if has_positional_tail:
                if name in values:
                    args.append(values.pop(name))
                elif parameter.default is not Parameter.empty:
                    args.append(parameter.default)
                else:
                    raise TypeError(f"Invalid constructor arguments at {name}: missing a required argument")
            elif name in values:
                kwargs[name] = values.pop(name)
        elif parameter.kind is Parameter.VAR_POSITIONAL:
            if name in values:
                try:
                    args.extend(values.pop(name))
                except TypeError as error:
                    raise TypeError(f"Invalid variadic value at {name}: {error}") from error
        elif parameter.kind is Parameter.KEYWORD_ONLY:
            if name in values:
                kwargs[name] = values.pop(name)
        elif parameter.kind is Parameter.VAR_KEYWORD:
            var_keyword = parameter

    if var_keyword is not None and var_keyword.name in values:
        extra = values.pop(var_keyword.name)
        if not isinstance(extra, Mapping):
            raise TypeError(f"Invalid variadic keyword value at {var_keyword.name}: expected mapping.")
        kwargs.update(extra)
    kwargs.update(values)

    try:
        sig.bind(*args, **kwargs)
    except TypeError as error:
        raise _binding_error(error) from error
    return tuple(args), kwargs


def validate_canonical_bound_arguments(bound_args: BoundArguments) -> BoundArguments:
    """Validate an import-free persisted bound record.

    Args:
        bound_args: Candidate immutable semantic record.

    Returns:
        The same validated record.

    Raises:
        TypeError: If a value is not recursively canonical.
    """

    for name, value in bound_args.items():
        _validate_canonical_value(value, (name,))
    return bound_args


def decode_bound_arguments(values: Mapping[str, Any] | Iterable[tuple[str, Any]]) -> BoundArguments:
    """Decode and validate a persisted bound record without resolving classes.

    Args:
        values: Persisted name/value mapping or sequence of pairs.

    Returns:
        A validated immutable semantic record.

    Raises:
        TypeError: If names or values violate the persisted canonical format.
        ValueError: If names are duplicated.
    """

    return validate_canonical_bound_arguments(BoundArguments(values))


def _constructor_signature(cls: type):
    sig = signature(cls.__init__)
    parameters = list(sig.parameters.values())
    if parameters and parameters[0].name == "self":
        parameters = parameters[1:]
    return sig.replace(parameters=parameters)


def _binding_error(error: TypeError) -> TypeError:
    text = str(error)
    start = text.find("'")
    end = text.find("'", start + 1)
    path = text[start + 1:end] if start >= 0 and end > start else "<arguments>"
    return TypeError(f"Invalid constructor arguments at {path}: {text}")


def _validate_canonical_value(value: Any, path: tuple[str | int, ...]) -> None:
    # Keep this import local so persisted decoding neither resolves a class nor
    # imports optional backends merely to hydrate a name/value record.
    from .canonical import (
        CANONICAL_DICT_KINDS,
        CANONICAL_SEQ_KINDS,
        NodeKind,
        _is_naked_core_type,
        iter_value_children,
        node_kind,
    )

    kind = node_kind(value)
    if kind in CANONICAL_SEQ_KINDS | CANONICAL_DICT_KINDS:
        for child_name, child in iter_value_children(value):
            _validate_canonical_value(child, path + (child_name,))
        return
    if kind in {
        NodeKind.POD,
        NodeKind.IDENTITY_VALUE,
        NodeKind.REFERENCE_VALUE,
        NodeKind.FROZEN_NDARRAY,
        NodeKind.CONCRETE_DEFINITION,
        NodeKind.DEFLINK,
        NodeKind.IMPORT_REF,
        NodeKind.SOURCE_SPEC,
    }:
        return
    location = "/".join(map(str, path))
    if kind is NodeKind.TYPE:
        if _is_naked_core_type(value):
            return
        raise TypeError(f"Non-canonical bound argument value at {location}: {type(value).__name__}.")
    if kind in {NodeKind.QUOTED_DEF, NodeKind.SELECTOR_SPEC}:
        from .utils.stable_hash import stable_hash_function

        try:
            stable_hash_function(value)
        except Exception as error:
            raise TypeError(
                f"ConcreteDefinition selector-as-data value at {location} "
                "must be stable-hashable."
            ) from error
        return
    raise TypeError(f"Non-canonical bound argument value at {location}: {type(value).__name__}.")
