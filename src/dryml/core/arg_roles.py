from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from inspect import Parameter, signature
from types import UnionType
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints

from .definition import ConcreteDefinition, Definition
from .links import DefLink, Ref
from .quoted import QuotedDef, SelectorSpec
from .selector import Selector
from .bound_args import BoundArguments


class ArgRole:
    """Constructor-argument canonicalization policy."""

    name = "materialize"

    def canonicalize(self, value: Any) -> Any:
        return value


@dataclass(frozen=True, slots=True)
class MaterializeArg(ArgRole):
    """Preserve the default DRYML materializing argument behavior."""

    name: str = "materialize"


@dataclass(frozen=True, slots=True)
class RefCDefArg(ArgRole):
    """Canonicalize Object/CDef inputs as non-materializing CDef references."""

    name: str = "ref_cdef"

    def canonicalize(self, value: Any) -> Any:
        from .object import Object

        if value is None:
            return None
        if isinstance(value, DefLink):
            return value
        if isinstance(value, Object):
            return Ref(value.definition)
        if isinstance(value, (ConcreteDefinition, Definition, Selector)):
            return Ref(value)
        raise TypeError(
            "RefCDef argument expects Object, Definition, ConcreteDefinition, Selector, or Ref; "
            f"got {type(value).__name__}."
        )


@dataclass(frozen=True, slots=True)
class SelectorArg(ArgRole):
    """Canonicalize selector inputs as quoted selector data."""

    name: str = "selector_arg"
    def canonicalize(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (QuotedDef, SelectorSpec)):
            return value
        if isinstance(value, Selector):
            return SelectorSpec(value)
        if isinstance(value, Definition):
            return QuotedDef(value)
        raise TypeError(
            "SelectorArg expects Definition, Selector, QuotedDef, or SelectorSpec; "
            f"got {type(value).__name__}."
        )


@dataclass(frozen=True, slots=True)
class ValueArg(ArgRole):
    """Explicit marker for ordinary value canonicalization."""

    name: str = "value"


RefCDef = Annotated[ConcreteDefinition, RefCDefArg()]

_ROLE_NAMES = {
    "materialize": MaterializeArg(),
    "ref_cdef": RefCDefArg(),
    "refcdef": RefCDefArg(),
    "selector_arg": SelectorArg(),
    "selectorarg": SelectorArg(),
    "value": ValueArg(),
}


def apply_arg_roles(cls: type, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Apply constructor argument-role policies after ``__prepare_args__``."""

    roles = resolve_arg_roles(cls)
    if not roles:
        return args, kwargs
    sig = signature(cls.__init__)
    params = list(sig.parameters.values())
    if params and params[0].name == "self":
        params = params[1:]
    bind_sig = sig.replace(parameters=params)
    bound = bind_sig.bind_partial(*args, **kwargs)
    values = dict(bound.arguments)
    changed = False
    for name, role in roles.items():
        if name not in values:
            continue
        values[name] = role.canonicalize(values[name])
        changed = True
    if not changed:
        return args, kwargs
    out_args: list[Any] = []
    out_kwargs: dict[str, Any] = {}
    consumed: set[str] = set()
    for param in params:
        if param.kind is Parameter.VAR_POSITIONAL:
            out_args.extend(values.get(param.name, ()))
            consumed.add(param.name)
        elif param.kind in {Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD}:
            if param.name in values:
                out_args.append(values[param.name])
                consumed.add(param.name)
        elif param.kind is Parameter.KEYWORD_ONLY:
            if param.name in values:
                out_kwargs[param.name] = values[param.name]
                consumed.add(param.name)
        elif param.kind is Parameter.VAR_KEYWORD:
            out_kwargs.update(values.get(param.name, {}))
            consumed.add(param.name)
    for name, value in values.items():
        if name not in consumed:
            out_kwargs[name] = value
    return tuple(out_args), out_kwargs


def apply_bound_arg_roles(cls: type, bound_args: BoundArguments) -> BoundArguments:
    """Apply declared argument roles to an already-bound semantic record.

    Args:
        cls: Constructor class declaring DRYML roles.
        bound_args: Fully or partially bound semantic argument record.

    Returns:
        A record with each declared role applied by parameter name.

    This path deliberately does not inspect or bind the constructor again. The
    V2 construction pipeline owns exactly one preparation and signature bind.
    """

    roles = resolve_arg_roles(cls)
    if not roles:
        return bound_args
    values = []
    for name, value in bound_args.items():
        if name in roles:
            try:
                value = roles[name].canonicalize(value)
            except (TypeError, ValueError) as error:
                raise type(error)(f"Invalid argument role value at {name}: {error}") from error
        values.append((name, value))
    return BoundArguments(values)


def apply_definition_arg_roles(definition: Definition) -> Definition:
    """Return a selector Definition with role-declared args frozen, without prepare/concretize."""

    if definition.cls is None or definition.args is None:
        return definition
    if not isinstance(definition.cls, type):
        # Avoid importing serialized selector classes during definition-only
        # query/index planning. Such selectors must already contain explicit
        # Ref/SelectorSpec wrappers if they need role-specific semantics.
        return definition
    cls = definition.cls
    args, kwargs = apply_arg_roles(cls, tuple(definition.args), dict(definition.kwargs))
    if args == tuple(definition.args) and kwargs == definition.kwargs:
        return definition
    return Definition(definition.cls, *args, **kwargs)


@lru_cache(maxsize=None)
def resolve_arg_roles(cls: type) -> dict[str, ArgRole]:
    """Resolve DRYML argument roles for a class constructor."""

    roles: dict[str, ArgRole] = {}
    sig = signature(cls.__init__)
    valid_names = {
        name for name, param in sig.parameters.items()
        if name != "self" and param.kind is not Parameter.VAR_KEYWORD
    }
    policy = getattr(cls, "__dryml_arg_roles__", {}) or {}
    for name, role_value in policy.items():
        if name not in valid_names:
            raise ValueError(f"Unknown DRYML argument role name {name!r} for {cls.__name__}.")
        roles[name] = normalize_role(role_value)
    try:
        hints = get_type_hints(cls.__init__, include_extras=True)
    except Exception:
        hints = getattr(cls.__init__, "__annotations__", {})
    for name, annotation in hints.items():
        if name == "return" or name not in valid_names:
            continue
        role = role_from_annotation(annotation)
        if role is not None:
            roles[name] = role
    return roles


def normalize_role(value: Any) -> ArgRole:
    if isinstance(value, ArgRole):
        return value
    if isinstance(value, str):
        key = value.replace("-", "_").lower()
        if key in _ROLE_NAMES:
            return _ROLE_NAMES[key]
    raise TypeError(f"Invalid DRYML argument role {value!r}.")


def role_from_annotation(annotation: Any) -> ArgRole | None:
    origin = get_origin(annotation)
    if origin is Annotated:
        for meta in get_args(annotation)[1:]:
            if isinstance(meta, ArgRole):
                return meta
            if isinstance(meta, str) and meta.replace("-", "_").lower() in _ROLE_NAMES:
                return normalize_role(meta)
    if origin in (Union, UnionType):
        found: list[ArgRole] = []
        for arg in get_args(annotation):
            if arg is type(None):
                continue
            role = role_from_annotation(arg)
            if role is not None:
                found.append(role)
        if len({role.name for role in found}) > 1:
            raise TypeError("Union annotations cannot declare multiple DRYML argument roles.")
        if found:
            return found[0]
    if annotation in (RefCDefArg, RefCDef):
        return RefCDefArg()
    if annotation is SelectorArg:
        return SelectorArg()
    return None
