from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from inspect import Parameter, signature
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from .definition import ConcreteDefinition, Definition, FrozenConcreteDefinition, FrozenDefinition


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
class FrozenCDefArg(ArgRole):
    """Canonicalize Object/CDef inputs as non-materializing CDef references."""

    name: str = "frozen_cdef"

    def canonicalize(self, value: Any) -> FrozenConcreteDefinition:
        from .object import Object

        if isinstance(value, FrozenConcreteDefinition):
            return value
        if isinstance(value, Object):
            return value.definition.freeze()
        if isinstance(value, ConcreteDefinition):
            return value.freeze()
        raise TypeError(
            "FrozenCDef argument expects Object, ConcreteDefinition, or "
            f"FrozenConcreteDefinition; got {type(value).__name__}."
        )


@dataclass(frozen=True, slots=True)
class FrozenDefArg(ArgRole):
    """Canonicalize Definition inputs as immutable selector snapshots."""

    name: str = "frozen_def"
    def canonicalize(self, value: Any) -> FrozenDefinition:
        if isinstance(value, FrozenDefinition):
            return value
        if isinstance(value, Definition):
            return value.freeze()
        raise TypeError(
            "FrozenDef argument expects Definition or FrozenDefinition; "
            f"got {type(value).__name__}."
        )


@dataclass(frozen=True, slots=True)
class ValueArg(ArgRole):
    """Explicit marker for ordinary value canonicalization."""

    name: str = "value"


FrozenCDef = Annotated[ConcreteDefinition, FrozenCDefArg()]
FrozenDef = Annotated[Definition, FrozenDefArg()]

_ROLE_NAMES = {
    "materialize": MaterializeArg(),
    "frozen_cdef": FrozenCDefArg(),
    "frozencdef": FrozenCDefArg(),
    "frozen_def": FrozenDefArg(),
    "frozendef": FrozenDefArg(),
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


def apply_definition_arg_roles(definition: Definition) -> Definition:
    """Return a selector Definition with role-declared args frozen, without prepare/concretize."""

    if definition.cls is None or definition.args is None:
        return definition
    if not isinstance(definition.cls, type):
        # Avoid importing serialized selector classes during definition-only
        # query/index planning. Such selectors must already contain explicit
        # frozen wrappers if they need frozen-reference semantics.
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
    if annotation in (FrozenCDefArg, FrozenCDef):
        return FrozenCDefArg()
    if annotation in (FrozenDefArg, FrozenDef):
        return FrozenDefArg()
    return None
