from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .freeze import FrozenDict, FrozenList, FrozenSet
from .symbol import ImportRef, SourceSpec, maybe_symbol_ref, resolve_symbol
from .utils.stable_hash import stable_hash_function


def _is_target_token(value: Any) -> bool:
    return isinstance(value, (str, ImportRef, SourceSpec)) or isinstance(value, type)


def _freeze_factory_value(value: Any, path: tuple[str, ...] = ()) -> Any:
    from .definition import ConcreteDefinition, Definition
    from .object import Object

    if isinstance(value, (Object, Definition, ConcreteDefinition)):
        loc = "/".join(path) or "<root>"
        raise TypeError(
            f"FactorySpec arguments cannot contain DRYML graph nodes at {loc}. "
            "Pass plain runtime construction values instead."
        )

    symbol = maybe_symbol_ref(value)
    if symbol is not None:
        return symbol

    if isinstance(value, Mapping):
        return FrozenDict(
            (k, _freeze_factory_value(v, path + (str(k),)))
            for k, v in value.items()
        )
    if isinstance(value, list):
        return FrozenList(_freeze_factory_value(v, path + (str(i),)) for i, v in enumerate(value))
    if isinstance(value, tuple):
        return tuple(_freeze_factory_value(v, path + (str(i),)) for i, v in enumerate(value))
    if isinstance(value, set):
        return FrozenSet(_freeze_factory_value(v, path + ("<set>",)) for v in value)
    if isinstance(value, frozenset):
        return frozenset(_freeze_factory_value(v, path + ("<frozenset>",)) for v in value)

    return value


def _resolve_factory_value(value: Any) -> Any:
    if isinstance(value, (ImportRef, SourceSpec)):
        return resolve_symbol(value)
    if isinstance(value, FrozenDict):
        return {k: _resolve_factory_value(v) for k, v in value.items()}
    if isinstance(value, FrozenList):
        return [_resolve_factory_value(v) for v in value]
    if isinstance(value, FrozenSet):
        return {_resolve_factory_value(v) for v in value}
    if isinstance(value, dict):
        return {k: _resolve_factory_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_factory_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_resolve_factory_value(v) for v in value)
    if isinstance(value, set):
        return {_resolve_factory_value(v) for v in value}
    if isinstance(value, frozenset):
        return frozenset(_resolve_factory_value(v) for v in value)

    return value


def _namespace_get(namespace: Any, name: str) -> Any | None:
    if namespace is None:
        return None
    if isinstance(namespace, Mapping):
        return namespace.get(name)
    return getattr(namespace, name, None)


def _resolve_string_target(target: str, *, namespace=None) -> Any:
    namespaced = _namespace_get(namespace, target)
    if namespaced is not None:
        return namespaced

    if ":" in target:
        return ImportRef.from_import_path(target).resolve()

    if "." in target:
        module, qualname = target.rsplit(".", 1)
        return ImportRef(module, qualname).resolve()

    raise ValueError(
        f"Cannot resolve factory target {target!r} without a namespace."
    )


@dataclass(frozen=True, slots=True)
class FactorySpec:
    """Leaf construction spec for non-DRYML runtime objects."""

    target: Any
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: FrozenDict = field(default_factory=lambda: FrozenDict({}))

    def __init__(self, target: Any, *args: Any, **kwargs: Any):
        symbol = maybe_symbol_ref(target)
        if symbol is not None:
            target = symbol
        elif not isinstance(target, (str, ImportRef, SourceSpec)):
            raise TypeError(
                "FactorySpec target must be a class, short name, import path, ImportRef, or SourceSpec."
            )

        object.__setattr__(self, "target", target)
        object.__setattr__(
            self,
            "args",
            tuple(_freeze_factory_value(arg, ("args", str(i))) for i, arg in enumerate(args)),
        )
        object.__setattr__(
            self,
            "kwargs",
            FrozenDict(
                (key, _freeze_factory_value(value, ("kwargs", str(key))))
                for key, value in kwargs.items()
            ),
        )

    @classmethod
    def coerce(cls, value: Any) -> "FactorySpec":
        if isinstance(value, cls):
            return value

        if _is_target_token(value):
            return cls(value)

        if not isinstance(value, (tuple, list)):
            raise TypeError(f"Cannot coerce {type(value).__name__} to FactorySpec.")

        if not value:
            raise ValueError("Factory tuple shorthand cannot be empty.")

        target = value[0]
        if not _is_target_token(target):
            raise TypeError("Factory tuple shorthand must start with a target class or symbol.")

        parts = list(value[1:])
        kwargs = {}
        has_kwargs = bool(parts and isinstance(parts[-1], Mapping))
        if has_kwargs:
            kwargs = dict(parts.pop())

        # Compatibility with the previous (target, args_tuple, kwargs_dict) form.
        if has_kwargs and len(parts) == 1 and isinstance(parts[0], (tuple, list)):
            args = tuple(parts[0])
        else:
            args = tuple(parts)

        return cls(target, *args, **kwargs)

    @classmethod
    def coerce_many(cls, values, *, strict: bool = False) -> tuple[Any, ...]:
        prepared = []
        for value in values:
            try:
                prepared.append(cls.coerce(value))
            except TypeError:
                if strict:
                    raise
                prepared.append(value)
        return tuple(prepared)

    def resolve_target(self, *, namespace=None):
        target = resolve_symbol(self.target)
        if isinstance(target, str):
            return _resolve_string_target(target, namespace=namespace)
        return target

    def build(self, *, namespace=None, instance_type=None):
        target = self.resolve_target(namespace=namespace)
        args = tuple(_resolve_factory_value(arg) for arg in self.args)
        kwargs = {
            key: _resolve_factory_value(value)
            for key, value in self.kwargs.items()
        }

        obj = target(*args, **kwargs)

        if instance_type is not None and not isinstance(obj, instance_type):
            raise TypeError(
                f"FactorySpec built {type(obj).__name__}, expected {instance_type.__name__}."
            )

        return obj

    def __stable_leaf_bytes__(self):
        digest = stable_hash_function((
            "dryml.core.FactorySpec",
            self.target,
            self.args,
            self.kwargs,
        ))
        return digest.encode("ascii")


__all__ = ["FactorySpec"]
