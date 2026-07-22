"""Immutable, definition-derived managed method output declarations."""

from __future__ import annotations

import inspect
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from dryml.core.definition import ConcreteDefinition, Definition
from dryml.core.links import DefLink
from dryml.core.symbol import resolve_symbol

from .errors import (
    DuplicateOutputError,
    InvalidSubjectPathError,
    ManagedDeclarationError,
    PrimaryOutputError,
    UnstableOutputsError,
)


_SLOT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


@dataclass(frozen=True, slots=True)
class ManagedOutput:
    """Declare one stable output slot of a managed method.

    Args:
        slot: Stable logical output name.
        primary: Whether this is the method's single primary result.
        kind: Framework-defined logical output kind.
        subject_path: Optional path through the producer CDef to the Object whose
            state the output represents. Paths may use constructor parameter
            names or explicit ``"args"``/``"kwargs"`` segments and never
            materialize the selected Object.
        representations: Stable compatible representation names.
    """

    slot: str
    primary: bool = False
    kind: str = "object"
    subject_path: tuple[str | int, ...] | None = None
    representations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.slot, str) or _SLOT_RE.fullmatch(self.slot) is None:
            raise ManagedDeclarationError(
                "Managed output slots must be non-empty identifier-like strings."
            )
        if not isinstance(self.primary, bool):
            raise TypeError("Managed output primary must be a bool.")
        if not isinstance(self.kind, str) or not self.kind:
            raise ManagedDeclarationError("Managed output kind must be a non-empty string.")
        path = _normalize_path(self.subject_path, allow_none=True)
        object.__setattr__(self, "subject_path", path)
        representations = tuple(self.representations)
        if any(not isinstance(item, str) or not item for item in representations):
            raise ManagedDeclarationError("Managed output representations must be non-empty strings.")
        if len(set(representations)) != len(representations):
            raise ManagedDeclarationError("Managed output representations must not contain duplicates.")
        object.__setattr__(self, "representations", representations)


@dataclass(frozen=True, slots=True, init=False)
class ManagedOutputs(Sequence[ManagedOutput]):
    """Validated immutable output set with exactly one primary slot."""

    _items: tuple[ManagedOutput, ...] = field(default_factory=tuple)

    def __init__(self, *items: ManagedOutput | Iterable[ManagedOutput]):
        if len(items) == 1 and not isinstance(items[0], ManagedOutput):
            items = tuple(items[0])
        normalized = tuple(items)
        if any(not isinstance(item, ManagedOutput) for item in normalized):
            raise TypeError("ManagedOutputs accepts only ManagedOutput declarations.")
        slots = tuple(item.slot for item in normalized)
        duplicates = tuple(dict.fromkeys(slot for slot in slots if slots.count(slot) > 1))
        if duplicates:
            raise DuplicateOutputError(
                f"Managed output slots are duplicated: {', '.join(duplicates)}."
            )
        primary = tuple(item for item in normalized if item.primary)
        if len(primary) != 1:
            raise PrimaryOutputError(
                f"Managed methods require exactly one primary output; found {len(primary)}."
            )
        object.__setattr__(self, "_items", normalized)

    def __getitem__(self, index):
        return self._items[index]

    def __len__(self) -> int:
        return len(self._items)

    @property
    def primary(self) -> ManagedOutput:
        """Return the declaration's single primary output."""

        return next(item for item in self._items if item.primary)

    @property
    def slots(self) -> tuple[str, ...]:
        """Return output slots in deterministic declaration order."""

        return tuple(item.slot for item in self._items)

    def get(self, slot: str) -> ManagedOutput | None:
        """Return the declaration for *slot*, or ``None`` when absent."""

        return next((item for item in self._items if item.slot == slot), None)


@dataclass(frozen=True, slots=True)
class DelegatedOutputs:
    """Derive outputs from an Object definition at a producer CDef path.

    The target class may expose ``__dryml_managed_outputs__`` as a
    :class:`ManagedOutputs` value or as a class/static method accepting its own
    CDef. Providers are evaluated twice and must return the same immutable
    contract, preventing runtime-dependent slot sets.
    """

    path: tuple[str | int, ...] | str
    provider: str = "__dryml_managed_outputs__"

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _normalize_path(self.path, allow_none=False))
        if not isinstance(self.provider, str) or not self.provider:
            raise ManagedDeclarationError("Delegated output provider must be a non-empty attribute name.")

    def resolve(self, producer: ConcreteDefinition) -> ManagedOutputs:
        """Resolve the delegated contract without building the target Object."""

        target = resolve_definition_path(producer, self.path)
        cls = resolve_symbol(target.cls)
        try:
            raw = inspect.getattr_static(cls, self.provider)
        except AttributeError as exc:
            raise ManagedDeclarationError(
                f"Delegated output target {cls.__name__} has no {self.provider!r} provider."
            ) from exc

        first = _call_output_provider(raw, cls, target)
        second = _call_output_provider(raw, cls, target)
        if first != second:
            raise UnstableOutputsError(
                f"Delegated output provider {cls.__name__}.{self.provider} returned unstable outputs."
            )
        return first


@dataclass(frozen=True, slots=True)
class ManagedMethodDeclaration:
    """Immutable declaration metadata attached to one managed descriptor."""

    outputs: ManagedOutputs | DelegatedOutputs
    resumable: bool = False
    checkpoint_schema: str | None = None
    early_completion: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.outputs, (ManagedOutputs, DelegatedOutputs)):
            object.__setattr__(self, "outputs", normalize_outputs(self.outputs))
        if not isinstance(self.resumable, bool) or not isinstance(self.early_completion, bool):
            raise TypeError("Managed method capabilities must be bool values.")
        if self.checkpoint_schema is not None and (
            not isinstance(self.checkpoint_schema, str) or not self.checkpoint_schema
        ):
            raise ManagedDeclarationError("checkpoint_schema must be a non-empty string or None.")

    def output_declarations(
        self,
        producer: ConcreteDefinition | Definition | Any | None = None,
    ) -> ManagedOutputs:
        """Return and validate outputs for an optional producer definition."""

        cdef = _as_cdef(producer) if producer is not None else None
        if isinstance(self.outputs, DelegatedOutputs):
            if cdef is None:
                raise ManagedDeclarationError(
                    "Delegated managed outputs require a producer definition."
                )
            outputs = self.outputs.resolve(cdef)
        else:
            outputs = self.outputs
        if cdef is not None:
            for output in outputs:
                if output.subject_path is not None:
                    resolve_definition_path(cdef, output.subject_path)
        return outputs


def normalize_outputs(value: Any) -> ManagedOutputs | DelegatedOutputs:
    """Normalize public output declaration shorthand."""

    if isinstance(value, (ManagedOutputs, DelegatedOutputs)):
        return value
    if isinstance(value, ManagedOutput):
        return ManagedOutputs(value)
    if isinstance(value, Mapping):
        declarations = []
        for slot, item in value.items():
            if isinstance(item, ManagedOutput):
                if item.slot != slot:
                    raise ManagedDeclarationError(
                        f"Output mapping key {slot!r} does not match declaration slot {item.slot!r}."
                    )
                declarations.append(item)
            elif item is None:
                declarations.append(ManagedOutput(slot))
            elif isinstance(item, Mapping):
                declarations.append(ManagedOutput(slot, **dict(item)))
            else:
                raise TypeError("Managed output mappings require declarations, mappings, or None values.")
        return ManagedOutputs(*declarations)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return ManagedOutputs(*value)
    raise TypeError("outputs must be ManagedOutputs, DelegatedOutputs, a mapping, or output iterable.")


def resolve_definition_path(
    producer: ConcreteDefinition,
    path: tuple[str | int, ...] | str,
) -> ConcreteDefinition:
    """Resolve a declared CDef path and require an Object-definition target."""

    if not isinstance(producer, ConcreteDefinition):
        raise TypeError("Managed output paths require an exact ConcreteDefinition producer.")
    normalized = _normalize_path(path, allow_none=False)
    current: Any = producer
    try:
        index = 0
        while index < len(normalized):
            part = normalized[index]
            if isinstance(current, DefLink):
                current = current.target
            if isinstance(current, (ConcreteDefinition, Definition)):
                if part == "args":
                    has_named_part = (
                        index + 1 < len(normalized) and isinstance(normalized[index + 1], str)
                    )
                    if has_named_part:
                        current = _definition_argument(current, normalized[index + 1])
                        index += 2
                        continue
                    current = current.args
                elif part == "kwargs":
                    if index + 1 < len(normalized):
                        key = normalized[index + 1]
                        if key in current.kwargs:
                            current = current.kwargs[key]
                        elif isinstance(key, str):
                            current = _definition_argument(current, key)
                        else:
                            raise KeyError(key)
                        index += 2
                        continue
                    current = current.kwargs
                elif isinstance(part, str):
                    current = _definition_argument(current, part)
                else:
                    raise KeyError(part)
            else:
                current = current[part]
            index += 1
        if isinstance(current, DefLink):
            current = current.target
    except (KeyError, IndexError, TypeError) as exc:
        raise InvalidSubjectPathError(
            f"Managed output path {normalized!r} does not exist on the producer definition."
        ) from exc
    if not isinstance(current, ConcreteDefinition):
        raise InvalidSubjectPathError(
            f"Managed output path {normalized!r} does not identify an Object definition."
        )
    return current


def _call_output_provider(raw: Any, cls: type, target: ConcreteDefinition) -> ManagedOutputs:
    if isinstance(raw, ManagedOutputs):
        return raw
    if isinstance(raw, classmethod):
        value = raw.__func__(cls, target)
    elif isinstance(raw, staticmethod):
        value = raw.__func__(target)
    elif inspect.isfunction(raw):
        value = raw(target)
    elif hasattr(raw, "output_declarations"):
        value = raw.output_declarations(target)
    else:
        raise ManagedDeclarationError(
            "Delegated output providers must be ManagedOutputs, class/static methods, "
            "or managed descriptors."
        )
    normalized = normalize_outputs(value)
    if isinstance(normalized, DelegatedOutputs):
        raise ManagedDeclarationError("Delegated output providers cannot return another delegation.")
    return normalized


def _normalize_path(
    value: tuple[str | int, ...] | list[str | int] | str | None,
    *,
    allow_none: bool,
) -> tuple[str | int, ...] | None:
    if value is None:
        if allow_none:
            return None
        raise ManagedDeclarationError("Managed definition paths cannot be None.")
    if isinstance(value, str):
        value = tuple(value.split("."))
    else:
        value = tuple(value)
    if not value or any(not isinstance(part, (str, int)) for part in value):
        raise ManagedDeclarationError("Managed definition paths must contain string or integer segments.")
    if isinstance(value[0], int):
        raise ManagedDeclarationError(
            "Managed definition paths must start with 'args', 'kwargs', or a constructor parameter name."
        )
    return value


def _definition_argument(definition: ConcreteDefinition | Definition, name: str) -> Any:
    if name in definition.kwargs:
        return definition.kwargs[name]
    cls = resolve_symbol(definition.cls)
    parameters = tuple(inspect.signature(cls.__init__).parameters.values())
    positional = tuple(
        parameter
        for parameter in parameters
        if parameter.name != "self" and parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    )
    for index, parameter in enumerate(positional):
        if parameter.name == name and definition.args is not None and index < len(definition.args):
            return definition.args[index]
    raise KeyError(name)


def _as_cdef(value: Any) -> ConcreteDefinition:
    if isinstance(value, ConcreteDefinition):
        return value
    if isinstance(value, Definition):
        return value.concretize()
    definition = getattr(value, "definition", None)
    if isinstance(definition, ConcreteDefinition):
        return definition
    raise TypeError("Managed declarations require an Object, Definition, or ConcreteDefinition.")


# Authoring aliases keep declarations concise while retaining explicit type names.
OutputDeclaration = ManagedOutput
OutputDeclarations = ManagedOutputs


__all__ = [
    "DelegatedOutputs",
    "ManagedMethodDeclaration",
    "ManagedOutput",
    "ManagedOutputs",
    "OutputDeclaration",
    "OutputDeclarations",
    "normalize_outputs",
    "resolve_definition_path",
]
