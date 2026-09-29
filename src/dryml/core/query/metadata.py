"""Bounded, typed metadata predicates for authoritative reference queries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
from typing import Any

from dryml.formats import canonical_json_bytes

from .codecs import QueryCodecError
from .model import QueryError


_SCOPES = frozenset(("object", "state", "lineage", "snapshot"))
_MAX_PATH = 32
_MAX_LEAVES = 256
_MAX_BOOLEAN_DEPTH = 32
_MAX_VALUE_DEPTH = 8
_MAX_NODES = 1024
_MAX_PREDICATE_NODES = 16_384
_MAX_PREDICATE_BYTES = 131_000
_MAX_ENTRIES = 64
_MAX_STRING = 4096
_MAX_INT_BITS = 4096
_MISSING = object()


@dataclass(frozen=True, slots=True)
class MetadataField:
    """Select one path in a documented detached metadata projection.

    Fields are inert values created with :func:`field`.  Their methods build
    immutable predicates; they neither inspect Stores nor materialize Objects.

    Args:
        scope: One of ``"object"``, ``"state"``, ``"lineage"`, or
            ``"snapshot"``.
        path: Mapping-field strings and sequence indexes beneath the scope root.

    Raises:
        TypeError: If direct construction receives a non-string scope or
            non-tuple path.
        ValueError: If the scope, a path component, or path length is invalid.

    Side Effects:
        None. Construction validates only detached selector data.
    """

    scope: str
    path: tuple[str | int, ...]

    def __post_init__(self) -> None:
        """Validate direct construction as strictly as the public helper."""

        if not isinstance(self.scope, str):
            raise TypeError("Metadata scope must be a string.")
        if not isinstance(self.path, tuple):
            raise TypeError("MetadataField path must be a tuple.")
        if self.scope not in _SCOPES:
            raise ValueError("Metadata scope must be object, state, lineage, or snapshot.")
        if len(self.path) > _MAX_PATH:
            raise ValueError("Metadata paths may contain at most 32 components.")
        for component in self.path:
            if isinstance(component, str):
                if len(component) > _MAX_STRING:
                    raise ValueError("Metadata path strings may contain at most 4096 characters.")
                continue
            if type(component) is int and component >= 0:
                continue
            raise ValueError("Metadata path components must be strings or non-negative integer indexes.")

    def exists(self) -> "MetadataPredicate":
        """Return a predicate requiring this path to be present.

        Returns:
            An inert immutable predicate. Present null values exist.

        Side Effects:
            None. This only constructs detached predicate data.
        """

        return _LeafPredicate("exists", self)

    def missing(self) -> "MetadataPredicate":
        """Return a predicate requiring this path to be absent.

        Returns:
            An inert immutable predicate. A present null value is not missing.

        Side Effects:
            None. This only constructs detached predicate data.
        """

        return _LeafPredicate("missing", self)

    def eq(self, value: Any) -> "MetadataPredicate":
        """Return a type-aware exact-equality predicate.

        Args:
            value: A bounded metadata value, or an aware datetime for a typed
                lifecycle timestamp field.

        Returns:
            An inert immutable predicate.

        Raises:
            TypeError: If the value has an unsupported type.
            ValueError: If the value is non-finite, naive, or exceeds bounds.

        Side Effects:
            None. This validates and detaches the operand without Store access.
        """

        return _LeafPredicate("eq", self, _operand_data(self, value, "eq"))

    def lt(self, value: int | float | datetime) -> "MetadataPredicate":
        """Return a strict typed ordering predicate.

        Args:
            value: A finite non-boolean numeric value, or an aware datetime for
                a typed lifecycle timestamp field.

        Returns:
            An inert immutable predicate.

        Raises:
            TypeError: If the operand is not numeric or a permitted datetime.
            ValueError: If it is non-finite or a naive datetime.

        Side Effects:
            None. This validates and detaches the operand without Store access.
        """

        return _LeafPredicate("lt", self, _operand_data(self, value, "order"))

    def le(self, value: int | float | datetime) -> "MetadataPredicate":
        """Return an inclusive typed ordering predicate.

        Args:
            value: A finite non-boolean numeric value, or an aware datetime for
                a typed lifecycle timestamp field.

        Returns:
            An inert immutable predicate.

        Raises:
            TypeError: If the operand is not numeric or a permitted datetime.
            ValueError: If it is non-finite or a naive datetime.

        Side Effects:
            None. This validates and detaches the operand without Store access.
        """

        return _LeafPredicate("le", self, _operand_data(self, value, "order"))

    def gt(self, value: int | float | datetime) -> "MetadataPredicate":
        """Return a strict typed ordering predicate.

        Args:
            value: A finite non-boolean numeric value, or an aware datetime for
                a typed lifecycle timestamp field.

        Returns:
            An inert immutable predicate.

        Raises:
            TypeError: If the operand is not numeric or a permitted datetime.
            ValueError: If it is non-finite or a naive datetime.

        Side Effects:
            None. This validates and detaches the operand without Store access.
        """

        return _LeafPredicate("gt", self, _operand_data(self, value, "order"))

    def ge(self, value: int | float | datetime) -> "MetadataPredicate":
        """Return an inclusive typed ordering predicate.

        Args:
            value: A finite non-boolean numeric value, or an aware datetime for
                a typed lifecycle timestamp field.

        Returns:
            An inert immutable predicate.

        Raises:
            TypeError: If the operand is not numeric or a permitted datetime.
            ValueError: If it is non-finite or a naive datetime.

        Side Effects:
            None. This validates and detaches the operand without Store access.
        """

        return _LeafPredicate("ge", self, _operand_data(self, value, "order"))

    def contains(self, value: Any) -> "MetadataPredicate":
        """Return a literal string or direct sequence-membership predicate.

        Args:
            value: A bounded metadata value compared directly, never as a regex
                or recursive subtree.

        Returns:
            An inert immutable predicate.

        Raises:
            TypeError: If the value has an unsupported type.
            ValueError: If the value is non-finite or exceeds bounds.

        Side Effects:
            None. This validates and detaches the operand without Store access.
        """

        return _LeafPredicate("contains", self, _operand_data(self, value, "contains"))

    def to_data(self) -> dict[str, Any]:
        """Return this closed canonical selector representation.

        Returns:
            A JSON-compatible scope and component-path mapping.

        Side Effects:
            None. The returned mapping is newly allocated detached data.
        """

        return {"scope": self.scope, "path": list(self.path)}


def field(scope: str, *path: str | int) -> MetadataField:
    """Create a selector into one documented metadata-query scope.

    Args:
        scope: Exactly ``"object"``, ``"state"``, ``"lineage"`, or
            ``"snapshot"``.
        *path: String mapping keys or non-negative integer sequence indexes.
            An empty path selects the detached scope-root projection.

    Returns:
        An immutable :class:`MetadataField` with no Store access or side effects.

    Raises:
        ValueError: If the scope or path is invalid, or it has over 32 components.

    Side Effects:
        None. This validates only detached selector components and never scans a
        Store.
    """

    if scope not in _SCOPES:
        raise ValueError("Metadata scope must be object, state, lineage, or snapshot.")
    if len(path) > _MAX_PATH:
        raise ValueError("Metadata paths may contain at most 32 components.")
    for component in path:
        if isinstance(component, str):
            if len(component) > _MAX_STRING:
                raise ValueError("Metadata path strings may contain at most 4096 characters.")
            continue
        if type(component) is int and component >= 0:
            continue
        raise ValueError("Metadata path components must be strings or non-negative integer indexes.")
    return MetadataField(scope, tuple(path))


class MetadataPredicate:
    """Immutable bounded Boolean expression over detached metadata projections.

    Expressions compose only with ``&``, ``|``, and ``~``. Evaluation occurs at
    a :class:`ReferenceQuery` terminal against Store authority, never on a live
    Object or state payload.

    Side Effects:
        Construction and composition are detached. Evaluation happens only at a
        ReferenceQuery terminal.
    """

    def __and__(self, other: "MetadataPredicate") -> "MetadataPredicate":
        """Return an immutable conjunction with ``other``.

        Args:
            other: Another metadata predicate.

        Returns:
            A bounded conjunction.

        Raises:
            TypeError: If ``other`` is not a MetadataPredicate.
            ValueError: If expression bounds would be exceeded.

        Side Effects:
            None. This only constructs detached predicate data.
        """

        return _BooleanPredicate("and", self, _require_predicate(other))

    def __or__(self, other: "MetadataPredicate") -> "MetadataPredicate":
        """Return an immutable disjunction with ``other``.

        Args:
            other: Another metadata predicate.

        Returns:
            A bounded disjunction.

        Raises:
            TypeError: If ``other`` is not a MetadataPredicate.
            ValueError: If expression bounds would be exceeded.

        Side Effects:
            None. This only constructs detached predicate data.
        """

        return _BooleanPredicate("or", self, _require_predicate(other))

    def __invert__(self) -> "MetadataPredicate":
        """Return an immutable complement of this predicate.

        Returns:
            A bounded negation.

        Raises:
            ValueError: If expression bounds would be exceeded.

        Side Effects:
            None. This only constructs detached predicate data.
        """

        return _NotPredicate(self)

    def __bool__(self) -> bool:
        """Reject truth testing, which would discard query composition.

        Raises:
            TypeError: Always, because predicates must remain explicit Boolean
                expression nodes.

        Side Effects:
            None.
        """

        raise TypeError("Metadata predicates must be combined with &, |, or ~.")

    def to_data(self) -> dict[str, Any]:
        """Return closed canonical predicate data.

        Returns:
            JSON-compatible tagged predicate data.

        Raises:
            QueryCodecError: If a subclass does not provide valid closed data.

        Side Effects:
            None. The returned data is detached from Store authority.
        """

        raise NotImplementedError

    def fingerprint(self) -> str:
        """Return the stable SHA-256 fingerprint of canonical predicate data.

        Returns:
            A lowercase hexadecimal digest.

        Raises:
            QueryCodecError: If predicate data is malformed or exceeds bounds.

        Side Effects:
            None. Hashing reads only canonical detached predicate data.
        """

        return hashlib.sha256(_canonical_bytes(self.to_data())).hexdigest()

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "MetadataPredicate":
        """Decode closed, bounded predicate data.

        Args:
            data: Tagged JSON-compatible predicate representation.

        Returns:
            An immutable MetadataPredicate.

        Raises:
            QueryCodecError: If data is malformed, unsupported, or noncanonical.

        Side Effects:
            None. Decoding validates detached data without Store access.
        """

        try:
            return _predicate_from_data(data)
        except QueryCodecError:
            raise
        except (TypeError, ValueError, RecursionError) as exc:
            raise QueryCodecError("Metadata predicate data violates expression bounds.") from exc


@dataclass(frozen=True, slots=True)
class _LeafPredicate(MetadataPredicate):
    operator: str
    field: MetadataField
    value: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.operator not in {"exists", "missing", "eq", "lt", "le", "gt", "ge", "contains"}:
            raise QueryCodecError("Unsupported metadata predicate operator.")
        if self.operator in {"exists", "missing"}:
            if self.value is not None:
                raise QueryCodecError("Presence predicates do not accept operands.")
        elif self.value is None:
            raise QueryCodecError("Metadata comparison predicates require an operand.")
        _predicate_bounds(self)

    def to_data(self) -> dict[str, Any]:
        data = {"kind": self.operator, "field": self.field.to_data()}
        if self.value is not None:
            data["value"] = self.value
        return data


@dataclass(frozen=True, slots=True)
class _BooleanPredicate(MetadataPredicate):
    operator: str
    left: MetadataPredicate
    right: MetadataPredicate

    def __post_init__(self) -> None:
        if self.operator not in {"and", "or"}:
            raise QueryCodecError("Unsupported metadata Boolean operator.")
        _require_predicate(self.left)
        _require_predicate(self.right)
        _predicate_bounds(self)

    def to_data(self) -> dict[str, Any]:
        return {"kind": self.operator, "left": self.left.to_data(), "right": self.right.to_data()}


@dataclass(frozen=True, slots=True)
class _NotPredicate(MetadataPredicate):
    operand: MetadataPredicate

    def __post_init__(self) -> None:
        _require_predicate(self.operand)
        _predicate_bounds(self)

    def to_data(self) -> dict[str, Any]:
        return {"kind": "not", "operand": self.operand.to_data()}


def predicate_requires_state(predicate: MetadataPredicate) -> bool:
    """Return whether a predicate selects state-only metadata scopes.

    Args:
        predicate: Valid MetadataPredicate to inspect.

    Returns:
        ``True`` if any leaf uses the ``state`` or ``snapshot`` scope.

    Raises:
        TypeError: If ``predicate`` is not a MetadataPredicate.

    Side Effects:
        None. This inspects detached predicate structure only.
    """

    return any(leaf.field.scope in {"state", "snapshot"} for leaf in _leaves(_require_predicate(predicate)))


def evaluate_metadata_predicate(predicate: MetadataPredicate, reference, repo, *, store=None) -> bool:
    """Evaluate a predicate against detached authority for one exact reference.

    Args:
        predicate: A bounded immutable MetadataPredicate.
        reference: An exact ObjectRef or StateRef selected structurally.
        repo: The managing Repo that resolves Store authority.
        store: Optional connected Store restricting both candidate and metadata
            authority.

    Returns:
        Whether the expression matches after every populated leaf is validated.

    Raises:
        QueryDomainError: If state-only scope is evaluated for an ObjectRef.
        QueryError: If a populated field has an incompatible ordering or
            containment type.
        MetadataConflictError: If connected authority disagrees.

    Side Effects:
        Reads detached metadata only. It never materializes Objects, opens state
        payloads, imports optional backends, or runs collectors or probes.
    """

    _require_predicate(predicate)
    context = _MetadataContext(reference, repo, store)
    values = {id(leaf): _evaluate_leaf(leaf, context) for leaf in _leaves(predicate)}
    return _reduce(predicate, values)


class _MetadataContext:
    def __init__(self, reference, repo, store):
        from ..reference_values import ObjectRef, StateRef

        if not isinstance(reference, (ObjectRef, StateRef)):
            raise TypeError("Metadata query candidates must be ObjectRef or StateRef values.")
        self.reference = reference
        self.repo = repo
        self.store = store
        self._object = _MISSING
        self._state = _MISSING
        self._lineage = _MISSING
        self._snapshot = _MISSING

    def value(self, selector: MetadataField):
        from ..reference_values import StateRef

        if selector.scope == "object":
            target = self.reference.object if isinstance(self.reference, StateRef) else self.reference
            return _traverse(self._current(target, "_object"), selector.path)
        if selector.scope == "state":
            self._require_state(selector.scope)
            return _traverse(self._current(self.reference, "_state"), selector.path)
        if selector.scope == "lineage":
            target = self.reference.object if isinstance(self.reference, StateRef) else self.reference
            if self._lineage is _MISSING:
                self._lineage = _lineage_projection(self.repo.get_lineage_metadata(target, store=self.store))
            return _traverse(self._lineage, selector.path)
        self._require_state(selector.scope)
        if self._snapshot is _MISSING:
            self._snapshot = _snapshot_projection(self.repo.get_snapshot_metadata(self.reference, store=self.store))
        return _traverse(self._snapshot, selector.path)

    def _current(self, target, cache_name):
        value = getattr(self, cache_name)
        if value is _MISSING:
            value = self.repo.get_metadata(target, store=self.store)
            if value is None:
                value = _MISSING
            setattr(self, cache_name, value)
        return value

    def _require_state(self, scope: str) -> None:
        from ..reference_values import StateRef
        from .model import QueryDomainError

        if not isinstance(self.reference, StateRef):
            raise QueryDomainError(f"Metadata scope {scope!r} requires StateRef candidates.")


def _evaluate_leaf(predicate: _LeafPredicate, context: _MetadataContext) -> bool:
    value = context.value(predicate.field)
    if predicate.operator == "exists":
        return value is not _MISSING
    if predicate.operator == "missing":
        return value is _MISSING
    if value is _MISSING:
        return False
    if predicate.operator == "eq":
        return _value_to_data(value) == predicate.value
    if predicate.operator in {"lt", "le", "gt", "ge"}:
        if value is None and _is_timestamp_field(predicate.field):
            return False
        if not _is_number(value):
            raise QueryError("Metadata ordering requires a populated numeric field.")
        operand = _value_from_data(predicate.value)
        if not _is_number(operand):
            raise QueryError("Metadata ordering operand is malformed.")
        return {
            "lt": value < operand,
            "le": value <= operand,
            "gt": value > operand,
            "ge": value >= operand,
        }[predicate.operator]
    if isinstance(value, str):
        needle = _value_from_data(predicate.value)
        if not isinstance(needle, str):
            raise QueryError("String metadata containment requires a string operand.")
        return needle in value
    if isinstance(value, (list, tuple)):
        needle = predicate.value
        return any(_value_to_data(item) == needle for item in value)
    raise QueryError("Metadata containment requires a populated string, list, or tuple field.")


def _reduce(predicate: MetadataPredicate, values: Mapping[int, bool]) -> bool:
    if isinstance(predicate, _LeafPredicate):
        return values[id(predicate)]
    if isinstance(predicate, _BooleanPredicate):
        left = _reduce(predicate.left, values)
        right = _reduce(predicate.right, values)
        return left and right if predicate.operator == "and" else left or right
    if isinstance(predicate, _NotPredicate):
        return not _reduce(predicate.operand, values)
    raise QueryCodecError("Unsupported metadata predicate type.")


def _leaves(predicate: MetadataPredicate):
    if isinstance(predicate, _LeafPredicate):
        yield predicate
    elif isinstance(predicate, _BooleanPredicate):
        yield from _leaves(predicate.left)
        yield from _leaves(predicate.right)
    elif isinstance(predicate, _NotPredicate):
        yield from _leaves(predicate.operand)
    else:
        raise QueryCodecError("Unsupported metadata predicate type.")


def _traverse(value, path):
    for component in path:
        if isinstance(component, str):
            if not isinstance(value, Mapping) or component not in value:
                return _MISSING
            value = value[component]
        else:
            if not isinstance(value, (list, tuple)) or component >= len(value):
                return _MISSING
            value = value[component]
    return value


def _lineage_projection(value) -> dict[str, Any]:
    return {
        "creation_status": value.creation_status,
        "created_at": _timestamp_seconds(value.created_at) if value.created_at is not None else None,
    }


def _snapshot_projection(value) -> dict[str, Any]:
    return {
        "saved_at": _timestamp_seconds(value.saved_at),
        "environment": _environment_projection(value.environment),
        "environment_status": value.environment_status,
        "requirements": _requirement_projection(value.requirements),
        "requirements_status": value.requirements_status,
        "requirements_coverage": value.requirements_coverage,
        "diagnostics": tuple(tuple(item) for item in value.diagnostics),
    }


def _environment_projection(value):
    if value is None:
        return None
    data = value.to_data()["payload"]
    return _plain_projection(data)


def _requirement_projection(value):
    if value is None:
        return None
    return _plain_projection(value._payload())


def _plain_projection(value):
    if isinstance(value, Mapping):
        return {key: _plain_projection(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_projection(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_plain_projection(item) for item in value)
    return value


def _is_timestamp_field(selector: MetadataField) -> bool:
    return (selector.scope, selector.path) in {
        ("lineage", ("created_at",)),
        ("snapshot", ("saved_at",)),
    }


def _operand_data(selector: MetadataField, value: Any, operation: str) -> dict[str, Any]:
    if _is_timestamp_field(selector) and isinstance(value, datetime):
        return _value_to_data(_timestamp_seconds(value))
    if _is_timestamp_field(selector) and _is_number(value):
        return _value_to_data(float(value))
    if isinstance(value, datetime):
        raise TypeError("Datetime operands are only valid for lifecycle timestamp fields.")
    if operation == "order":
        if not _is_number(value):
            raise TypeError("Metadata ordering operands must be finite non-boolean numbers.")
    return _value_to_data(value)


def _timestamp_seconds(value: datetime) -> float:
    if not isinstance(value, datetime):
        raise QueryCodecError("Persisted lifecycle timestamps must be datetime values.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Metadata query timestamps must be timezone-aware.")
    seconds = value.astimezone(timezone.utc).timestamp()
    if not math.isfinite(seconds):
        raise ValueError("Metadata query timestamps must be finite.")
    return seconds


def _is_number(value: Any) -> bool:
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _require_predicate(value) -> MetadataPredicate:
    if not isinstance(value, MetadataPredicate):
        raise TypeError("Metadata Boolean operands must be MetadataPredicate values.")
    _predicate_bounds(value)
    return value


def _predicate_bounds(predicate: MetadataPredicate) -> None:
    leaves = 0

    def visit(item, depth):
        nonlocal leaves
        if depth > _MAX_BOOLEAN_DEPTH:
            raise ValueError("Metadata Boolean expressions may be at most 32 levels deep.")
        if isinstance(item, _LeafPredicate):
            leaves += 1
            if leaves > _MAX_LEAVES:
                raise ValueError("Metadata predicates may contain at most 256 leaves.")
            return
        if isinstance(item, _BooleanPredicate):
            visit(item.left, depth + 1)
            visit(item.right, depth + 1)
            return
        if isinstance(item, _NotPredicate):
            visit(item.operand, depth + 1)
            return
        raise QueryCodecError("Unsupported metadata predicate type.")

    visit(predicate, 0)
    try:
        encoded = _canonical_bytes(predicate.to_data())
    except QueryCodecError as exc:
        raise ValueError("Metadata predicate exceeds canonical codec bounds.") from exc
    if len(encoded) > _MAX_PREDICATE_BYTES:
        raise ValueError("Metadata predicate exceeds canonical codec size bound.")


def _field_from_data(data: Any) -> MetadataField:
    if not isinstance(data, Mapping) or set(data) != {"scope", "path"}:
        raise QueryCodecError("Metadata field data must contain scope and path.")
    path = data["path"]
    if not isinstance(path, list):
        raise QueryCodecError("Metadata field path must be a list.")
    try:
        return field(data["scope"], *path)
    except (TypeError, ValueError) as exc:
        raise QueryCodecError("Metadata field data is invalid.") from exc


def _predicate_from_data(data: Any) -> MetadataPredicate:
    if not isinstance(data, Mapping) or not isinstance(data.get("kind"), str):
        raise QueryCodecError("Metadata predicate data must contain a kind.")
    kind = data["kind"]
    if kind in {"exists", "missing"} and set(data) == {"kind", "field"}:
        return _LeafPredicate(kind, _field_from_data(data["field"]))
    if kind in {"eq", "lt", "le", "gt", "ge", "contains"} and set(data) == {"kind", "field", "value"}:
        value = data["value"]
        selector = _field_from_data(data["field"])
        try:
            operand = _value_from_data(value)
            canonical = _operand_data(
                selector, operand,
                "order" if kind in {"lt", "le", "gt", "ge"} else kind,
            )
        except (TypeError, ValueError) as exc:
            raise QueryCodecError("Metadata predicate operand is invalid.") from exc
        if canonical != value:
            raise QueryCodecError("Metadata predicate operand is noncanonical.")
        return _LeafPredicate(kind, selector, value)
    if kind in {"and", "or"} and set(data) == {"kind", "left", "right"}:
        return _BooleanPredicate(kind, _predicate_from_data(data["left"]), _predicate_from_data(data["right"]))
    if kind == "not" and set(data) == {"kind", "operand"}:
        return _NotPredicate(_predicate_from_data(data["operand"]))
    raise QueryCodecError("Metadata predicate data has unsupported fields or kind.")


def _canonical_bytes(value: Any) -> bytes:
    try:
        return canonical_json_bytes(
            value, max_depth=64, max_nodes=_MAX_PREDICATE_NODES, max_entries=_MAX_ENTRIES,
            max_string=_MAX_STRING, max_int_bits=_MAX_INT_BITS,
        )
    except Exception as exc:
        raise QueryCodecError("Metadata query data is malformed or exceeds codec bounds.") from exc


def _value_to_data(value: Any, *, depth: int = 0, count: list[int] | None = None) -> dict[str, Any]:
    if count is None:
        count = [0]
    count[0] += 1
    if depth > _MAX_VALUE_DEPTH or count[0] > _MAX_NODES:
        raise ValueError("Metadata query values exceed nesting bounds.")
    if value is None:
        return {"kind": "null"}
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if type(value) is int:
        if value.bit_length() > _MAX_INT_BITS:
            raise ValueError("Metadata query integer exceeds size bound.")
        return {"kind": "int", "value": value}
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Metadata query floats must be finite.")
        return {"kind": "float", "value": value}
    if isinstance(value, str):
        if len(value) > _MAX_STRING:
            raise ValueError("Metadata query string exceeds size bound.")
        return {"kind": "str", "value": value}
    if isinstance(value, list):
        if len(value) > _MAX_ENTRIES:
            raise ValueError("Metadata query list exceeds entry bound.")
        return {"kind": "list", "items": [_value_to_data(item, depth=depth + 1, count=count) for item in value]}
    if isinstance(value, tuple):
        if len(value) > _MAX_ENTRIES:
            raise ValueError("Metadata query tuple exceeds entry bound.")
        return {"kind": "tuple", "items": [_value_to_data(item, depth=depth + 1, count=count) for item in value]}
    if isinstance(value, Mapping):
        if len(value) > _MAX_ENTRIES or any(not isinstance(key, str) for key in value):
            raise TypeError("Metadata query mappings require bounded string keys.")
        return {"kind": "mapping", "items": [[key, _value_to_data(value[key], depth=depth + 1, count=count)] for key in sorted(value)]}
    raise TypeError(f"Metadata query values do not support {type(value).__name__}.")


def _value_from_data(data: Any, *, depth: int = 0, count: list[int] | None = None):
    if count is None:
        count = [0]
    count[0] += 1
    if depth > _MAX_VALUE_DEPTH or count[0] > _MAX_NODES:
        raise QueryCodecError("Metadata query value exceeds nesting bounds.")
    if not isinstance(data, Mapping) or not isinstance(data.get("kind"), str):
        raise QueryCodecError("Metadata query value data must contain a kind.")
    kind = data["kind"]
    if kind == "null" and set(data) == {"kind"}:
        return None
    if kind == "bool" and set(data) == {"kind", "value"} and isinstance(data["value"], bool):
        return data["value"]
    if kind == "int" and set(data) == {"kind", "value"} and type(data["value"]) is int:
        _value_to_data(data["value"])
        return data["value"]
    if kind == "float" and set(data) == {"kind", "value"} and type(data["value"]) is float and math.isfinite(data["value"]):
        return data["value"]
    if kind == "str" and set(data) == {"kind", "value"} and isinstance(data["value"], str):
        _value_to_data(data["value"])
        return data["value"]
    if kind in {"list", "tuple"} and set(data) == {"kind", "items"} and isinstance(data["items"], list):
        if len(data["items"]) > _MAX_ENTRIES:
            raise QueryCodecError("Metadata query value exceeds entry bound.")
        items = [_value_from_data(item, depth=depth + 1, count=count) for item in data["items"]]
        return items if kind == "list" else tuple(items)
    if kind == "mapping" and set(data) == {"kind", "items"} and isinstance(data["items"], list):
        if len(data["items"]) > _MAX_ENTRIES:
            raise QueryCodecError("Metadata query value exceeds entry bound.")
        keys = []
        output = {}
        for item in data["items"]:
            if not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str):
                raise QueryCodecError("Metadata query mapping item is invalid.")
            keys.append(item[0])
            output[item[0]] = _value_from_data(item[1], depth=depth + 1, count=count)
        if keys != sorted(set(keys)):
            raise QueryCodecError("Metadata query mapping keys must be unique and sorted.")
        return output
    raise QueryCodecError("Metadata query value data is unsupported.")


__all__ = ["MetadataField", "MetadataPredicate", "evaluate_metadata_predicate", "field", "predicate_requires_state"]
