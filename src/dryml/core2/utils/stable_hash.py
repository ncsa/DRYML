import hashlib
import struct
from dataclasses import dataclass
from inspect import isclass
from datetime import date, datetime, time
from decimal import Decimal
from uuid import UUID
from pathlib import Path
from enum import Enum

import numpy as np

from .types import is_dictlike, is_collection
from .recurse import cycle_detect
from .graph import GraphHasher, GraphCtx

def stable_int_hash(s: str, *, bits: int = 64) -> int:
    # blake2b is fast and stable; digest_size controls output size
    digest = hashlib.blake2b(s.encode("utf-8"), digest_size=bits // 8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)

# ---------- LEAF ENCODING (DETERMINISTIC) ----------

def _stable_leaf_bytes(value) -> bytes:
    """
    Canonical byte representation for a single *leaf* value.
    No use of Python's built-in hash().
    """

    # Classes: use module + qualname
    if isclass(value):
        s = f"class:{value.__module__}.{value.__qualname__}"
        return s.encode("utf-8")

    # Enums: type + name
    if isinstance(value, Enum):
        s = f"enum:{type(value).__module__}.{type(value).__qualname__}:{value.name}"
        return s.encode("utf-8")

    # None
    if value is None:
        return b"N"

    # Booleans first (bool is a subclass of int)
    if isinstance(value, bool):
        return b"B1" if value else b"B0"

    # Plain ints
    if isinstance(value, int):
        # Stable decimal representation
        return b"I" + str(value).encode("ascii")

    # Floats (Python float = IEEE754 binary64)
    if isinstance(value, float):
        # binary representation to avoid repr/locale issues
        return b"F" + struct.pack(">d", value)

    # Strings
    if isinstance(value, str):
        return b"S" + value.encode("utf-8")

    # Raw bytes / byte-like
    if isinstance(value, (bytes, bytearray, memoryview)):
        return b"Y" + bytes(value)

    # NumPy arrays: include shape + dtype + raw bytes
    if isinstance(value, np.ndarray):
        return (
            b"A"
            + str(value.shape).encode("ascii")
            + b"|"
            + str(value.dtype).encode("ascii")
            + b"|"
            + value.tobytes()
        )

    # NumPy scalar
    if isinstance(value, np.generic):
        return (
            b"Ng"
            + str(value.dtype).encode("ascii")
            + b"|"
            + value.tobytes()
        )

    # Datetime-like
    if isinstance(value, (datetime, date, time)):
        # isoformat is deterministic for these
        return b"D" + value.isoformat().encode("ascii")

    # Decimal
    if isinstance(value, Decimal):
        # normalize to canonical form, then 'f' to avoid sci-notation variability
        return b"De" + format(value.normalize(), "f").encode("ascii")

    # UUID
    if isinstance(value, UUID):
        return b"U" + value.hex.encode("ascii")

    # Path
    if isinstance(value, Path):
        # string form is stable enough across runs
        return b"P" + str(value).encode("utf-8")

    # custom implemented stable hash
    if hasattr(value, "__stable_leaf_bytes__"):
        return value.__stable_leaf_bytes__()

    # If you want to support more types, add them explicitly above.
    # Falling back to repr() or pickle here *would* risk non-determinism.
    raise TypeError(f"Unsupported leaf type for stable hashing: {type(value)!r}")


def stable_hash_value(value) -> str:
    """
    Deterministic hash for a single leaf value (no containers).
    """
    return hashlib.sha256(_stable_leaf_bytes(value)).hexdigest()


# ----------------------------------------------------------------------
# graph hasher
# ----------------------------------------------------------------------

class StableHashGraphHasher(GraphHasher):
    def is_atomic(self, obj, ctx: GraphCtx) -> bool:
        from ..canonical import node_kind, NodeKind

        kind = node_kind(obj)
        return kind in {
            NodeKind.POD,
            NodeKind.TYPE,
            NodeKind.IDENTITY_VALUE,
            NodeKind.NDARRAY,
            NodeKind.FROZEN_NDARRAY,
            NodeKind.IMPORT_REF,
            NodeKind.SOURCE_SPEC,
        }

    def hash_atomic(self, obj, ctx: GraphCtx) -> str:
        return stable_hash_value(obj)

    def should_track_cycle(self, obj, ctx: GraphCtx) -> bool:
        from ..canonical import node_kind, NodeKind

        kind = node_kind(obj)
        return kind in {
            NodeKind.LIST,
            NodeKind.TUPLE,
            NodeKind.SET,
            NodeKind.DICT,
            NodeKind.FROZEN_LIST,
            NodeKind.FROZEN_TUPLE,
            NodeKind.FROZEN_SET,
            NodeKind.FROZEN_DICT,
            NodeKind.DEFINITION,
            NodeKind.CONCRETE_DEFINITION,
            NodeKind.DEFLINK,
            NodeKind.QUOTED_DEF,
            NodeKind.SELECTOR_SPEC,
            NodeKind.SELECTOR,
            NodeKind.PAR,
            NodeKind.OBJECT,
        }

    def dispatch(self, obj, ctx: GraphCtx) -> str:
        from ..canonical import node_kind, NodeKind
        from ..definition import Definition, ConcreteDefinition
        from ..freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
        from ..links import DefLink
        from ..object import Object
        from ..params import Par
        from ..quoted import QuotedDef, SelectorSpec
        from ..selector import Selector

        kind = node_kind(obj)

        if kind is NodeKind.OBJECT:
            return self.hash(obj.definition, ctx.child("definition"))

        if kind in {NodeKind.LIST, NodeKind.FROZEN_LIST}:
            return self._hash_sequence("builtins.list", obj, ctx)

        if kind in {NodeKind.TUPLE, NodeKind.FROZEN_TUPLE}:
            return self._hash_sequence("builtins.tuple", obj, ctx)

        if kind in {NodeKind.SET, NodeKind.FROZEN_SET}:
            return self._hash_set("builtins.set", obj, ctx)

        if kind in {NodeKind.DICT, NodeKind.FROZEN_DICT}:
            return self._hash_mapping("builtins.dict", obj, ctx)

        if isinstance(obj, (Definition, ConcreteDefinition)):
            type_marker = f"{type(obj).__module__}.{type(obj).__qualname__}"
            items = [(k, obj[k]) for k in obj]
            return self._hash_mapping(type_marker, dict(items), ctx)

        if isinstance(obj, DefLink):
            type_marker = f"{type(obj).__module__}.{type(obj).__qualname__}"
            return self._hash_mapping(type_marker, {"kind": obj.kind.value, "target": obj.target}, ctx)

        if isinstance(obj, QuotedDef):
            type_marker = f"{type(obj).__module__}.{type(obj).__qualname__}"
            return self._hash_mapping(type_marker, {"value": obj.value}, ctx)

        if isinstance(obj, SelectorSpec):
            type_marker = f"{type(obj).__module__}.{type(obj).__qualname__}"
            return self._hash_mapping(type_marker, {"selector": obj.selector}, ctx)

        if isinstance(obj, Selector):
            type_marker = f"{type(obj).__module__}.{type(obj).__qualname__}"
            return self._hash_mapping(type_marker, {"root": obj.root, "strict": obj.strict, "cls_policy": obj.cls_policy}, ctx)

        if isinstance(obj, Par):
            type_marker = f"{type(obj).__module__}.{type(obj).__qualname__}"
            return self._hash_mapping(type_marker, {"stable_key": obj.stable_key()}, ctx)

        raise TypeError(f"Unsupported type {type(obj)} for stable hashing")

    def _hash_sequence(self, type_marker: str, seq, ctx: GraphCtx) -> str:
        """
        Order-sensitive hashing.
        """
        hasher = hashlib.sha256()
        hasher.update(b"T" + type_marker.encode("utf-8"))
        hasher.update(b"|" + str(len(seq)).encode("ascii"))

        for i, v in enumerate(seq):
            child_hash = self.hash(v, ctx.child(i))
            hasher.update(b"I" + str(i).encode("ascii"))
            hasher.update(b"V" + child_hash.encode("utf-8"))

        return hasher.hexdigest()

    def _hash_set(self, type_marker: str, st, ctx: GraphCtx) -> str:
        """
        Order-insensitive hashing by child hash.
        """
        hasher = hashlib.sha256()
        hasher.update(b"T" + type_marker.encode("utf-8"))
        hasher.update(b"|" + str(len(st)).encode("ascii"))

        child_hashes = sorted(self.hash(v, ctx.child(i)) for i, v in enumerate(st))
        for child_hash in child_hashes:
            hasher.update(b"V" + child_hash.encode("utf-8"))

        return hasher.hexdigest()

    def _hash_mapping(self, type_marker: str, mp, ctx: GraphCtx) -> str:
        """
        Order-insensitive by key hash, key-sensitive, value-sensitive.
        """
        hasher = hashlib.sha256()
        hasher.update(b"T" + type_marker.encode("utf-8"))
        hasher.update(b"|" + str(len(mp)).encode("ascii"))

        key_val_hashes = []
        for k, v in mp.items():
            key_hash = self.hash(k, ctx.child("<key>"))
            val_hash = self.hash(v, ctx.child(k if isinstance(k, (str, int)) else str(k)))
            key_val_hashes.append((key_hash, val_hash))

        key_val_hashes.sort(key=lambda kv: kv[0])

        for key_hash, val_hash in key_val_hashes:
            hasher.update(b"K" + key_hash.encode("utf-8"))
            hasher.update(b"V" + val_hash.encode("utf-8"))

        return hasher.hexdigest()


def stable_hash_function(structure, cache=None) -> str:
    ctx = GraphCtx(memo={} if cache is None else cache)
    return StableHashGraphHasher().hash(structure, ctx)


@dataclass(frozen=True, slots=True)
class StableHashLimits:
    """Budgets for :func:`bounded_stable_hash_function`.

    The bounded hasher produces the same digest as :func:`stable_hash_function`
    while charging graph work before recursion or byte encoding.  Limits count
    the root at depth zero, every incoming parent/child edge, every value
    occurrence (including memo hits), and every byte passed to a SHA-256 update.
    """

    max_depth: int = 128
    max_occurrences: int = 100_000
    max_edges: int = 200_000
    max_encoded_bytes: int = 4_194_304
    max_integer_bits: int = 4_096
    max_string_chars: int = 4_096


class StableHashLimitError(ValueError):
    """Raised before a bounded stable-hash budget would be exceeded."""

    def __init__(self, limit_name: str, limit: int, observed_lower_bound: int):
        super().__init__(f"stable hash {limit_name} limit exceeded")
        self.limit_name = limit_name
        self.limit = limit
        self.observed_lower_bound = observed_lower_bound


@dataclass(slots=True)
class _HashBudget:
    limits: StableHashLimits
    occurrences: int = 0
    edges: int = 0
    encoded_bytes: int = 0

    def charge_value(self, depth: int, incoming_edge: bool) -> None:
        self._check("depth", self.limits.max_depth, depth)
        self.occurrences += 1
        self._check("occurrences", self.limits.max_occurrences, self.occurrences)
        if incoming_edge:
            self.edges += 1
            self._check("edges", self.limits.max_edges, self.edges)

    def charge_bytes(self, size: int) -> None:
        observed = self.encoded_bytes + size
        self._check("encoded_bytes", self.limits.max_encoded_bytes, observed)
        self.encoded_bytes = observed

    @staticmethod
    def _check(name: str, limit: int, observed: int) -> None:
        if observed > limit:
            raise StableHashLimitError(name, limit, observed)


class _BoundedStableHasher:
    """StableHashGraphHasher-compatible implementation with shared budgets."""

    def __init__(self, limits: StableHashLimits):
        self.budget = _HashBudget(limits)
        self.limits = limits
        self.active_ids: set[int] = set()

    def hash(self, value) -> str:
        return self._hash(value, depth=0, incoming_edge=False, memo={})

    def _hash(self, value, *, depth: int, incoming_edge: bool, memo: dict[int, str]) -> str:
        from ..canonical import NodeKind, node_kind

        self.budget.charge_value(depth, incoming_edge)
        kind = node_kind(value)
        self._require_explicit_identity_decomposition(value, kind)
        atomic = kind in {
            NodeKind.POD,
            NodeKind.TYPE,
            NodeKind.IDENTITY_VALUE,
            NodeKind.NDARRAY,
            NodeKind.FROZEN_NDARRAY,
            NodeKind.IMPORT_REF,
            NodeKind.SOURCE_SPEC,
        }
        if atomic:
            return self._hash_atomic(value, depth=depth)

        key = id(value)
        if key in memo:
            return memo[key]
        if key in self.active_ids:
            from .graph.hasher import GraphHashError

            raise GraphHashError("Cycle detected while bounded stable hashing")
        self.active_ids.add(key)
        try:
            digest = self._hash_non_atomic(value, kind=kind, depth=depth, memo=memo)
            memo[key] = digest
            return digest
        finally:
            self.active_ids.remove(key)

    @staticmethod
    def _require_explicit_identity_decomposition(value, kind) -> None:
        """Reject unsupported atomic leaf variants before representation hooks run."""

        from ..canonical import NodeKind
        from ..cardinality import Cardinality
        from ..config import ConfigRef
        from ..dtype import DType
        from ..factory import FactorySpec
        from ..symbol import ImportRef, SourceSpec
        from ..tensor_spec import TensorSpec

        # ``node_kind`` intentionally classifies native scalar subclasses as
        # POD so Definitions retain their normal canonical surface.  The
        # bounded hasher cannot accept those subclasses, however: the legacy
        # leaf encoder uses ``str(value)`` for integers and ``value.encode()``
        # for strings, either of which can dispatch to an unbounded custom
        # hook.  Admit only exact native POD leaves before any leaf encoder or
        # representation/conversion operation is reached.
        if (
            kind is NodeKind.POD
            and isinstance(value, (int, float, str, bytes))
            and type(value) not in {bool, int, float, str, bytes}
        ):
            raise TypeError("Unsupported Python POD subclass for bounded stable hashing")

        # Enum members have an explicit generic decomposition below. The other
        # allowlisted identity values are only bounded for their exact current
        # types; accepting subclasses would permit an unbounded leaf override.
        if kind is NodeKind.IDENTITY_VALUE and isinstance(value, Enum):
            return
        if kind is NodeKind.IDENTITY_VALUE and type(value) not in {
            DType, TensorSpec, Cardinality, ConfigRef, FactorySpec,
        }:
            raise TypeError(
                "Unsupported identity-value type for bounded stable hashing: "
                f"{type(value)!r}"
            )
        if kind in {NodeKind.IMPORT_REF, NodeKind.SOURCE_SPEC} and type(value) not in {
            ImportRef, SourceSpec,
        }:
            raise TypeError(
                "Unsupported atomic leaf type for bounded stable hashing: "
                f"{type(value)!r}"
            )

    def _hash_atomic(self, value, *, depth: int) -> str:
        from ..cardinality import Cardinality
        from ..config import CONFIG_MISSING, ConfigRef
        from ..dtype import DType
        from ..factory import FactorySpec
        from ..freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
        from ..symbol import ImportRef, SourceSpec
        from ..tensor_spec import TensorSpec

        if isinstance(value, type):
            module = type.__getattribute__(value, "__module__")
            qualname = type.__getattribute__(value, "__qualname__")
            for field_value in (module, qualname):
                if not isinstance(field_value, str) or len(field_value) > self.limits.max_string_chars:
                    observed = len(field_value) if isinstance(field_value, str) else self.limits.max_string_chars + 1
                    raise StableHashLimitError("string_chars", self.limits.max_string_chars, observed)
            return self._leaf_digest(f"class:{module}.{qualname}".encode("utf-8"))
        if isinstance(value, np.ndarray):
            prefix = (
                b"A" + str(value.shape).encode("ascii") + b"|"
                + str(value.dtype).encode("ascii") + b"|"
            )
            observed = self.budget.encoded_bytes + len(prefix) + int(value.nbytes)
            self.budget._check("encoded_bytes", self.limits.max_encoded_bytes, observed)
            return self._leaf_digest(prefix + value.tobytes())
        if isinstance(value, np.generic):
            prefix = b"Ng" + str(value.dtype).encode("ascii") + b"|"
            observed = self.budget.encoded_bytes + len(prefix) + int(value.nbytes)
            self.budget._check("encoded_bytes", self.limits.max_encoded_bytes, observed)
            return self._leaf_digest(prefix + value.tobytes())
        if isinstance(value, bytes):
            # Check the complete leaf size before constructing a prefixed copy.
            # Definitions may legitimately contain multi-megabyte byte payloads,
            # so relying on ``_stable_leaf_bytes`` here would allocate first and
            # enforce the encoded-byte budget afterwards.
            observed = self.budget.encoded_bytes + 1 + len(value)
            self.budget._check("encoded_bytes", self.limits.max_encoded_bytes, observed)
            hasher = hashlib.sha256()
            self._update(hasher, b"Y")
            self._update(hasher, value)
            return hasher.hexdigest()

        if isinstance(value, FactorySpec):
            synthetic = ("dryml.core2.FactorySpec", value.target, value.args, value.kwargs)
            inner = self._hash(synthetic, depth=depth + 1, incoming_edge=True, memo={})
            return self._leaf_digest(inner.encode("ascii"))

        if isinstance(value, ConfigRef):
            self._account_only(value.key, depth=depth + 1, incoming_edge=True)
            if value.default is CONFIG_MISSING:
                self._account_only("<missing>", depth=depth + 1, incoming_edge=True)
                leaf = f"ConfigRef:{value.key}:<missing>".encode("utf-8")
            else:
                default_digest = self._hash(value.default, depth=depth + 1, incoming_edge=True, memo={})
                leaf = f"ConfigRef:{value.key}:{default_digest}".encode("utf-8")
            return self._leaf_digest(leaf)

        fields = None
        if isinstance(value, DType):
            fields = (value.kind, value.bits)
        elif isinstance(value, TensorSpec):
            fields = (
                value.dtype, value.shape, value.batch, value.backend, value.layout,
                value.axis_names, value.batch_axis_name, value.ragged_rank,
                value.row_splits_dtype, value.sparse_format,
            )
        elif isinstance(value, Cardinality):
            fields = (value.kind, value.value)
        elif isinstance(value, ImportRef):
            fields = (value.module, value.qualname)
        elif isinstance(value, SourceSpec):
            fields = (value.kind, value.source, value.name, value.imports)
        elif isinstance(value, Enum):
            enum_type = type(value)
            fields = (
                type.__getattribute__(enum_type, "__module__"),
                type.__getattribute__(enum_type, "__qualname__"),
                object.__getattribute__(value, "_name_"),
            )
        if fields is not None:
            for field_value in fields:
                self._account_only(field_value, depth=depth + 1, incoming_edge=True)

        self._precheck_atomic(value)
        if isinstance(value, Enum):
            enum_type = type(value)
            leaf = (
                f"enum:{type.__getattribute__(enum_type, '__module__')}."
                f"{type.__getattribute__(enum_type, '__qualname__')}:"
                f"{object.__getattribute__(value, '_name_')}"
            ).encode("utf-8")
        else:
            leaf = _stable_leaf_bytes(value)
        return self._leaf_digest(leaf)

    def _precheck_atomic(self, value) -> None:
        if type(value) is int and value.bit_length() > self.limits.max_integer_bits:
            raise StableHashLimitError("integer_bits", self.limits.max_integer_bits, value.bit_length())
        if type(value) is str and len(value) > self.limits.max_string_chars:
            raise StableHashLimitError("string_chars", self.limits.max_string_chars, len(value))
        if isinstance(value, np.ndarray):
            self.budget._check("encoded_bytes", self.limits.max_encoded_bytes, self.budget.encoded_bytes + int(value.nbytes))
        if isinstance(value, np.generic):
            self.budget._check("encoded_bytes", self.limits.max_encoded_bytes, self.budget.encoded_bytes + int(value.nbytes))

    def _leaf_digest(self, data: bytes) -> str:
        hasher = hashlib.sha256()
        self._update(hasher, data)
        return hasher.hexdigest()

    def _hash_non_atomic(self, value, *, kind, depth: int, memo: dict[int, str]) -> str:
        from ..canonical import NodeKind
        from ..definition import ConcreteDefinition, Definition
        from ..links import DefLink
        from ..object import Object
        from ..params import Par
        from ..quoted import QuotedDef, SelectorSpec
        from ..selector import Selector

        if kind is NodeKind.OBJECT:
            return self._hash(value.definition, depth=depth + 1, incoming_edge=True, memo=memo)
        if kind in {NodeKind.LIST, NodeKind.FROZEN_LIST}:
            return self._hash_sequence("builtins.list", value, depth=depth, memo=memo)
        if kind in {NodeKind.TUPLE, NodeKind.FROZEN_TUPLE}:
            return self._hash_sequence("builtins.tuple", value, depth=depth, memo=memo)
        if kind in {NodeKind.SET, NodeKind.FROZEN_SET}:
            return self._hash_set("builtins.set", value, depth=depth, memo=memo)
        if kind in {NodeKind.DICT, NodeKind.FROZEN_DICT}:
            return self._hash_mapping("builtins.dict", value, depth=depth, memo=memo)
        if isinstance(value, (Definition, ConcreteDefinition)):
            marker = f"{type(value).__module__}.{type(value).__qualname__}"
            # Definition and ConcreteDefinition already implement Mapping.
            # Traversing that view directly avoids materializing an unbounded
            # intermediate copy before the occurrence/edge budgets are charged.
            return self._hash_mapping(marker, value, depth=depth, memo=memo)
        if isinstance(value, DefLink):
            marker = f"{type(value).__module__}.{type(value).__qualname__}"
            return self._hash_mapping(marker, {"kind": value.kind.value, "target": value.target}, depth=depth, memo=memo)
        if isinstance(value, QuotedDef):
            marker = f"{type(value).__module__}.{type(value).__qualname__}"
            return self._hash_mapping(marker, {"value": value.value}, depth=depth, memo=memo)
        if isinstance(value, SelectorSpec):
            marker = f"{type(value).__module__}.{type(value).__qualname__}"
            return self._hash_mapping(marker, {"selector": value.selector}, depth=depth, memo=memo)
        if isinstance(value, Selector):
            marker = f"{type(value).__module__}.{type(value).__qualname__}"
            return self._hash_mapping(marker, {"root": value.root, "strict": value.strict, "cls_policy": value.cls_policy}, depth=depth, memo=memo)
        if isinstance(value, Par):
            marker = f"{type(value).__module__}.{type(value).__qualname__}"
            return self._hash_mapping(marker, {"stable_key": value.stable_key()}, depth=depth, memo=memo)
        raise TypeError(f"Unsupported type {type(value)} for stable hashing")

    def _hash_sequence(self, marker: str, sequence, *, depth: int, memo: dict[int, str]) -> str:
        hasher = hashlib.sha256()
        self._update(hasher, b"T" + marker.encode("utf-8"))
        self._update(hasher, b"|" + str(len(sequence)).encode("ascii"))
        for index, child in enumerate(sequence):
            digest = self._hash(child, depth=depth + 1, incoming_edge=True, memo=memo)
            self._update(hasher, b"I" + str(index).encode("ascii"))
            self._update(hasher, b"V" + digest.encode("utf-8"))
        return hasher.hexdigest()

    def _hash_set(self, marker: str, values, *, depth: int, memo: dict[int, str]) -> str:
        hasher = hashlib.sha256()
        self._update(hasher, b"T" + marker.encode("utf-8"))
        self._update(hasher, b"|" + str(len(values)).encode("ascii"))
        digests = sorted(
            self._hash(child, depth=depth + 1, incoming_edge=True, memo=memo)
            for child in values
        )
        for digest in digests:
            self._update(hasher, b"V" + digest.encode("utf-8"))
        return hasher.hexdigest()

    def _hash_mapping(self, marker: str, mapping, *, depth: int, memo: dict[int, str]) -> str:
        hasher = hashlib.sha256()
        self._update(hasher, b"T" + marker.encode("utf-8"))
        self._update(hasher, b"|" + str(len(mapping)).encode("ascii"))
        pairs = []
        for key, child in mapping.items():
            key_digest = self._hash(key, depth=depth + 1, incoming_edge=True, memo=memo)
            child_digest = self._hash(child, depth=depth + 1, incoming_edge=True, memo=memo)
            pairs.append((key_digest, child_digest))
        pairs.sort(key=lambda pair: pair[0])
        for key_digest, child_digest in pairs:
            self._update(hasher, b"K" + key_digest.encode("utf-8"))
            self._update(hasher, b"V" + child_digest.encode("utf-8"))
        return hasher.hexdigest()

    def _account_only(self, value, *, depth: int, incoming_edge: bool) -> None:
        """Charge identity metadata recursively without writing a child digest."""

        from ..canonical import node_kind
        from ..cardinality import Cardinality
        from ..config import CONFIG_MISSING, ConfigRef
        from ..dtype import DType
        from ..factory import FactorySpec
        from ..freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
        from ..symbol import ImportRef, SourceSpec
        from ..tensor_spec import TensorSpec

        self.budget.charge_value(depth, incoming_edge)
        self._require_explicit_identity_decomposition(value, node_kind(value))
        if type(value) is int:
            bits = value.bit_length()
            if bits > self.limits.max_integer_bits:
                raise StableHashLimitError("integer_bits", self.limits.max_integer_bits, bits)
            self.budget.charge_bytes(len(str(value).encode("ascii")))
            return
        if type(value) is str:
            if len(value) > self.limits.max_string_chars:
                raise StableHashLimitError("string_chars", self.limits.max_string_chars, len(value))
            self.budget.charge_bytes(len(value.encode("utf-8")))
            return
        if value is None or type(value) is bool or type(value) is float:
            self.budget.charge_bytes(len(_stable_leaf_bytes(value)))
            return
        if isinstance(value, (bytes, bytearray, memoryview)):
            self.budget.charge_bytes(len(value))
            return
        if isinstance(value, (dict, FrozenDict)):
            for key, child in value.items():
                self._account_only(key, depth=depth + 1, incoming_edge=True)
                self._account_only(child, depth=depth + 1, incoming_edge=True)
            return
        if isinstance(value, (tuple, list, set, frozenset, FrozenList, FrozenTuple, FrozenSet)):
            for child in value:
                self._account_only(child, depth=depth + 1, incoming_edge=True)
            return
        if isinstance(value, Enum):
            enum_type = type(value)
            for child in (
                type.__getattribute__(enum_type, "__module__"),
                type.__getattribute__(enum_type, "__qualname__"),
                object.__getattribute__(value, "_name_"),
            ):
                self._account_only(child, depth=depth + 1, incoming_edge=True)
            leaf = (
                f"enum:{type.__getattribute__(enum_type, '__module__')}."
                f"{type.__getattribute__(enum_type, '__qualname__')}:"
                f"{object.__getattribute__(value, '_name_')}"
            ).encode("utf-8")
            self.budget.charge_bytes(len(leaf))
            return
        fields = None
        if isinstance(value, DType):
            fields = (value.kind, value.bits)
        elif isinstance(value, TensorSpec):
            fields = (
                value.dtype, value.shape, value.batch, value.backend, value.layout,
                value.axis_names, value.batch_axis_name, value.ragged_rank,
                value.row_splits_dtype, value.sparse_format,
            )
        elif isinstance(value, Cardinality):
            fields = (value.kind, value.value)
        elif isinstance(value, ImportRef):
            fields = (value.module, value.qualname)
        elif isinstance(value, SourceSpec):
            fields = (value.kind, value.source, value.name, value.imports)
        if fields is not None:
            for child in fields:
                self._account_only(child, depth=depth + 1, incoming_edge=True)
            self.budget.charge_bytes(len(_stable_leaf_bytes(value)))
            return
        if isinstance(value, FactorySpec):
            synthetic = ("dryml.core2.FactorySpec", value.target, value.args, value.kwargs)
            self._account_only(synthetic, depth=depth + 1, incoming_edge=True)
            self.budget.charge_bytes(64)
            return
        if isinstance(value, ConfigRef):
            self._account_only(value.key, depth=depth + 1, incoming_edge=True)
            if value.default is CONFIG_MISSING:
                self._account_only("<missing>", depth=depth + 1, incoming_edge=True)
                self.budget.charge_bytes(len(b"ConfigRef::" + value.key.encode("utf-8") + b"<missing>"))
            else:
                self._account_only(value.default, depth=depth + 1, incoming_edge=True)
                self.budget.charge_bytes(len(b"ConfigRef::" + value.key.encode("utf-8")) + 64)
            return
        # Identity fields may themselves contain an accepted identity value or
        # symbol reference. Charge their final existing leaf bytes after safe
        # decomposition by the normal atomic path in an isolated temporary hash.
        self._precheck_atomic(value)
        self.budget.charge_bytes(len(_stable_leaf_bytes(value)))

    def _update(self, hasher, data: bytes) -> None:
        self.budget.charge_bytes(len(data))
        hasher.update(data)


def bounded_stable_hash_function(structure, *, limits: StableHashLimits | None = None) -> str:
    """Return the existing stable digest while enforcing traversal budgets.

    Unlike a preflight check, budget accounting occurs in the same traversal
    that computes the digest, so a Definition cannot pass validation and then
    trigger an unmetered second hash walk. Native Python POD subclasses are
    rejected before their representation or conversion hooks can run.
    """

    selected = limits or StableHashLimits()
    for name in (
        "max_depth", "max_occurrences", "max_edges", "max_encoded_bytes",
        "max_integer_bits", "max_string_chars",
    ):
        value = getattr(selected, name)
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    return _BoundedStableHasher(selected).hash(structure)


__all__ = [
    "StableHashLimitError",
    "StableHashLimits",
    "bounded_stable_hash_function",
    "stable_hash_function",
    "stable_hash_value",
    "stable_int_hash",
]
