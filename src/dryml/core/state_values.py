"""Detached immutable state selections and rooted facet-map adapters.

These values describe constraints, not payload availability, publication, or a
Store read lifetime. Constructing or inspecting them performs no IO.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Generic, TypeVar

from .freeze import FrozenDict

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class StateKey:
    """Exact compound address of private state, independent of ObjectId.

    Args:
        cdef_digest (str): Lowercase SHA-256 creation-definition digest.
        codec_hash (str): Codec-qualified content hash, ``codec-<sha256>``.
    Raises:
        ValueError: If either component is malformed.
    Side Effects:
        None; this address does not assert publication or availability.
    """

    cdef_digest: str
    codec_hash: str

    def __post_init__(self):
        from .reference_values import _validate_state_hash

        if not isinstance(self.cdef_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", self.cdef_digest):
            raise ValueError("StateKey cdef_digest must be 64 lowercase hexadecimal digits.")
        _validate_state_hash(self.codec_hash)


@dataclass(frozen=True, slots=True, eq=False)
class LocalState(Generic[T]):
    """One exact detached state selection and its retained creation evidence.

    Args:
        key (StateKey): Full creation/content address.
        creation_cdef (ConcreteDefinition): Immutable normalized creation view.
            The caller supplies evidence; digest/codec/application admission is
            performed by the consuming operation, not by metadata authoring.
    Raises:
        TypeError: If the key or construction evidence has an unsupported type.
    Side Effects:
        None. No payload is opened and no donor ObjectId is retained.
    """

    key: StateKey
    creation_cdef: object

    def __post_init__(self):
        from .definition import ConcreteDefinition

        if not isinstance(self.key, StateKey):
            raise TypeError("LocalState key must be a StateKey.")
        if not isinstance(self.creation_cdef, ConcreteDefinition):
            raise TypeError("LocalState creation_cdef must be a ConcreteDefinition.")

    @property
    def codec(self) -> str:
        """Return the codec name carried by the content component, without IO."""
        return self.key.codec_hash.split("-", 1)[0]

    def validate_for(self, destination, *, codec):
        """Validate origin evidence and destination compatibility before state hooks.

        Args:
            destination (ConcreteNode | ConcreteDefinition): Normalized receiver
                configuration; its ObjectIds and state selections are ignored.
            codec (str): Codec admitted by the consuming destination implementation.
                This must equal the selected state's opaque codec name.
        Returns:
            LocalState: This exact donor selection, without rehoming its StateKey
            or changing either side's retained construction evidence.
        Raises:
            TypeError: For nonconcrete evidence/destination or an invalid codec type.
            ValueError: For malformed creation authority/digest, incompatible graph
                configuration, or a codec not admitted by the destination.
        Side Effects:
            Reuses bounded immutable origin/destination preparation, but checks
            this selection's digest and admitted codec on every call. No target
            resolution, arbitrary compatibility callback, payload lookup/read,
            restore hook or publication.
            Success is metadata admission, not payload availability/authentication.
        """
        from .node_views import CREATION_VIEW, COMPATIBILITY_VIEW, view_equal
        from .node_prepare import _owned_preparation

        if type(codec) is not str:
            raise TypeError("The admitted codec must be a string.")
        if self.codec != codec:
            raise ValueError("Selected state codec is not admitted by the destination.")
        origin = _owned_preparation(self.creation_cdef)
        creation = origin._prepared_view(self.creation_cdef, CREATION_VIEW)
        normalized = creation.roots[0]
        if normalized is not self.creation_cdef:
            original = origin._prepared_view(self.creation_cdef)
            normalized_exact = origin._prepared_view(normalized)
            if not original.exact_equal(self.creation_cdef, normalized_exact, normalized):
                raise ValueError("Selected state needs normalized creation evidence without own facets.")
        if creation.digest(normalized, domain=CREATION_VIEW.name) != self.key.cdef_digest:
            raise ValueError("Selected state creation evidence does not match its creation digest.")
        if not view_equal(self.creation_cdef, destination, COMPATIBILITY_VIEW):
            raise ValueError("Selected state is incompatible with the destination configuration.")
        return self

    def __eq__(self, other):
        from .node_prepare import NodePreparation

        if not isinstance(other, LocalState) or self.key != other.key:
            return False
        a, b = NodePreparation((self.creation_cdef,)), NodePreparation((other.creation_cdef,))
        return a.exact_equal(self.creation_cdef, b, other.creation_cdef)

    def __hash__(self):
        return hash(self.key)


class _FacetSpec(Mapping):
    __slots__ = ("_root",)
    _facet = ""

    def __init__(self, root):
        from .node import Node

        if not isinstance(root, Node):
            raise TypeError("Facet specifications require a Node root.")
        object.__setattr__(self, "_root", root)

    def __setattr__(self, name, value):
        raise AttributeError("Facet specifications are immutable rooted adapters.")

    @property
    def root(self):
        """Return the immutable scope that owns these derived assignments."""
        return self._root

    def _mapping(self):
        from .node import _owned_nodes

        return FrozenDict((path, getattr(node, self._facet))
                          for path, node in _owned_nodes(self.root)
                          if getattr(node, self._facet) is not None)

    def __iter__(self):
        return iter(self._mapping())

    def __len__(self):
        return len(self._mapping())

    def __getitem__(self, path):
        from .utils.graph.path import normalize_path

        return self._mapping()[normalize_path(path)]

    def items(self):
        """Return one derived path/facet item view from a single ownership walk."""
        return self._mapping().items()

    def values(self):
        """Return a read-only facet values view from one ownership walk.

        Values retain primary-path order and their ObjectId or LocalState types.
        Each call derives fresh immutable metadata without IO or authority caching.
        """
        return self._mapping().values()

    def __reduce__(self):
        return type(self), (self.root,)


class IdentitySpec(_FacetSpec):
    """Read-only rooted path/ObjectId adapter over a Node's owned closure.

    Args:
        root (Node): Scope whose present identities are inspected.
    Returns:
        A Mapping with one primary typed path per assigned occurrence; shared
        nodes appear once and paths are rebased to ``root``.
    Raises:
        TypeError: If root is not a Node.
        GraphPathError: If a requested key is malformed or cannot be addressed.
        KeyError: If a requested path has no primary identity assignment.
    Side Effects:
        None. Ref and quotation metadata do not enlarge the ownership closure.
    """
    __slots__ = ()
    _facet = "object_id"


class StateSpec(_FacetSpec):
    """Read-only rooted path/LocalState adapter, without donor identities.

    Args:
        root (Node): Scope whose present private-state selections are inspected.
    Returns:
        A Mapping with one rebased primary typed path per selected occurrence.
    Raises:
        TypeError: If root is not a Node.
        GraphPathError: If a requested key is malformed or cannot be addressed.
        KeyError: If a requested path has no primary state selection.
    Side Effects:
        None. References and quotations remain outside materializing ownership.
    """
    __slots__ = ()
    _facet = "state"
