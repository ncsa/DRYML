"""Authority-verified queries over lightweight ObjectRef and StateRef values.

Reference queries deliberately verify immutable Store records. Persistent and
memory query indexes may cache candidate facts, but never define the answer:
every result in this module is reconstructed from Store authority.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Iterator

from ..cdef_graph import EdgeKind
from ..definition import ConcreteDefinition, Definition
from ..links import DefLink
from ..reference_values import ObjectId, ObjectRef, StateRef
from ..utils.graph.path import GraphPath, graph_path_sort_key, normalize_path
from ..utils.graph.value import iter_value_edges
from .query import _query_match


def _reference_key(value: ObjectRef | StateRef) -> tuple[str, str]:
    """Return a total canonical ordering key for a lightweight reference."""

    return ("state" if isinstance(value, StateRef) else "object", value.digest())


def _owner_key(value: Any) -> tuple[str, str]:
    if isinstance(value, (ObjectRef, StateRef)):
        return _reference_key(value)
    if isinstance(value, ConcreteDefinition):
        return ("definition", value.graph_hash())
    return (type(value).__qualname__, repr(value))


@dataclass(frozen=True, slots=True)
class ReferenceOccurrence:
    """One exact lightweight reference occurrence in immutable authority.

    Attributes:
        owner: Complete aggregate reference or Definition record containing the
            value.  Complete reference owners retain durable aggregate identity.
        path: Typed GraphPath from ``owner`` to ``value``.
        value: Exact ObjectRef or StateRef found at the path.
    """

    owner: Any
    path: GraphPath
    value: ObjectRef | StateRef


def _occurrence_key(item: ReferenceOccurrence) -> tuple[Any, ...]:
    return (_owner_key(item.owner), graph_path_sort_key(item.path), _reference_key(item.value))


def _iter_embedded_references(value: Any, *, owner: Any, path: GraphPath = GraphPath(), active: set[int] | None = None):
    """Yield exact reference leaves with typed paths without resolving symbols."""

    if active is None:
        active = set()
    if isinstance(value, (ObjectRef, StateRef)):
        yield ReferenceOccurrence(owner, path, value)
        return
    if isinstance(value, DefLink):
        # Ref and Mat remain distinguishable in the typed path through their
        # enclosing parameter; neither is projected to a structural CDef.
        yield from _iter_embedded_references(value.target, owner=owner, path=path, active=active)
        return
    if isinstance(value, ConcreteDefinition):
        marker = id(value)
        if marker in active:
            return
        active.add(marker)
        try:
            for edge in iter_value_edges(value):
                yield from _iter_embedded_references(edge.value, owner=owner, path=path.child(edge.segment), active=active)
        finally:
            active.remove(marker)
        return
    for edge in iter_value_edges(value):
        yield from _iter_embedded_references(edge.value, owner=owner, path=path.child(edge.segment), active=active)


class ReferenceResultSet:
    """Deterministic result set of exact reference occurrences.

    Occurrences are deduplicated only when owner, typed path, and complete
    reference identity are all canonical replicas.  Equal values belonging to
    different owners are intentionally retained as separate entries.
    """

    def __init__(self, repo, occurrences: Iterable[ReferenceOccurrence]):
        self.repo = repo
        unique = {_occurrence_key(item): item for item in occurrences}
        self._occurrences = tuple(unique[key] for key in sorted(unique))

    def __iter__(self) -> Iterator[ReferenceOccurrence]:
        """Iterate occurrences in canonical owner/path/value order."""

        return iter(self._occurrences)

    def __len__(self) -> int:
        """Return the number of distinct owner/path/value occurrences."""

        return len(self._occurrences)

    def count(self) -> int:
        """Return the number of distinct occurrences."""

        return len(self)

    def exists(self) -> bool:
        """Return whether this result has at least one occurrence."""

        return bool(self._occurrences)

    def one(self) -> ReferenceOccurrence:
        """Return the sole occurrence or raise when cardinality is not one."""

        if len(self) != 1:
            from .model import QueryCardinalityError
            raise QueryCardinalityError(f"Expected exactly one reference occurrence, found {len(self)}.")
        return self._occurrences[0]

    def one_or_none(self) -> ReferenceOccurrence | None:
        """Return the sole occurrence, ``None``, or raise for multiple entries."""

        if len(self) > 1:
            from .model import QueryCardinalityError
            raise QueryCardinalityError(f"Expected zero or one reference occurrence, found {len(self)}.")
        return self._occurrences[0] if self._occurrences else None


class _ReferenceValueResultSet:
    """Shared deterministic behavior for complete reference projections.

    Values deduplicate only canonical replicas of the same complete lightweight
    identity. Retained occurrences remain available to recover the owners and
    typed paths that produced each projection.
    """

    value_type: type

    def __init__(self, repo, values: Iterable[ObjectRef | StateRef], occurrences: Iterable[ReferenceOccurrence] = ()):
        self.repo = repo
        unique = {_reference_key(value): value for value in values}
        self._values = tuple(unique[key] for key in sorted(unique))
        self._occurrences = tuple(occurrences)

    def __iter__(self):
        """Iterate complete references in canonical identity order."""

        return iter(self._values)

    def __len__(self) -> int:
        """Return the number of complete canonical references."""

        return len(self._values)

    def count(self) -> int:
        """Return the number of complete canonical references."""

        return len(self)

    def exists(self) -> bool:
        """Return whether at least one complete reference matched."""

        return bool(self._values)

    def one(self):
        """Return the sole reference or raise when cardinality is not one."""

        if len(self) != 1:
            from .model import QueryCardinalityError
            raise QueryCardinalityError(f"Expected exactly one reference result, found {len(self)}.")
        return self._values[0]

    def one_or_none(self):
        """Return the sole reference, ``None``, or raise for multiple entries."""

        if len(self) > 1:
            from .model import QueryCardinalityError
            raise QueryCardinalityError(f"Expected zero or one reference result, found {len(self)}.")
        return self._values[0] if self._values else None

    def occurrences(self) -> ReferenceResultSet:
        """Project retained owner/path/value occurrences without collapsing owners."""

        values = set(self._values)
        return ReferenceResultSet(self.repo, (item for item in self._occurrences if item.value in values))


class ObjectRefResultSet(_ReferenceValueResultSet):
    """Deterministic projection of complete ObjectRefs from Store authority.

    The terminal returns lightweight immutable identities, never live Objects
    or local-state payloads. Equal replicas deduplicate by complete ObjectRef
    identity, while :meth:`occurrences` retains distinct owners and paths.
    """

    value_type = ObjectRef


class StateRefResultSet(_ReferenceValueResultSet):
    """Deterministic projection of complete StateRefs from Store authority.

    The terminal returns exact immutable state identities without restoration.
    Equal replicas deduplicate by complete StateRef identity, while
    :meth:`occurrences` retains distinct owners and paths.
    """

    value_type = StateRef


class ReferenceQuery:
    """Composable authority query for ObjectRef and StateRef metadata.

    The query never materializes Objects, opens local-state payloads, imports
    optional backends, or trusts a derived index.  Filters are exact predicates
    over immutable Definition, Declaration, StateRef, and alias records.
    """

    def __init__(self, repo, *, definition=None, object_id=None, namespace=None, object_ref=None, contains=None, alias=None, path=None, state_hash=None):
        """Create an immutable reference-query builder.

        Args:
            repo: Managing Repo whose connected Store authority is scanned.
            definition: Optional structural referenced-definition constraint.
            object_id: Optional complete ObjectId containment constraint.
            namespace: Optional ObjectId namespace-prefix constraint.
            object_ref: Optional exact complete ObjectRef constraint.
            contains: Optional proper closed ObjectRef subtree constraint.
            alias: Optional Store-local object or state alias constraint.
            path: Optional typed occurrence-path constraint.
            state_hash: Optional exact local-state hash constraint.

        Side Effects:
            None until a terminal is evaluated. Terminals may rebuild derived
            SQLite reference rows, then verify results from Store authority.
        """
        self.repo = repo
        self._definition = definition
        self._object_id = object_id
        self._namespace = namespace
        self._object_ref = object_ref
        self._contains = contains
        self._alias = alias
        self._path = path
        self._state_hash = state_hash

    def references(self) -> "ReferenceQuery":
        """Return this query, allowing uniform fluent use from Repo and DefinitionQuery."""

        return self

    def object_id(self, value: ObjectId) -> "ReferenceQuery":
        """Restrict results to aggregate references containing one ObjectId.

        Args:
            value: Complete durable ObjectId to locate in owned subtrees.

        Returns:
            A refined immutable query.

        Raises:
            TypeError: If ``value`` is not an ObjectId.
        """

        if not isinstance(value, ObjectId):
            raise TypeError("object_id filter requires an ObjectId.")
        return self._replace(object_id=value)

    def namespace(self, prefix) -> "ReferenceQuery":
        """Restrict results to ObjectIds beginning with a namespace prefix.

        Args:
            prefix: Iterable of validated namespace components.

        Returns:
            A refined immutable query.

        Raises:
            ValueError: If a namespace component is invalid.
        """

        prefix = tuple(prefix)
        ObjectId._trusted(prefix, __import__("uuid").uuid4())
        return self._replace(namespace=prefix)

    def exact(self, value: ObjectRef) -> "ReferenceQuery":
        """Restrict results to one complete exact ObjectRef identity.

        Args:
            value: Complete ObjectRef including topology and ObjectIds.

        Returns:
            A refined immutable query.

        Raises:
            TypeError: If ``value`` is not an ObjectRef.
        """

        if not isinstance(value, ObjectRef):
            raise TypeError("exact reference filter requires an ObjectRef.")
        return self._replace(object_ref=value)

    def contains(self, value: ObjectRef) -> "ReferenceQuery":
        """Restrict aggregate references to those containing a closed subtree.

        Args:
            value: Complete ObjectRef that must occur at an owned materializing
                path in the aggregate graph.

        Returns:
            A refined authority query.
        """

        if not isinstance(value, ObjectRef):
            raise TypeError("contains filter requires an ObjectRef.")
        return self._replace(contains=value)

    def definition(self, value) -> "ReferenceQuery":
        """Restrict results to references whose complete CDef matches ``value``.

        Args:
            value: Definition selector or exact ConcreteDefinition.

        Returns:
            A refined immutable query.

        Raises:
            TypeError: If ``value`` is neither Definition nor ConcreteDefinition.
        """

        if not isinstance(value, (Definition, ConcreteDefinition)):
            raise TypeError("definition filter requires a Definition or ConcreteDefinition.")
        return self._replace(definition=value)

    def alias(self, value: str) -> "ReferenceQuery":
        """Restrict results to complete references named by a Store-local alias.

        Args:
            value: Non-empty object or scoped state alias.

        Returns:
            A refined immutable query.

        Raises:
            ValueError: If ``value`` is not a non-empty string.
            RepoLoadError: At terminal evaluation when Stores disagree.
        """

        if not isinstance(value, str) or not value:
            raise ValueError("alias filter requires a non-empty string.")
        return self._replace(alias=value)

    def path(self, value) -> "ReferenceQuery":
        """Restrict embedded reference occurrences to one exact typed path.

        Args:
            value: GraphPath or supported typed path input.

        Returns:
            A refined immutable query whose reference projections are limited to
            matching occurrences.

        Raises:
            GraphPathError: If ``value`` is not a valid canonical path.
        """

        return self._replace(path=normalize_path(value))

    def state_hash(self, value: str) -> "ReferenceQuery":
        """Restrict StateRef results to exact local state hashes.

        Args:
            value: Complete codec-prefixed local-state hash.

        Returns:
            A refined immutable query.

        Raises:
            TypeError: If ``value`` is not a string.
        """

        if not isinstance(value, str):
            raise TypeError("state_hash filter requires a state hash string.")
        return self._replace(state_hash=value)

    def object_refs(self) -> ObjectRefResultSet:
        """Return matching complete aggregate ObjectRefs in canonical order.

        Returns:
            Deduplicated complete ObjectRef identities with retained occurrences.

        Raises:
            RepoLoadError: If authoritative ObjectId or alias sources conflict.
        """

        objects, states, occurrences = self._scan(include_embedded=True)
        return ObjectRefResultSet(self.repo, objects, occurrences)

    def objects(self) -> ObjectRefResultSet:
        """Alias for :meth:`object_refs`; this terminal returns no live Objects."""

        return self.object_refs()

    def state_refs(self) -> StateRefResultSet:
        """Return matching exact StateRefs in canonical order.

        Returns:
            Deduplicated complete StateRef identities with retained occurrences.

        Raises:
            RepoLoadError: If authoritative ObjectId or alias sources conflict.
        """

        _, states, occurrences = self._scan(include_embedded=True)
        return StateRefResultSet(self.repo, states, occurrences)

    def states(self) -> StateRefResultSet:
        """Alias for :meth:`state_refs`."""

        return self.state_refs()

    def occurrences(self) -> ReferenceResultSet:
        """Return matching owner/path/value occurrences without owner collapse.

        Returns:
            Deterministically ordered exact occurrences retaining each owner and
            typed path.

        Raises:
            RepoLoadError: If authoritative ObjectId or alias sources conflict.
        """

        _, _, occurrences = self._scan(include_embedded=True)
        return ReferenceResultSet(self.repo, occurrences)

    def _replace(self, **values) -> "ReferenceQuery":
        data = {
            "definition": self._definition, "object_id": self._object_id,
            "namespace": self._namespace, "object_ref": self._object_ref,
            "contains": self._contains, "alias": self._alias, "path": self._path, "state_hash": self._state_hash,
        }
        data.update(values)
        return ReferenceQuery(self.repo, **data)

    def _scan(self, *, include_embedded: bool = False):
        self._refresh_derived_reference_rows()
        roots: list[ObjectRef] = []
        states: list[StateRef] = []
        occurrences: list[ReferenceOccurrence] = []
        authority_by_id: dict[ObjectId, ObjectRef] = {}
        authority_sources: dict[ObjectId, list[str]] = {}

        for store in self.repo.stores:
            for record in store.iter_declaration_records():
                roots.append(record.object_ref)
                source = repr(store)
                for object_id in record.object_ref.objects.values():
                    authority_sources.setdefault(object_id, []).append(source)
            for record in store.iter_state_ref_records():
                states.append(record.state_ref)
                roots.append(record.state_ref.object)
                source = repr(store)
                for object_id in record.state_ref.object.objects.values():
                    authority_sources.setdefault(object_id, []).append(source)
            if include_embedded:
                for record in store.iter_definition_records():
                    occurrences.extend(_iter_embedded_references(record.definition, owner=record.definition))

        for ref in roots:
            for path, object_id in ref.objects.items():
                subtree = ref.at(path)
                existing = authority_by_id.setdefault(object_id, subtree)
                if existing != subtree:
                    from ..repo import RepoLoadError
                    sources = ", ".join(dict.fromkeys(authority_sources.get(object_id, ())))
                    raise RepoLoadError(
                        f"ObjectId {object_id!s} has incompatible closed-subtree authority in: {sources}."
                    )
            occurrences.append(ReferenceOccurrence(ref, GraphPath(), ref))
        for state in states:
            occurrences.append(ReferenceOccurrence(state, GraphPath(), state))
            occurrences.append(ReferenceOccurrence(state, GraphPath(), state.object))

        alias_objects: set[ObjectRef] | None = None
        alias_states: set[StateRef] | None = None
        if self._alias is not None:
            alias_objects, alias_states = self._alias_targets(roots)

        alias_reference_objects = None if alias_objects is None else alias_objects | {state.object for state in alias_states}
        root_values = [ref for ref in roots if self._matches_object(ref, alias_reference_objects)]
        state_values = [state for state in states if self._matches_state(state, alias_objects, alias_states)]
        allowed = set(root_values) | set(state_values)
        if include_embedded:
            occurrences = [item for item in occurrences if (
                (item.value in allowed and self._matches_occurrence(item))
                or self._matches_embedded(item)
            )]
            root_values.extend(
                item.value for item in occurrences
                if isinstance(item.value, ObjectRef) and self._matches_object(item.value, alias_reference_objects)
            )
            state_values.extend(
                item.value for item in occurrences
                if isinstance(item.value, StateRef) and self._matches_state(item.value, alias_objects, alias_states)
            )
        else:
            occurrences = [item for item in occurrences if item.value in allowed and self._matches_occurrence(item)]
        if self._path is not None:
            # A path identifies reference occurrences, so value projections must
            # be derived from those occurrences rather than unrelated root refs.
            root_values = [
                item.value for item in occurrences
                if isinstance(item.value, ObjectRef) and self._matches_object(item.value, alias_reference_objects)
            ]
            state_values = [
                item.value for item in occurrences
                if isinstance(item.value, StateRef) and self._matches_state(item.value, alias_objects, alias_states)
            ]
        return root_values, state_values, occurrences

    def _refresh_derived_reference_rows(self) -> None:
        """Refresh persistent reference projections before authority verification.

        SQLite rebuilds cache complete reference, ObjectId, state, and alias
        facts.  The following scan deliberately remains the answer source, so a
        derived-row fault cannot hide a Store conflict or change a result.
        """

        seen = set()
        for store in self.repo.stores:
            key = store.catalog_key()
            if key in seen:
                continue
            seen.add(key)
            index = store.open_query_index()
            if index is not None:
                index.refresh("auto")

    def _alias_targets(self, roots):
        objects: set[ObjectRef] = set()
        states: set[StateRef] = set()
        try:
            objects.add(self.repo.get_alias(self._alias))
        except KeyError:
            pass
        for ref in set(roots):
            try:
                states.add(self.repo.resolve_state_selector(ref.state(self._alias)))
            except KeyError:
                continue
        return objects, states

    def _matches_object(self, ref, alias_objects) -> bool:
        if self._object_id is not None and self._object_id not in ref.objects.values():
            return False
        if self._namespace is not None and not any(item.namespace[:len(self._namespace)] == self._namespace for item in ref.objects.values()):
            return False
        if self._object_ref is not None and ref != self._object_ref:
            return False
        if self._contains is not None and not any(
                path and ref.at(path) == self._contains for path in ref.objects):
            return False
        if self._definition is not None and not _query_match(self._definition, ref.definition, strict=False, class_match="selector"):
            return False
        if alias_objects is not None and ref not in alias_objects:
            return False
        return True

    def _matches_state(self, state, alias_objects, alias_states) -> bool:
        if not self._matches_object(state.object, None):
            return False
        if self._state_hash is not None and self._state_hash not in state.states.values():
            return False
        return alias_states is None or state in alias_states or state.object in alias_objects

    def _matches_occurrence(self, item) -> bool:
        return self._path is None or item.path == self._path

    def _matches_embedded(self, item) -> bool:
        if not self._matches_occurrence(item):
            return False
        if isinstance(item.value, StateRef):
            return self._matches_state(item.value, None, None)
        return self._matches_object(item.value, None)


__all__ = ["ObjectRefResultSet", "ReferenceOccurrence", "ReferenceQuery", "ReferenceResultSet", "StateRefResultSet"]
