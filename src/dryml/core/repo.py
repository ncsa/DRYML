from __future__ import annotations

import os
import glob
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable
from contextlib import contextmanager
from io import IOBase
from pathlib import Path
import weakref
from contextvars import ContextVar
from collections.abc import Iterable, Mapping
from collections import defaultdict
import numpy as np
import atexit
import time
from uuid import uuid4

from .definition import Definition, ConcreteDefinition
from .object import Object, Serializable
from .store.store import Store
from .policies import InstancePolicy, CachePolicy, RepoGraphOptions, RepoLoadOptions
from .repo_graph import manage_revision
from .canonical import from_canonical
from .config import CONFIG_MISSING, ConfigError, ConfigRef
from .query.federation import RepoQueryIndex
from .query.memory import AggregateMemoryQueryIndex
from .query.result import ObjectResultSet


def _node_key(key):
    """Normalize a runtime map key to private CDef node identity."""

    if isinstance(key, ConcreteDefinition):
        from .cdef_identity import cdef_node_key

        return cdef_node_key(key)
    return key


def _unique_objects(objects):
    """Return candidates deduplicated by live Object identity."""

    unique = []
    seen = set()
    for obj in objects:
        obj_id = id(obj)
        if obj_id not in seen:
            seen.add(obj_id)
            unique.append(obj)
    return tuple(unique)


def _fork_rekey_reference(reference, namespace):
    """Rekey an exact reference graph, including materializing embedded refs.

    The CDef codec is used as the immutable graph reconstruction boundary.  A
    single old-to-new ObjectId map preserves sharing across the root and every
    embedded materializing ObjectRef/StateRef without using local state bytes as
    identity evidence.
    """
    from .cdef_codec import decode_cdef_graph, encode_cdef_graph
    from .reference_values import ObjectId, ObjectRef, StateRef, _expected_paths

    replacements = {}
    object_cache = {}
    state_cache = {}

    def replacement(old):
        result = replacements.get(old)
        if result is None:
            result = ObjectId._trusted(
                old.namespace if namespace is None else namespace, uuid4()
            )
            replacements[old] = result
        return result

    def transform_value(data):
        kind = data.get("kind")
        if kind == "object_ref":
            data["value"] = rekey_object(ObjectRef.from_data(data["value"])).to_data()
        elif kind == "state_ref":
            data["value"] = rekey_state(StateRef.from_data(data["value"])).to_data()
        elif kind == "link":
            transform_value(data["target"])
        elif kind == "dict":
            for _, value in data["items"]:
                transform_value(value)
        elif kind in {"list", "tuple", "set"}:
            for value in data["items"]:
                transform_value(value)

    def rekey_definition(definition):
        data = encode_cdef_graph(definition)
        for node in data["nodes"]:
            transform_value(node["parameters"])
        return decode_cdef_graph(data)

    def rekey_object(old):
        cached = object_cache.get(old.digest())
        if cached is not None:
            return cached
        definition = rekey_definition(old.definition)
        expected, _ = _expected_paths(definition)
        objects = {}
        for key, path in expected.items():
            if key[0] == "object-id":
                objects[path] = key[1]
            else:
                objects[path] = replacement(old.objects[path])
        result = ObjectRef(definition, objects)
        object_cache[old.digest()] = result
        return result

    def rekey_state(old):
        cached = state_cache.get(old.digest())
        if cached is not None:
            return cached
        result = StateRef(rekey_object(old.object), old.states)
        state_cache[old.digest()] = result
        return result

    if isinstance(reference, StateRef):
        root = rekey_state(reference)
    elif isinstance(reference, ObjectRef):
        root = rekey_object(reference)
    else:
        raise TypeError("Fork rekeying requires an ObjectRef or StateRef.")
    return root, tuple(state_cache.values())


class _NodeMap(dict):
    """Dictionary which never collapses equal but independent CDefs."""

    def __contains__(self, key):
        return super().__contains__(_node_key(key))

    def __getitem__(self, key):
        return super().__getitem__(_node_key(key))

    def get(self, key, default=None):
        return super().get(_node_key(key), default)

    def __setitem__(self, key, value):
        super().__setitem__(_node_key(key), value)

    def pop(self, key, *args):
        return super().pop(_node_key(key), *args)


class _CandidateCache:
    """Live Objects grouped by private node identity without structural collapse."""

    def __init__(self, *, weak: bool = False):
        self._values = defaultdict(list)
        self._weak = weak

    def _live(self, key):
        values = self._values.get(key, ())
        if not self._weak:
            return tuple(values)
        live = tuple(obj for ref in values if (obj := ref()) is not None)
        if live:
            self._values[key] = [weakref.ref(obj) for obj in live]
        else:
            self._values.pop(key, None)
        return live

    def _keys(self, cdef):
        node = _node_key(cdef)
        return tuple(key for key in self._values if key[1] is node)

    def add(self, cdef, obj):
        key = (getattr(obj, "_realization_scope", None), _node_key(cdef))
        values = self._live(key)
        if all(existing is not obj for existing in values):
            self._values[key].append(weakref.ref(obj) if self._weak else obj)

    def discard(self, cdef, obj=None):
        if obj is None:
            for key in self._keys(cdef):
                self._values.pop(key, None)
            return None
        for key in self._keys(cdef):
            remaining = [item for item in self._live(key) if item is not obj]
            if remaining:
                self._values[key] = [weakref.ref(item) for item in remaining] if self._weak else remaining
            else:
                self._values.pop(key, None)

    def get(self, cdef, default=None):
        """Return the sole exact private-node candidate, if present."""

        values = self.candidates(cdef)
        return values[0] if len(values) == 1 else default

    def candidates(self, cdef):
        """Return all live candidates for one exact private CDef node."""

        return tuple(
            obj for key in self._keys(cdef) for obj in self._live(key)
        )

    def has_unique(self, cdef):
        """Report whether exactly one candidate exists without returning it."""

        count = 0
        for key in self._keys(cdef):
            count += len(self._live(key))
            if count > 1:
                return False
        return count == 1

    def __contains__(self, cdef):
        return bool(self.candidates(cdef))

    def __getitem__(self, cdef):
        result = self.get(cdef)
        if result is None:
            raise KeyError(cdef)
        return result

    def pop(self, cdef, default=None):
        values = self.candidates(cdef)
        self.discard(cdef)
        if len(values) != 1:
            return default
        return values[0]

    def clear(self):
        self._values.clear()

    def items(self):
        for key in tuple(self._values):
            for obj in self._live(key):
                yield obj.definition, obj

    def keys(self):
        for cdef, _ in self.items():
            yield cdef

    def __iter__(self):
        """Iterate cached CDefs with candidate multiplicity preserved."""

        return self.keys()

    def __len__(self):
        return sum(len(self._live(key)) for key in tuple(self._values))


class RepoSaveError(Exception):
    pass


class RepoLoadError(Exception):
    pass


class RepoGraphError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class _ClaimLease:
    """One live first-construction fence retained until initial StateRef publication."""

    store: Store
    object_ref: object
    generation: int
    owner: str


SelectorType = Callable | Definition | ConcreteDefinition
RevisionType = dict[ConcreteDefinition, str]


class Repo:
    # Trackers
    _num_saves: int
    _num_constructions: int

    # Caches
    # Links particular concrete definition with particular object
    weak_obj_cache: weakref.WeakValueDictionary[ConcreteDefinition, Object]
    strong_obj_cache: dict[ConcreteDefinition, Object]
    obj_default_store: dict[ConcreteDefinition, Store]

    # known to exist in stores
    light_index: set[ConcreteDefinition]

    # Links particular Definition object with a concrete definition (Definitions are resolved )
    cdef_cache: weakref.WeakValueDictionary[str, ConcreteDefinition]

    # Main definition
    main_def: ConcreteDefinition | None

    # Backing stores
    stores: list[Store]

    # Object config store
    obj_config: dict[ConcreteDefinition, Any]

    # Runtime config values, intentionally not persisted by stores.
    config: dict[str, Any]

    # User-facing alias index
    alias_index: dict[str, ConcreteDefinition]

    # Settings
    save_objs_on_deletion: bool = False


    # Helper class for saving objects
    def __init__(
            self, stores=None, config: Mapping[str, Any] | None = None,
            *, clock: Callable[[], float] | None = None,
            lease_duration: float = 30.0,
            owner_token_factory: Callable[[], str] | None = None):
        # Initialize caches
        self.weak_obj_cache = _CandidateCache(weak=True)
        self.strong_obj_cache = _CandidateCache()
        self.obj_default_store = _NodeMap()
        self.light_index = set()
        self.cdef_cache = weakref.WeakValueDictionary()
        self.obj_config = {}
        self.config = dict(config or {})
        if not isinstance(lease_duration, (int, float)) or not 0 < lease_duration <= 3600:
            raise ValueError("lease_duration must be a positive bounded number of seconds.")
        self._clock = clock or time.time
        self._lease_duration = float(lease_duration)
        self._owner_token_factory = owner_token_factory or (lambda: uuid4().hex)
        self.alias_index = {}
        self._aliases_dirty = False
        # Compatibility facade and live cache overlay. Store-owned indexes handle
        # persistent sources; this aggregate remains the memory backend and cache
        # source for existing APIs and `known()` cache federation.
        self._query_catalog = AggregateMemoryQueryIndex(self)

        # Some helper variables for monitoring
        self._num_saves = 0
        self._num_constructions = 0

        # Initialize the main def
        self.main_def = None

        # Multiple stores, optional
        self.stores = []
        if stores is not None:
            if not isinstance(stores, (tuple, list)):
                stores = [stores]

            for store in stores:
                if not isinstance(store, Store):
                    self.stores.append(make_store(store))
                else:
                    self.stores.append(store)
        self._query_index = RepoQueryIndex(self)

        # Main remains a structural reference in current Store authority.
        if len(self.stores) > 0:
            for store in self.stores:
                main = store.read_main_ref()
                if main is None:
                    continue
                definition = store.read_definition_record(main.definition_digest)
                if definition is None:
                    raise RepoLoadError("Main reference points to a missing DefinitionRecord.")
                self.main_def = definition.definition
                break

    # Store Methods

    @property
    def default_store(self):
        return self.stores[0] if len(self.stores) > 0 else None

    def set_default_store(self, store: "Store"):
        if not isinstance(store, Store):
            store = make_store(store)
        if store not in self.stores:
            self.stores.insert(0, store)
        else:
            # Find and move to front
            store_idx = self.stores.index(store)
            self.stores.insert(0, self.stores.pop(store_idx))
        self._query_index.refresh_bindings()

    def add_store(self, store: "Store", make_default=False):
        if not isinstance(store, Store):
            store = make_store(store)
        if make_default or self.default_store is None:
            self.stores.insert(0, store)
        else:
            self.stores.append(store)
        self._query_index.refresh_bindings()

    def _ensure_store(self, store):
        if store is None:
            return None
        store = make_store(store)
        if store not in self.stores:
            self.add_store(store)
        return store

    def cache_strong(self, obj: Object) -> None:
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_cache_strong"):
            self.strong_obj_cache.add(obj.__cdef__, obj)
            self._query_catalog.register_cached(obj.__cdef__)

    def cache_weak(self, obj: Object) -> None:
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_cache_weak"):
            self.weak_obj_cache.add(obj.__cdef__, obj)
            self._query_catalog.register_cached(obj.__cdef__)

    # --- helpers you already have ---
    def _cached_candidates(self, cdef, *, reuse_weak: bool) -> tuple[Object, ...]:
        """Return distinct live candidates for one exact private CDef node."""

        candidates = self.strong_obj_cache.candidates(cdef)
        if reuse_weak:
            candidates += self.weak_obj_cache.candidates(cdef)
        return _unique_objects(candidates)

    def get_cached(self, cdef, *, reuse_weak: bool = True):
        """Return a cached Object after live-object admission.

        Args:
            cdef: Exact private-node definition used as the cache key; this is
                not a structural-equality lookup.
            reuse_weak: Whether the weak cache may satisfy the lookup.

        Returns:
            The sole cached Object, or ``None`` when no reusable entry exists
            or candidates are ambiguous across the selected cache tiers.

        Raises:
            RuntimeTransitionError: If strict orchestration prohibits returning
                the live Object.
        """
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_get_cached"):
            candidates = self._cached_candidates(cdef, reuse_weak=reuse_weak)
            return candidates[0] if len(candidates) == 1 else None

    def has_cached(self, cdef, *, reuse_weak: bool = True) -> bool:
        """Return cache availability without acquiring a live Object.

        Args:
            cdef: Exact private-node definition used as the cache key; this is
                not a structural-equality lookup.
            reuse_weak: Whether weak-cache availability counts.

        Returns:
            Whether exactly one reusable candidate exists across the selected
            strong and weak cache tiers.

        Side Effects:
            None. This metadata-only check is available during strict
            orchestration and does not retain the cached Object.
        """
        return len(self._cached_candidates(cdef, reuse_weak=reuse_weak)) == 1

    def pin(self, obj):
        """Promote to strong cache."""
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_pin"):
            cdef = obj.__cdef__
            self.strong_obj_cache.add(cdef, obj)
            self.weak_obj_cache.discard(cdef, obj)
            self._query_catalog.register_cached(cdef)

    def unpin(self, obj_or_cdef):
        """Demote to weak cache."""
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_unpin"):
            cdef = obj_or_cdef if isinstance(obj_or_cdef, ConcreteDefinition) else obj_or_cdef.__cdef__
            obj = self.strong_obj_cache.get(cdef)
            if obj is not None:
                self.strong_obj_cache.discard(cdef, obj)
                self.weak_obj_cache.add(cdef, obj)
                self._query_catalog.register_cached(cdef)

    def _load_aliases_from_stores(self) -> None:
        """Retired CDef alias caches have no projection from reference authority."""
        return None

    @staticmethod
    def _validate_alias(alias: str) -> None:
        if not isinstance(alias, str):
            raise TypeError("Object aliases must be strings.")
        if alias == "":
            raise ValueError("Object aliases cannot be empty strings.")

    def _object_target_cdef(self, target: Object | Definition | ConcreteDefinition) -> ConcreteDefinition:
        if isinstance(target, Object):
            return target.definition
        if isinstance(target, ConcreteDefinition):
            return target
        if isinstance(target, Definition):
            return target.concretize(repo=self)
        raise TypeError(
            "Object target must be an Object, Definition, or ConcreteDefinition."
        )

    def set_object_store(self, target: Object | Definition | ConcreteDefinition, store) -> Store:
        if isinstance(target, Object):
            from dryml.runtime import materialization_admission

            with materialization_admission(operation="repo_set_object_store_live_object"):
                return self._set_object_store(target, store)
        return self._set_object_store(target, store)

    def _set_object_store(self, target: Object | Definition | ConcreteDefinition, store) -> Store:
        """Bind an already-admitted target definition to a Store."""

        store = self._ensure_store(store)
        if store is None:
            raise ValueError("No store provided for object store binding.")
        cdef = self._object_target_cdef(target)
        self.obj_default_store[cdef] = store
        if isinstance(target, Object):
            target._store_affinity = store
        return store

    def location(
            self,
            target: Object | Definition | ConcreteDefinition,
            *,
            store=None,
            require_exists: bool = False) -> str:
        if isinstance(target, Object):
            from dryml.runtime import materialization_admission

            with materialization_admission(operation="repo_location_live_object"):
                return self._location(target, store=store, require_exists=require_exists)
        return self._location(target, store=store, require_exists=require_exists)

    def _location(
            self,
            target: Object | Definition | ConcreteDefinition,
            *,
            store=None,
            require_exists: bool = False) -> str:
        """Resolve an already-admitted target to its selected Store path."""

        cdef = self._object_target_cdef(target)

        if store is not None:
            store = self.set_object_store(cdef, store)
        else:
            store = self.obj_default_store.get(cdef)
            if store is None:
                store = self._first_store_with(cdef)
                if store is not None:
                    self.obj_default_store[cdef] = store
            if store is None:
                store = self.default_store

        if store is None:
            raise RuntimeError("No store available for object location.")
        if require_exists and not store.has(cdef):
            raise RuntimeError("Object is not saved in the selected store.")
        return store.object_dir(cdef)

    def _selected_writable_store(self, store, operation: str) -> Store:
        """Select one writable Store or reject ambiguous reference mutation."""
        if store is not None:
            selected = self._ensure_store(store)
            selected.preflight_publication(operation)
            return selected
        candidates = [candidate for candidate in self.stores if candidate.publication_capabilities.writable]
        if len(candidates) != 1:
            raise RepoSaveError(f"{operation} requires an explicit Store or exactly one writable Repo Store.")
        candidates[0].preflight_publication(operation)
        return candidates[0]

    def _reference_authoritative_in(self, store: Store, reference) -> bool:
        """Return whether a complete ObjectRef is authoritative in one Store."""
        from .store.records import DeclarationRecord

        declaration = store.read_declaration_record(reference.digest())
        if declaration is not None:
            if declaration.object_ref != reference:
                raise RepoLoadError("Declaration digest collision has incompatible ObjectRef authority.")
            return True
        return any(record.state_ref.object == reference for record in store.iter_state_ref_records())

    def _alias_records(self, alias: str, *, state_scope=None):
        records = []
        for store in self.stores:
            record = (
                store.read_state_alias(state_scope.digest(), alias)
                if state_scope is not None else store.read_object_alias(alias)
            )
            if record is not None:
                records.append((store, record))
        return records

    def _single_alias_target(self, alias: str, *, state_scope=None):
        self._validate_alias(alias)
        records = self._alias_records(alias, state_scope=state_scope)
        if not records:
            raise KeyError(f"Repo has no alias {alias!r}.")
        targets = {
            record.state_ref_digest if state_scope is not None else record.object_ref.digest()
            for _, record in records
        }
        if len(targets) != 1:
            detail = ", ".join(
                f"{store!r}={record.state_ref_digest if state_scope is not None else record.object_ref.digest()}"
                for store, record in records
            )
            raise RepoLoadError(f"Alias {alias!r} conflicts across connected Stores: {detail}.")
        return records[0]

    def set_alias(self, alias: str, target, *, store=None, save_live: bool = True):
        """Publish a Store-local alias for existing ObjectRef authority.

        Args:
            alias: Non-empty path-safe alias text.
            target: Existing ``ObjectRef`` or ``StateRef``, or a live ``Object``.
            store: Explicit writable target Store, required when writable Repo
                Stores are ambiguous.
            save_live: Whether a live target may be saved before aliasing.

        Returns:
            The authoritative ObjectRef selected for the alias.

        Raises:
            TypeError: If the alias or target type is invalid.
            ValueError: If the alias is empty or a live target is not saveable.
            RepoLoadError: If the target lacks same-Store declaration or StateRef
                authority.

        Side Effects:
            May save a live graph, then atomically replaces one mutable alias
            record. It never creates declaration or StateRef authority itself.

        Concurrency:
            Mutation holds the selected Store writer lock. A stale claimant cannot
            create StateRef authority through this method.

        Store Requirements:
            The selected Store must be writable and provide atomic mutable-record
            replacement and writer serialization.
        """
        from .reference_values import ObjectRef, StateRef
        from .store.records import ObjectAliasRecord

        self._validate_alias(alias)
        selected = self._selected_writable_store(store, "write object alias")
        if isinstance(target, Object):
            if not save_live:
                raise ValueError("Object aliases require an authoritative ObjectRef; save the live object first.")
            target = self.save_object(target, store=selected).object
        elif isinstance(target, StateRef):
            target = target.object
        if not isinstance(target, ObjectRef):
            raise TypeError("Object aliases target ObjectRef, StateRef, or a saved live Object.")
        with selected.writer_lock():
            if not self._reference_authoritative_in(selected, target):
                raise RepoLoadError("Object aliases require same-Store declaration or StateRef authority.")
            selected.write_object_alias(ObjectAliasRecord(alias, target))
        return target

    def get_alias(self, alias: str):
        """Resolve an ObjectRef alias, accepting only identical Store replicas.

        Args:
            alias: Non-empty object alias to resolve.

        Returns:
            The complete authoritative ObjectRef.

        Raises:
            KeyError: If no connected Store defines the alias.
            RepoLoadError: If connected Stores define conflicting targets.

        Side Effects:
            None. This reads authoritative mutable records and never constructs a
            live Object or consults a derived index.

        Concurrency:
            Readers observe a complete old or new record through atomic Store
            replacement; identical replicated values are deduplicated.

        Store Requirements:
            Every connected Store must implement object-alias reads.
        """
        _, record = self._single_alias_target(alias)
        return record.object_ref

    def delete_alias(self, alias: str, *, store=None):
        """Retire unsupported legacy CDef alias deletion.

        Mutable reference deletion was not part of U6's authority protocol; it
        is rejected rather than silently updating retired alias state.
        """
        raise NotImplementedError("Deleting reference aliases is not implemented by the current Store protocol.")

    def aliases(self) -> dict[str, object]:
        """Return the legacy in-memory alias cache, which is intentionally empty."""
        return {}

    def set_state_alias(self, alias: str, state_ref, *, store=None):
        """Publish a state alias scoped by the complete ObjectRef identity.

        Args:
            alias: Non-empty path-safe state alias text.
            state_ref: Existing exact StateRef to name.
            store: Explicit writable target Store when Repo Stores are ambiguous.

        Returns:
            The supplied StateRef.

        Raises:
            TypeError: If ``state_ref`` is not a StateRef.
            RepoLoadError: If its exact immutable record is not in the selected
                Store.

        Side Effects:
            Atomically replaces one Store-local mutable state-alias record; it
            does not publish state, mutate a CDef, or create ObjectRef authority.

        Concurrency:
            The selected Store writer lock serializes replacement with competing
            aliases and StateRef publication.

        Store Requirements:
            The selected Store must be writable and carry the exact StateRef.
        """
        from .reference_values import StateRef
        from .store.records import StateAliasRecord

        self._validate_alias(alias)
        if not isinstance(state_ref, StateRef):
            raise TypeError("State aliases require an exact StateRef target.")
        selected = self._selected_writable_store(store, "write state alias")
        with selected.writer_lock():
            record = selected.read_state_ref_record(state_ref.digest())
            if record is None or record.state_ref != state_ref:
                raise RepoLoadError("State aliases require same-Store exact StateRef authority.")
            selected.write_state_alias(StateAliasRecord(alias, state_ref.object, state_ref.digest()))
        return state_ref

    def resolve_state_selector(self, selector):
        """Resolve one soft StateSelectorRef through non-conflicting Store refs.

        Args:
            selector: StateSelectorRef containing complete ObjectRef scope and
                Store-local alias text.

        Returns:
            The exact immutable StateRef named by the selector.

        Raises:
            TypeError: If ``selector`` is not a StateSelectorRef.
            KeyError: If no connected Store has the scoped alias.
            RepoLoadError: If aliases conflict or their target record is missing
                or has a different ObjectRef.

        Side Effects:
            None. Selector resolution reads authority only and never materializes
            state or changes an alias.

        Concurrency:
            A reader accepts only one identical target across connected Stores.

        Store Requirements:
            Connected Stores must expose state aliases and StateRef records.
        """
        from .reference_values import StateSelectorRef

        if not isinstance(selector, StateSelectorRef):
            raise TypeError("resolve_state_selector requires a StateSelectorRef.")
        store, record = self._single_alias_target(selector.alias, state_scope=selector.object)
        state = store.read_state_ref_record(record.state_ref_digest)
        if state is None or state.state_ref.object != selector.object:
            raise RepoLoadError("State alias points to missing or incompatible StateRef authority.")
        return state.state_ref

    def load_alias(self, alias: str, **kwargs):
        """Materialize an object alias's structural recipe pending U7 restore.

        Alias lookup remains ObjectRef-authoritative.  U7 will replace this
        temporary structural materialization with exact StateRef restoration;
        keeping it here preserves ConfigRef's current public construction path
        without projecting the alias back to a retired CDef record.
        """
        if "restore_state" in kwargs:
            raise TypeError(
                "Object aliases name ObjectRefs, not state; exact restoration is unavailable until U7."
            )
        return self.load_object(self.get_alias(alias).definition, restore_state=False, **kwargs)

    def _references(self):
        """Yield every Store-authoritative ObjectRef once with its Store/source."""
        seen = set()
        for store in self.stores:
            for record in store.iter_declaration_records():
                reference = record.object_ref
                key = (reference.digest(), store.catalog_key())
                if key not in seen:
                    seen.add(key)
                    yield store, reference, "declaration"
            for record in store.iter_state_ref_records():
                reference = record.state_ref.object
                key = (reference.digest(), store.catalog_key())
                if key not in seen:
                    seen.add(key)
                    yield store, reference, "state-ref"

    def find_object_refs(self, object_id=None, *, namespace=None, contains=None):
        """Scan current authority for complete refs matching durable identity facts.

        Args:
            object_id: Optional complete ``ObjectId`` to match.
            namespace: Optional tuple namespace prefix to match exactly at the
                beginning of a contained ObjectId namespace.
            contains: Optional ObjectRef that must occur as a closed subtree.

        Returns:
            Deterministically ordered ``(Store, ObjectRef)`` pairs.  This direct
            scan deliberately remains correct without a derived query index.

        Raises:
            TypeError: If an identity, namespace prefix, or containment reference
                has an unsupported type.
            ValueError: If a namespace prefix violates ObjectId validation.

        Side Effects:
            None. This scans declaration and StateRef authority; local-state bytes
            and query indexes do not participate in identity resolution.

        Concurrency:
            Each scanned record is complete immutable authority. Results may span
            Store generations when another writer publishes during the scan.

        Store Requirements:
            Connected Stores must expose iterable declaration and StateRef records.
        """
        from .reference_values import ObjectId, ObjectRef

        if object_id is not None and not isinstance(object_id, ObjectId):
            raise TypeError("object_id must be an ObjectId.")
        if namespace is not None:
            namespace = tuple(namespace)
            # Validate the supplied prefix without allocating an ID.
            ObjectId._trusted(namespace, uuid4())
        if contains is not None and not isinstance(contains, ObjectRef):
            raise TypeError("contains must be an ObjectRef.")
        matches = []
        for store, reference, _ in self._references():
            if object_id is not None and object_id not in reference.objects.values():
                continue
            if namespace is not None and not any(value.namespace[:len(namespace)] == namespace for value in reference.objects.values()):
                continue
            if contains is not None and not any(reference.at(path) == contains for path in reference.objects):
                continue
            matches.append((store, reference))
        return tuple(matches)

    def lookup_object_ref(self, object_id):
        """Resolve one ObjectId to its canonical closed ObjectRef subtree.

        Args:
            object_id: Complete durable ObjectId to resolve.

        Returns:
            The canonical closed ObjectRef subtree named by the identity.

        Raises:
            TypeError: If ``object_id`` is not an ObjectId.
            KeyError: If no connected authoritative record names it.
            RepoLoadError: If records use the same ID for incompatible subtrees.

        Side Effects:
            None. The lookup scans immutable declaration and StateRef authority,
            never a derived index or local state bytes.

        Concurrency:
            Immutable records make each candidate stable; concurrent publication
            can add a later candidate but cannot rewrite an observed one.

        Store Requirements:
            Connected Stores must expose authoritative record iteration.
        """
        candidates = []
        for store, reference in self.find_object_refs(object_id):
            for path, candidate_id in reference.objects.items():
                if candidate_id == object_id:
                    candidates.append((store, reference.at(path)))
        if not candidates:
            raise KeyError(f"No Store authority names ObjectId {object_id!s}.")
        target = candidates[0][1]
        if any(candidate != target for _, candidate in candidates[1:]):
            details = ", ".join(f"{store!r}={reference.digest()}" for store, reference in candidates)
            raise RepoLoadError(f"ObjectId {object_id!s} has incompatible closed-subtree authority: {details}.")
        return target

    find_object_ref = lookup_object_ref

    def _assert_compatible_object_ids(self, reference) -> None:
        """Reject reuse of an ObjectId with a different authoritative closure."""
        for path, object_id in reference.objects.items():
            try:
                existing = self.lookup_object_ref(object_id)
            except KeyError:
                continue
            if existing != reference.at(path):
                raise RepoLoadError(
                    f"ObjectId {object_id!s} is already authoritative for an incompatible closed subtree."
                )

    def _declaration_reference(self, cdef, namespace):
        """Allocate only CDef-owned unidentified Serializable node IDs."""
        from .reference_values import ObjectId, ObjectRef, _expected_paths

        by_key, _ = _expected_paths(cdef)
        objects = {}
        allocated = 0
        for key, path in by_key.items():
            if key[0] == "object-id":
                objects[path] = key[1]
            else:
                objects[path] = ObjectId(namespace)
                allocated += 1
        if not objects:
            raise ValueError("Cannot declare an all-ephemeral graph; build its CDef normally.")
        if not allocated:
            raise ValueError("Declaration has no new durable lineage; build its CDef normally.")
        return ObjectRef(cdef, objects)

    def _register_declaration(self, reference, store, *, allow_preallocated: bool = False):
        """Publish Definition, available claim, then declaration under one fence."""
        from .reference_values import ObjectRef
        from .store.records import ClaimRecord, DeclarationRecord, DefinitionRecord

        if not isinstance(reference, ObjectRef) or not reference.objects:
            raise ValueError("Declarations require a complete non-empty ObjectRef.")
        store.preflight_publication("declare ObjectRef")
        with store.writer_lock():
            self._assert_compatible_object_ids(reference)
            store.write_definition_record(DefinitionRecord(reference.definition))
            existing = store.read_declaration_record(reference.digest())
            claim = store.read_claim_record(reference.digest())
            if existing is not None:
                if existing.object_ref != reference:
                    raise RepoLoadError("Declaration digest collision has incompatible ObjectRef authority.")
                if claim is None:
                    raise RepoLoadError("Declaration exists without ClaimRecord; Store authority is corrupt.")
                return reference
            # An interrupted claim without a declaration is not authority. Its
            # complete record can safely be replaced by registration's fence.
            if claim is None or claim.object_digest != reference.digest():
                store.write_claim_record(ClaimRecord(reference.digest(), 0, "available"))
            elif claim.status != "available":
                raise RepoLoadError("Unregistered ObjectRef claim is not available for recovery.")
            store.write_declaration_record(DeclarationRecord(reference))
        return reference

    def declare_object(self, cdef: ConcreteDefinition, *, store=None, namespace=None):
        """Preallocate and register one first-construction ObjectRef.

        Args:
            cdef: V2 ConcreteDefinition with at least one new owned Serializable
                lineage.
            store: Explicit writable registration Store when Repo Stores are
                ambiguous.
            namespace: Optional ObjectId namespace; ``None`` uses the active
                allocation scope and ``()`` is explicitly empty.

        Returns:
            Complete registered non-empty ObjectRef.

        Raises:
            TypeError: If ``cdef`` is not concrete.
            ValueError: If the graph is all-ephemeral or already has no new
                durable lineage to allocate.
            RepoLoadError: If an existing ObjectId has incompatible authority.

        Side Effects:
            Under one writer lock, installs/verifies DefinitionRecord, writes an
            available ClaimRecord, then writes DeclarationRecord as the only
            registration boundary.

        Concurrency:
            Writer serialization prevents competing registration from reusing an
            ID with a different closed subtree.

        Store Requirements:
            The selected Store must support immutable install, atomic claim
            replacement, and writer serialization.
        """
        if not isinstance(cdef, ConcreteDefinition):
            raise TypeError("declare_object requires a ConcreteDefinition.")
        selected = self._selected_writable_store(store, "declare ObjectRef")
        reference = self._declaration_reference(cdef, namespace)
        return self._register_declaration(reference, selected)

    def _matching_state_ref(self, store, reference):
        """Return a matching complete StateRef record, if initial completion exists."""
        for record in store.iter_state_ref_records():
            if record.state_ref.object == reference:
                return record
        return None

    def _acquire_claim(self, reference, store):
        """Acquire or recover one declaration claim under the Store writer fence."""
        from .store.records import ClaimRecord

        with store.writer_lock():
            declaration = store.read_declaration_record(reference.digest())
            claim = store.read_claim_record(reference.digest())
            if declaration is None:
                raise RepoLoadError("build_object_ref requires a registered declaration in its selected Store.")
            if declaration.object_ref != reference or claim is None:
                raise RepoLoadError("Declaration and ClaimRecord authority is missing or incompatible.")
            if claim.object_digest != reference.digest():
                raise RepoLoadError("ClaimRecord does not match its declaration.")
            completed = self._matching_state_ref(store, reference)
            if completed is not None:
                if claim.status != "completed" or claim.state_ref_digest != completed.digest:
                    store.write_claim_record(ClaimRecord(reference.digest(), claim.generation, "completed", state_ref_digest=completed.digest))
                return None
            now = self._clock()
            if claim.status == "completed":
                raise RepoLoadError("Declared ObjectRef is already completed; choose its exact StateRef.")
            if claim.status == "claimed" and claim.lease_until > now:
                raise RepoLoadError("Declared ObjectRef has an active first-construction claim.")
            owner = self._owner_token_factory()
            if not isinstance(owner, str) or not owner:
                raise ValueError("owner_token_factory must return a non-empty string.")
            generation = claim.generation + 1
            store.write_claim_record(ClaimRecord(reference.digest(), generation, "claimed", owner, now + self._lease_duration))
            return _ClaimLease(store, reference, generation, owner)

    def _pending_declaration_references(self, reference):
        """Return materializing nested declarations in dependency-first order.

        Args:
            reference: Registered parent ``ObjectRef`` to inspect.

        Returns:
            Tuples of materializing ``GraphPath`` and unique nested ``ObjectRef``
            values, ordered with each nested dependency before its parent.

        Raises:
            TypeError: If ``reference`` is not an ObjectRef.
            RepoLoadError: If a materializing nested ObjectRef has no unambiguous
                declaration Store or materializing declarations form a cycle.

        Side Effects:
            None. The direct authority scan neither claims nor constructs a node.

        Store Requirements:
            Each nested ObjectRef must be registered in exactly one connected
            Store. Ref-only links and embedded StateRefs are not declarations.
        """
        from .cdef_graph import EdgeKind
        from .definition import ConcreteDefinition
        from .links import DefLink
        from .reference_values import ObjectRef, StateRef
        from .utils.graph.path import GraphPath
        from .utils.graph.value import iter_value_edges

        if not isinstance(reference, ObjectRef):
            raise TypeError("Pending declaration traversal requires an ObjectRef.")
        result = []
        seen = {reference.digest()}
        active = set()

        def visit_value(value, path):
            if isinstance(value, ObjectRef):
                visit_reference(value, path)
            elif isinstance(value, StateRef):
                return
            elif isinstance(value, ConcreteDefinition):
                visit_cdef(value, path)
            elif isinstance(value, DefLink):
                if value.kind is EdgeKind.MATERIALIZE:
                    visit_value(value.target, path)
            else:
                for edge in iter_value_edges(value):
                    visit_value(edge.value, path.child(edge.segment))

        def visit_cdef(cdef, path):
            for edge in iter_value_edges(cdef):
                visit_value(edge.value, path.child(edge.segment))

        def visit_reference(child, path):
            digest = child.digest()
            if digest in active:
                raise RepoLoadError("Materializing declaration references cannot form a cycle.")
            if digest in seen:
                return
            active.add(digest)
            try:
                self._selected_declaration_store(child, None)
                visit_cdef(child.definition, path)
                result.append((path, child))
                seen.add(digest)
            finally:
                active.remove(digest)

        visit_cdef(reference.definition, GraphPath())
        return tuple(result)

    def _renew_claim(self, lease: _ClaimLease) -> None:
        """Extend a live first-build lease only when its full fence still matches."""
        from .store.records import ClaimRecord

        with lease.store.writer_lock():
            declaration = lease.store.read_declaration_record(lease.object_ref.digest())
            claim = lease.store.read_claim_record(lease.object_ref.digest())
            now = self._clock()
            if declaration is None or declaration.object_ref != lease.object_ref or claim is None:
                raise RepoLoadError("Cannot renew a missing declaration claim.")
            if (claim.object_digest != lease.object_ref.digest()
                    or claim.generation != lease.generation or claim.owner != lease.owner
                    or claim.status != "claimed" or claim.lease_until <= now):
                raise RepoLoadError("Cannot renew a stale first-construction claim.")
            lease.store.write_claim_record(ClaimRecord(claim.object_digest, claim.generation, "claimed", claim.owner, now + self._lease_duration))

    def _abandon_claim(self, lease: _ClaimLease) -> bool:
        """Release exactly one matching live generation; stale leases do nothing."""
        from .store.records import ClaimRecord

        with lease.store.writer_lock():
            declaration = lease.store.read_declaration_record(lease.object_ref.digest())
            claim = lease.store.read_claim_record(lease.object_ref.digest())
            if declaration is None or declaration.object_ref != lease.object_ref or claim is None:
                return False
            if (claim.object_digest != lease.object_ref.digest()
                    or claim.generation != lease.generation or claim.owner != lease.owner
                    or claim.status != "claimed" or claim.lease_until <= self._clock()):
                return False
            lease.store.write_claim_record(ClaimRecord(claim.object_digest, claim.generation, "available"))
            return True

    def abandon_object_ref(self, live_object: Object) -> bool:
        """Abandon the exact live graph's current initial-construction claim.

        Args:
            live_object: Object returned by ``build_object_ref``.

        Returns:
            Whether this call released its still-current claim generation.

        Raises:
            None. Missing, completed, or stale claims return ``False``.

        Side Effects:
            Atomically returns only this live generation to ``available`` and
            clears the live object's lease when successful.

        Concurrency:
            Generation and owner comparisons under the Store writer lock prevent
            stale builders from abandoning a successor's claim.

        Store Requirements:
            The original declaration Store must remain writable and connected.
        """
        lease = getattr(live_object, "_claim_lease", None)
        if not isinstance(lease, _ClaimLease):
            return False
        result = self._abandon_claim(lease)
        if result:
            live_object._claim_lease = None
        return result

    def build_object_ref(self, reference, *, store=None):
        """Construct one registered ObjectRef through its unique active claim.

        Args:
            reference: Complete non-empty registered ObjectRef.
            store: Its explicit declaration Store, or omitted only when exactly
                one connected Store contains the declaration.

        Returns:
            Fresh live Object graph retaining the exact reference and active claim
            generations through initial StateRef publication.

        Raises:
            ValueError: If ``reference`` is empty or not an ObjectRef.
            RepoLoadError: If registration is missing, ambiguous, completed,
                actively claimed, stale, or nested declaration authority fails.

        Side Effects:
            Acquires unique nested declarations before their parent, constructs
            the graph, attaches exact IDs and leases, and releases only claims
            acquired by this attempt in reverse order on failure.

        Concurrency:
            Each acquire and renewal compares declaration digest, generation,
            owner, status, and lease under Store writer serialization.

        Store Requirements:
            The declaration Store for every pending materializing ObjectRef must
            be connected, unambiguous, writable, and writer-serialized.
        """
        from .reference_values import ObjectRef
        from .repo_plan import apply_exact_reference_identity

        if not isinstance(reference, ObjectRef) or not reference.objects:
            raise ValueError("build_object_ref requires a non-empty ObjectRef.")
        selected = self._selected_declaration_store(reference, store)
        acquired = []
        try:
            dependencies = self._pending_declaration_references(reference)
            for _, dependency in dependencies:
                dependency_store = self._selected_declaration_store(dependency, None)
                dependency_lease = self._acquire_claim(dependency, dependency_store)
                if dependency_lease is None:
                    raise RepoLoadError("Nested ObjectRef is already completed; choose an exact StateRef.")
                acquired.append(dependency_lease)
            lease = self._acquire_claim(reference, selected)
            if lease is None:
                raise RepoLoadError("Declared ObjectRef is complete; load its exact StateRef in U7.")
            acquired.append(lease)
            obj = self.load_object(
                reference.definition, restore_state=False, build_missing=True
            )
            apply_exact_reference_identity(obj, reference)
            obj._store_affinity = selected
            obj._claim_lease = lease
            obj._claim_leases = tuple(acquired)
            pending = []
            for path, dependency in dependencies:
                try:
                    dependency_obj = obj.graph_at(path)
                except Exception as error:
                    raise RepoLoadError(
                        f"Nested ObjectRef at {path!s} did not retain a live construction binding."
                    ) from error
                if not isinstance(dependency_obj, Object):
                    raise RepoLoadError(
                        f"Nested ObjectRef at {path!s} did not materialize an Object."
                    )
                child_lease = next(item for item in acquired if item.object_ref == dependency)
                dependency_obj._claim_lease = child_lease
                pending.append((child_lease, dependency_obj))
            obj._pending_claim_dependencies = tuple(pending)
            self._renew_claim(lease)
            return obj
        except BaseException:
            for acquired_lease in reversed(acquired):
                self._abandon_claim(acquired_lease)
            raise

    def _selected_declaration_store(self, reference, store):
        """Resolve one declaration Store explicitly or from unambiguous authority."""
        if store is not None:
            selected = self._ensure_store(store)
            declaration = selected.read_declaration_record(reference.digest())
            if declaration is None or declaration.object_ref != reference:
                raise RepoLoadError("Selected Store does not contain this ObjectRef declaration.")
            return selected
        matches = [candidate for candidate in self.stores if (record := candidate.read_declaration_record(reference.digest())) is not None and record.object_ref == reference]
        if len(matches) != 1:
            raise RepoLoadError("build_object_ref requires an explicit declaration Store or one unambiguous connected Store.")
        return matches[0]

    def _complete_initial_state_ref(self, state_ref, store) -> None:
        """Fence initial StateRef publication against the live claim generation."""
        from .store.records import ClaimRecord

        lease = getattr(self, "_publishing_claim_lease", None)
        if not isinstance(lease, _ClaimLease) or lease.store is not store or lease.object_ref != state_ref.object:
            return
        declaration = store.read_declaration_record(state_ref.object.digest())
        claim = store.read_claim_record(state_ref.object.digest())
        now = self._clock()
        if declaration is None or declaration.object_ref != state_ref.object or claim is None:
            raise RepoSaveError("Initial StateRef publication lost declaration authority.")
        if (claim.generation != lease.generation or claim.owner != lease.owner
                or claim.status != "claimed" or claim.lease_until <= now):
            raise RepoSaveError("Initial StateRef publication lost its claim generation.")

    def _mark_initial_state_ref_complete(self, state_ref, store) -> None:
        """Replace a verified live claim with completed StateRef authority."""
        from .store.records import ClaimRecord

        lease = getattr(self, "_publishing_claim_lease", None)
        if not isinstance(lease, _ClaimLease) or lease.store is not store or lease.object_ref != state_ref.object:
            return
        declaration = store.read_declaration_record(state_ref.object.digest())
        claim = store.read_claim_record(state_ref.object.digest())
        now = self._clock()
        if (declaration is None or declaration.object_ref != state_ref.object or claim is None
                or claim.object_digest != lease.object_ref.digest()
                or claim.generation != lease.generation or claim.owner != lease.owner
                or claim.status != "claimed" or claim.lease_until <= now):
            raise RepoSaveError("Initial StateRef was published but its claim fence changed.")
        store.write_claim_record(ClaimRecord(claim.object_digest, claim.generation, "completed", state_ref_digest=state_ref.digest()))
        self._publishing_claim_lease = None

    def fork_object_ref(self, reference, *, store=None, namespace=None):
        """Rekey a non-empty ObjectRef and register a state-free declaration.

        Args:
            reference: Complete non-empty source ObjectRef.
            store: Explicit writable declaration Store when Repo Stores are
                ambiguous.
            namespace: Replacement namespace for every new ObjectId, or ``None``
                to preserve each source namespace independent of active scopes.

        Returns:
            A graph-isomorphic ObjectRef with fresh nonces and a new available
            ClaimRecord/DeclarationRecord boundary.

        Raises:
            ValueError: If the reference is empty or namespace validation fails.
            TypeError: If the reference has an unsupported type.
            RepoLoadError: If existing identity authority conflicts.

        Side Effects:
            Allocates IDs and publishes the fork declaration only after the
            selected Store can install its DefinitionRecord and claim.

        Concurrency:
            Registration runs under one Store writer lock; the fork has no
            authority if publication fails before DeclarationRecord installation.

        Store Requirements:
            The selected Store must provide current writable declaration and claim
            publication semantics.
        """
        from .reference_values import ObjectRef

        if not isinstance(reference, ObjectRef) or not reference.objects:
            raise ValueError("fork_object_ref requires a non-empty ObjectRef.")
        selected = self._selected_writable_store(store, "fork ObjectRef")
        fork, _ = _fork_rekey_reference(reference, namespace)
        return self._register_declaration(fork, selected, allow_preallocated=True)

    def fork_state_ref(self, state_ref, *, store=None, namespace=None, federated: bool = False):
        """Rekey verified state authority and publish only after closure staging.

        Args:
            state_ref: Complete non-empty authoritative StateRef to fork.
            store: Explicit writable target Store when Repo Stores are ambiguous.
            namespace: Replacement namespace for all newly allocated IDs, or
                ``None`` to preserve source namespaces despite active scopes.
            federated: Whether verified dependency state may remain in connected
                source Stores instead of being copied into ``store``.

        Returns:
            New exact StateRef with fresh ObjectIds and the source local-state
            hashes.

        Raises:
            ValueError: If the source is not a non-empty StateRef.
            RepoLoadError: If the root or materializing seed authority/local state
                is missing, conflicting, or fails verification.
            StoreAuthorityError: If target publication or copying fails.

        Side Effects:
            Verifies every local state before allocating fork authority. A
            non-federated target receives verified local states, DefinitionRecords,
            and embedded seed records before the final StateRef boundary.

        Concurrency:
            Final record publication is serialized by the target Store writer
            lock. Pre-boundary failure leaves no fork StateRef authority.

        Store Requirements:
            Source Stores must remain connected for closure verification. The
            target must provide writable immutable-record and local-state install
            semantics; federated forks retain their connected source dependencies.
        """
        from .reference_values import StateRef
        from .store.records import DefinitionRecord, StateRefRecord
        from .repo_plan import _embedded_state_refs, _find_local_state

        if not isinstance(state_ref, StateRef) or not state_ref.object.objects:
            raise ValueError("fork_state_ref requires a non-empty StateRef.")
        selected = self._selected_writable_store(store, "fork StateRef")
        # Validate the root and every materializing exact seed before allocating
        # new identities. A non-federated fork must carry this complete closure.
        references = []
        seen_references = set()

        def collect(reference):
            if reference.digest() in seen_references:
                return
            source_record = None
            for candidate in self.stores:
                record = candidate.read_state_ref_record(reference.digest())
                if record is not None:
                    if record.state_ref != reference:
                        raise RepoLoadError("StateRef digest collision has incompatible authority.")
                    source_record = record
                    break
            if source_record is None:
                raise RepoLoadError("Fork source lacks an authoritative exact StateRef record.")
            seen_references.add(reference.digest())
            references.append(reference)
            for _, seed in _embedded_state_refs(reference.definition):
                collect(seed)

        collect(state_ref)
        sources = []
        for reference in references:
            for path, state_hash in reference.states.items():
                definition = reference.object.at(path).definition
                source = _find_local_state(self, definition, state_hash)
                if source is None:
                    raise RepoLoadError(f"Fork source lacks verified local state at {path!s}.")
                sources.append((definition, state_hash, source))
        fork, forked_seeds = _fork_rekey_reference(state_ref, namespace)
        with selected.writer_lock():
            # DefinitionRecords are graph-aware, but every local-state entry is
            # independently verified against its node definition before copying.
            for definition, state_hash, source in sources:
                selected.write_definition_record(DefinitionRecord(definition))
                if not federated and source is not selected:
                    selected.copy_local_state_from(source, definition, state_hash)
            if not federated:
                for seed in forked_seeds:
                    if seed != fork:
                        selected.write_definition_record(DefinitionRecord(seed.definition))
                        selected.write_state_ref_record(StateRefRecord(seed))
            selected.write_definition_record(DefinitionRecord(fork.definition))
            selected.write_state_ref_record(StateRefRecord(fork))
        return fork

    def set_config(self, key: str, value: Any) -> None:
        if not isinstance(key, str):
            raise TypeError("Config keys must be strings.")
        if key == "":
            raise ValueError("Config keys cannot be empty.")
        self.config[key] = value

    def update_config(self, values: Mapping[str, Any]) -> None:
        for key, value in values.items():
            self.set_config(key, value)

    def get_config(self, key: str, default=CONFIG_MISSING) -> Any:
        if not isinstance(key, str):
            raise TypeError("Config keys must be strings.")
        if key in self.config:
            return self.config[key]

        cur = self.config
        found_nested = True
        for part in key.split("."):
            if isinstance(cur, Mapping) and part in cur:
                cur = cur[part]
            else:
                found_nested = False
                break
        if found_nested:
            return cur

        if default is not CONFIG_MISSING:
            return default
        raise ConfigError(f"Repo config has no value for {key!r}.")

    def resolve_config(self, value: Any) -> Any:
        from .definition import ConcreteDefinition, Definition

        if isinstance(value, ConfigRef):
            if value.has_default:
                return self.get_config(value.key, default=value.default)
            return self.get_config(value.key)

        if isinstance(value, (ConcreteDefinition, Definition)):
            return value

        if isinstance(value, Mapping):
            return {k: self.resolve_config(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(self.resolve_config(v) for v in value)
        if isinstance(value, list):
            return [self.resolve_config(v) for v in value]
        if isinstance(value, (set, frozenset)):
            return type(value)(self.resolve_config(v) for v in value)

        return value

    def has_cdef_light(self, cdef: ConcreteDefinition) -> bool:
        """Return whether current definition authority exists in any Store."""
        from .store.records import DefinitionRecord

        digest = DefinitionRecord(cdef).digest
        return any(store.read_definition_record(digest) is not None for store in self.stores)

    def hydrate_from_stores(self):
        """
        Ask each store to enumerate all cdefs it has.
        Populate obj_cache[cdef] = None for those not already present.
        """
        self._query_catalog.refresh(True)

    def refresh_index(self, *, force: bool = True):
        self._query_index.refresh(True if force else "auto")
        self._query_catalog.refresh(True if force else "auto")
        return self

    def index_status(self, store=None):
        if store is not None:
            store = make_store(store)
        return self._query_index.index_status(store=store)

    def rebuild_index(self, store=None):
        if store is not None:
            store = make_store(store)
        self._query_index.rebuild(store=store)
        return self

    def validate_index(self, store=None, *, thorough: bool = False):
        if store is not None:
            store = make_store(store)
        return self._query_index.validate(store=store, thorough=thorough)

    def __len__(self):
        return len(self.strong_obj_cache)

    def save_object(
            self,
            obj,
            *,
            main: bool = False,
            store=None,
            alias: str | None = None,
            deep_capture: bool = False,
            federated: bool = False,
            report_stores: bool = False,
            _capture_memo: set[object] | None = None):
        """Publish one live graph as immutable local states and a StateRef.

        Args:
            obj: Root object whose graph is saved.
            main: Whether its concrete definition becomes the main reference.
            store: Optional target Store.
            alias: Optional object alias written after StateRef publication.
            deep_capture: Whether every owned Serializable node is serialized.
            federated: Whether reusable dependencies may remain external.
            report_stores: Whether to return an ephemeral StoreReport.

        Returns:
            The complete StateRef, or it with the requested StoreReport.

        Raises:
            TypeError: If ``alias`` is not a string.
            ValueError: If ``alias`` is empty.
            StoreAuthorityError: If Store publication rejects authoritative data.

        Side Effects:
            Publishes every local state and the enclosing StateRef before main or
            object-alias references can change.
        """
        from dryml.runtime import materialization_admission
        from .store.records import DefinitionRecord, MainRefRecord, ObjectAliasRecord

        with materialization_admission(operation="repo_save_object"):
            if alias is not None:
                self._validate_alias(alias)
            store = self._ensure_store(store) or self.default_store
            if store is None:
                raise RepoSaveError("No Store available to save object.")
            lease = getattr(obj, "_claim_lease", None)
            if isinstance(lease, _ClaimLease):
                if store is not lease.store:
                    raise RepoSaveError("The initial StateRef must be published in the declaration Store.")
                self._renew_claim(lease)
            capture_memo = set() if _capture_memo is None else _capture_memo
            # Complete nested declarations before publishing an enclosing graph.
            # The nested immutable StateRef then supplies reusable state for
            # adoption, so deep capture does not serialize it a second time.
            for dependency_lease, dependency_obj in getattr(obj, "_pending_claim_dependencies", ()):
                if dependency_lease is lease:
                    continue
                self.save_object(
                    dependency_obj,
                    store=dependency_lease.store,
                    deep_capture=deep_capture,
                    federated=federated,
                    _capture_memo=capture_memo,
                )
                from .repo_plan import build_save_plan

                capture_memo.update(
                    action.obj.object_id
                    for action in build_save_plan(
                        self, dependency_obj, store=dependency_lease.store
                    ).actions
                    if isinstance(action.obj, Serializable)
                )
            self.add_objects(obj, store=store)
            from .repo_plan import build_save_plan, execute_save_plan

            plan = build_save_plan(self, obj, store=store)
            previous_lease = getattr(self, "_publishing_claim_lease", None)
            self._publishing_claim_lease = lease
            try:
                result = execute_save_plan(
                    self, plan, store=store, deep_capture=deep_capture,
                    federated=federated, report_stores=report_stores,
                    capture_memo=capture_memo,
                )
            except BaseException:
                for pending_lease in reversed(
                        getattr(obj, "_claim_leases", (lease,) if lease else ())):
                    self._abandon_claim(pending_lease)
                raise
            finally:
                self._publishing_claim_lease = previous_lease
            state_ref = result[0] if report_stores else result
            if isinstance(lease, _ClaimLease):
                obj._claim_lease = None
            if main:
                store.write_main_ref(MainRefRecord(DefinitionRecord(obj.definition).digest))
                self.main_def = obj.definition
            if alias is not None:
                self.set_alias(alias, state_ref.object, store=store, save_live=False)
            return result

    def save(
            self,
            obj: Object,
            *,
            main: bool = False,
            store=None,
            alias: str | None = None,
            deep_capture: bool = False,
            federated: bool = False,
            report_stores: bool = False):
        """Publish one object graph through the direct StateRef save surface."""
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_save"):
            result = self.save_object(
                obj, main=main, store=store, alias=alias,
                deep_capture=deep_capture, federated=federated,
                report_stores=report_stores,
            )
            self.flush()
            return result

    def _first_store_with(self, cdef):
        from .store.records import DefinitionRecord

        digest = DefinitionRecord(cdef).digest
        for st in self.stores:
            if st.read_definition_record(digest) is not None:
                return st
        return None

    def _load_options(
            self,
            *,
            options: RepoLoadOptions | None = None,
            instance: InstancePolicy = "reuse",
            restore_state: bool = True,
            build_missing: bool = False,
            reuse_weak: bool = True,
            cache: CachePolicy = "weak",
            revision: RevisionType | str | None = None) -> RepoLoadOptions:
        if options is not None:
            return options
        return RepoLoadOptions(
            instance=instance,
            restore_state=restore_state,
            build_missing=build_missing,
            reuse_weak=reuse_weak,
            cache=cache,
            revision=revision,
        )

    def _candidate_cdefs(self, *, reuse_weak: bool = True) -> set[ConcreteDefinition]:
        cdefs = {obj.definition for _, obj in self.strong_obj_cache.items()}
        if reuse_weak:
            cdefs.update(obj.definition for _, obj in self.weak_obj_cache.items())
        cdefs.update(self.light_index)
        return cdefs

    @staticmethod
    def _selector_tuple(selector):
        if type(selector) is list:
            return tuple(selector)
        if type(selector) is tuple:
            return selector
        return (selector,)

    # -------------------------------------------------------------------------
    # Core: realize arbitrary structure into runtime Python + Objects
    # -------------------------------------------------------------------------
    def _realize(
        self,
        x: Any,
        *,
        instance: InstancePolicy = "reuse",
        restore_state: bool = True,
        build_missing: bool = False,
        reuse_weak: bool = True,
        cache: CachePolicy = "weak",
        revision: RevisionType | None = None,
        options: RepoLoadOptions | None = None,
        memo: dict | None = None,
        path: list[str | int] | None = None,
    ):
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_realize"):
            load_options = self._load_options(
                options=options,
                instance=instance,
                restore_state=restore_state,
                build_missing=build_missing,
                reuse_weak=reuse_weak,
                cache=cache,
                revision=revision,
            )
            return from_canonical(
                x,
                repo=self,
                options=load_options,
                memo=memo,
                path=path,
            )

    # -------------------------------------------------------------------------
    # Core: turn a ConcreteDefinition into a live Object under load knobs
    # -------------------------------------------------------------------------
    def _materialize_cdef(
        self,
        cdef,
        revision: RevisionType | str | None = None,
        *,
        options: RepoLoadOptions | None = None,
        instance: InstancePolicy = "reuse",
        restore_state: bool = True,
        build_missing: bool = False,
        reuse_weak: bool = True,
        cache: CachePolicy = "weak",
        # internal
        memo: dict | None = None,   # cdef->obj memo for this realization pass
        path: list[str | int] | None = None,
    ):
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_materialize_cdef"):
            if memo is None:
                memo = {}
            if path is None:
                path = ["<root>"]

            load_options = self._load_options(
                options=options,
                instance=instance,
                restore_state=restore_state,
                build_missing=build_missing,
                reuse_weak=reuse_weak,
                cache=cache,
                revision=revision,
            )
            revision = manage_revision(cdef, load_options.revision)
            from .materialization import build_materialization_plan, execute_materialization_plan

            plan = build_materialization_plan(
                self,
                cdef,
                load_options,
                revision=revision,
                memo=memo,
                path=path,
            )
            return execute_materialization_plan(
                self,
                plan,
                memo=memo,
                revision=revision,
                root=cdef,
            )


    def load_object(
        self,
        x: object,
        *,
        instance: InstancePolicy = "reuse",
        restore_state: bool = True,
        build_missing: bool = False,
        reuse_weak: bool = True,
        cache: CachePolicy = "weak",
        revision: RevisionType|str | None = None,
        options: RepoLoadOptions | None = None,
    ):
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_load_object"):
            from .repo_plan import _NodeBindings, realization_scope

            with realization_scope():
                load_options = self._load_options(
                    options=options,
                    instance=instance,
                    restore_state=restore_state,
                    build_missing=build_missing,
                    reuse_weak=reuse_weak,
                    cache=cache,
                    revision=revision,
                )
                memo = _NodeBindings()
                return self._realize(
                    x,
                    options=load_options,
                    path=[""],
                    memo=memo,
                )

    def load(self, cdef: ConcreteDefinition, **kwargs) -> Object:
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_load"):
            if not isinstance(cdef, ConcreteDefinition):
                raise TypeError("Repo.load requires an exact ConcreteDefinition.")
            if kwargs.get("options") is not None:
                kwargs["options"] = replace(kwargs["options"], build_missing=False)
            kwargs["build_missing"] = False
            return self.load_object(cdef, **kwargs)

    def load_or_build(self, x: object, **kwargs):
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_load_or_build"):
            if isinstance(x, Object):
                x = x.definition
            elif isinstance(x, Definition):
                x = x.concretize(repo=self)
            elif not isinstance(x, ConcreteDefinition):
                raise TypeError("Repo.load_or_build requires a Definition, ConcreteDefinition, or Object.")
            if kwargs.get("options") is not None:
                kwargs["options"] = replace(kwargs["options"], build_missing=True)
            kwargs["build_missing"] = True
            return self.load_object(x, **kwargs)


    def __contains__(
            self, item: Object | ConcreteDefinition, weak=True):
        # if weak is true, check both strong and weak caches
        if isinstance(item, ConcreteDefinition):
            cdef = item
        elif isinstance(item, Object):
            cdef = item.definition
        else:
            raise TypeError(
                f"Unsupported type {type(item)} for Repo.__contains__!")

        # “Strong” membership: known in cache and either loaded or known to exist
        in_cache = cdef in self.strong_obj_cache
        if not in_cache and weak:
            in_cache = cdef in self.weak_obj_cache
        in_store = cdef in self.light_index or bool(self._query_catalog.stores_for_cdef(cdef))
        return in_cache or in_store

    def __getitem__(
            self, key: ConcreteDefinition):
        """
        Easy access to objects within.

        if unpack is true, plain objects are returned
        """
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_getitem"):
            if not isinstance(key, ConcreteDefinition):
                raise TypeError("Repo.__getitem__ requires a ConcreteDefinition key.")
            result = self.query(key).known().objects()
            if len(result) == 0:
                raise KeyError(f"Repo doesn't contain an object with definition {key}")
            return result.one()

    def query(self, selector=None):
        from .query import DefinitionQuery

        return DefinitionQuery.from_source(self, selector)

    def definition_graph(self, value) -> "ConcreteDefinitionGraph":
        from .cdef_graph import ConcreteDefinitionGraph

        def cdef_from(item):
            if isinstance(item, Object):
                return item.definition
            if isinstance(item, ConcreteDefinition):
                return item
            if isinstance(item, Definition):
                raise TypeError("definition_graph() requires exact ConcreteDefinition values; concretize Definitions first.")
            raise TypeError(f"definition_graph() cannot inspect {type(item).__name__}.")

        if isinstance(value, (Object, ConcreteDefinition, Definition)):
            return ConcreteDefinitionGraph.from_root(cdef_from(value))
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
            return ConcreteDefinitionGraph.from_roots(cdef_from(item) for item in value)
        raise TypeError(f"definition_graph() cannot inspect {type(value).__name__}.")

    def find_defs(
            self,
            selector=None,
            *,
            scope: str = "stored",
            refresh="auto",
            class_match: str = "selector"):
        q = self.query(selector).class_match(class_match).refresh(refresh)
        if scope == "stored":
            return q.stored().defs()
        if scope == "known":
            return q.known().defs()
        if scope == "cached":
            return q.cached().defs()
        if scope == "nested":
            return q.nested().definitions().defs()
        raise ValueError("scope must be 'stored', 'known', 'cached', or 'nested'.")

    def find_occurrences(
            self,
            selector=None,
            *,
            refresh="auto",
            class_match: str = "selector",
            max_occurrences: int | None = None):
        return (
            self.query(selector)
            .class_match(class_match)
            .refresh(refresh)
            .nested()
            .max_occurrences(max_occurrences)
            .execute()
        )

    def find_owner_defs(
            self,
            selector=None,
            *,
            refresh="auto",
            class_match: str = "selector"):
        return self.query(selector).class_match(class_match).refresh(refresh).nested().owners().defs()

    def find(
            self,
            selector=None,
            *,
            scope: str = "stored",
            refresh="auto",
            class_match: str = "selector",
            **load_options):
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_find"):
            q = self.query(selector).class_match(class_match).refresh(refresh)
            if scope == "stored":
                q = q.stored()
            elif scope == "known":
                q = q.known()
            elif scope == "cached":
                q = q.cached()
            else:
                raise ValueError("scope must be 'stored', 'known', or 'cached'.")
            return q.objects(**load_options)

    def find_owners(
            self,
            selector=None,
            *,
            refresh="auto",
            class_match: str = "selector",
            **load_options):
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_find_owners"):
            return (
                self.query(selector)
                .class_match(class_match)
                .refresh(refresh)
                .nested()
                .owners()
                .objects(**load_options)
            )

    def get(self,
            selector:  SelectorType | tuple[SelectorType] | list[SelectorType] | None = None,
            sel_args=None, sel_kwargs=None,
            instance: InstancePolicy = "reuse",
            restore_state: bool = True,
            build_missing: bool = False,
            reuse_weak: bool = True,
            cache: CachePolicy = "weak",
            revision: RevisionType | str | None = None,
            options: RepoLoadOptions | None = None,
            verbose: bool = True) -> ObjectResultSet:
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_get"):
            if sel_args is None:
                sel_args = []
            if sel_kwargs is None:
                sel_kwargs = {}
            load_options = self._load_options(
                options=options,
                instance=instance,
                restore_state=restore_state,
                build_missing=build_missing,
                reuse_weak=reuse_weak,
                cache=cache,
                revision=revision,
            )
            if load_options.build_missing:
                raise ValueError("Repo.get selects existing objects only; use Repo.load_or_build for construction.")
            selectors = self._selector_tuple(selector)
            if isinstance(load_options.revision, str):
                raise ValueError("plain string revisions aren't supported in `get`.")
            selected_objects: dict[ConcreteDefinition, Object] = {}
            for sel in selectors:
                if isinstance(sel, Callable) and not isinstance(sel, (Definition, ConcreteDefinition)):
                    for cdef, obj in self.strong_obj_cache.items():
                        if sel(obj, *sel_args, **sel_kwargs):
                            selected_objects[cdef] = obj
                    continue

                objs = (
                    self.query(sel)
                    .known()
                    .reuse_weak(load_options.reuse_weak)
                    .objects(options=load_options)
                )
                selected_objects.update(objs)

            return ObjectResultSet(self, selected_objects, domain="known")

    def apply(self,
              func, func_args=None, func_kwargs=None,
              selector: Optional[Callable] = None,
              sel_args=None, sel_kwargs=None,
              verbose: bool = False,
              options: RepoLoadOptions | None = None,
              **kwargs):
        """
        Apply a function to all objects tracked by the repo.
        We can also use a Selector to apply only to specific models
        **kwargs is passed to self.get
        """
        if func_args is None:
            func_args = []
        if func_kwargs is None:
            func_kwargs = {}

        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_apply"):
            # Create apply function
            def apply_func(obj):
                return func(obj, *func_args, **func_kwargs)

            # Get object list
            objs = self.get(
                selector=selector,
                sel_args=sel_args, sel_kwargs=sel_kwargs,
                options=options,
                **kwargs)

            obj_iter = objs.items()
            if verbose:
                obj_iter = tqdm(obj_iter)
            return {
                obj_def: apply_func(obj) for obj_def, obj in obj_iter
            }

    def _graph_options(
            self,
            *,
            options: RepoGraphOptions | None = None,
            include_root: bool = True,
            order: str = "post",
            missing: str = "raise",
            dedupe: bool = True,
            instance: InstancePolicy = "reuse",
            restore_state: bool = True,
            build_missing: bool = False,
            reuse_weak: bool = True,
            cache: CachePolicy = "weak",
            revision: RevisionType | str | None = None) -> RepoGraphOptions:
        if options is not None:
            return options
        load_options = self._load_options(
            instance=instance,
            restore_state=restore_state,
            build_missing=build_missing,
            reuse_weak=reuse_weak,
            cache=cache,
            revision=revision,
        )
        return RepoGraphOptions(
            load=load_options,
            include_root=include_root,
            order=order,
            missing=missing,
            dedupe=dedupe,
        )

    def iter_graph(
            self,
            root,
            *,
            options: RepoGraphOptions | None = None,
            include_root: bool = True,
            order: str = "post",
            missing: str = "raise",
            dedupe: bool = True,
            instance: InstancePolicy = "reuse",
            restore_state: bool = True,
            build_missing: bool = False,
            reuse_weak: bool = True,
            cache: CachePolicy = "weak",
            revision: RevisionType | str | None = None):
        graph_options = self._graph_options(
            options=options,
            include_root=include_root,
            order=order,
            missing=missing,
            dedupe=dedupe,
            instance=instance,
            restore_state=restore_state,
            build_missing=build_missing,
            reuse_weak=reuse_weak,
            cache=cache,
            revision=revision,
        )
        from .repo_plan import iter_graph_objects
        from dryml.runtime import materialization_admission

        def guarded_iterator():
            with materialization_admission(operation="repo_iter_graph"):
                yield from iter_graph_objects(self, root, graph_options)

        return guarded_iterator()

    def apply_graph(
            self,
            root,
            func,
            *,
            options: RepoGraphOptions | None = None,
            include_root: bool = True,
            order: str = "post",
            missing: str = "raise",
            dedupe: bool = True,
            instance: InstancePolicy = "reuse",
            restore_state: bool = True,
            build_missing: bool = False,
            reuse_weak: bool = True,
            cache: CachePolicy = "weak",
            revision: RevisionType | str | None = None):
        graph_options = self._graph_options(
            options=options,
            include_root=include_root,
            order=order,
            missing=missing,
            dedupe=dedupe,
            instance=instance,
            restore_state=restore_state,
            build_missing=build_missing,
            reuse_weak=reuse_weak,
            cache=cache,
            revision=revision,
        )
        from .repo_plan import apply_graph_objects
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_apply_graph"):
            return apply_graph_objects(self, root, func, graph_options)

    def set_main_def(self, main_def: ConcreteDefinition, store=None):
        """Stage a concrete main definition in this Repo and selected Store.

        Args:
            main_def: ``ConcreteDefinition`` to make the default root.
            store: Optional Store to receive the staged reference; defaults to
                this Repo's default Store.

        Raises:
            TypeError: If ``main_def`` is not a concrete definition.
            ValueError: If no Store is available.

        Side Effects:
            Changes the Repo cache only after validation and marks the Store for
            its explicit reference-publication path.
        """
        if not isinstance(main_def, ConcreteDefinition):
            raise TypeError("Main definition must be a ConcreteDefinition.")
        if store is None:
            store = self.default_store
        if store is None:
            raise ValueError("No store available to set main definition!")
        from .store.records import DefinitionRecord, MainRefRecord

        record = store.write_definition_record(DefinitionRecord(main_def))
        store.write_main_ref(MainRefRecord(record.digest))
        self.main_def = main_def

    def add_objects(self, *args, store=None):
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_add_objects"):
            store = self._ensure_store(store)
            from .repo_plan import add_objects

            add_objects(self, args, store=store)

    def flush(self):
        """Publish this Repo's pending aliases and commit each configured Store.

        Raises:
            StoreAuthorityError: If a reference payload is malformed or a
                concurrent Store update conflicts with this Repo's alias change.
            OSError: If a Store cannot publish its authoritative bytes.

        Side Effects:
            Valid aliases are merged with non-conflicting concurrent DirStore
            changes, published to each Store, and cleared from the dirty state
            only after every Store commits successfully.
        """
        for store in self.stores:
            store.commit()
        self._aliases_dirty = False

    def close(self, flush=True):
        if flush:
            self.flush()
        self._query_index.close()

    def __del__(self):
        if self.save_objs_on_deletion:
            self.save()
            self.close(flush=True)

    def clear_cache(self, strong=False, weak=True):
        if strong:
            self.strong_obj_cache.clear()
        if weak:
            self.weak_obj_cache.clear()

    @staticmethod
    def dir_store_inspect(root_path: str):
        files = glob.glob(os.path.join(root_path, '**/def.pkl'), recursive=True)
        # Strip root directory
        return list(map(lambda f: f[len(root_path)+1:], files))


def make_store(store):
    """Normalize a Store, path, or seekable binary file into a Store.

    Args:
        store: Existing Store, directory/archive path, or seekable file-like
            object that supplies ``read``, ``write``, ``seek``, and ``truncate``.

    Returns:
        The existing Store, a directory Store, or a ZIP-backed Store.

    Raises:
        ValueError: If ``store`` is not a supported path or file-like object.

    Side Effects:
        Path and file-like inputs may open or initialize their corresponding
        Store backend.
    """
    from .store.store import Store
    if isinstance(store, Store):
        return store

    elif isinstance(store, IOBase) or all(
            callable(getattr(store, name, None))
            for name in ("read", "write", "seek", "truncate")):
        from .store.zip import ZipStore
        # file-like => zip-backed store in a temp dir
        return ZipStore(store)

    elif isinstance(store, (str, Path)):
        from .store.dir import DirStore
        from .store.zip import ZipStore
        path = os.fspath(store)
        if os.path.isdir(path):
            store = DirStore(store)
        else:
            # treat as zip file path (may or may not exist yet)
            store = ZipStore(store)
        return store
    else:
        raise ValueError(f"Cannot open a store pointing to location {store!r}")


# Context management for explicit default repo authority.
_current_repo: ContextVar["Repo|None"] = ContextVar("_current_repo", default=None)


# This cleanup system is required because we use a 'heavy'
# hash function which wants to import types at runtime.
# This causes a crash at cleanup, so we explicitly cleanup
# repos so they aren't left until after the module import
# system is cleaned up.
def global_repo_cleanup():
    from .session import close_configured_repo

    close_configured_repo()
atexit.register(global_repo_cleanup)


# Get the current default repo
def get_default_repo() -> "Repo | None":
    """Return only explicitly active Repo authority, if any.

    Returns:
        The innermost context-local or session-configured Repo, or ``None``.

    Side Effects:
        None. This function never creates a Repo or falls back to process-global
        mutable state.
    """

    r = _current_repo.get()
    if r is not None:
        return r

    from .session import current_repo

    r = current_repo()
    return r


# Context manager for isolated repo
@contextmanager
def default_repo(r: Repo|None=None):
    """Install an explicit Repo for the dynamic extent of the context.

    Args:
        r: Repo to install. ``None`` creates a temporary in-memory Repo.

    Yields:
        The installed Repo.

    Side Effects:
        Restores the prior context-local authority and closes a Repo created for
        this context on exit.
    """

    close_repo = r is None
    if close_repo:
        r = Repo()
    tok = _current_repo.set(r)
    try:
        yield r
    finally:
        _current_repo.reset(tok)
        if close_repo:
            r.close()


@contextmanager
def manage_repo(repo=None):
    """
    Handle all the following cases:

      * repo is None:
          - reuse an explicitly context-local/session Repo when present
          - otherwise create, install, and auto-close a fresh in-memory Repo

      * repo is a Repo:
          - use it as-is, do not close it at the end

      * repo is a Store:
          - Use the store as is

      * repo is an IOBase:
          - treat it as a zip container
          - create ZipStore(repo), Repo([ZipStore])
          - auto-close (commit+cleanup) at the end

      * repo is a str/Path:
          - if it points to an existing directory: DirStore(path)
          - else: ZipStore(path)
          - Repo([store])
          - auto-close at the end

      * repo is a list containing the previous types
          - Create a repo backed with multiple stores
    """
    close_repo = False

    if repo is None:
        repo_obj = get_default_repo()
        if repo_obj is None:
            repo_obj = Repo()
            close_repo = True

    elif isinstance(repo, Repo):
        # user-supplied repo, don't manage its lifetime
        repo_obj = repo

    else:
        if isinstance(repo, list):
            # Check there are no Repos or Nones.
            for el in repo:
                if el is None or isinstance(el, Repo):
                    raise ValueError("Store list can't contain a None or Repo object.")
            stores = [
                make_store(store)
                for store in repo
            ]
        else:
            stores = [ make_store(repo) ]
            
        repo_obj = Repo(stores=stores)
        close_repo = True

    with default_repo(repo_obj):
        try:
            yield repo_obj
        finally:
            if close_repo:
                repo_obj.close()


# Saving and Loading
def save_object(
        obj,
        repo=None,
        *,
        main=False,
        store=None,
        alias: str | None = None,
        deep_capture: bool = False,
        federated: bool = False,
        report_stores: bool = False):
    """Publish one Object graph through its current immutable StateRef boundary."""
    from dryml.runtime import materialization_admission

    with materialization_admission(operation="global_save_object"):
        with manage_repo(repo=repo) as sub_repo:
            main = main or ((repo is not sub_repo) and isinstance(obj, Object))
            return sub_repo.save_object(
                obj, main=main, store=store, alias=alias,
                deep_capture=deep_capture, federated=federated,
                report_stores=report_stores,
            )


def load_alias(alias: str, repo=None, **kwargs):
    from dryml.runtime import materialization_admission

    with materialization_admission(operation="global_load_alias"):
        with manage_repo(repo=repo) as repo:
            return repo.load_alias(alias, **kwargs)


def load_object(
        cdef=None, repo=None,
        revision: RevisionType|str|None=None,
        **kwargs):
    from dryml.runtime import materialization_admission

    with materialization_admission(operation="global_load_object"):
        with manage_repo(repo=repo) as repo:
            if cdef is None:
                cdef = repo.main_def
                if cdef is None:
                    raise ValueError("When cdef is None, the repo must have a main def, we didn't find one.")
            return repo.load_object(cdef, revision=revision, **kwargs)
