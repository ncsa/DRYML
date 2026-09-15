from __future__ import annotations

import os
import glob
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable
from contextlib import ExitStack, contextmanager, nullcontext
from io import IOBase
from pathlib import Path
import weakref
from contextvars import ContextVar
from collections.abc import Iterable, Mapping
from collections import defaultdict
import atexit
import time
import stat
from threading import RLock
from uuid import uuid4

from .definition import Definition, ConcreteDefinition
from .cdef_graph import ConcreteDefinitionGraph
from .object import Object
from .store.store import Store, StoreCapabilityError
from .policies import CachePolicy, LiveReusePolicy, RepoGraphOptions
from .canonical import from_canonical
from .config import CONFIG_MISSING, ConfigError, ConfigRef
from .query.federation import RepoQueryIndex
from .query.memory import AggregateMemoryQueryIndex
from .query.result import ObjectResultSet

if TYPE_CHECKING:
    from .repo_definition import RepoDefinition


_active_object_ref_builds: ContextVar[frozenset[tuple[int, str]]] = ContextVar(
    "dryml_active_object_ref_builds", default=frozenset()
)


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
    return root, dict(state_cache)


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
        self._keys_by_node = defaultdict(set)
        self._weak = weak

    def _drop_key(self, key):
        self._values.pop(key, None)
        keys = self._keys_by_node.get(key[1])
        if keys is not None:
            keys.discard(key)
            if not keys:
                self._keys_by_node.pop(key[1], None)

    def _remove_dead_ref(self, key, dead_ref):
        values = self._values.get(key)
        if values is None:
            return
        self._values[key] = [ref for ref in values if ref is not dead_ref]
        if not self._values[key]:
            self._drop_key(key)

    def _live(self, key):
        values = self._values.get(key, ())
        if not self._weak:
            return tuple(values)
        retained = []
        live = []
        for ref in values:
            obj = ref()
            if obj is not None:
                retained.append(ref)
                live.append(obj)
        if retained:
            self._values[key] = retained
        else:
            self._drop_key(key)
        return tuple(live)

    def _keys(self, cdef):
        node = _node_key(cdef)
        return tuple(self._keys_by_node.get(node, ()))

    def add(self, cdef, obj):
        key = (getattr(obj, "_realization_scope", None), _node_key(cdef))
        values = self._live(key)
        if all(existing is not obj for existing in values):
            self._keys_by_node[key[1]].add(key)
            if self._weak:
                self._values[key].append(
                    weakref.ref(
                        obj,
                        lambda ref, cache=self, item=key: cache._remove_dead_ref(
                            item, ref
                        ),
                    )
                )
            else:
                self._values[key].append(obj)

    def discard(self, cdef, obj=None):
        if obj is None:
            for key in self._keys(cdef):
                self._drop_key(key)
            return None
        for key in self._keys(cdef):
            if self._weak:
                remaining = [
                    ref for ref in self._values.get(key, ())
                    if ref() is not None and ref() is not obj
                ]
            else:
                remaining = [
                    item for item in self._values.get(key, ()) if item is not obj
                ]
            if remaining:
                self._values[key] = remaining
            else:
                self._drop_key(key)

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
        self._keys_by_node.clear()

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
    """Save failure carrying immutable partial publication evidence when available.

    Args:
        message: Human-readable save failure summary.
        report: Optional immutable :class:`StoreReport` for work planned before
            capture or publication failed.

    ``report`` distinguishes completed, failed, unattempted, and uncertain Store
    boundaries. Completed immutable authority is deliberately preserved rather
    than rolled back across Stores. Failures before planning have ``report=None``;
    callers inspect the original chained cause for backend details.
    """

    def __init__(self, message: str, *, report=None) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True, slots=True)
class ReferenceDeclarationEvidence:
    """One deduplicated declaration identity observed in a stable Store cut.

    Attributes:
        object_ref: Complete declared identity whose definition matched the
            requested complete topology.
        stores: Connected Store handles carrying equal declaration replicas.
        claim_statuses: Valid associated ClaimRecord statuses in Store order.

    Declaration identity is evidence for reference selection only.  It does not
    grant construction permission or acquire a claim.
    """

    object_ref: object
    stores: tuple[Store, ...]
    claim_statuses: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReferenceStateEvidence:
    """One deduplicated exact StateRef observed in a stable Store cut.

    Attributes:
        state_ref: Complete immutable snapshot authority.
        stores: Connected Store handles carrying equal immutable replicas.

    State-only evidence can satisfy an explicitly supplied ObjectRef lookup but
    does not become declaration evidence for CDef strengthening.
    """

    state_ref: object
    stores: tuple[Store, ...]


@dataclass(frozen=True, slots=True)
class ReferenceEvidence:
    """Authoritative declaration and snapshot facts from one Repo evidence cut.

    Attributes:
        declarations: Topology-matching declared identities with validated claims.
        states: Topology-matching immutable snapshots, including state-only facts.

    The value contains no live Objects, payload bytes, query-index results, or
    construction reservations.  It is intended for read-only authority selection.
    """

    declarations: tuple[ReferenceDeclarationEvidence, ...]
    states: tuple[ReferenceStateEvidence, ...]


def _record_late_publication(
        report, *, store, phase, state_ref, operation, checker, error_checker=None):
    """Run one post-authority boundary and return an updated immutable report."""

    from .repo_plan import SavePublication

    publication = SavePublication(store, None, None, state_ref, phase, "unattempted")

    def updated(status):
        publications = list(report.publications)
        for index, existing in enumerate(publications):
            if (
                    existing.store is store and existing.phase == phase
                    and existing.path is None and existing.status == "unattempted"):
                publications[index] = replace(existing, state_ref=state_ref, status=status)
                break
        else:
            publications.append(replace(publication, status=status))
        return replace(report, publications=publications)

    try:
        operation()
    except BaseException as error:
        try:
            observed = (checker if error_checker is None else error_checker)()
            if observed is True:
                status = "completed"
            elif observed is False:
                status = "failed"
            else:
                status = "uncertain"
        except Exception:
            status = "uncertain"
        failure = updated(status)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            error.report = failure
            raise
        if isinstance(error, RepoSaveError):
            if error.report is None:
                error.report = failure
            raise
        raise RepoSaveError("Save publication failed.", report=failure) from error
    try:
        completed = checker()
    except Exception:
        completed = None
    if completed is not True:
        status = "failed" if completed is False else "uncertain"
        raise RepoSaveError("Save publication did not survive its read-back boundary.", report=updated(status))
    return updated("completed")


def _commit_publication_checker(store):
    """Return one post-commit proof checker for the supported Zip backend.

    A buffered Zip receipt is authoritative only when a dirty transaction began
    from its recorded archive baseline, then leaves a clean handle whose baseline
    equals the current destination identity. Other Store implementations expose no
    portable commit receipt, so an exception at their commit boundary is uncertain.
    """

    from .store.zip import ZipStore

    if type(store) is not ZipStore:
        return None
    was_dirty = store._archive_dirty
    baseline = store._archive_baseline
    try:
        destination = store._archive_identity()
    except Exception:
        return None

    def checker():
        if not was_dirty or destination != baseline:
            return None
        current = store._archive_identity()
        if not store._archive_dirty and current == store._archive_baseline:
            return True
        if store._archive_dirty and current == destination:
            return False
        return None

    return checker


def _commit_save_report(repo, state_ref, report, *, stores=None, dirty_only: bool = False):
    """Commit selected Stores and append each observed commit boundary.

    Args:
        repo: Repo whose alias-dirty marker is finalized after a successful commit.
        state_ref: Exact receipt associated with this bounded publication work.
        report: Immutable report to extend with observed commit boundaries.
        stores: Optional bounded Store sequence; omitted retains the Repo-wide
            historical flush behavior.
        dirty_only: Commit only Stores exposing a true buffered dirty marker.

    Returns:
        The report extended with each attempted commit boundary.

    Raises:
        RepoSaveError: If a selected commit fails or cannot survive read-back.

    Side Effects:
        Commits only the selected Stores. ``dirty_only`` avoids flushing unrelated
        buffered Store work.
    """

    selected = tuple(repo.stores if stores is None else stores)
    if dirty_only:
        selected = tuple(
            store for store in selected if getattr(store, "_archive_dirty", False)
        )
    # Freeze every required boundary before starting work so a first failure
    # leaves a complete ledger, including later commits that were not attempted.
    if selected:
        from .repo_plan import SavePublication

        planned_stores = {
            publication.store
            for publication in report.publications
            if publication.phase == "commit" and publication.status == "unattempted"
        }
        report = replace(
            report,
            publications=(*report.publications, *(
                SavePublication(store, None, None, state_ref, "commit", "unattempted")
                for store in selected
                if store not in planned_stores
            )),
        )
    for commit_store in selected:
        error_checker = _commit_publication_checker(commit_store)
        report = _record_late_publication(
            report, store=commit_store, phase="commit", state_ref=state_ref,
            operation=commit_store.commit,
            checker=error_checker or (lambda: True),
            error_checker=error_checker or (lambda: None),
        )
    if not dirty_only:
        repo._aliases_dirty = False
    return report


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
    """Coordinate live Objects, connected Stores, and save-routing policy.

    Repo owns live-object/query bindings and handles it opens from Store
    specifications, while supplied Store handles remain borrowed. A configured
    ``save_routing`` publishes retained per-object or closure placements; an
    unconfigured Repo retains default-Store closure behavior. Save/export context
    snapshots make configuration changes apply to later operations, and close
    rejects while a retained save or managed topology lease needs resources.
    """
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
            owner_token_factory: Callable[[], str] | None = None,
            save_routing=None,
            _state_io: bool = False):
        """Create a Repo over supplied Store handles and optional routing policy.

        Args:
            stores: One Store, Store specification, or ordered collection of
                them.  Configured routing requires one handle per physical
                built-in destination; repeated occurrences of one handle dedupe.
            config: Initial runtime-only configuration values.
            clock: Optional claim-clock function.
            lease_duration: Bounded claim duration in seconds.
            owner_token_factory: Optional claim-owner token factory.
            save_routing: ``None``, a detached SaveRouting, or a placement
                shorthand.  Shorthands configure empty first-match rules.
            _state_io: Internal authority-only view mode.

        Raises:
            TypeError: If routing input is not a supported policy or shorthand.
            ValueError: If a routing policy is invalid, disconnected, or names
                ambiguous built-in physical Store handles.

        Side Effects:
            Store specifications may open or initialize their backend and become
            Repo-owned handles; supplied Store instances remain borrowed. No
            routing validation publishes data or creates an implicit destination
            Store.

        Concurrency:
            Construction installs one initial Store/routing view. Later supported
            configuration changes are synchronized; direct mutation of ``stores``
            is not a concurrent configuration interface.
        """
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
        self._clock_explicit = clock is not None
        self._owner_token_factory_explicit = owner_token_factory is not None
        self._clock = clock or time.time
        self._lease_duration = float(lease_duration)
        self._owner_token_factory = owner_token_factory or (lambda: uuid4().hex)
        self._state_io = _state_io
        self._configuration_lock = RLock()
        self._configuration_version = 0
        self._save_context_leases = 0
        self._topology_leases = 0
        self._save_routing = None
        self._closing = False
        self._closed = False
        # Handles supplied by callers remain borrowed.  Paths and file-like
        # specifications opened by this Repo are released on successful close.
        self._owned_stores: list[Store] = []
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
        candidate_stores = []
        try:
            if stores is not None:
                if not isinstance(stores, (tuple, list)):
                    stores = [stores]

                for store in stores:
                    if not isinstance(store, Store):
                        store = make_store(store)
                        self._adopt_owned_stores((store,))
                    candidate_stores.append(store)
        except BaseException:
            self._close_owned_stores(suppress_errors=True)
            raise
        # Legacy unconfigured federation can retain separate Store handles.
        # Routing installation validates that topology before it becomes a save
        # destination policy.
        try:
            self.stores = list(self._normalize_store_handles(candidate_stores, reject_physical=False))
            self._query_index = RepoQueryIndex(self, authority_only=_state_io)
            self.set_save_routing(save_routing)

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
        except BaseException:
            if hasattr(self, "_query_index"):
                try:
                    self._query_index.close()
                except BaseException:
                    pass
            self._close_owned_stores(suppress_errors=True)
            raise

    # Store Methods

    @property
    def default_store(self):
        """Return the first connected Store, or ``None`` when no Store exists."""

        with self._configuration_lock:
            return self.stores[0] if self.stores else None

    @property
    def save_routing(self):
        """Return the normalized routing policy, or ``None`` for legacy closure.

        Returns:
            A normalized immutable SaveRouting policy, or ``None`` for the
            default-Store closure policy.

        The returned value is safe to retain. It configures future selection only
        and does not save Objects, create Stores, or alter an active save.
        """

        with self._configuration_lock:
            return self._save_routing

    def set_default_store(self, store: "Store"):
        """Make one connected Store the default in one configuration update.

        Args:
            store: Existing Store handle or an accepted Store specification.

        Side Effects:
            May open a supplied Store specification, publishes a new source and
            default ordering for later retained contexts, and refreshes query
            bindings.  It never changes a context already retained by a save.

        Concurrency:
            The replacement is synchronized with retained save/export contexts;
            active managed topology leases reject changes to the physical Store
            set.

        Raises:
            ValueError: If routing is configured and distinct built-in handles
                name one physical Store.

        Returns:
            ``None``.
        """

        self.add_store(store, make_default=True)

    def add_store(self, store: "Store", make_default=False):
        """Register one Store for later Repo operations without duplicating handles.

        Args:
            store: Existing Store handle or an accepted Store specification.
            make_default: Whether the Store becomes the first/default handle.

        Side Effects:
            May open a supplied Store specification and publishes a new source
            ordering for future retained contexts.  The same handle is retained
            once; no data is saved or copied.

        Concurrency:
            The registration is synchronized with retained save/export contexts;
            active managed topology leases reject changes to the physical Store
            set.

        Raises:
            ValueError: If routing is configured and distinct built-in handles
                name one physical Store.

        Returns:
            ``None``.
        """

        self._register_store(store, make_default=make_default)

    def _register_store(self, store: "Store", *, make_default: bool = False) -> Store:
        """Connect ``store`` and return its normalized handle for internal use.

        Public registration methods intentionally return ``None``. Internal
        Store-selection paths use this helper to retain the normalized handle
        without widening that public API.
        """

        with self._configuration_lock:
            if self._closing or self._closed:
                raise RuntimeError("Cannot change Store configuration after Repo close begins.")
            # Do not coerce a new specification while managed work has frozen the
            # physical state set: coercion can initialize persistent storage.
            if self._topology_leases and (
                    not isinstance(store, Store)
                    or not any(existing is store for existing in self.stores)):
                raise RuntimeError("Cannot change Store topology while an active topology lease retains resources.")
        opened = False
        if not isinstance(store, Store):
            store = make_store(store)
            opened = True
        try:
            with self._configuration_lock:
                if self._closing or self._closed:
                    raise RuntimeError("Cannot change Store configuration after Repo close begins.")
                existing = next((item for item in self.stores if item is store), None)
                if existing is not None and not make_default:
                    return store
                candidate = [item for item in self.stores if item is not store]
                if make_default or not candidate:
                    candidate.insert(0, store if existing is None else existing)
                elif existing is None:
                    candidate.append(store)
                else:
                    candidate.append(existing)
                candidate = list(self._normalize_store_handles(
                    candidate, reject_physical=self._save_routing is not None,
                ))
                previous_stores = self.stores
                if self._topology_leases and {
                    self._physical_store_key(item) or ("opaque", id(item))
                    for item in candidate
                } != {
                    self._physical_store_key(item) or ("opaque", id(item))
                    for item in previous_stores
                }:
                    raise RuntimeError("Cannot change Store topology while an active topology lease retains resources.")
                previous_version = self._configuration_version
                self.stores = candidate
                self._configuration_version += 1
                try:
                    self._query_index.refresh_bindings()
                except BaseException:
                    self.stores = previous_stores
                    self._configuration_version = previous_version
                    raise
        except BaseException:
            if opened:
                self._adopt_owned_stores((store,))
                self._close_one_owned_store(store, suppress_errors=True)
            raise
        if opened:
            self._adopt_owned_stores((store,))
        return store

    @staticmethod
    def _physical_store_key(store: Store):
        """Return stable identity for built-in persistent Store handles.

        Direct Stores identify their canonical root with filesystem evidence.
        Path-backed ZipStores identify their retained persistent archive path,
        never their temporary extraction directory or archive inode.  File-like
        ZipStores and custom Stores deliberately remain opaque in this
        registration boundary.
        """

        from .store.dir import DirStore
        from .store.zip import ZipStore

        if type(store) is ZipStore:
            archive_path = store.archive_path
            if archive_path is None:
                return None
            path = os.path.normcase(os.path.realpath(archive_path))
            parent = os.path.dirname(path) or "."
            try:
                evidence = os.stat(parent)
            except OSError as error:
                raise ValueError("ZipStore archive parent cannot be inspected.") from error
            if not stat.S_ISDIR(evidence.st_mode):
                raise ValueError("ZipStore archive parent is not a directory.")
            return ("zip", path)
        if type(store) is DirStore:
            path = os.path.normcase(os.path.realpath(store.base_dir))
            try:
                evidence = os.stat(path)
            except OSError as error:
                raise ValueError("DirStore root cannot be inspected.") from error
            if not stat.S_ISDIR(evidence.st_mode):
                raise ValueError("DirStore root is not a directory.")
            return ("dir", path, evidence.st_dev, evidence.st_ino)
        return None

    @classmethod
    def _normalize_store_handles(cls, stores, *, reject_physical: bool = True):
        """Deduplicate handles and optionally reject physical built-in aliases."""

        selected = []
        handles = set()
        physical = {}
        for store in stores:
            if not isinstance(store, Store):
                raise TypeError("Repo Store registrations must be Store instances.")
            handle = id(store)
            if handle in handles:
                continue
            handles.add(handle)
            key = cls._physical_store_key(store)
            existing = physical.get(key) if key is not None else None
            if existing is not None and existing is not store:
                if reject_physical:
                    raise ValueError("Distinct Store handles cannot name one physical destination.")
            if key is not None:
                physical[key] = store
            selected.append(store)
        return tuple(selected)

    def _adopt_owned_stores(self, stores) -> None:
        """Record newly opened handles exactly once for Repo lifetime cleanup."""

        for store in stores:
            if not any(existing is store for existing in self._owned_stores):
                self._owned_stores.append(store)

    def _close_one_owned_store(self, store: Store, *, suppress_errors: bool) -> None:
        """Close one owned handle, retaining it when cleanup cannot finish."""

        try:
            store.close()
        except BaseException:
            if not suppress_errors:
                raise
        else:
            self._owned_stores[:] = [
                existing for existing in self._owned_stores if existing is not store
            ]

    def _close_owned_stores(self, *, suppress_errors: bool = False) -> None:
        """Release owned handles, retaining failed cleanup handles for retry.

        Args:
            suppress_errors: Preserve an earlier setup/reconstruction failure by
                ignoring cleanup errors. Ordinary ``close()`` calls leave this
                false and propagate the first cleanup failure.
        """

        errors = []
        for store in reversed(tuple(self._owned_stores)):
            try:
                store.close()
            except BaseException as error:
                errors.append(error)
            else:
                self._owned_stores[:] = [
                    existing for existing in self._owned_stores if existing is not store
                ]
        if errors and not suppress_errors:
            raise errors[0]

    def set_save_routing(self, routing) -> None:
        """Atomically replace or adjust the internal immutable save-routing policy.

        Args:
            routing: ``None`` for default-Store closure behavior, a SaveRouting
                whose route Store handles are already connected, or
                ``"per-object"``/``"closure"`` to change only graph placement
                while preserving current rules and match mode.

        Raises:
            TypeError: If ``routing`` is neither a supported policy nor string.
            ValueError: If a mode is unknown or a route destination is not an
                exact connected Store handle.

        Side Effects:
            Publishes one replacement policy for later retained contexts.  It
            never opens, creates, publishes to, moves, or deletes a Store.

        Returns:
            ``None``.

        Concurrency:
            The complete replacement is atomic with respect to retained save and
            export contexts. Direct concurrent mutation of ``stores`` is not a
            supported configuration API.
        """

        from .repo_plan import SaveRouting

        with self._configuration_lock:
            if self._closing or self._closed:
                raise RuntimeError("Cannot change save routing after Repo close begins.")
            if routing is None:
                normalized = None
            elif isinstance(routing, SaveRouting):
                normalized = routing
            elif isinstance(routing, str):
                if routing not in {"per-object", "closure"}:
                    raise ValueError("Save routing graph mode must be 'per-object' or 'closure'.")
                previous = self._save_routing or SaveRouting()
                normalized = SaveRouting(previous.routes, previous.match_mode, routing)
            else:
                raise TypeError("Save routing must be SaveRouting, a graph-mode string, or None.")
            if normalized is not None:
                stores = self._normalize_store_handles(self.stores)
                connected = {id(store) for store in stores}
                if any(id(store) not in connected for _, store in normalized.routes):
                    raise ValueError("Save routing destinations must be connected Store handles.")
            self._save_routing = normalized
            self._configuration_version += 1

    @contextmanager
    def _retain_save_context(self):
        """Retain one immutable source/default/routing view until the caller exits.

        Yields:
            SaveRoutingContext containing ordered source Stores, physical
            representatives, default Store, and routing policy captured together.

        Raises:
            RepoSaveError: If direct unsupported Store-list mutation disconnects
                an installed routing destination.

        Side Effects:
            Acquires a lightweight Repo resource lease.  ``close()`` rejects
            while any such context remains active; no Store is opened, created,
            or published during retention.
        """

        from .repo_plan import SaveRoutingContext

        with self._configuration_lock:
            if self._closing or self._closed:
                raise RuntimeError("Cannot retain a save context after Repo close begins.")
            stores = self._normalize_store_handles(
                self.stores, reject_physical=True,
            )
            routing = self._save_routing
            if routing is not None:
                connected = {id(store) for store in stores}
                if any(id(store) not in connected for _, store in routing.routes):
                    raise RepoSaveError("Save routing contains a disconnected destination.")
            context = SaveRoutingContext(
                stores, stores[0] if stores else None, routing,
                self._configuration_version,
            )
            self._save_context_leases += 1
        try:
            yield context
        finally:
            with self._configuration_lock:
                self._save_context_leases -= 1

    @contextmanager
    def retain_topology(self):
        """Retain the connected physical Store set without selecting save routes.

        Returns:
            A context manager covering the current connected physical Store set.

        Raises:
            RuntimeError: If the Repo is closing or closed.

        Side Effects:
            Prevents Store addition and Repo close until release. Routing and
            default-order changes remain valid because they do not alter the
            physical connected set.
        """

        with self._configuration_lock:
            if self._closing or self._closed:
                raise RuntimeError("Cannot retain topology after Repo close begins.")
            self._normalize_store_handles(self.stores, reject_physical=True)
            self._topology_leases += 1
        try:
            yield self
        finally:
            with self._configuration_lock:
                self._topology_leases -= 1

    def _select_save_destinations(self, context, value) -> tuple[Store, ...]:
        """Select retained routing destinations for one Object without publishing.

        Args:
            context: Active SaveRoutingContext retained from this Repo.
            value: Object or ConcreteDefinition to match with existing Selector
                semantics.

        Returns:
            Ordered, handle-deduplicated destination Stores.  An unconfigured
            policy and an empty configured policy both select the retained default.

        Raises:
            TypeError: If ``context`` or ``value`` has an unsupported type.
            RepoSaveError: If no matching or default destination exists.

        Side Effects:
            Invokes trusted Selector matching only.  It never opens, creates,
            validates publication capability, or publishes to a Store.
        """

        from .repo_plan import SaveRoutingContext

        if not isinstance(context, SaveRoutingContext):
            raise TypeError("Save destination selection requires a SaveRoutingContext.")
        if isinstance(value, Object):
            target = value.definition
        elif isinstance(value, ConcreteDefinition):
            target = value
        else:
            raise TypeError("Save destination selection requires an Object or ConcreteDefinition.")
        routing = context.routing
        selected = []
        if routing is not None:
            for selector, store in routing.routes:
                if selector.matches(target):
                    if not any(candidate is store for candidate in selected):
                        selected.append(store)
                    if routing.match_mode == "first":
                        break
        if not selected and context.default_store is not None:
            selected.append(context.default_store)
        if not selected:
            raise RepoSaveError("No Store available for save destination selection.")
        return tuple(selected)

    @staticmethod
    def _validate_save_mode(mode, *, name, choices) -> None:
        """Reject an invalid per-save routing override before Store selection.

        Args:
            mode: Optional caller-supplied override.
            name: Public keyword name used in the validation error.
            choices: Closed supported string values for that keyword.

        Raises:
            TypeError: If a non-``None`` override is not a string.
            ValueError: If the string is not one of ``choices``.
        """

        if mode is None:
            return
        if not isinstance(mode, str):
            raise TypeError(f"{name} must be a string or None.")
        if mode not in choices:
            values = " or ".join(repr(choice) for choice in choices)
            raise ValueError(f"{name} must be {values}.")

    def _save_context_with_overrides(self, context, *, match_mode, graph_mode):
        """Return a save-local effective routing context without mutating Repo policy.

        An unconfigured Repo retains its historical default-Store closure by
        materializing an empty, closure-mode policy only in this retained view.

        Args:
            context: Active immutable Repo configuration snapshot.
            match_mode: Optional ``"first"`` or ``"all"`` override.
            graph_mode: Optional ``"per-object"`` or ``"closure"`` override.

        Returns:
            A context whose routing policy contains the effective save-local modes.

        Raises:
            TypeError: If either supplied mode has the wrong type.
            ValueError: If either supplied mode is unsupported.
        """

        from .repo_plan import SaveRouting, SaveRoutingContext

        if not isinstance(context, SaveRoutingContext):
            raise TypeError("Save mode overrides require a SaveRoutingContext.")
        self._validate_save_mode(match_mode, name="match_mode", choices=("first", "all"))
        self._validate_save_mode(
            graph_mode, name="graph_mode", choices=("per-object", "closure"),
        )
        installed = context.routing
        routing = SaveRouting(
            () if installed is None else installed.routes,
            ("first" if installed is None else installed.match_mode)
            if match_mode is None else match_mode,
            ("closure" if installed is None else installed.graph_mode)
            if graph_mode is None else graph_mode,
        )
        return replace(context, routing=routing)

    def _ensure_store(self, store):
        if store is None:
            return None
        return self._register_store(store)

    def cache_strong(self, obj: Object) -> None:
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_cache_strong"):
            if getattr(obj, "_restore_failed", False):
                return
            self.strong_obj_cache.add(obj.__cdef__, obj)
            self._query_catalog.register_cached(obj.__cdef__)

    def cache_weak(self, obj: Object) -> None:
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_cache_weak"):
            if getattr(obj, "_restore_failed", False):
                return
            self.weak_obj_cache.add(obj.__cdef__, obj)
            self._query_catalog.register_cached(obj.__cdef__)

    # --- helpers you already have ---
    def _cached_candidates(self, cdef) -> tuple[Object, ...]:
        """Return distinct live candidates for one exact private CDef node."""

        candidates = self.strong_obj_cache.candidates(cdef) + self.weak_obj_cache.candidates(cdef)
        from .state import is_reserved

        return _unique_objects(
            candidate for candidate in candidates
            if not getattr(candidate, "_restore_failed", False) and not is_reserved(candidate)
        )

    def _all_live_candidates(self) -> tuple[Object, ...]:
        """Return all distinct cached live Objects without structural lookup.

        Exact StateRef reuse filters this complete live set by ObjectId, graph
        topology, bindings, and reservation availability.  It intentionally does
        not use CDef equality as an identity key.
        """
        from .state import is_reserved

        return _unique_objects(
            obj for obj in (
                tuple(obj for _, obj in self.strong_obj_cache.items())
                + tuple(obj for _, obj in self.weak_obj_cache.items())
            ) if not getattr(obj, "_restore_failed", False) and not is_reserved(obj)
        )

    def _evict_live(self, obj: Object) -> None:
        """Remove one externally mutable failed exact-reuse candidate from caches."""
        self.strong_obj_cache.discard(obj.definition, obj)
        self.weak_obj_cache.discard(obj.definition, obj)

    def _state_graph_evidence(self, obj: Object):
        """Return the exact live nodes and ObjectIds covered by state IO.

        The retained save plan is the common authoritative binding check for
        graph reservations, ordinary saves, and targeted restores. A stateless
        root remains a live node while stateful descendants provide ObjectId
        ownership.
        """

        from .repo_plan import build_save_plan

        if not isinstance(obj, Object):
            raise TypeError("State graph operations require a live Object root.")
        plan = build_save_plan(self, obj)
        if any(getattr(node, "_restore_failed", False) for node in plan.nodes):
            raise RepoSaveError("State graph contains an invalidated restore target.")
        return plan, plan.nodes, plan.object_ids

    def reserve_state_graph(self, obj: Object):
        """Reserve one exact live state graph for this process and thread.

        Args:
            obj: Live root with complete retained ObjectRef/runtime bindings.

        Returns:
            An active :class:`StateGraphReservation` context manager covering the
            exact root, stateful ObjectIds, and live state nodes.

        Raises:
            RepoSaveError: If bindings are incomplete, a target was invalidated,
                or any covered ObjectId/live identity is already reserved.

        Side Effects:
            Installs a process-local nonblocking ownership token. Store authority
            is not opened or mutated.
        """

        plan, nodes, object_ids = self._state_graph_evidence(obj)
        return self._reserve_state_graph_evidence(plan, nodes, object_ids)

    def _reserve_state_graph_evidence(self, plan, nodes, object_ids):
        """Reserve one already-captured exact live state graph.

        Managed admission captures graph evidence after retaining its topology and
        must reuse that exact capture for reservation rather than rebuilding a
        potentially changed live graph.
        """

        from .state import reserve

        reservation = reserve(plan.object_ref, nodes, object_ids)
        # The reservation is the exact route-neutral evidence boundary.  A
        # later save using this token must not rebuild bindings against a
        # potentially changed routing configuration.
        reservation._save_plan = plan
        return reservation

    @classmethod
    def _for_state_io(cls, stores):
        """Create a non-owning authority-only Repo view over caller Stores.

        Args:
            stores: Existing Store instances selected by the caller for exact
                state authority and any immutable seed reads.

        Returns:
            A call-owned Repo with independent memory caches and query federation
            disabled from opening or registering persistent indexes.

        Side Effects:
            Does not close, commit, or otherwise take ownership of caller Stores
            or their existing query-index connections.
        """

        return cls(stores, _state_io=True)

    def get_cached(self, cdef):
        """Return a cached Object after live-object admission.

        Args:
            cdef: Exact private-node definition used as the cache key; this is
                not a structural-equality lookup.
        Returns:
            The sole cached Object, or ``None`` when no reusable entry exists
            or candidates are ambiguous across the selected cache tiers.

        Raises:
            RuntimeTransitionError: If strict orchestration prohibits returning
                the live Object.
        """
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_get_cached"):
            candidates = self._cached_candidates(cdef)
            return candidates[0] if len(candidates) == 1 else None

    def has_cached(self, cdef) -> bool:
        """Return cache availability without acquiring a live Object.

        Args:
            cdef: Exact private-node definition used as the cache key; this is
                not a structural-equality lookup.
        Returns:
            Whether exactly one reusable candidate exists across the selected
            strong and weak cache tiers.

        Side Effects:
            None. This metadata-only check is available during strict
            orchestration and does not retain the cached Object.
        """
        return len(self._cached_candidates(cdef)) == 1

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

    def resolve_object_alias(self, alias: str):
        """Resolve a Store object alias to its complete ObjectRef.

        Args:
            alias: Non-empty Store-local object alias.

        Returns:
            The authoritative ObjectRef selected from identical Store replicas.

        Raises:
            KeyError: If no connected Store defines ``alias``.
            RepoLoadError: If connected Stores disagree about its target.

        Side Effects:
            None. This reads mutable alias authority and does not construct an
            Object or select a StateRef.
        """
        return self.get_alias(alias)

    def delete_alias(self, alias: str, *, store=None):
        """Retire unsupported legacy CDef alias deletion.

        Mutable reference deletion is outside the current authority protocol; it
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

    def resolve_state_alias(self, object_ref, alias: str):
        """Resolve one ObjectRef-scoped state alias to an exact StateRef.

        Args:
            object_ref: Complete ObjectRef scope for the state alias.
            alias: Non-empty Store-local state alias.

        Returns:
            The immutable StateRef selected from non-conflicting Stores.

        Raises:
            TypeError: If ``object_ref`` is not an ObjectRef.
            KeyError: If no connected Store defines the scoped alias.
            RepoLoadError: If replica authority conflicts or is incomplete.

        Side Effects:
            None. Alias resolution reads authority only and never restores state.
        """
        from .reference_values import ObjectRef, StateSelectorRef

        if not isinstance(object_ref, ObjectRef):
            raise TypeError("resolve_state_alias requires an ObjectRef scope.")
        return self.resolve_state_selector(StateSelectorRef(object_ref, alias))

    def reference_evidence(self, cdef: ConcreteDefinition) -> ReferenceEvidence:
        """Read matching declaration and StateRef facts under one stable cut.

        Args:
            cdef: Complete CDef topology for which reference authority is sought.

        Returns:
            Deduplicated matching declaration and StateRef evidence. Equal
            replicas remain represented by every contributing Store handle.

        Raises:
            TypeError: If ``cdef`` is not a ConcreteDefinition.
            RepoLoadError: If a required authoritative record is inaccessible,
            corrupt, incompatible, or lacks its associated ClaimRecord.
            StoreCapabilityError: If any connected Store cannot provide a stable
            metadata-read fence.

        Side Effects:
            Acquires all connected Store authority fences in deterministic order.
            It never queries derived indexes, acquires claims, creates identities,
            loads payloads, or writes Store records.

        Concurrency:
            Cooperating publication is excluded over the complete scan. Duplicate
            handles in one Store lock domain acquire one fence while retaining
            their separate read provenance in the returned evidence.
        """
        from .store.records import ClaimRecord

        if not isinstance(cdef, ConcreteDefinition):
            raise TypeError("reference_evidence requires a ConcreteDefinition.")
        groups: dict[str, list[Store]] = {}
        for store in self.stores:
            key = store.authority_fence_key()
            groups.setdefault(key, []).append(store)
        declarations: list[ReferenceDeclarationEvidence] = []
        states: list[ReferenceStateEvidence] = []

        def add_declaration(reference, store, claim):
            for index, current in enumerate(declarations):
                if current.object_ref == reference:
                    declarations[index] = ReferenceDeclarationEvidence(
                        reference, current.stores + (store,),
                        current.claim_statuses + (claim.status,),
                    )
                    return
            declarations.append(
                ReferenceDeclarationEvidence(reference, (store,), (claim.status,))
            )

        def add_state(reference, store):
            for index, current in enumerate(states):
                if current.state_ref == reference:
                    states[index] = ReferenceStateEvidence(
                        reference, current.stores + (store,)
                    )
                    return
            states.append(ReferenceStateEvidence(reference, (store,)))

        try:
            with ExitStack() as fences:
                for key in sorted(groups):
                    fences.enter_context(groups[key][0].authority_read_fence())
                for store in self.stores:
                    for record in store.iter_declaration_records():
                        reference = record.object_ref
                        if not reference.definition.graph_equal(cdef):
                            continue
                        claim = store.read_claim_record(reference.digest())
                        if not isinstance(claim, ClaimRecord):
                            raise RepoLoadError(
                                "Matching declaration lacks authoritative ClaimRecord evidence."
                            )
                        if claim.object_digest != reference.digest():
                            raise RepoLoadError(
                                "Matching declaration has incompatible ClaimRecord evidence."
                            )
                        add_declaration(reference, store, claim)
                    for record in store.iter_state_ref_records():
                        state_ref = record.state_ref
                        if state_ref.definition.graph_equal(cdef):
                            add_state(state_ref, store)
        except (RepoLoadError, StoreCapabilityError):
            raise
        except Exception as error:
            raise RepoLoadError("Authoritative reference evidence could not be read.") from error
        return ReferenceEvidence(tuple(declarations), tuple(states))

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
        expected = set(reference.objects.values())
        existing_by_id = defaultdict(list)
        for store, authoritative, _ in self._references():
            for path, object_id in authoritative.objects.items():
                if object_id in expected:
                    existing_by_id[object_id].append(
                        (store, authoritative.at(path))
                    )
        for path, object_id in reference.objects.items():
            candidates = existing_by_id.get(object_id, ())
            if not candidates:
                continue
            expected_subtree = reference.at(path)
            if any(existing != expected_subtree for _, existing in candidates):
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
            definition_record = store.write_definition_record(
                DefinitionRecord(reference.definition), stored_root=False
            )
            existing = store.read_declaration_record(reference.digest())
            claim = store.read_claim_record(reference.digest())
            if existing is not None:
                if existing.object_ref != reference:
                    raise RepoLoadError("Declaration digest collision has incompatible ObjectRef authority.")
                if claim is None:
                    raise RepoLoadError("Declaration exists without ClaimRecord; Store authority is corrupt.")
                store.write_definition_record(definition_record, stored_root=True)
                return reference
            # An interrupted claim without a declaration is not authority. Its
            # complete record can safely be replaced by registration's fence.
            if claim is None or claim.object_digest != reference.digest():
                store.write_claim_record(ClaimRecord(reference.digest(), 0, "available"))
            elif claim.status != "available":
                raise RepoLoadError("Unregistered ObjectRef claim is not available for recovery.")
            store.write_declaration_record(DeclarationRecord(reference))
            store.write_definition_record(definition_record, stored_root=True)
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
                raise RepoLoadError("Declared ObjectRef is complete; load its exact StateRef.")
            acquired.append(lease)
            active = _active_object_ref_builds.get()
            token = _active_object_ref_builds.set(
                active | {
                    (id(self), item.object_ref.digest())
                    for item in acquired
                }
            )
            try:
                obj = self._load_structural(reference.definition)
            finally:
                _active_object_ref_builds.reset(token)
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

    def _complete_initial_state_ref(self, state_ref, store, lease) -> None:
        """Fence initial StateRef publication against the live claim generation."""
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

    def _mark_initial_state_ref_complete(self, state_ref, store, lease) -> None:
        """Replace a verified live claim with completed StateRef authority."""
        from .store.records import ClaimRecord

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

    def _preflight_routed_claims(self, plan, routed):
        """Validate and renew every pending claim before routed save hooks run.

        Claims retain declaration-Store authority.  A route that excludes that
        Store is rejected rather than silently adding a destination, and nested
        claims retain their construction dependency order.
        """
        root = plan.binding.roots[0].obj
        candidates = [root]
        candidates.extend(action.obj for action in plan.actions if isinstance(action.obj, Object))
        action_for_object = {id(action.obj): action for action in plan.actions}
        leases = []
        seen = set()

        def add(lease, obj=None):
            if not isinstance(lease, _ClaimLease) or id(lease) in seen:
                return
            action = action_for_object.get(id(obj)) if obj is not None else None
            if action is None:
                action = next(
                    (
                        item for item in plan.actions
                        if getattr(item.obj, "object_ref", None) == lease.object_ref
                    ),
                    None,
                )
            if action is None:
                raise RepoSaveError("Pending declaration has no retained routed save action.")
            if not any(destination is lease.store for destination in routed.destinations[action.path]):
                raise RepoSaveError("Pending declaration Store is excluded by the effective save destinations.")
            seen.add(id(lease))
            leases.append((lease, action))

        # build_object_ref records nested leases first, then the enclosing lease.
        for lease in getattr(root, "_claim_leases", ()):
            add(lease)
        for candidate in candidates:
            add(getattr(candidate, "_claim_lease", None), candidate)
            for lease, dependency in getattr(candidate, "_pending_claim_dependencies", ()):
                add(lease, dependency)
        for lease, _ in leases:
            self._renew_claim(lease)
        return tuple(leases)

    def _clear_completed_routed_claim(self, plan, lease) -> None:
        """Drop only one confirmed completed routed claim from live metadata."""
        candidates = _unique_objects(
            (plan.binding.roots[0].obj, *(
                action.obj for action in plan.actions if isinstance(action.obj, Object)
            ))
        )
        for candidate in candidates:
            if getattr(candidate, "_claim_lease", None) is lease:
                candidate._claim_lease = None
            candidate._claim_leases = tuple(
                item for item in getattr(candidate, "_claim_leases", ()) if item is not lease
            )
            candidate._pending_claim_dependencies = tuple(
                (item, dependency)
                for item, dependency in getattr(candidate, "_pending_claim_dependencies", ())
                if item is not lease
            )

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
                sources.append((reference, path, definition, state_hash, source))
        fork, forked_references = _fork_rekey_reference(state_ref, namespace)
        with selected.writer_lock():
            # DefinitionRecords are graph-aware, but every local-state entry is
            # independently verified against its node definition before copying.
            for reference, path, definition, state_hash, source in sources:
                target_definition = forked_references[
                    reference.digest()
                ].object.at(path).definition
                selected.write_definition_record(
                    DefinitionRecord(target_definition), stored_root=False
                )
                if not target_definition.graph_equal(definition):
                    selected.rebind_local_state_from(
                        source, definition, target_definition, state_hash
                    )
                elif not federated and source is not selected:
                    selected.copy_local_state_from(source, definition, state_hash)
            for seed in forked_references.values():
                if seed != fork:
                    selected.write_definition_record(
                        DefinitionRecord(seed.definition), stored_root=False
                    )
                    selected.write_state_ref_record(StateRefRecord(seed))
            selected.write_definition_record(
                DefinitionRecord(fork.definition), stored_root=False
            )
            selected.write_state_ref_record(StateRefRecord(fork))
            selected.write_definition_record(
                DefinitionRecord(fork.definition), stored_root=True
            )
        return fork

    def set_config(self, key: str, value: Any) -> None:
        """Set one runtime configuration value in the next export/save snapshot.

        Args:
            key: Non-empty string configuration name.
            value: Caller-owned runtime value.  Portable definition export later
                rejects values outside its bounded JSON configuration codec.

        Raises:
            TypeError: If ``key`` is not a string.
            ValueError: If ``key`` is empty.

        Side Effects:
            Atomically replaces this key and publishes a new configuration
            version.  Existing retained save/export snapshots are unchanged.
        """

        if not isinstance(key, str):
            raise TypeError("Config keys must be strings.")
        if key == "":
            raise ValueError("Config keys cannot be empty.")
        with self._configuration_lock:
            if self._closing or self._closed:
                raise RuntimeError("Cannot change configuration after Repo close begins.")
            self.config[key] = value
            self._configuration_version += 1

    def update_config(self, values: Mapping[str, Any]) -> None:
        """Atomically apply a complete mapping of runtime configuration updates.

        Args:
            values: Mapping with non-empty string keys and caller-owned values.

        Raises:
            TypeError: If ``values`` is not a mapping or any key is not a string.
            ValueError: If any key is empty.

        Side Effects:
            Validates all keys before changing configuration, then publishes one
            new version for later retained save/export snapshots.  It does not
            validate portability or alter an active snapshot.
        """

        if not isinstance(values, Mapping):
            raise TypeError("Config updates must be a mapping.")
        with self._configuration_lock:
            if self._closing or self._closed:
                raise RuntimeError("Cannot change configuration after Repo close begins.")
            for key, value in values.items():
                if not isinstance(key, str):
                    raise TypeError("Config keys must be strings.")
                if key == "":
                    raise ValueError("Config keys cannot be empty.")
            self.config.update(values)
            self._configuration_version += 1

    def to_definition(self) -> "RepoDefinition":
        """Return a detached, inert portable configuration snapshot.

        The snapshot includes only supported Store descriptors, routing, and
        declarative settings. It never traverses Store/cache data, commits an
        archive, resolves symbols, activates a session, or reconstructs live
        resources. Exported ``config`` values are caller-owned and may be
        sensitive; errors avoid rendering arbitrary supplied values.

        Returns:
            A detached :class:`RepoDefinition` containing only portable
            configuration descriptors.

        Raises:
            RepoDefinitionError: If configuration contains a nonportable Store,
                selector value, explicit runtime factory, dirty archive, or the
                Repo is closing/closed.
        """

        from .repo_definition import definition_from_repo

        return definition_from_repo(self)

    @classmethod
    def from_definition(cls, definition: "RepoDefinition") -> "Repo":
        """Reconnect a detached configuration using fresh existing Store handles.

        Args:
            definition: Fully validated inert portable Repo configuration.

        Returns:
            A new Repo that owns the Store handles opened for reconstruction.

        Raises:
            TypeError: If ``definition`` is not a RepoDefinition.
            RepoDefinitionError: If descriptors, selectors, or existing Store
                authority cannot be reconstructed.  Storage failures are chained.

        Side Effects:
            Opens only existing supported Store authority. It neither installs a
            session Repo nor creates missing storage. The returned Repo owns its
            newly opened handles; ``close(flush=False)`` releases them without a
            commit, including after caller-managed failed work.
        """

        from .repo_definition import repo_from_definition

        return repo_from_definition(definition)

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
            store = self._ensure_store(store)
        return self._query_index.index_status(store=store)

    def rebuild_index(self, store=None):
        if store is not None:
            store = self._ensure_store(store)
        self._query_index.rebuild(store=store)
        return self

    def validate_index(self, store=None, *, thorough: bool = False):
        if store is not None:
            store = self._ensure_store(store)
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
            match_mode: str | None = None,
            graph_mode: str | None = None,
            report_stores: bool = False,
            _capture_memo: set[object] | None = None,
            reservation=None,
            _save_context=None,
            _commit_stores=()):
        """Publish one live graph as immutable local states and a StateRef.

        Args:
            obj: Root object whose graph is saved.
            main: Whether its concrete definition becomes the main reference.
            store: Optional whole-graph closure target Store. It bypasses routing
                and replication for this save.
            alias: Optional object alias written after StateRef publication.
            deep_capture: Whether every owned Serializable node is serialized.
            match_mode: Optional ``"first"`` or ``"all"`` routing override.
            graph_mode: Optional ``"per-object"`` or ``"closure"`` placement
                override. Both overrides are local to this save.
            report_stores: Whether to return an ephemeral StoreReport.
            _capture_memo: Private retained capture memo. It is accepted for
                internal state-operation compatibility and does not alter routing.
            reservation: Optional active exact graph reservation reused by an
                enclosing state operation; callers normally omit this.

        Returns:
            The complete StateRef, or ``(StateRef, StoreReport)`` when requested.

        Raises:
            TypeError: If ``alias`` or a supplied mode has the wrong type.
            ValueError: If ``alias`` is empty or a supplied mode is unsupported.
            RepoSaveError: If routing, claims, capture, or publication fails.
            StoreAuthorityError: If Store publication rejects authoritative data.

        Side Effects:
            Publishes every local state and the enclosing StateRef before main or
            object-alias references can change. Configured per-object saves also
            publish exact child StateRef projections at their selected Stores.
            A completed StateRef and claim
            remain authoritative if later derived-index or mutable-reference
            registration fails; completed live claim metadata is still cleared.
            Once that authority is complete, installs the StateRef as ``obj``'s
            read-only last-state receipt before any derived index, main, or alias
            update that may later raise.

        Concurrency:
            Retains one routing/default/Store-order view, preserves declaration
            claim fencing, and reports partial cross-Store work rather than
            claiming transactional rollback. ``store=`` selects one complete
            closure even when routed modes were supplied.
        """
        from dryml.runtime import materialization_admission
        from .store.records import DefinitionRecord, MainRefRecord

        with materialization_admission(operation="repo_save_object"):
            if alias is not None:
                self._validate_alias(alias)
            self._validate_save_mode(match_mode, name="match_mode", choices=("first", "all"))
            self._validate_save_mode(
                graph_mode, name="graph_mode", choices=("per-object", "closure"),
            )
            selected_store = self._ensure_store(store)
            owns_reservation = reservation is None
            if owns_reservation:
                plan, nodes, object_ids = self._state_graph_evidence(obj)
                from .state import reserve

                reservation = reserve(obj.object_ref, nodes, object_ids)
                reservation._save_plan = plan
            else:
                plan = getattr(reservation, "_save_plan", None)
                if plan is None or plan.binding.roots[0].obj is not obj:
                    plan, nodes, object_ids = self._state_graph_evidence(obj)
                else:
                    from .repo_plan import validate_retained_save_plan

                    validate_retained_save_plan(plan, obj)
                    if reservation.object_ref != plan.object_ref:
                        raise RepoSaveError("State graph reservation does not cover this exact ObjectRef.")
                    nodes, object_ids = plan.nodes, plan.object_ids
                reservation._covers(nodes, object_ids)
            context_scope = (
                nullcontext(_save_context)
                if _save_context is not None else self._retain_save_context()
            )
            with (reservation if owns_reservation else nullcontext()), context_scope as context:
                context = self._save_context_with_overrides(
                    context, match_mode=match_mode, graph_mode=graph_mode,
                )
                lease = None
                try:
                    lease = getattr(obj, "_claim_lease", None)
                    from .repo_plan import (
                        _unique_stores,
                        build_routed_save_plan,
                        execute_routed_save_plan,
                        _register_retained_save_plan,
                    )

                    _register_retained_save_plan(self, plan)
                    routed = build_routed_save_plan(
                        self, plan, context, store=selected_store,
                    )
                    index_destinations = _unique_stores(
                        destination
                        for destinations in routed.destinations.values()
                        for destination in destinations
                    )
                    late_publications = []
                    if not self._state_io:
                        late_publications.append(("index", index_destinations))
                    if main:
                        late_publications.append(("main", routed.root_destinations))
                    if alias is not None:
                        late_publications.append(("alias", routed.root_destinations))
                    if _commit_stores:
                        late_publications.append(("commit", tuple(_commit_stores)))
                    result = execute_routed_save_plan(
                        self,
                        routed,
                        deep_capture=deep_capture,
                        report_stores=True,
                        late_publications=tuple(late_publications),
                    )
                except BaseException:
                    for pending_lease in reversed(
                            getattr(obj, "_claim_leases", (lease,) if lease else ())):
                        self._abandon_claim(pending_lease)
                    raise
                state_ref, report = result
                if isinstance(lease, _ClaimLease):
                    obj._claim_lease = None
                obj._pending_claim_dependencies = ()
                obj._claim_leases = ()
                # StateRef publication is authoritative; only then may the derived
                # query index expose this root. A registration failure leaves the
                # Store authority intact and the sidecar explicitly dirty.
                if not self._state_io:
                    roots_by_store = defaultdict(list)
                    state_refs_by_store = defaultdict(list)
                    for snapshot in report.snapshots:
                        for destination in snapshot.stores:
                            roots_by_store[destination].append(snapshot.state_ref.definition)
                            state_refs_by_store[destination].append(snapshot.state_ref)
                    for index_store in roots_by_store:
                        report = _record_late_publication(
                            report, store=index_store, phase="index", state_ref=state_ref,
                            operation=lambda index_store=index_store: self._query_index.register_saved_graph(
                                plan.graph,
                                {index_store: tuple(roots_by_store[index_store])},
                                {index_store: tuple(state_refs_by_store[index_store])},
                            ),
                            checker=lambda index_store=index_store: not getattr(
                                index_store, "query_index_is_dirty", lambda: False
                            )(),
                        )
                if main:
                    main_record = MainRefRecord(DefinitionRecord(obj.definition).digest)
                    for root_store in report.target_stores:
                        report = _record_late_publication(
                            report, store=root_store, phase="main", state_ref=state_ref,
                            operation=lambda root_store=root_store, main_record=main_record: root_store.write_main_ref(main_record),
                            checker=lambda root_store=root_store, main_record=main_record: root_store.read_main_ref() == main_record,
                        )
                    self.main_def = obj.definition
                if alias is not None:
                    from .store.records import ObjectAliasRecord

                    alias_record = ObjectAliasRecord(alias, state_ref.object)
                    for root_store in report.target_stores:
                        report = _record_late_publication(
                            report, store=root_store, phase="alias", state_ref=state_ref,
                            operation=lambda root_store=root_store, alias_record=alias_record: root_store.write_object_alias(alias_record),
                            checker=lambda root_store=root_store, alias_record=alias_record: root_store.read_object_alias(alias) == alias_record,
                        )
                return (state_ref, report) if report_stores else state_ref

    def save(
            self,
            obj: Object,
            *,
            main: bool = False,
            store=None,
            alias: str | None = None,
            deep_capture: bool = False,
            match_mode: str | None = None,
            graph_mode: str | None = None,
            report_stores: bool = False):
        """Publish one object graph and flush its Repo.

        Args:
            obj: Live Object root whose retained runtime graph will be saved.
            main: Whether to update the target Store's main-definition reference.
            store: Explicit whole-graph closure Store or Store specification.
            alias: Optional Store-local ObjectRef alias to update after publication.
            deep_capture: Whether to serialize every owned live Serializable node.
            match_mode: Optional ``"first"`` or ``"all"`` routing override.
            graph_mode: Optional ``"per-object"`` or ``"closure"`` placement
                override local to this save.
            report_stores: Whether to pair the StateRef with a StoreReport.

        Returns:
            The immutable StateRef, or ``(StateRef, StoreReport)`` when
            ``report_stores`` is true.

        Raises:
            RepoSaveError: If graph bindings, claims, codecs, or hooks fail.
            StoreAuthorityError: If Store preflight or publication fails.
            TypeError: If a supplied mode has the wrong type.
            ValueError: If no writable target Store can be selected or a mode is unsupported.

        Side Effects:
            Publishes immutable definition, local-state, root-membership, and
            StateRef authority, optionally updates main/alias refs, then flushes
            every configured Store after successful publication. The completed
            top-level StateRef becomes the live root's last-state receipt before
            any derived index, main-reference, or alias update can fail.

        Concurrency:
            Store publication and initial-claim completion use writer locks and
            the exact claim lease carried by this save.
        """
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_save"):
            with self._retain_save_context() as context:
                result = self.save_object(
                    obj, main=main, store=store, alias=alias,
                    deep_capture=deep_capture, match_mode=match_mode,
                    graph_mode=graph_mode,
                    report_stores=True, _save_context=context,
                    _commit_stores=context.stores,
                )
                state_ref, report = result
                report = _commit_save_report(self, state_ref, report, stores=context.stores)
            return (state_ref, report) if report_stores else state_ref

    def _first_store_with(self, cdef):
        from .store.records import DefinitionRecord

        digest = DefinitionRecord(cdef).digest
        for st in self.stores:
            if st.read_definition_record(digest) is not None:
                return st
        return None

    def _candidate_cdefs(self) -> set[ConcreteDefinition]:
        cdefs = {obj.definition for _, obj in self.strong_obj_cache.items()}
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
        cache: CachePolicy = "weak",
        memo: dict | None = None,
        path: list[str | int] | None = None,
    ):
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_realize"):
            return from_canonical(
                x,
                repo=self,
                cache=cache,
                memo=memo,
                path=path,
            )

    def materialize_boundary(self, roots, *, cache=None, reuse_live=None,
                             reservation=None, reserved_live=None):
        """Realize selected materializing boundary roots under one admission cut.

        Args:
            roots: Ordered selected values from one signature argument or return
                boundary.  Ref-valued delivery must not call this API.
            cache: ``None`` preserves Repo's ``"weak"`` default; supplied values
                are forwarded unchanged to structural realization.
            reuse_live: ``None`` preserves exact-load ``"matching"``; supplied
                values are forwarded unchanged to exact realization.
            reservation: Optional active state-graph reservation retained by an
                outer lifecycle that owns overlapping live state.
            reserved_live: Optional mapping from exact StateRef or ObjectRef
                digests to live Objects covered by ``reservation``. These entries
                are admitted only after complete aggregate authority preflight.

        Returns:
            A tuple of runtime values in ``roots`` order, preserving aliases across
            all roots and ordinary supported containers.

        Raises:
            RepoLoadError: If exact authority is incomplete or incompatible,
                claim admission fails, or a constructor/restore fails.
            TypeError: If ``roots`` is not an ordered materializing boundary.

        Side Effects:
            May acquire first-construction claims, reserve/reuse live objects,
            construct, restore, and populate caches.  It never saves or creates
            declaration authority.  A failed call abandons only its own acquired
            claim generations in reverse order.
        """

        from .cdef_graph import EdgeKind
        from .canonical import to_canonical
        from .cdef_identity import cdef_node_key
        from .definition import Definition
        from .links import DefLink
        from .materialization import (
            build_exact_state_load_plan,
            execute_exact_state_load_plan,
        )
        from .reference_values import ObjectRef, StateRef
        from .repo_plan import AggregateMaterializationPlan, _NodeBindings, realization_scope
        from .utils.graph.value import iter_value_edges

        if not isinstance(roots, (tuple, list)):
            raise TypeError("materialize_boundary roots must be an ordered sequence.")
        if reserved_live is not None and (reservation is None or not isinstance(reserved_live, Mapping)):
            raise TypeError("reserved live materialization requires a reservation and mapping.")
        reserved_live = {} if reserved_live is None else dict(reserved_live)

        def concretize_definitions(value, memo):
            """Lower materializing Definition values before authority admission."""

            key = id(value)
            if key in memo:
                return memo[key]
            if isinstance(value, Definition):
                result = to_canonical(value, repo=self)
            elif isinstance(value, list):
                result = []
                memo[key] = result
                result.extend(concretize_definitions(item, memo) for item in value)
            elif isinstance(value, tuple):
                result = tuple(concretize_definitions(item, memo) for item in value)
            elif isinstance(value, dict):
                result = {
                    item: concretize_definitions(child, memo)
                    for item, child in value.items()
                }
            else:
                result = value
            memo[key] = result
            return result

        roots = tuple(concretize_definitions(root, {}) for root in roots)
        plan = AggregateMaterializationPlan(
            tuple(roots), "weak" if cache is None else cache,
            "matching" if reuse_live is None else reuse_live,
        )
        if plan.reuse_live not in {"matching", "greedy", "never"}:
            raise ValueError("reuse_live must be 'matching', 'greedy', or 'never'.")

        # The visitor uses object identity for private CDef nodes, while exact
        # reference identities deliberately deduplicate by digest.
        object_refs, state_refs = {}, {}
        visited = set()

        def visit(value):
            if isinstance(value, StateRef):
                state_refs.setdefault(value.digest(), value)
                visit(value.object)
                return
            if isinstance(value, ObjectRef):
                object_refs.setdefault(value.digest(), value)
                visit(value.definition)
                return
            if isinstance(value, ConcreteDefinition):
                key = cdef_node_key(value)
                if key in visited:
                    return
                visited.add(key)
                for edge in iter_value_edges(value):
                    visit(edge.value)
                return
            if isinstance(value, DefLink):
                if value.kind is EdgeKind.MATERIALIZE:
                    visit(value.target)
                return
            for edge in iter_value_edges(value):
                visit(edge.value)

        for root in plan.roots:
            visit(root)

        # Every exact state closure is validated before any constructor, restore,
        # claim, or live reservation can occur.  Compare effective local actions,
        # not embedded seed StateRefs, so an enclosing StateRef remains decisive.
        exact_plans = {
            digest: build_exact_state_load_plan(self, reference)
            for digest, reference in state_refs.items()
        }
        state_actions = {}
        for exact in exact_plans.values():
            effective_actions = {}
            for action in exact.actions:
                # An enclosing exact StateRef is authoritative over an embedded
                # seed for the same ObjectId. Compare only the effective action
                # from each root before combining roots in this admission.
                if action.reference == exact.state_ref or action.object_id not in effective_actions:
                    effective_actions[action.object_id] = action
            for action in effective_actions.values():
                previous = state_actions.setdefault(action.object_id, action)
                if previous.state_hash != action.state_hash:
                    raise RepoLoadError(
                        "Materializing boundary has incompatible effective StateRef demands."
                    )

        # Managed lifecycle ownership may retain the receiver while this generic
        # boundary preflights its checkpoint beside ordinary Mat roots. Validate
        # that every admitted overlap is genuinely covered before bypassing the
        # generic candidate search, which correctly excludes reserved live state.
        for exact in exact_plans.values():
            reserved = reserved_live.get(exact.state_ref.digest())
            if reserved is None:
                continue
            if not isinstance(reserved, Object) or reserved.object_ref != exact.state_ref.object:
                raise RepoLoadError("Reserved live materialization does not match exact StateRef authority.")
            _, nodes, object_ids = self._state_graph_evidence(reserved)
            reservation._covers(nodes, object_ids)

        # An ObjectRef shares any exact root selected for the same identity.  For
        # the remaining identities, choose live first, then one saved authority,
        # and only then the registered construction claim.
        state_for_object = {
            exact.state_ref.object.digest(): exact for exact in exact_plans.values()
        }
        results, claim_roots = {}, []
        for digest, reference in object_refs.items():
            if digest in state_for_object:
                continue
            reserved = reserved_live.get(digest)
            if reserved is not None:
                if not isinstance(reserved, Object) or reserved.object_ref != reference:
                    raise RepoLoadError("Reserved live materialization does not match ObjectRef authority.")
                _, nodes, object_ids = self._state_graph_evidence(reserved)
                reservation._covers(nodes, object_ids)
                results[digest] = reserved
                continue
            live = tuple(
                candidate for candidate in self._all_live_candidates()
                if getattr(candidate, "object_ref", None) == reference
            )
            if len(live) == 1:
                results[digest] = live[0]
                continue
            if len(live) > 1:
                raise RepoLoadError("Materializing ObjectRef has ambiguous live authority.")
            evidence = self.reference_evidence(reference.definition)
            matches = [item.state_ref for item in evidence.states if item.state_ref.object == reference]
            if len(matches) > 1:
                raise RepoLoadError("Materializing ObjectRef has ambiguous saved StateRef authority.")
            if matches:
                exact = build_exact_state_load_plan(self, matches[0])
                exact_plans.setdefault(matches[0].digest(), exact)
                state_for_object[digest] = exact
            else:
                claim_roots.append(reference)

        # Establish the complete dependency closure and acquire each live claim
        # before the first root executes.  The DFS is deterministic and retains
        # child-before-parent ordering required by current completion semantics.
        claim_order, claim_stores, seen_claims = [], {}, set()

        def add_claim(reference):
            if reference.digest() in seen_claims:
                return
            seen_claims.add(reference.digest())
            for _, dependency in self._pending_declaration_references(reference):
                add_claim(dependency)
            matches = [
                store for store in self.stores
                if (record := store.read_declaration_record(reference.digest())) is not None
                and record.object_ref == reference
            ]
            if not matches:
                raise RepoLoadError("Materializing ObjectRef lacks registered declaration authority.")
            claim_stores[reference.digest()] = matches[0]
            claim_order.append(reference)

        for reference in sorted(claim_roots, key=lambda item: item.digest()):
            add_claim(reference)

        with realization_scope() as scope:
            leases = {}
            try:
                for reference in claim_order:
                    lease = self._acquire_claim(reference, claim_stores[reference.digest()])
                    if lease is None:
                        raise RepoLoadError("Declared ObjectRef completed during aggregate admission.")
                    leases[reference.digest()] = lease
                for lease in leases.values():
                    self._renew_claim(lease)

                def abandon_claims():
                    for lease in reversed(tuple(leases.values())):
                        self._abandon_claim(lease)

                if leases:
                    scope.add_claim_cleanup(abandon_claims)

                reference_memo = {}
                for exact in exact_plans.values():
                    reserved = reserved_live.get(exact.state_ref.digest())
                    results[exact.state_ref.object.digest()] = (
                        reserved if reserved is not None else execute_exact_state_load_plan(
                            self, exact, reuse_live=plan.reuse_live, cache=plan.cache,
                            _reference_memo=reference_memo,
                        )
                    )

                cdef_memo = _NodeBindings()

                def build_claim(reference):
                    known = results.get(reference.digest())
                    if known is not None:
                        return known
                    lease = leases[reference.digest()]
                    active = _active_object_ref_builds.get()
                    token = _active_object_ref_builds.set(
                        active | {(id(self), item.object_ref.digest()) for item in leases.values()}
                    )
                    try:
                        obj = self._materialize_cdef(
                            reference.definition, cache=plan.cache, memo=cdef_memo,
                        )
                    finally:
                        _active_object_ref_builds.reset(token)
                    from .repo_plan import apply_exact_reference_identity

                    apply_exact_reference_identity(obj, reference)
                    obj._store_affinity = lease.store
                    obj._claim_lease = lease
                    obj._claim_leases = tuple(leases[item.digest()] for item in claim_order)
                    results[reference.digest()] = obj
                    return obj

                for reference in claim_roots:
                    build_claim(reference)

                def realize(value, memo):
                    key = id(value)
                    if key in memo:
                        return memo[key]
                    if isinstance(value, Object):
                        return value
                    if isinstance(value, ConcreteDefinition):
                        result = self._materialize_cdef(value, cache=plan.cache, memo=cdef_memo)
                    elif isinstance(value, StateRef):
                        result = results[value.object.digest()]
                    elif isinstance(value, ObjectRef):
                        result = results.get(value.digest()) or reserved_live.get(value.digest()) or build_claim(value)
                    elif isinstance(value, DefLink):
                        result = value.target if value.kind is EdgeKind.REF else realize(value.target, memo)
                    elif isinstance(value, list):
                        result = []
                        memo[key] = result
                        result.extend(realize(item, memo) for item in value)
                    elif isinstance(value, tuple):
                        result = tuple(realize(item, memo) for item in value)
                    elif isinstance(value, dict):
                        result = {item: realize(child, memo) for item, child in value.items()}
                    else:
                        result = value
                    memo[key] = result
                    return result

                memo = {}
                return tuple(realize(root, memo) for root in plan.roots)
            except BaseException:
                # ``realization_scope`` runs this same cleanup after any nested
                # construction failure; retain it here for failures before scope
                # exit and keep reverse generation-safe abandonment authoritative.
                raise

    # -------------------------------------------------------------------------
    # Core: turn a ConcreteDefinition into a live Object under load knobs
    # -------------------------------------------------------------------------
    def _materialize_cdef(
        self,
        cdef,
        *,
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

            from .materialization import build_materialization_plan, execute_materialization_plan

            plan = build_materialization_plan(
                self,
                cdef,
                cache=cache,
                memo=memo,
                path=path,
            )
            return execute_materialization_plan(
                self,
                plan,
                memo=memo,
                root=cdef,
            )

    def _materialize_object_ref(
            self, reference, *, cache: CachePolicy = "weak", memo=None,
            path=None):
        """Materialize an ObjectRef only through its declaration claim.

        Nested ObjectRefs whose claims were acquired by ``build_object_ref`` use
        that enclosing operation's context-local authorization. Every other call
        enters the public declaration-and-claim path.

        Args:
            reference: Exact ObjectRef requested by a materializing CDef edge.
            cache: Cache policy for an already-authorized nested realization.
            memo: Current private-node realization memo.
            path: Current constructor path used for diagnostics.

        Returns:
            A live graph carrying the supplied exact ObjectIds.

        Raises:
            TypeError: If ``reference`` is not an ObjectRef.
            RepoLoadError: If declaration or claim authority is unavailable.

        Side Effects:
            May acquire declaration claims, construct Objects, and update caches.
        """
        from .reference_values import ObjectRef
        from .repo_plan import apply_exact_reference_identity

        if not isinstance(reference, ObjectRef):
            raise TypeError("ObjectRef materialization requires an ObjectRef.")
        if (id(self), reference.digest()) not in _active_object_ref_builds.get():
            realized = self.build_object_ref(reference)
            from .repo_plan import current_realization_scope

            scope = current_realization_scope()
            if scope is not None:
                def abandon_realized_claims():
                    first_error = None
                    leases = getattr(realized, "_claim_leases", ())
                    for lease in reversed(leases):
                        try:
                            self._abandon_claim(lease)
                        except BaseException as error:
                            if first_error is None:
                                first_error = error
                    if first_error is not None:
                        raise first_error

                scope.add_claim_cleanup(abandon_realized_claims)
            return realized
        realized = self._materialize_cdef(
            reference.definition, cache=cache, memo=memo, path=path
        )
        apply_exact_reference_identity(realized, reference)
        return realized


    def _load_structural(
        self,
        x: object,
        *,
        cache: CachePolicy = "weak",
        require_store: bool = False,
    ):
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_load_object"):
            from .repo_plan import _NodeBindings, realization_scope

            with realization_scope():
                memo = _NodeBindings()
                if isinstance(x, ConcreteDefinition) and require_store and self._first_store_with(x) is None:
                    raise RepoLoadError("No connected Store contains this structural CDef; use load_or_build to construct it.")
                return self._realize(
                    x,
                    cache=cache,
                    path=[""],
                    memo=memo,
                )

    def load_object(self, x: Definition | ConcreteDefinition | Object, *, cache: CachePolicy = "weak") -> Object:
        """Load existing structural authority without restoring mutable state.

        Args:
            x: Definition, CDef, or Object projected to its structural CDef.
            cache: Cache tier for the resulting realization.

        Returns:
            A structural live Object.

        Raises:
            TypeError: If ``x`` is an exact reference or unsupported root type.
            RepoLoadError: If no connected Store has the structural authority.

        Side Effects:
            May construct and cache live objects. Exact snapshots are loaded only
            by :meth:`load_state_ref`.
        """
        from .reference_values import ObjectRef, StateRef, StateSelectorRef

        if isinstance(x, (ObjectRef, StateRef, StateSelectorRef)):
            raise TypeError("Repo.load_object is structural; use build_object_ref or load_state_ref for exact references.")
        if isinstance(x, Object):
            x = x.definition
        if isinstance(x, Definition):
            x = x.concretize(repo=self)
        if not isinstance(x, ConcreteDefinition):
            raise TypeError("Repo.load_object requires a Definition, ConcreteDefinition, or Object.")
        if self._first_store_with(x) is None:
            raise RepoLoadError("No connected Store contains this structural CDef; use load_or_build to construct it.")
        return self._load_structural(x, require_store=True, cache=cache)

    def load_state_ref(
            self,
            state_ref,
            *,
            reuse_live: LiveReusePolicy = "matching",
            cache: CachePolicy = "weak") -> Object:
        """Exactly restore one authoritative StateRef after complete preflight.

        Args:
            state_ref: Immutable StateRef record that must be byte-equivalent to
                current authority in a connected Store.
            reuse_live: ``"matching"`` reuses one matching live checkpoint,
                ``"greedy"`` restores one unique candidate in place, and
                ``"never"`` always builds fresh.
            cache: Cache tier populated only after the complete realization has
                succeeded.

        Returns:
            A fully restored root preserving the StateRef's ObjectIds and graph
            topology.

        Raises:
            RepoLoadError: If exact authority preflight, construction, or restore
                fails. No fresh partial cache entry is published.

        Side Effects:
            May restore a uniquely eligible greedy live Object. A failed greedy
            restore clears its state hash and evicts it from Repo caches.
            A successful top-level exact load installs ``state_ref`` as the
            returned root's last-state receipt; descendants receive no synthetic
            receipt.
        """
        from dryml.runtime import materialization_admission
        from .materialization import build_exact_state_load_plan, execute_exact_state_load_plan
        from .reference_values import StateRef

        if not isinstance(state_ref, StateRef):
            raise TypeError("load_state_ref requires a StateRef.")
        with materialization_admission(operation="repo_load_state_ref"):
            plan = build_exact_state_load_plan(self, state_ref)
            return execute_exact_state_load_plan(
                self, plan, reuse_live=reuse_live, cache=cache,
            )

    def restore_state_ref_into(self, obj: Object, state_ref, *, reservation=None):
        """Restore an authoritative snapshot into the exact supplied live graph.

        Args:
            obj: Existing live root whose ObjectRef and retained runtime bindings
                must exactly match ``state_ref``.
            state_ref: Complete authoritative StateRef to restore.
            reservation: Optional active reservation returned for this exact graph.

        Returns:
            The requested ``state_ref`` after every local restore hook succeeds.

        Raises:
            RepoLoadError: If authority, topology, paths, IDs, or payloads fail
                preflight, or if a hook fails. A post-hook failure invalidates all
                covered live nodes and requires a fresh exact load.
            RepoSaveError: If graph ownership is unavailable or ``reservation``
                is inactive, foreign, or does not cover the supplied graph.

        Side Effects:
            Runs local hooks dependency-first without candidate search or object
            replacement. Successful completion updates only ``obj.last_state_ref``.
        """

        from .materialization import build_exact_state_load_plan
        from .reference_values import StateRef

        if not isinstance(obj, Object):
            raise TypeError("restore_state_ref_into requires a live Object target.")
        if not isinstance(state_ref, StateRef):
            raise TypeError("restore_state_ref_into requires a StateRef.")
        if getattr(obj, "_restore_failed", False):
            raise RepoLoadError("Cannot restore an invalidated live target; load a fresh exact graph.")

        # Authority validation, retained binding validation, and reservation all
        # complete before the first user restore hook can run.
        plan = build_exact_state_load_plan(self, state_ref)
        _, nodes, object_ids = self._state_graph_evidence(obj)
        if obj.object_ref != state_ref.object:
            raise RepoLoadError("Target object does not carry the requested exact ObjectRef.")
        owns_reservation = reservation is None
        if owns_reservation:
            reservation = self.reserve_state_graph(obj)
        else:
            reservation._covers(nodes, object_ids)

        locks = []
        hooks_started = False
        try:
            targets = {}
            for path, object_id in state_ref.object.objects.items():
                try:
                    target = obj.graph_at(path)
                except Exception as error:
                    raise RepoLoadError(
                        f"Target graph lacks retained binding at {path!s}."
                    ) from error
                if not isinstance(target, Object) or target.object_id != object_id:
                    raise RepoLoadError(
                        f"Target graph identity does not match ObjectRef at {path!s}."
                    )
                targets[object_id] = target
            if set(targets) != set(object_ids):
                raise RepoLoadError("Target graph has incomplete stateful ObjectId bindings.")

            # Existing per-instance exclusion remains meaningful for code outside
            # the graph-token protocol. Acquire every lock before invalidation or
            # hook entry so a failure here leaves the target usable.
            for object_id in sorted(targets, key=str):
                lock = getattr(targets[object_id], "_save_load_reservation", None)
                if lock is None or not lock.acquire(blocking=False):
                    raise RepoLoadError("Target local state is already reserved by save or restore.")
                locks.append(lock)

            # A StateRef embeds materializing StateRefs below ordinary CDef graph
            # edges. Keep the full preflight closure, selecting the outer action
            # where it explicitly supersedes a seed state for the same ObjectId.
            actions = {}
            for action in plan.actions:
                if action.object_id not in targets:
                    raise RepoLoadError("Exact StateRef closure has an unbound live ObjectId.")
                if action.object_id not in actions or action.reference == state_ref:
                    actions[action.object_id] = action
            if set(actions) != set(object_ids):
                raise RepoLoadError("Exact StateRef preflight did not retain every live state action.")

            reference_depths = {state_ref.digest(): 0}

            def visit_reference_values(value, depth):
                from .cdef_graph import EdgeKind
                from .links import DefLink
                from .reference_values import StateRef
                from .utils.graph.value import iter_value_edges

                if isinstance(value, StateRef):
                    digest = value.digest()
                    reference_depths[digest] = max(reference_depths.get(digest, 0), depth)
                    for parameter in value.definition.parameters.values():
                        visit_reference_values(parameter, depth + 1)
                    return
                if isinstance(value, DefLink):
                    if value.kind is EdgeKind.MATERIALIZE:
                        visit_reference_values(value.target, depth)
                    return
                for edge in iter_value_edges(value):
                    visit_reference_values(edge.value, depth)

            for parameter in state_ref.definition.parameters.values():
                visit_reference_values(parameter, 1)
            ordered = sorted(
                actions.values(),
                key=lambda action: (
                    -reference_depths.get(action.reference.digest(), 0),
                    -len(action.path), str(action.path), str(action.object_id),
                ),
            )
            obj._last_state_ref = None
            for action in ordered:
                target = targets[action.object_id]
                codec = action.state_hash.split("-", 1)[0]
                hooks_started = True
                target.restore_state_from_dir(
                    os.path.join(os.fspath(action.payload), "data"), codec=codec
                )
                target._last_state_hash = action.state_hash
            obj._last_state_ref = state_ref
            return state_ref
        except RepoLoadError:
            if hooks_started:
                self._invalidate_restoration_target(nodes)
            raise
        except BaseException as error:
            if hooks_started:
                self._invalidate_restoration_target(nodes)
            raise RepoLoadError(f"Targeted exact restore failed: {error}") from error
        finally:
            for lock in reversed(locks):
                lock.release()
            if owns_reservation:
                reservation.release()

    def _invalidate_restoration_target(self, nodes) -> None:
        """Conservatively retire live instances after an in-place hook failure."""

        for node in nodes:
            node._restore_failed = True
            node._last_state_hash = None
            node._last_state_ref = None
            self._evict_live(node)

    def load(self, cdef: ConcreteDefinition, *, cache: CachePolicy = "weak") -> Object:
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_load"):
            return self.load_object(cdef, cache=cache)

    def load_or_build(self, x: Definition | ConcreteDefinition | Object, *, cache: CachePolicy = "weak") -> Object:
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_load_or_build"):
            if isinstance(x, Object):
                x = x.definition
            elif isinstance(x, Definition):
                x = x.concretize(repo=self)
            elif not isinstance(x, ConcreteDefinition):
                raise TypeError("Repo.load_or_build requires a Definition, ConcreteDefinition, or Object.")
            return self._load_structural(x, cache=cache)


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

    def references(self):
        """Start an authority-verified query over ObjectRefs and StateRefs.

        Returns:
            A ReferenceQuery over immutable Definition, Declaration, StateRef,
            and alias authority in connected Stores.

        Side Effects:
            None until a terminal is evaluated. Evaluation never materializes an
            Object or opens local-state payloads.
        """

        from .query.reference import ReferenceQuery

        return ReferenceQuery(self)

    def definition_graph(self, value) -> "ConcreteDefinitionGraph":
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
            cache: CachePolicy = "weak",
            verbose: bool = True) -> ObjectResultSet:
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_get"):
            if sel_args is None:
                sel_args = []
            if sel_kwargs is None:
                sel_kwargs = {}
            selectors = self._selector_tuple(selector)
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
                    .objects(cache=cache)
                )
                selected_objects.update(objs)

            return ObjectResultSet(self, selected_objects, domain="known")

    def apply(self,
              func, func_args=None, func_kwargs=None,
              selector: Callable | None = None,
              sel_args=None, sel_kwargs=None,
              verbose: bool = False,
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
                **kwargs)

            obj_iter = objs.items()
            if verbose:
                from tqdm import tqdm

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
            dedupe: bool = True) -> RepoGraphOptions:
        if options is not None:
            return options
        return RepoGraphOptions(
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
            dedupe: bool = True):
        graph_options = self._graph_options(
            options=options,
            include_root=include_root,
            order=order,
            missing=missing,
            dedupe=dedupe,
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
            dedupe: bool = True):
        graph_options = self._graph_options(
            options=options,
            include_root=include_root,
            order=order,
            missing=missing,
            dedupe=dedupe,
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
        """Flush and detach Repo query bindings unless an active save retains them.

        Args:
            flush: Whether to commit configured Stores before detaching bindings.

        Raises:
            RuntimeError: If an active retained save context still depends on this
                Repo's resources.

        Side Effects:
            Optionally commits configured Stores, closes Repo-owned query
            bindings, and closes each Store handle opened by this Repo exactly
            once. Supplied Store handles remain borrowed and are not closed.
            A failed commit or owned-handle cleanup leaves the Repo open for
            inspection/retry; ``flush=False`` skips commits.
        """

        with self._configuration_lock:
            if self._save_context_leases:
                raise RuntimeError("Cannot close Repo while an active save context retains resources.")
            if self._topology_leases:
                raise RuntimeError("Cannot close Repo while an active topology lease retains resources.")
            if self._closed:
                return
            self._closing = True
        try:
            if flush and not self._state_io:
                self.flush()
            self._query_index.close()
            self._close_owned_stores()
            if self._state_io:
                self.clear_cache(strong=True, weak=True)
        except BaseException:
            with self._configuration_lock:
                self._closing = False
            raise
        with self._configuration_lock:
            self._closing = False
            self._closed = True

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
            stores = repo
        else:
            stores = [repo]

        # Let Repo perform coercion so the temporary wrapper owns every handle
        # it opens while preserving caller-supplied Store handles as borrowed.
        repo_obj = Repo(stores=stores)
        close_repo = True

    with default_repo(repo_obj):
        failure = None
        try:
            yield repo_obj
        except BaseException as error:
            failure = error
            raise
        finally:
            if close_repo:
                try:
                    repo_obj.close(flush=not getattr(repo_obj, "_skip_cleanup_flush", False))
                except BaseException:
                    if failure is None:
                        raise
                    if hasattr(failure, "add_note"):
                        failure.add_note(
                            "Temporary Repo cleanup failed after a save failure."
                        )


# Saving and Loading
def save_object(
        obj,
        repo=None,
        *,
        main=False,
        store=None,
        alias: str | None = None,
        deep_capture: bool = False,
        match_mode: str | None = None,
        graph_mode: str | None = None,
        report_stores: bool = False):
    """Publish one Object graph through its current immutable StateRef boundary.

    Args:
        obj: Live Object graph root to publish.
        repo: Optional Repo or Store authority used for publication.
        main: Whether to update the target Store's main definition after StateRef
            authority is complete.
        store: Optional explicit whole-graph target Store.
        alias: Optional object alias written after StateRef publication.
        deep_capture: Whether to serialize every owned Serializable node.
        match_mode: Optional ``"first"`` or ``"all"`` routing override.
        graph_mode: Optional ``"per-object"`` or ``"closure"`` placement
            override local to this save.
        report_stores: Whether to return a StoreReport with the StateRef.

    Returns:
        The complete StateRef, or ``(StateRef, StoreReport)`` when requested.

    Raises:
        RepoSaveError: If bindings, claims, local state, or StateRef publication
            cannot complete.
        StoreAuthorityError: If the selected Store rejects authoritative writes.
        TypeError: If a supplied mode has the wrong type.
        ValueError: If a supplied mode is unsupported.

    Side Effects:
        Publishes immutable graph state and installs the completed StateRef as
        ``obj.last_state_ref`` before derived index, main, or alias work. Later
        failures propagate without clearing that valid receipt.

    Lifetime and Concurrency:
        A supplied Repo/Store remains borrowed. When this convenience creates a
        temporary Repo, it commits that Repo's configured Stores before return and
        closes only handles it opened. The delegated save retains one routing and
        Store-order view and preserves partial cross-Store evidence on failure.
    """
    from dryml.runtime import materialization_admission

    with materialization_admission(operation="global_save_object"):
        temporary_repo = not isinstance(repo, Repo) and (
            repo is not None or get_default_repo() is None
        )
        with manage_repo(repo=repo) as sub_repo:
            if temporary_repo:
                # This entry point owns the only flush.  Cleanup discards a
                # failed buffered save instead of retrying and masking it.
                sub_repo._skip_cleanup_flush = True
            main = main or ((repo is not sub_repo) and isinstance(obj, Object))
            with sub_repo._retain_save_context() as context:
                result = sub_repo.save_object(
                    obj, main=main, store=store, alias=alias,
                    deep_capture=deep_capture, match_mode=match_mode,
                    graph_mode=graph_mode,
                    report_stores=True, _save_context=context,
                    _commit_stores=context.stores if temporary_repo else (),
                )
                state_ref, report = result
                if temporary_repo:
                    report = _commit_save_report(
                        sub_repo, state_ref, report, stores=context.stores,
                    )
            return (state_ref, report) if report_stores else state_ref


def load_object(
        cdef: ConcreteDefinition | None = None, repo=None,
        *, cache: CachePolicy = "weak") -> Object:
    """Load existing structural CDef authority through a managed Repo.

    Args:
        cdef: CDef to construct, or ``None`` to select the Store main CDef.
        repo: Explicit Repo or Store-like source.
        cache: Cache tier for the resulting structural realization.

    Returns:
        A live structural Object without exact state restoration.

    Raises:
        ValueError: If neither ``cdef`` nor a Store main CDef is available.
        RepoLoadError: If the selected CDef lacks current Store authority.
    """
    from dryml.runtime import materialization_admission

    with materialization_admission(operation="global_load_object"):
        with manage_repo(repo=repo) as repo:
            if cdef is None:
                cdef = repo.main_def
                if cdef is None:
                    raise ValueError("When cdef is None, the repo must have a main def, we didn't find one.")
            return repo.load_object(cdef, cache=cache)


def load_state_ref(state_ref, repo=None, *, reuse_live: LiveReusePolicy = "matching", cache: CachePolicy = "weak") -> Object:
    """Exactly load one immutable StateRef through a managed Repo.

    Args:
        state_ref: Exact immutable snapshot authority.
        repo: Explicit Repo or Store-like source containing the full closure.
        reuse_live: Policy for eligible already-live exact nodes.
        cache: Cache tier for the completed realization.

    Returns:
        A graph restored from the requested exact StateRef.

    Raises:
        RepoLoadError: If exact authority or its local-state closure is missing.

    Side Effects:
        May construct or reuse live Objects. On success, installs ``state_ref``
        as the returned top-level root's last-state receipt only; descendants do
        not receive projected or synthetic receipts.
    """
    with manage_repo(repo=repo) as sub_repo:
        return sub_repo.load_state_ref(state_ref, reuse_live=reuse_live, cache=cache)
