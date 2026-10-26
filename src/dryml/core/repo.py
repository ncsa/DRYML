"""Coordinate live objects, Store authority, publication, and query topology.

Repo operations expose authority, lifecycle, and publication failures rather than
partial success. Topology retention leases connected Store handles without reading
inventory; inherited-process retention raises RuntimeError before owner locking.
Forked children construct fresh Repos instead of resetting parent locks or caches.
Exact preflight uses explicit owner-bound read cuts; neither borrowed sources nor
live Object caches extend a cut's payload-validity evidence across operations.
The same facility retains bounded immutable content only after successful cuts;
every independent operation establishes fresh Store authority before using it.
"""

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
from collections import defaultdict, OrderedDict
import hashlib
import atexit
import stat
import sys
from threading import RLock, current_thread
from uuid import uuid4

from ..paths import local_path_key
from .definition import Definition, ConcreteDefinition
from .cdef_graph import ConcreteDefinitionGraph
from .object import Object
from .store.store import (
    Store, StoreAuthorityError, StoreCapabilityError,
)
from .policies import CachePolicy, LiveReusePolicy, RepoGraphOptions
from .canonical import from_canonical
from ._content_facts import _primitive_size, _context_matches
from .config import CONFIG_MISSING, ConfigError, ConfigRef
from .query.federation import RepoQueryIndex
from .query.memory import AggregateMemoryQueryIndex
from .query.result import ObjectResultSet

if TYPE_CHECKING:
    from .repo_definition import RepoDefinition


def _content_hash(source):
    return hashlib.sha256(source).digest()


@dataclass(frozen=True)
class _ContentFact:
    bucket: bytes
    context: tuple
    source: bytes
    prepared: Any
    charge: int


class _ValidationEvidence:
    """One Repo's bounded content facts and separate cut-local authority facility.

    Only bookkeeping runs under the lock. Pending reservations consume the same
    slots/bytes as residents before any optional owned preparation is allocated.
    The fixed per-entry surcharge bounds LRU, ticket, bucket and cut bookkeeping;
    primitive sizing deliberately overcounts shared scalars and container slots.
    """

    max_entries = 256
    max_bytes = 32 * 1024 * 1024
    max_entry_bytes = 2 * 1024 * 1024
    max_bucket = 4

    def __init__(self, repo):
        self.repo = repo
        self.epoch = 0
        self._lock = RLock()
        self._resident = OrderedDict()
        self._pending = {}
        self._serial = 0
        self.accounted_bytes = 0
        self.high_water_bytes = 0
        self.high_water_entries = 0

    def check(self):
        if self.repo._configuration_pid != os.getpid():
            raise RuntimeError("Content evidence inherited from another process; open a fresh Repo.")
        if self.repo._closing or self.repo._closed:
            raise RuntimeError("Content evidence owner is closing/closed.")

    def clear(self):
        if self.repo._configuration_pid != os.getpid():
            raise RuntimeError("Cannot clear inherited content evidence; open a fresh Repo.")
        with self._lock:
            old = self._resident, self._pending
            self._resident = OrderedDict()
            # A preparing copy still owns its reservation until its creator
            # returns. Clear retires its epoch, not its in-flight memory charge.
            self._pending = {ticket: item for ticket, item in self._pending.items()
                             if item[2] is None}
            self.accounted_bytes = sum(item[1] for item in self._pending.values())
            self.epoch += 1
        del old

    def read_cut(self, repo, *, selected_stores=()):
        return _AuthorityReadCut(self, repo, selected_stores=selected_stores)

    def lookup(self, cut, context, source):
        cut.check()
        if type(context) is not tuple or type(source) is not bytes:
            raise TypeError("Content facts require tuple context and exact source bytes.")
        _primitive_size(context)
        bucket = _content_hash(source)
        with self._lock:
            candidates = tuple((ticket, fact) for ticket, fact in self._resident.items()
                               if fact.bucket == bucket)
        for ticket, fact in candidates:
            if fact.source == source and _context_matches(fact.context, context):
                cut.check()
                with self._lock:
                    if self.epoch == cut.epoch and self._resident.get(ticket) is fact:
                        self._resident.move_to_end(ticket)
                return fact.prepared
        return None

    def stage(self, cut, context, source, owned_size, prepare):
        """Reserve then prepare outside locks; admit only on successful cut exit.

        ``context`` is a closed primitive tuple including kind, policy, versions
        and implementation identities. ``owned_size`` bounds the prepared value;
        ``prepare`` must return owned immutable, token/resource-free content.
        Admission failure omits optional preparation, never ordinary validation.
        """
        cut.check()
        if type(context) is not tuple or type(source) is not bytes:
            raise TypeError("Content facts require tuple context and exact source bytes.")
        charge = 2048 + _primitive_size(context) + sys.getsizeof(source) + owned_size
        if cut._failed or charge > self.max_entry_bytes or charge > self.max_bytes:
            return
        bucket = _content_hash(source)
        retired = []
        with self._lock:
            if self.epoch != cut.epoch:
                return
            bucket_count = sum(fact.bucket == bucket for fact in self._resident.values()) + sum(
                item[0] == bucket for item in self._pending.values())
            while (len(self._resident) + len(self._pending) >= self.max_entries
                   or self.accounted_bytes + charge > self.max_bytes
                   or bucket_count >= self.max_bucket):
                victim = next((ticket for ticket, fact in self._resident.items()
                               if bucket_count < self.max_bucket or fact.bucket == bucket), None)
                if victim is None:
                    break
                fact = self._resident.pop(victim)
                self.accounted_bytes -= fact.charge
                bucket_count -= fact.bucket == bucket
                retired.append(fact)
                del fact
            admitted = (len(self._resident) + len(self._pending) < self.max_entries
                        and self.accounted_bytes + charge <= self.max_bytes
                        and bucket_count < self.max_bucket)
            if admitted:
                self._serial += 1
                ticket = self._serial
                self._pending[ticket] = (bucket, charge, None)
                cut._pending.add(ticket)
                self.accounted_bytes += charge
                self.high_water_bytes = max(self.high_water_bytes, self.accounted_bytes)
                self.high_water_entries = max(self.high_water_entries, len(self._resident) + len(self._pending))
        del retired
        if not admitted:
            return
        try:
            prepared = prepare()
            fact = _ContentFact(bucket, context, source, prepared, charge)
            with self._lock:
                if self.epoch == cut.epoch and ticket in self._pending:
                    self._pending[ticket] = (bucket, charge, fact)
                    accepted = True
                else:
                    accepted = False
            if not accepted:
                del fact, prepared
                with self._lock:
                    item = self._pending.pop(ticket, None)
                    if item is not None:
                        self.accounted_bytes -= charge
                cut._pending.discard(ticket)
        except BaseException:
            cut.discard()
            raise

    def finish(self, cut, *, success):
        """Atomically publish staged facts, or drop reservations after failure."""
        retired = []
        with self._lock:
            for ticket in cut._pending:
                item = self._pending.pop(ticket, None)
                if item is None:
                    continue
                _, charge, fact = item
                if (success and self.epoch == cut.epoch and fact is not None
                        and not self.repo._closing and not self.repo._closed
                        and not cut.repo._closing and not cut.repo._closed):
                    self._resident[ticket] = fact
                else:
                    self.accounted_bytes -= charge
                    retired.append(item)
            cut._pending.clear()
        del retired


class _AuthorityReadCut:
    """One owner-bound topology/fence lease, never an ambient validity cache.

    Prepared immutable content may be shared across cuts, but never substitutes
    for current Store authority or payload validation.
    """

    def __init__(self, facility, repo, *, selected_stores=()):
        self.facility = facility
        self.repo = repo
        self.pid = os.getpid()
        self.thread = current_thread()
        self.epoch = facility.epoch
        self.active = False
        self._entered = False
        self.stores = ()
        self.selected_stores = tuple(selected_stores)
        self._stack = None
        self._pending = set()
        self._failed = False

    def _check_owner(self):
        if self.pid != os.getpid() or self.repo._configuration_pid != os.getpid():
            raise RuntimeError("Read cut inherited from another process; open a fresh Repo.")
        if self.thread is not current_thread():
            raise RuntimeError("Read cut belongs to another thread.")
        self.facility.check()

    def check(self, store=None):
        self._check_owner()
        if not self.active or self.repo._closing or self.repo._closed:
            raise RuntimeError("Read cut is inactive or Repo is closing/closed.")
        if self.epoch != self.facility.epoch:
            raise RuntimeError("Read cut evidence was cleared.")
        if store is not None and not any(store is item for item in self.stores):
            raise ValueError("Store is not connected in this read cut.")

    def __enter__(self):
        self._check_owner()
        if self._entered:
            raise RuntimeError("Read cuts cannot be re-entered.")
        if self.repo._closing or self.repo._closed:
            raise RuntimeError("Cannot open read cut after Repo close begins.")
        with self.repo._configuration_lock:
            if self.repo._closing or self.repo._closed:
                raise RuntimeError("Cannot open read cut after Repo close begins.")
            self.stores = tuple(self.repo._normalize_store_handles(
                self.repo.stores, reject_physical=False,
            ))
            for store in self.selected_stores:
                if not isinstance(store, Store):
                    raise TypeError("Store selection must be a connected Store handle.")
                if not any(store is item for item in self.stores):
                    raise ValueError("Store selection must be a connected Store handle.")
            self.repo._topology_leases += 1
            self.epoch = self.facility.epoch
            self._entered = True
        self._stack = ExitStack()
        try:
            self._stack.enter_context(self.repo._authority_read_fences(self.stores))
            self.active = True
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def __exit__(self, *exc):
        # Foreign unwind must leave the owner's fences and lease untouched.
        self._check_owner()
        if self._stack is None:
            return False
        stack, self._stack = self._stack, None
        self.active = False
        try:
            result = stack.__exit__(*exc)
            self.facility.finish(self, success=exc[0] is None and not self._failed)
            return result
        finally:
            self.active = False
            self.discard()
            with self.repo._configuration_lock:
                self.repo._topology_leases -= 1

    def discard(self):
        """Drop partial evidence after failure without releasing an owner's lease."""
        self._check_owner()
        self._failed = True
        self.facility.finish(self, success=False)

    def prepared(self, context, source):
        """Return exact-matched immutable content, never a validity verdict."""
        return self.facility.lookup(self, context, source)

    def stage_prepared(self, context, source, owned_size, prepare):
        self.facility.stage(self, context, source, owned_size, prepare)

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


class MetadataConflictError(RepoLoadError):
    """Raised when selected Stores disagree about metadata for one exact target."""


class RepoGraphError(Exception):
    pass


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
    Topology access through a Repo inherited after fork raises RuntimeError before
    acquiring its parent-process lock; children must construct a fresh Repo.
    One internal validation facility owns both explicit authority cuts and bounded
    immutable snapshot JSON and eligible token-neutral graph preparation facts.
    Successful cuts may retain primitive content; later reads still check current
    file shape/bytes, fresh framed dill and domain associations. Graph realization
    allocates fresh tokens, symbols, containers and arrays on every independent read.
    Retention does not change live Object weak/strong cache policies or expose
    shared mutable public values. ``clear_cache`` and accepted owner close discard
    facts independently of those live-cache policies.
    Saved-root retain/release and affected-full-key cleanup preserve metadata
    history. Federated inventory is metadata-only; cleanup reports each target
    independently and never claims distributed rollback or payload availability.
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
            *, save_routing=None,
            _state_io: bool = False):
        """Create a Repo over supplied Store handles and optional routing policy.

        Args:
            stores: One Store, Store specification, or ordered collection of
                them.  Configured routing requires one handle per physical
                built-in destination; repeated occurrences of one handle dedupe.
            config: Initial runtime-only configuration values.
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
        self._state_io = _state_io
        self._configuration_lock = RLock()
        self._configuration_pid = os.getpid()
        self._configuration_version = 0
        self._save_context_leases = 0
        self._topology_leases = 0
        self._save_routing = None
        self._closing = False
        self._closed = False
        self._validation_evidence = _ValidationEvidence(self)
        self._content_owner_lease = None
        # Handles supplied by callers remain borrowed.  Paths and file-like
        # specifications opened by this Repo are released on successful close.
        self._owned_stores: list[Store] = []
        self.alias_index = {}
        self._aliases_dirty = False
        # U2 keeps allocation facts process-local until U3 adds Store authority.
        self._lineage_candidates = {}
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
                    try:
                        node_main = store.read_node_name("main")
                    except StoreCapabilityError:
                        node_main = None
                    if node_main is not None:
                        self.main_def = store.read_node(node_main.root_digest, kind=node_main.root_kind)
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
            from .session import _resource_cache_retains

            if (
                    _resource_cache_retains(self)
                    and (not isinstance(store, Store) or not any(existing is store for existing in self.stores))):
                raise RuntimeError(
                    "Cannot change Store topology while the Session resource cache retains Repo."
                )
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
            path = local_path_key(archive_path)
            parent = os.path.dirname(path) or "."
            try:
                evidence = os.stat(parent)
            except OSError as error:
                raise ValueError("ZipStore archive parent cannot be inspected.") from error
            if not stat.S_ISDIR(evidence.st_mode):
                raise ValueError("ZipStore archive parent is not a directory.")
            return ("zip", path)
        if type(store) is DirStore:
            path = local_path_key(store.base_dir)
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
    def retain_topology(self, *, allow_physical_duplicates: bool = False):
        """Retain connected Store handles without selecting save routes.

        Args:
            allow_physical_duplicates: Permit distinct handles for one physical
                destination. Query capture uses authority fence keys to merge
                equivalent DirStore views while preserving independent ZipStore
                transaction views. Other callers retain strict rejection.

        Returns:
            A context manager covering the current connected Store handles.

        Raises:
            RuntimeError: If the Repo is closing, closed, or inherited from another
                process. Open a fresh Repo in a forked child; inherited topology
                is rejected before acquiring the parent-process configuration lock.

        Side Effects:
            Prevents Store addition and Repo close until release. Routing and
            default-order changes remain valid because they do not alter the
            connected handle set.
        """

        if self._configuration_pid != os.getpid():
            raise RuntimeError("Cannot retain Repo topology inherited from another process; open a fresh Repo.")
        if self._closing or self._closed:
            raise RuntimeError("Cannot retain topology after Repo close begins.")
        with self._configuration_lock:
            if self._closing or self._closed:
                raise RuntimeError("Cannot retain topology after Repo close begins.")
            self._normalize_store_handles(
                self.stores, reject_physical=not allow_physical_duplicates,
            )
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

    def _cache_exact_graph(self, root: Object, cache: CachePolicy) -> None:
        """Publish an exact graph to the requested cache after identity rebinding.

        Identity-bound Node construction first replaces provisional ObjectIds and
        lineage facts. This helper preserves normal cache behavior without
        exposing a provisional fact table to cache consumers.
        """
        if cache == "none":
            return
        if cache not in {"weak", "strong"}:
            raise ValueError(f"Unknown cache policy: {cache!r}")
        objects = _unique_objects(
            value for value in getattr(root, "_runtime_projection", {}).values()
            if isinstance(value, Object)
        )
        for obj in objects:
            if cache == "strong":
                self.cache_strong(obj)
            else:
                self.cache_weak(obj)

    # --- helpers you already have ---
    def _cached_candidates(self, cdef) -> tuple[Object, ...]:
        """Return distinct live candidates for one exact private CDef node."""

        candidates = self.strong_obj_cache.candidates(cdef) + self.weak_obj_cache.candidates(cdef)
        from .state import is_reserved

        return _unique_objects(
            candidate for candidate in candidates
            if not getattr(candidate, "_restore_failed", False) and not is_reserved(candidate)
        )

    def _all_live_candidates(self, *, reservation=None) -> tuple[Object, ...]:
        """Return all distinct cached live Objects without structural lookup.

        Exact saved-Node reuse filters this complete live set by ObjectId, graph
        topology, bindings, and reservation availability.  It intentionally does
        not use CDef equality as an identity key. An internal exact-load identity
        reservation may expose only candidates covered by that caller-owned token.
        """
        from .state import is_reserved

        def available(obj):
            if not is_reserved(obj):
                return True
            if reservation is None:
                return False
            try:
                reservation._covers_object_ids((obj.object_id,))
            except RepoSaveError:
                return False
            return True

        return _unique_objects(
            obj for obj in (
                tuple(obj for _, obj in self.strong_obj_cache.items())
                + tuple(obj for _, obj in self.weak_obj_cache.items())
            ) if not getattr(obj, "_restore_failed", False) and available(obj)
        )

    def _allocate_unique_object_ids(self, count, *, excluded):
        """Allocate IDs absent from graph/cache metadata and connected anchors.

        Retain current Store handles during reads, without reserving IDs or
        recording candidates. Existing cached graph IDs and live runtime bindings
        count even if their Objects are reserved or unrestorable. Each missing ID
        gets at most 32 UUID candidates; exhaustion raises RuntimeError. Store
        authority/capability/IO failures propagate instead of establishing absence.
        This is observational checking, not a cross-process allocation transaction.
        """
        from .node import _all_nodes
        from .reference_values import ObjectId

        with self.retain_topology(allow_physical_duplicates=True):
            known = set(excluded) | set(self._lineage_candidates)
            roots = list(self.light_index) + list(self.cdef_cache.values())
            for cache in (self.strong_obj_cache, self.weak_obj_cache):
                for definition, obj in cache.items():
                    roots.append(definition)
                    for live in (obj, *getattr(obj, "_runtime_projection", {}).values()):
                        identity = getattr(live, "object_id", None)
                        if identity is not None:
                            known.add(identity)
            known.update(node.object_id for root in roots for node in _all_nodes(root, quoted=True)
                         if node.object_id is not None)
            assigned = []
            for _ in range(count):
                for _ in range(32):
                    identity = ObjectId()
                    if identity in known:
                        continue
                    if any(store.read_declaration_anchor(identity) is not None for store in self.stores):
                        known.add(identity)
                        continue
                    known.add(identity)
                    assigned.append(identity)
                    break
                else:
                    raise RuntimeError("Cannot allocate a unique ObjectId after 32 candidates.")
            return tuple(assigned)

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
            obj: Live root with complete retained Node/runtime bindings.

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
    def _for_state_io(cls, stores, *, _content_owner=None):
        """Create a non-owning authority-only Repo view over caller Stores.

        Args:
            stores: Existing Store instances selected by the caller for exact
                state authority and any immutable seed reads.
            _content_owner: Optional originating Repo whose bounded content
                facility is explicitly borrowed. A topology lease retains its
                lifetime; this view still owns distinct authority cuts.

        Returns:
            A call-owned Repo with independent memory caches and query federation
            disabled from independently opening/registering persistent indexes.
            Store-owned save mutations still maintain enabled projection before
            complete return; a borrowed view cannot bypass Store coherence.

        Side Effects:
            Does not close, commit, or otherwise take ownership of caller Stores
            or their existing query-index connections.
            A borrowing view detaches on successful close without clearing the
            originating facility. Explicit clear advances the shared epoch.
        """

        if _content_owner is None:
            return cls(stores, _state_io=True)
        lease = _content_owner.retain_topology(allow_physical_duplicates=True)
        lease.__enter__()
        try:
            result = cls(stores, _state_io=True)
            result._validation_evidence = _content_owner._validation_evidence
            result._content_owner_lease = lease
            return result
        except BaseException:
            lease.__exit__(*sys.exc_info())
            raise

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

    def _selected_writable_physical_store(self, store, operation: str, *, record_family=None) -> Store:
        """Select one connected writable physical Store for current-state CAS.

        Ordinary reference mutation retains its historical handle-level selection.
        Current-state coordination instead collapses identical direct physical
        handles, so replicas do not manufacture another mutable authority.
        """

        if store is not None:
            selected = self._connected_metadata_store(store)
            selected.preflight_publication(operation, record_family=record_family)
            return selected
        candidates = {}
        for candidate in self.stores:
            if not candidate.publication_capabilities.writable:
                continue
            key = self._physical_store_key(candidate)
            candidates.setdefault(("opaque", id(candidate)) if key is None else key, candidate)
        if len(candidates) != 1:
            raise RepoSaveError(
                f"{operation} requires an explicit Store or exactly one writable physical Repo Store."
            )
        selected = next(iter(candidates.values()))
        selected.preflight_publication(operation, record_family=record_family)
        return selected

    def _connected_metadata_store(self, store: Store) -> Store:
        """Require one already-connected Store without opening or registering it."""

        if not isinstance(store, Store):
            raise TypeError("Metadata Store selection must be a connected Store handle.")
        if not any(candidate is store for candidate in self.stores):
            raise ValueError("Metadata Store selection must be a connected Store handle.")
        return store

    @staticmethod
    def _metadata_target(target):
        """Validate one metadata attachment target without touching live Objects."""

        from .node import Node

        if not isinstance(target, Node):
            raise TypeError("metadata target must be a Node.")
        return target

    @contextmanager
    def _authority_read_fences(self, stores=None):
        """Hold each distinct selected Store authority domain in stable order."""

        groups = {}
        for store in self.stores if stores is None else stores:
            groups.setdefault(store.authority_fence_key(), store)
        with ExitStack() as fences:
            for key in sorted(groups):
                fences.enter_context(groups[key].authority_read_fence())
            yield

    def _read_cut(self, *, selected_stores=()):
        """Capture selections/topology before fences; substeps pass the cut explicitly."""
        return self._validation_evidence.read_cut(self, selected_stores=selected_stores)

    def get_metadata(self, target, *, store=None, scope=None):
        """Return fresh detached annotations for a current concrete Node target.

        Args:
            target: Concrete Node object projection or saved graph.
            store: Optional already-connected Store restricting the authority cut.
            scope: object/state, or None to infer state when the Node has selected
                state and object otherwise. Use state explicitly for stateless saved
                graphs whose object and saved descriptions have equal content.

        Returns:
            A detached whole mapping, or ``None`` when the known target has no
            current mapping in the selected Store(s).

        Raises:
            KeyError: If no selected Store holds ``target``.
            MetadataConflictError: If selected target-holding Stores differ,
            including absent versus present mappings.
            StoreAuthorityError: If target or metadata authority is malformed.
            ValueError: If scope is outside object/state/None.

        This is metadata-only inspection: it never materializes Objects, opens a
        Store, reads a payload, or runs environment observation.
        """

        target = self._metadata_target(target)
        from .metadata import _node_annotation_scope
        return self._node_metadata_values(target, store=store, scope=_node_annotation_scope(scope))

    def set_metadata(self, target, values, *, store=None, scope=None) -> None:
        """Atomically replace one selected current metadata mapping.

        Args:
            target: Concrete Node object projection or saved graph.
            values: Complete bounded metadata mapping that replaces any current
                mapping in the selected Store.
            store: Optional connected target-holding Store. It is required when
                more than one writable connected Store holds ``target``.
            scope: object/state or None, with the inference in get_metadata.

        Returns:
            ``None`` after atomic replacement succeeds.

        Raises:
            TypeError: If the target, Store, or mapping has an unsupported type.
            ValueError: If values violate metadata bounds or ``store`` is not
                connected.
            KeyError: If no selected Store holds ``target``.
            RepoSaveError: If destination selection is ambiguous.
            StoreAuthorityError: If target authority is malformed.
            StoreCapabilityError: If the selected backend cannot mutate metadata.

        Side Effects:
            Validates and detaches ``values`` before mutation, then replaces one
            whole mapping under the selected Store's writer fence. Other Stores,
            captured snapshot annotations, payloads, and reference identity remain
            unchanged.
        """

        from .metadata import encode_metadata_mapping

        target = self._metadata_target(target)
        # Validate and detach before selection can mutate authority.
        detached = encode_metadata_mapping(values)
        from .metadata import decode_metadata_mapping

        selected = self._node_metadata_mutation_store(target, store, scope)
        selected.write_metadata(target, decode_metadata_mapping(detached), scope=scope)

    def delete_metadata(self, target, *, store=None, scope=None) -> bool:
        """Atomically remove one selected current mapping and return prior presence.

        Args:
            target: Concrete Node object projection or saved graph.
            store: Optional connected target-holding Store. It is required when
                more than one writable connected Store holds ``target``.
            scope: object/state or None, with the inference in get_metadata.

        Returns:
            ``True`` when a current mapping was removed, otherwise ``False``.

        Raises:
            TypeError: If the target or Store has an unsupported type.
            ValueError: If ``store`` is not connected.
            KeyError: If no selected Store holds ``target``.
            RepoSaveError: If destination selection is ambiguous.
            StoreAuthorityError: If target authority is malformed.
            StoreCapabilityError: If the selected backend cannot mutate metadata.

        Side Effects:
            Removes only current metadata in the selected Store under its writer
            fence. The target, captured snapshot copies, payloads, lifecycle facts,
            and other Stores remain unchanged.
        """

        target = self._metadata_target(target)
        return self._node_metadata_mutation_store(target, store, scope).delete_metadata(target, scope=scope)

    def get_lineage_metadata(self, target, *, store=None):
        """Return immutable creation evidence or a valid unknown marker.

        Args:
            target: Current Node object projection.
            store: Optional connected Store restricting the authority cut.

        Returns:
            Detached NodeLineageMetadata. A valid target with no creation evidence
            returns an explicit unknown marker rather than an inferred timestamp.

        Raises:
            TypeError: If ``target`` is not a Node or ``store`` has an
                unsupported type.
            ValueError: If ``store`` is not connected.
            KeyError: If no selected Store holds ``target``.
            MetadataConflictError: If selected Stores contain different facts.
            StoreAuthorityError: If present lineage or snapshot authority is
                malformed.

        Side Effects:
            Reads lineage sidecars and snapshot metadata only. It does not load
            payloads, materialize Objects, run probes, or modify authority.
        """

        return self._node_metadata_values(self._metadata_target(target), store=store, scope="lineage")

    def get_snapshot_metadata(self, target, *, store=None):
        """Return detached first-capture metadata for one exact saved graph.

        Args:
            target: Saved Node whose complete captured evidence is required.
            store: Optional connected Store restricting the authority cut.

        Returns:
            Detached NodeSnapshotMetadata from the complete captured
            association. Later current-metadata edits are never merged into it.

        Raises:
            TypeError: If ``target`` is not a Node or ``store`` has an
                unsupported type.
            ValueError: If ``store`` is not connected.
            KeyError: If no selected Store holds ``target``.
            MetadataConflictError: If selected Stores contain different captured
                evidence.
            StoreAuthorityError: If matching snapshot authority is malformed or
                incomplete.

        Side Effects:
            Validates metadata association without opening payload bytes,
            materializing Objects, running probes, or modifying Store authority.
        """

        return self._node_metadata_values(self._metadata_target(target), store=store, scope="capture")

    def _node_metadata_values(self, target, *, store, scope):
        import json
        from .metadata import NodeLineageMetadata, timestamp_from_seconds, decode_metadata_mapping, _node_snapshot_metadata
        selected = (self._connected_metadata_store(store),) if store is not None else tuple(self.stores)
        values = []
        for candidate in selected:
            with candidate._metadata_read():
                if not candidate._holds_node_metadata(target, scope):
                    continue
                record = candidate._read_node_metadata(target, scope)
                data = None if record is None else json.loads(record.content)
                if scope == "capture":
                    if data is None:
                        raise StoreAuthorityError("Saved graph lacks captured metadata authority.")
                    value = _node_snapshot_metadata(target, data)
                elif scope == "lineage":
                    value = NodeLineageMetadata(target.object_view(), "unknown" if data is None else data["creation_status"],
                                                None if data is None else timestamp_from_seconds(data["created_at"]))
                else:
                    value = None if data is None else decode_metadata_mapping(data["values"])
                values.append((None if record is None else record.content, value))
        if not values:
            raise KeyError("No connected Store holds the metadata target.")
        if any(item[0] != values[0][0] for item in values[1:]):
            raise MetadataConflictError("Selected Stores disagree about Node metadata.")
        return values[0][1]

    def _node_metadata_mutation_store(self, target, store, scope):
        from .metadata import _node_annotation_scope
        area = _node_annotation_scope(scope)
        selected = (self._connected_metadata_store(store),) if store is not None else tuple(self.stores)
        holders = []
        for candidate in selected:
            with candidate._metadata_read():
                if candidate._holds_node_metadata(target, area) and candidate.publication_capabilities.writable:
                    holders.append(candidate)
        if not holders:
            raise KeyError("No writable connected Store holds the metadata target.")
        if len(holders) != 1:
            raise RepoSaveError("Node metadata mutation requires an explicit Store when holders are ambiguous.")
        return holders[0]

    def _record_lineage_candidate(self, object_id, created_at=None):
        """Retain one process-local creation fact for a newly allocated ObjectId.

        U2 deliberately records candidates only. U3 owns immutable Store
        publication and reopening, so this helper neither reads nor writes Store
        authority.
        """
        from .reference_values import ObjectId
        from .snapshot_capture import current_utc_time

        if not isinstance(object_id, ObjectId):
            raise TypeError("lineage candidates require an ObjectId")
        if created_at is None:
            created_at = current_utc_time()
        self._lineage_candidates.setdefault(object_id, created_at)
        return self._lineage_candidates[object_id]

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
        """Reconnect a detached configuration using existing Store handles.

        Args:
            definition: Fully validated inert portable Repo configuration.

        Returns:
            A new Repo. Without an active Session resource cache, it owns the
            Store handles opened for reconstruction; with caching, an equivalent
            live Repo is reused and a cache-created Repo borrows cached Stores.

        Raises:
            TypeError: If ``definition`` is not a RepoDefinition.
            RepoDefinitionError: If descriptors, selectors, or existing Store
                authority cannot be reconstructed.  Storage failures are chained.
            RepoReconstructionError: If cleanup after a reconstruction failure
                cannot close a fresh handle. The error privately retains only
                those handles for bounded idempotent cleanup retry.

        Side Effects:
            Opens only existing supported Store authority. It neither installs a
            session Repo nor creates missing storage. The returned Repo owns its
            newly opened noncached handles; ``close(flush=False)`` releases them
            without a commit, including after caller-managed failed work. Cached
            Repos and Store handles remain leased and owned by the active cache,
            so callers cannot close them before cache teardown.
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

    def hydrate_from_stores(self):
        """
        Ask each store to enumerate all cdefs it has.
        Populate obj_cache[cdef] = None for those not already present.
        """
        self._query_catalog.refresh(True)

    def refresh_index(self, *, force: bool = True):
        """Refresh configured persistent indexes and the Repo's memory catalog.

        This is an immediate, repeatable backend operation, not a shared Query
        one-shot request. Correct ready SQLite sidecars validate without
        replacement; use the Store's ``rebuild_query_index()`` when replacement
        is required.

        Args:
            force: bool selecting one explicit reconciliation cycle per call
                when True. False requests backend readiness recovery and memory
                hydration of unseen eligible Stores; unlike Query auto execution,
                this method explicitly visits configured backends.

        Returns:
            This Repo after both backend operations return successfully.

        Raises:
            QueryIndexUnavailable: If an explicit cycle cannot reach a backend.
            QueryIndexError: If backend reconciliation or memory hydration fails.
                Earlier successful backend work is not rolled back.

        Side Effects:
            May read Store inventories, replace derived sidecars, acknowledge
            captured dirty tokens, and rehydrate the memory catalog. Store
            authority is never rewritten.
        """
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
            states=None,
            main: bool = False,
            store=None,
            alias: str | None = None,
            deep_capture: bool = False,
            match_mode: str | None = None,
            graph_mode: str | None = None,
            report_stores: bool = False,
            source_store=None,
            source_stores=None,
            annotations=None,
            _capture_memo: set[object] | None = None,
            reservation=None,
            _save_context=None,
            _snapshot_observer=None,
            _commit_stores=()):
        """Publish one live graph as a format-4 SavedNode and retained user root.

        Args:
            obj: Root object whose graph is saved.
            states: Optional partial StateSpec or rooted path/LocalState map.
                Explicit compatible available selections skip capture; all other
                distinct live stateful nodes capture once on every plain save.
            main: Whether its concrete definition becomes the main reference.
            store: Optional whole-graph closure target Store. It bypasses routing
                and replication for this save.
            alias: Optional saved-Node name in the same Store batch.
            deep_capture: Retained spelling; plain saves already capture every
                stateful node without an explicit state selection.
            match_mode: Optional ``"first"`` or ``"all"`` routing override.
            graph_mode: Optional ``"per-object"`` or ``"closure"`` placement
                override. Both overrides are local to this save.
            report_stores: Whether to return an ephemeral StoreReport.
            source_store: Optional connected Store for selected local-state sources.
            source_stores: Optional full-StateKey/connected-Store source overrides.
            annotations: Optional SaveAnnotations replacing supplied current object
                and state maps and copying them into a first immutable capture.
            _capture_memo: Private retained capture memo. It is accepted for
                internal state-operation compatibility and does not alter routing.
            reservation: Optional active exact graph reservation reused by an
                enclosing state operation; callers normally omit this.

        Returns:
            The complete SavedNode, or ``(SavedNode, StoreReport)`` when requested.

        Raises:
            TypeError: If ``alias`` or a supplied mode has the wrong type.
            ValueError: If ``alias`` is empty or a supplied mode is unsupported.
            RepoSaveError: If routing, capture, or publication fails.
            StoreAuthorityError: If Store publication rejects authoritative data.
            MetadataConflictError: If selected complete sources or destination
                write-once snapshot evidence disagree.

        Side Effects:
            Captures outside writer fences. Each destination journals shared Node
            blocks, CDef-owned state variants, provisional physical grants,
            declaration anchors, memberships and names in one batch. Enabled
            projection acknowledges that cut before complete return. Confirmed
            graph metadata remains in partial reports after failure; last_state_ref
            advances only after its full closure has confirmed protected providers.
            Incomplete recovery preserves the prior receipt. Prepared local batches
            support alternating Store routes without splitting the per-Store commit.
            If unrelated cooperating writes invalidate detached read-back evidence,
            that observation retries at most eight times without recapturing or
            republishing. Exhausted read-back remains an unconfirmed save failure.
            Durable user retention is established before association; uncertain
            grants are preserved. Buffered/custom backends lacking this seam
            fail before capture. Load the
            returned SavedNode through this Repo's fresh materialization boundary.

        Concurrency:
            Retains one routing/default/Store-order view and reports partial
            cross-Store work rather than
            claiming transactional rollback. ``store=`` selects one complete
            closure even when routed modes were supplied.
        """
        from dryml.runtime import materialization_admission

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

                reservation = reserve(obj._identity_node, nodes, object_ids)
                reservation._save_plan = plan
            else:
                plan = getattr(reservation, "_save_plan", None)
                if plan is None or plan.binding.roots[0].obj is not obj:
                    plan, nodes, object_ids = self._state_graph_evidence(obj)
                else:
                    from .repo_plan import validate_retained_save_plan

                    validate_retained_save_plan(plan, obj)
                    if reservation.object_ref != plan.object_ref:
                        raise RepoSaveError("State graph reservation does not cover this exact identity Node.")
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
                from .repo_plan import (
                    build_routed_save_plan,
                    execute_routed_save_plan,
                    _register_retained_save_plan,
                )

                _register_retained_save_plan(self, plan)
                routed = build_routed_save_plan(
                    self, plan, context, store=selected_store,
                )
                result = execute_routed_save_plan(
                    self,
                    routed,
                    states=states, main=main, alias=alias,
                    deep_capture=deep_capture,
                    report_stores=True,
                    snapshot_observer=_snapshot_observer,
                    annotations=annotations, source_store=source_store,
                    source_stores=source_stores,
                )
                state_ref, report = result
                if main:
                    self.main_def = obj.definition
                return (state_ref, report) if report_stores else state_ref

    def inspect_states(self, cdef_digest, *, available_only=False):
        """Inventory one creation bucket in each connected Store, without payload IO.

        cdef_digest is a carried creation digest; available_only is bool. Returns
        tuple[StateInventoryEntry] in Store order, distinguishing historical
        descriptors from currently available variants. No Objects are constructed
        and no future availability is promised. Invalid/corrupt/unsupported Store
        evidence raises its original error; no partial inventory is called complete.
        """
        from .store.retention import StateInventoryEntry
        with self.retain_topology():
            return tuple(StateInventoryEntry(store.retention_domain, descriptor)
                         for store in self.stores
                         for descriptor in store.inspect_states(cdef_digest, available_only=available_only))

    def delete_states(self, keys, *, store=None):
        """Try immediate full-key deletion through the connected Store federation.

        keys is an iterable of StateKeys; optional store selects one connected
        Store. Returns tuple[StateCleanupOutcome], in Store then sorted key order,
        including every requested target. Independent targets continue after an
        ordinary failure. Retained/in-use is immediate and queues nothing; physical
        garbage failure carries incarnation retry evidence. Metadata/history remains.
        Malformed keys/Store selection and interruptions raise; per-target authority,
        capability and index errors are failed outcomes, never global success.
        """
        from .store.retention import RetentionResource, _keys, reclaim_resources
        keys = _keys(set(keys))
        with self.retain_topology():
            targets = tuple(self.stores) if store is None else (self._ensure_store(store),)
            resources = tuple(RetentionResource(target.retention_domain, keys) for target in targets)
            return reclaim_resources(resources, targets)

    def retain(self, saved, *, owner_store=None):
        """Retain an existing SavedNode without capture, restore or serialization.

        saved must be complete and freshly available in this Repo. owner_store is
        an optional connected durable Store (default: default_store). Returns its
        RetentionIntentRecord, an explicit independent root handle for release().
        Only this graph's selected provider/full-key closure is visited. Protection
        precedes adoption; errors propagate and preserve provisional intent/grants
        for explicit reconciliation, never infer abandonment or remove old roots.
        """
        from .materialization import build_exact_state_load_plan
        with self.retain_topology():
            owner = self.default_store if owner_store is None else self._ensure_store(owner_store)
            retention = owner.retention
            plan = build_exact_state_load_plan(self, saved)
            targets = {}
            for action in plan.actions:
                targets.setdefault(action.store, set()).add(action.state.key)
            token = str(uuid4())
            retention.prepare(token, saved.exact_digest(), targets.items())
            retention.protect(token, self.stores)
            return retention.adopt(token, self.stores)

    def release(self, root, *, owner_store=None):
        """Release one owned retained-root handle and reclaim its affected keys.

        root is a RetentionIntentRecord from retain(), or a UUID token from a save
        publication report. owner_store defaults to this Repo's default Store and
        must match the handle's owner. Returns tuple[StateCleanupOutcome]. Other
        roots/readers remain effective. A same-caller active key lease returns
        in-use without retiring the root; retry explicitly after that scope ends.
        Grant/owner/index failures raise with cleanup_outcomes attached; independently
        acknowledged targets still reclaim and unsafe dependents are unattempted.
        Deletion-phase authority/index failures are explicit failed outcomes.
        Only physical garbage failure is a successful cleanup-pending outcome.
        No graph/result/history metadata or private payload is serialized or removed
        to repair derived state. Interrupted work preserves its exact resource ledger.
        """
        from .store.retention import (
            RetentionIntentRecord, RetentionReleasePending, StateCleanupOutcome,
            _held, reclaim_resources,
        )
        with self.retain_topology():
            owner = self.default_store if owner_store is None else self._ensure_store(owner_store)
            token = root.token if isinstance(root, RetentionIntentRecord) else root
            intent = owner.retention.read(token)
            if intent is None or isinstance(root, RetentionIntentRecord) and (
                    root.owner_domain != owner.retention_domain or root.resources != intent.resources
                    or root.graph_digest != intent.graph_digest):
                raise StoreAuthorityError("Retained root is missing or belongs to another owner.")
            if _held():
                return tuple(StateCleanupOutcome(resource.domain, key, "in-use")
                             for resource in intent.resources for key in resource.keys)
            error = None
            try:
                released = owner.retention.release(token, self.stores)
            except Exception as failure:
                error = failure
                released = getattr(failure, "retention_intent", None)
            except BaseException as failure:
                failure.cleanup_outcomes = tuple(StateCleanupOutcome(resource.domain, key, "unattempted")
                    for resource in intent.resources for key in resource.keys)
                raise
            confirmed = () if released is None else released.released_targets
            blocked = {resource.domain for resource in intent.resources} - set(confirmed)
            outcomes = reclaim_resources(intent.resources, self.stores, blocked=blocked)
            if error is not None:
                failures = dict(getattr(error, "retention_failures", ()))
                outcomes = tuple(replace(item, status="failed", error=failures[item.domain], phase="release")
                                 if item.domain in failures else item for item in outcomes)
                error.cleanup_outcomes = outcomes
                raise error
            if released.disposition != "released":
                raise RetentionReleasePending(released, outcomes)
            return outcomes

    def describe(self, obj):
        """Return a ConcreteNode realization view from retained live construction.

        obj is a usable live Object. The result carries existing identities and
        explicit construction pins, never newly captured state or a last-save
        receipt. Missing/invalid retained bindings raise RepoSaveError; unsupported
        inputs raise TypeError. This performs pure graph preparation only, with no
        construction, payload IO, publication or ObjectId allocation.
        """
        from .repo_plan import build_save_plan
        return build_save_plan(self, obj).node

    def build(self, node, *, cache="weak"):
        """Build a ConcreteNode, allocating only missing IDs and selected state once.

        node must already be concretized; cache is weak/strong/none. Returns the
        live root. TypeError/ValueError reject invalid metadata and RepoLoadError
        reports unavailable state or realization failures. Unspecified state is
        initialized normally. Selected state preflights before effects; hooks run
        under active key leases without Store fences. Post-hook failure requires
        fresh materialization, never automatic rollback.
        """
        from .materialization import realize_node
        return realize_node(self, node, cache=cache)

    def apply_state(self, obj, states, *, source_store=None, source_stores=None):
        """Apply rooted LocalState selections, leaving unspecified live state untouched.

        obj is a usable live Object; states is a StateSpec or path/LocalState map.
        Optional source overrides select connected Stores by full StateKey.
        Returns None, changes no identities/sharing and captures/publishes nothing.
        Validation errors before hooks leave the graph usable. Hook failure raises
        RepoLoadError and invalidates the graph, requiring fresh materialization.
        """
        from .materialization import prepare_restoration
        with prepare_restoration(self, obj, states, source_store=source_store,
                                 source_stores=source_stores) as prepared:
            prepared.apply()

    def prepare_restore(self, obj, node, *, reservation=None, source_store=None, source_stores=None):
        """Return an owned one-shot full restoration handoff without running hooks.

        obj is a live graph; node must be SavedNode metadata matching its original
        canonical object projection. Optional reservation covers this exact graph;
        source overrides select full StateKeys in connected Stores. The returned
        PreparedRestoration must be applied or closed on this thread. It retains
        live ownership and source leases, validates before hooks and adopts supplied
        construction pins only on success. Type/Value/RepoLoad errors reject invalid
        preparation without invalidating a usable graph.
        """
        from .materialization import prepare_restoration
        return prepare_restoration(self, obj, saved=node, reservation=reservation,
                                   source_store=source_store, source_stores=source_stores)

    def restore(self, obj, node, *, reservation=None, source_store=None, source_stores=None):
        """Fully restore a projection-compatible SavedNode into an existing live graph.

        Parameters follow prepare_restore. Returns the supplied node after hooks
        succeed once dependency-first and construction pins are adopted. Live
        identities, ObjectIds and sharing are preserved. Before-hook validation
        failure leaves the graph usable; post-hook failure raises RepoLoadError,
        invalidates the graph and requires fresh materialization. No rollback,
        capture, publication or last-save receipt shortcut occurs.
        """
        with self.prepare_restore(obj, node, reservation=reservation, source_store=source_store,
                                  source_stores=source_stores) as prepared:
            return prepared.apply()

    def save(
            self,
            obj: Object,
            *,
            states=None,
            main: bool = False,
            store=None,
            alias: str | None = None,
            deep_capture: bool = False,
            match_mode: str | None = None,
            graph_mode: str | None = None,
            report_stores: bool = False,
            source_store=None, source_stores=None, annotations=None):
        """Publish a retained SavedNode through save_object and flush its Repo.

        Args:
            obj: Live Object root whose retained runtime graph will be saved.
            states: Optional rooted partial StateSpec/path-to-LocalState assignments;
                only the remaining distinct live stateful nodes capture.
            main: Whether to update the target Store's main-definition reference.
            store: Explicit whole-graph closure Store or Store specification.
            alias: Optional Store-local name for the saved graph.
            deep_capture: Whether to serialize every owned live Serializable node.
            match_mode: Optional ``"first"`` or ``"all"`` routing override.
            graph_mode: Optional ``"per-object"`` or ``"closure"`` placement
                override local to this save.
            report_stores: Whether to pair the SavedNode with a StoreReport.
            source_store: Optional connected Store for explicit local-state sources.
            source_stores: Optional full-StateKey/connected-Store source overrides.
            annotations: Optional SaveAnnotations replacing supplied current object
                and state maps and copying them into a first immutable capture.

        Returns:
            The immutable SavedNode, or ``(SavedNode, StoreReport)`` when
            ``report_stores`` is true.

        Raises:
            RepoSaveError: If graph bindings, reservations, codecs, or hooks fail.
            StoreAuthorityError: If Store preflight or publication fails.
            MetadataConflictError: If selected sources or immutable destination
                snapshot evidence disagree.
            TypeError: If a supplied mode has the wrong type.
            ValueError: If no writable target Store can be selected or a mode is unsupported.

        Side Effects:
            Uses save_object's one-batch publication, index and retention contract,
            then flushes configured Stores. Buffered durable ownership raises
            StoreCapabilityError before capture.

        Concurrency:
            Store publication uses short writer fences; native live reservations
            and full-key leases protect the save's ownership and payload use.
        """
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_save"):
            # Retain an explicit target before the flush scope is captured so a
            # newly connected buffered archive cannot be omitted from its commit.
            selected_store = self._ensure_store(store)
            with self._retain_save_context() as context:
                result = self.save_object(
                    obj, states=states, main=main, store=selected_store, alias=alias,
                    deep_capture=deep_capture, match_mode=match_mode,
                    graph_mode=graph_mode,
                    source_store=source_store, source_stores=source_stores,
                    annotations=annotations,
                    report_stores=True, _save_context=context,
                    _commit_stores=context.stores,
                )
                state_ref, report = result
                report = _commit_save_report(self, state_ref, report, stores=context.stores)
            return (state_ref, report) if report_stores else state_ref

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
                             declaration_stores=None):
        """Realize selected Node boundary roots after aggregate state admission.

        Args:
            roots: Ordered selected values from one signature argument or return
                boundary.  Ref-valued delivery must not call this API.
            cache: ``None`` preserves Repo's ``"weak"`` default; supplied values
                are forwarded unchanged to structural realization.
            reuse_live: ``None`` preserves exact-load ``"matching"``; supplied
                values are forwarded unchanged to exact realization.
            declaration_stores: Optional mapping from exact Node digests to
                connected Stores, pinning saved metadata and all
                selected full StateKeys to that connected source with no replica
                fallback. Conflicting source demands fail before hooks.

        Returns:
            A tuple of runtime values in ``roots`` order, preserving aliases across
            all roots and supported container families, including distinct built-in
            ``frozenset`` and tagged ``FrozenSet`` containers.

        Raises:
            RepoLoadError: If exact authority is incomplete or incompatible,
                a selected Store is disconnected, stale, or incompatible, or
                construction/restore fails.
            TypeError: If ``roots`` or ``declaration_stores`` is malformed.

        Side Effects:
            May reserve/reuse live objects, construct, restore, and populate
            caches. Selected full keys are validated under native reader leases.
            A post-hook failure invalidates affected live graphs; no rollback is
            claimed. This method publishes no Store authority.
        """

        from dryml.runtime import materialization_admission
        from .materialization import materialize_nodes, preflight_symbolic_materialization

        if not isinstance(roots, (tuple, list)):
            raise TypeError("materialize_boundary roots must be an ordered sequence.")
        if declaration_stores is None:
            declaration_stores = {}
        if not isinstance(declaration_stores, Mapping):
            raise TypeError("materialize_boundary declaration_stores must be a mapping.")
        declaration_stores = dict(declaration_stores)
        for digest, store in declaration_stores.items():
            if not isinstance(digest, str) or not isinstance(store, Store):
                raise TypeError("materialize_boundary declaration_stores must map digests to Stores.")
            if not any(store is connected for connected in self.stores):
                raise RepoLoadError("Selected declaration Store is not connected to this Repo.")

        preflight_symbolic_materialization(*roots, repo=self)
        with materialization_admission(operation="repo_node_boundary"):
            return materialize_nodes(self, roots, cache="weak" if cache is None else cache,
                                     reuse_live="matching" if reuse_live is None else reuse_live,
                                     declaration_stores=declaration_stores)

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
        class_manifest=None,
    ):
        from dryml.runtime import materialization_admission
        from .node import Node

        if isinstance(cdef, Node):
            from .materialization import realize_node
            from .cdef_identity import cdef_node_key
            key = cdef_node_key(cdef)
            if memo is not None and key in memo:
                return memo[key]
            result = realize_node(self, cdef, cache=cache)
            if memo is not None:
                memo[key] = result
            return result

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
                class_manifest=class_manifest,
            )

    def _load_structural(
        self,
        x: object,
        *,
        cache: CachePolicy = "weak",
    ):
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_load_object"):
            from .repo_plan import _NodeBindings, realization_scope

            with realization_scope():
                memo = _NodeBindings()
                return self._realize(
                    x,
                    cache=cache,
                    path=[""],
                    memo=memo,
                )

    def load_object(self, x, *, cache: CachePolicy = "weak") -> Object:
        """Load a SavedNode with fresh full-key validation and reader leases.

        Args:
            x: A complete SavedNode, including stateless saved graphs.
            cache: Cache tier for the resulting realization.

        Returns:
            A live Object. Node inputs restore their complete selected state.

        Raises:
            ValueError: If a Node lacks saved completeness.
            TypeError: If ``x`` is not a Node.
            RepoLoadError: If selected state is unavailable or cannot be restored.

        Side Effects:
            May construct, restore and cache live objects. Each Node use freshly
            validates selected full keys before hooks and holds active leases.
        """
        return self.load(x, cache=cache)

    def load_state_ref(
            self,
            state_ref,
            *,
            reuse_live: LiveReusePolicy = "matching",
            cache: CachePolicy = "weak",
            source_store=None,
            source_stores=None) -> Object:
        """Restore complete SavedNode metadata with fresh selected-state preflight.

        Args:
            state_ref: Immutable complete SavedNode with full LocalState selections.
            reuse_live: ``"matching"`` reuses one matching live checkpoint,
                ``"greedy"`` restores one unique candidate in place, and
                ``"never"`` always builds fresh.
            cache: Cache tier populated only after the complete realization has
                succeeded.
            source_store: Optional connected Store for selected full keys.
            source_stores: Optional full-StateKey-to-connected-Store overrides,
                confined to the selected closure.

        Returns:
            A fully restored root preserving the SavedNode's ObjectIds and graph
            topology.

        Raises:
            ValueError: If saved completeness or source scope is invalid.
            RepoLoadError: If exact authority preflight, including selected
                snapshot or local payload authority, construction, or restore
                fails. No fresh partial cache entry is published.
            StoreAuthorityError: If selected Node capture/declaration metadata is
                missing or inconsistent. No constructor or restore hook runs.
            TypeError: If ``source_store`` is not a Store, ``source_stores`` is
                not a mapping, or a mapping key/value is not a StateKey/Store.

        Side Effects:
            Restores selected nodes once, including eligible reused live Objects.
            A failed post-hook realization invalidates affected live nodes and
            requires fresh materialization; no rollback is claimed.
            A successful top-level exact load installs ``state_ref`` as the
            returned root's last-state receipt; descendants receive no synthetic
            receipt. Legacy persisted snapshot formats are not converted or
            substituted for missing Node state. Annotation/lineage metadata feeds
            remain a separate consumer boundary.
        """
        from .node import Node, require_saved
        if isinstance(state_ref, Node):
            from .materialization import realize_node
            return realize_node(self, require_saved(state_ref), exact=True,
                                reuse_live=reuse_live, cache=cache,
                                source_store=source_store, source_stores=source_stores)
        raise TypeError("load_state_ref requires a complete SavedNode; legacy snapshots are incompatible.")

    def _load_state_ref_with_identity_reservation(
            self, state_ref, *, reuse_live: LiveReusePolicy = "matching",
            cache: CachePolicy = "weak") -> Object:
        """Exactly load one state while atomically owning all target ObjectIds."""

        from .materialization import realize_node
        from .node import _owned_nodes, require_saved
        from .state import reserve

        require_saved(state_ref)
        identities = tuple(node.object_id for _, node in _owned_nodes(state_ref) if node._stateful_role)
        with reserve(state_ref, (), identities) as reservation:
            return realize_node(self, state_ref, exact=True, reuse_live=reuse_live,
                                cache=cache, _identity_reservation=reservation)

    def restore_state_ref_into(
            self, obj: Object, state_ref, *, reservation=None,
            source_store=None, source_stores=None):
        """Restore a SavedNode into the supplied projection-compatible live graph.

        Args:
            obj: Existing live root whose original object projection and retained
                live bindings match the supplied saved graph.
            state_ref: Complete SavedNode to restore without replacing identities.
            reservation: Optional active reservation returned for this exact graph.
            source_store: Optional connected Store for selected full keys.
            source_stores: Optional full-StateKey-to-connected-Store overrides.

        Returns:
            The requested ``state_ref`` after every local restore hook succeeds.

        Raises:
            RepoLoadError: If authority, including selected snapshot/local payload
                evidence, topology, paths, IDs, or payloads fail preflight, or if
                a hook fails. A post-hook failure invalidates all covered live
                nodes and requires a fresh exact load.
            RepoSaveError: If graph ownership is unavailable or ``reservation``
                is inactive, foreign, or does not cover the supplied graph.
            MetadataConflictError: If unqualified replicas or a live cached
                identity disagree with selected captured lineage evidence,
                including an uncached supplied target.
            TypeError: If ``source_store`` is not a Store, ``source_stores`` is
                not a mapping, or a mapping key/value is not a StateKey/Store.
            ValueError: If a selected Store is disconnected, root selectors
                differ, or a mapping names a StateKey outside the exact closure.

        Side Effects:
            Runs local hooks once dependency-first under active key leases, without
            writer fences, candidate search or Object replacement. On success,
            adopts supplied construction pins and updates the root receipt. Failure
            after hook entry invalidates the graph; recover by fresh materialization,
            not rollback. Validation failure before hooks leaves it usable.
        """

        return self.restore(obj, state_ref, reservation=reservation,
                            source_store=source_store, source_stores=source_stores)

    def _invalidate_restoration_target(self, nodes) -> None:
        """Conservatively retire live instances after an in-place hook failure."""

        affected = {id(node): node for node in nodes}
        candidates = _unique_objects(
            obj for _, obj in (*self.strong_obj_cache.items(), *self.weak_obj_cache.items()))
        for candidate in candidates:
            if any(id(bound) in affected for bound in getattr(candidate, "_runtime_projection", {}).values()):
                affected[id(candidate)] = candidate
        for node in affected.values():
            node._restore_failed = True
            node._last_state_hash = None
            node._last_state_ref = None
            self._evict_live(node)

    def load(self, cdef: ConcreteDefinition, *, cache: CachePolicy = "weak") -> Object:
        """Load a complete SavedNode with fresh full-key state validation.

        cdef is the supplied SavedNode; cache is weak/strong/none. Returns a live
        graph with the selected ObjectIds and alias partition. Incomplete Nodes
        raise TypeError/ValueError before effects. Missing/incompatible/corrupt
        selected state raises RepoLoadError without initialization or fallback.
        Selected hooks run once dependency-first under active leases and outside
        writer fences, even on eligible cache reuse. Construction-only recipes
        belong to build(), rather than a saved-state load.
        """
        from .node import require_saved

        return self.load_state_ref(require_saved(cdef), cache=cache)

    def load_or_build(self, x: Definition | ConcreteDefinition | Object, *, cache: CachePolicy = "weak") -> Object:
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_load_or_build"):
            from .node import Node
            if isinstance(x, Node):
                return self.build(x.concretize(repo=self), cache=cache)
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
        """Return the one retained known Object with an exact CDef.

        Args:
            key: Exact ConcreteDefinition to select from stored roots and live
                retained cache entries.

        Returns:
            The matching live Object, reusing a retained cache instance when
            available.

        Raises:
            TypeError: If ``key`` is not a ConcreteDefinition.
            KeyError: If no retained lookup candidate matches ``key``.

        Side Effects:
            May read direct stored-root authority and structurally materialize a
            stored root that is not already live.
        """
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="repo_getitem"):
            if not isinstance(key, ConcreteDefinition):
                raise TypeError("Repo.__getitem__ requires a ConcreteDefinition key.")
            result = self._retained_lookup_objects(key)
            if len(result) == 0:
                raise KeyError(f"Repo doesn't contain an object with definition {key}")
            return result.one()

    def _retained_lookup_objects(self, selector, *, cache: CachePolicy = "weak") -> ObjectResultSet:
        """Materialize the fixed V3 candidate universe retained by get/item.

        Stored roots and explicit cache CDefs are selected without exposing
        merely represented nested identities. Exact CDef selection uses V3's
        direct stored-root reader; other authoritative root families remain
        eligible through complete Store inspection when direct proof is absent.
        """

        from .generator import GeneratorSelector
        from .query.query import IdentityQuery
        from .query.result import ObjectResultSet
        from .selector import Selector

        if isinstance(selector, Object):
            selector = selector.definition
        if selector is not None and not isinstance(
            selector, (Definition, ConcreteDefinition, Selector, GeneratorSelector),
        ):
            raise TypeError(
                "Query source must be Selector, Definition, ConcreteDefinition, Object, "
                f"or None, not {type(selector).__name__}."
            )
        candidates = (
            IdentityQuery.from_retained_repo_lookup(self)
            .sel(selector)
            .cdefs()
            .collect()
        )
        objects = {}
        for cdef in candidates:
            objects[cdef] = self.load_object(cdef, cache=cache)
        return ObjectResultSet(self, objects, domain="known")

    def query(self, *, weak: bool = True):
        """Create an unevaluated Query V3 identity universe for this Repo.

        Args:
            weak: Include weak and strong cache entries when true; include only
                strong cache entries when false. Store knowledge is retained in
                either case.

        Returns:
            An :class:`~dryml.core.query.IdentityQuery` over identity knowledge
            captured from connected Stores and retained caches. Call ``sel()``
            for structural or exact identity restrictions, then use an explicit
            terminal such as ``collect()``, ``count()``, or ``one()``.

        Raises:
            TypeError: If ``weak`` is not an exact bool.
            StoreAuthorityError: At terminal evaluation when connected Store
                authority is malformed, incomplete, or cannot provide the
                required complete inventory.

        Side Effects:
            Construction has none. A terminal reads only query authority and
            retained cache facts; it never constructs Objects, allocates
            ObjectIds, opens payloads, saves, observes host state, or runs work.
        """
        from .query import IdentityQuery

        if type(weak) is not bool:
            raise TypeError("Repo.query weak must be an exact bool.")
        return IdentityQuery.from_repo(self, weak=weak)

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

    def get(self,
            selector:  SelectorType | tuple[SelectorType] | list[SelectorType] | None = None,
            sel_args=None, sel_kwargs=None,
            cache: CachePolicy = "weak",
            verbose: bool = True) -> ObjectResultSet:
        """Return retained root/cache Objects selected by definitions or callables.

        Definition-like selectors evaluate against stored roots plus the retained
        strong/weak cache universe. Callable selectors retain their historical
        strong-cache-only behavior and receive ``sel_args`` and ``sel_kwargs``.

        Args:
            selector: One selector, callable, or exact list/tuple of selectors.
            sel_args: Positional arguments supplied to callable selectors.
            sel_kwargs: Keyword arguments supplied to callable selectors.
            cache: Cache policy for structurally materialized stored candidates.
            verbose: Retained compatibility option; does not alter selection.

        Returns:
            A mapping-like ObjectResultSet in the retained ``"known"`` domain.

        Raises:
            TypeError: If a definition-like selector has an unsupported type.
            QueryDomainError: If an exact reference is used outside containment.

        Side Effects:
            May read stored-root records and materialize selected stored CDefs.
        """
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
                    self._retained_lookup_objects(sel, cache=cache)
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

        Returns:
            None after cleanup, or immediately when already closed.

        Raises:
            RuntimeError: If an active retained save context still depends on this
                Repo's resources, or an active Session resource cache retains this
                Repo, a read cut retains topology, or the Repo is inherited from
                another process.

        Side Effects:
            Optionally commits configured Stores, closes Repo-owned query
            bindings, and closes each Store handle opened by this Repo exactly
            once. Supplied Store handles remain borrowed and are not closed.
            A failed commit or cleanup leaves the Repo open for inspection/retry;
            cleanup still attempts every owned Store handle. ``flush=False``
            skips commits and never triggers deletion-save publication.
            Accepted close discards derived validation evidence before cleanup;
            a failed close leaves that evidence empty when the Repo reopens.
            An internal content borrower instead detaches on successful close;
            its owner retains facts and cannot close while the borrower is open.
        """

        from .session import _assert_resource_close_allowed

        if self._configuration_pid != os.getpid():
            raise RuntimeError("Cannot close inherited Repo; open a fresh Repo.")
        _assert_resource_close_allowed(self)
        with self._configuration_lock:
            if self._save_context_leases:
                raise RuntimeError("Cannot close Repo while an active save context retains resources.")
            if self._topology_leases:
                raise RuntimeError("Cannot close Repo while an active topology lease retains resources.")
            if self._closed:
                return
            self._closing = True
            if self._content_owner_lease is None:
                self._validation_evidence.clear()
        try:
            if flush and not self._state_io:
                self.flush()
            errors = []
            try:
                self._query_index.close()
            except BaseException as error:
                errors.append(error)
            try:
                self._close_owned_stores()
            except BaseException as error:
                errors.append(error)
            if errors:
                raise errors[0]
            if self._state_io:
                self.strong_obj_cache.clear()
                self.weak_obj_cache.clear()
        except BaseException:
            with self._configuration_lock:
                self._closing = False
            raise
        if self._content_owner_lease is not None:
            lease, self._content_owner_lease = self._content_owner_lease, None
            self._validation_evidence = _ValidationEvidence(self)
            lease.__exit__(None, None, None)
        with self._configuration_lock:
            self._closing = False
            self._closed = True

    def __del__(self):
        """Best-effort deletion cleanup for an explicitly configured Repo.

        A Repo with ``save_objs_on_deletion`` saves a detached strong-cache
        snapshot in cache order until its first failure, then closes owned
        resources without flushing. Failures are reported through Python's
        unraisable hook because destructors cannot safely propagate them.
        """

        if (
                not getattr(self, "save_objs_on_deletion", False)
                or getattr(self, "_closed", True)
                or getattr(self, "_closing", True)
                or not hasattr(self, "_query_index")):
            return

        failed = False
        for _, obj in tuple(self.strong_obj_cache.items()):
            try:
                self.save(obj)
            except BaseException:
                failed = True
                break
        try:
            self.close(flush=False)
        except BaseException:
            failed = True
        if failed:
            # Raising delegates valid UnraisableHookArgs construction to CPython
            # and prevents the original exception's details from escaping.
            raise RuntimeError("Repo deletion cleanup failed.") from None

    def clear_cache(self, strong=False, weak=True):
        """Discard derived validation evidence and selected live Object cache tiers.

        Args:
            strong (bool): Clear strong live candidates when true.
            weak (bool): Clear weak live candidates when true (the default).

        Returns:
            None. Existing live Objects and borrowed Store sources remain valid.

        Raises:
            RuntimeError: If called on a Repo inherited from another process.

        Side Effects:
            Invalidates active read-cut evidence independently of live-tier flags.
            Empties retained content facts and advances their shared epoch,
            preventing in-flight preparation from repopulating cleared facts.
            In-flight optional copies keep their reservations until preparation
            finishes; ordinary decoding working memory is independent.
        """
        if self._configuration_pid != os.getpid():
            raise RuntimeError("Cannot clear inherited Repo; open a fresh Repo.")
        with self._configuration_lock:
            self._validation_evidence.clear()
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
        states=None,
        main=False,
        store=None,
        alias: str | None = None,
        deep_capture: bool = False,
        match_mode: str | None = None,
        graph_mode: str | None = None,
        report_stores: bool = False,
        source_store=None, source_stores=None, annotations=None):
    """Publish one Object graph through Repo.save_object's SavedNode boundary.

    Args:
        obj: Live Object graph root to publish.
        states: Optional partial StateSpec or rooted LocalState assignments;
            explicit selections skip capture and other distinct nodes capture once.
        repo: Optional Repo or Store authority used for publication.
        main: Whether to update the target Store's main definition after saved Node
            authority is complete.
        store: Optional explicit whole-graph target Store.
        alias: Optional Store-local name for the saved graph.
        deep_capture: Whether to serialize every owned Serializable node.
        match_mode: Optional ``"first"`` or ``"all"`` routing override.
        graph_mode: Optional ``"per-object"`` or ``"closure"`` placement
            override local to this save.
        report_stores: Whether to return a StoreReport with the SavedNode.
        source_store: Optional connected Store for explicit local-state sources.
        source_stores: Optional full-StateKey/connected-Store source overrides.
        annotations: Optional SaveAnnotations replacing supplied current object
            and state maps and copying them into a first immutable capture.

    Returns:
        The complete SavedNode, or ``(SavedNode, StoreReport)`` when requested.

    Raises:
        RepoSaveError: If bindings, reservations, local state, or saved Node publication
            cannot complete.
        StoreAuthorityError: If the selected Store rejects authoritative writes.
        TypeError: If a supplied mode has the wrong type.
        ValueError: If source selection or a supplied mode is unsupported.
        MetadataConflictError: If completed source or destination snapshot evidence
            conflicts before publication.

    Side Effects:
        Delegates one-batch state/graph/name/index publication and user retention
        to Repo.save_object. Partial failures preserve confirmed saved graphs and
        provisional grants. Buffered durable ownership fails before capture.

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
            # See Repo.save: the temporary wrapper owns this commit scope.
            selected_store = sub_repo._ensure_store(store)
            with sub_repo._retain_save_context() as context:
                result = sub_repo.save_object(
                    obj, states=states, main=main, store=selected_store, alias=alias,
                    deep_capture=deep_capture, match_mode=match_mode,
                    graph_mode=graph_mode,
                    source_store=source_store, source_stores=source_stores,
                    annotations=annotations,
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
    """Load SavedNode state or legacy plain-CDef authority through a Repo lifetime.

    Args:
        cdef: Complete SavedNode, or ``None`` to select Store main metadata.
            A legacy plain CDef uses the separate structural adapter.
        repo: Explicit Repo or Store-like source.
        cache: Cache tier for the resulting structural realization.

    Returns:
        A live Object; Node inputs restore complete selected state as in Repo.load.

    Raises:
        ValueError: If no main metadata is available, or a Node is not saved-complete.
        RepoLoadError: If the selected CDef lacks current Store authority.

    Node use freshly validates full-key sources under native leases before hooks;
    hooks may mutate eligible live candidates. Post-hook failure requires fresh
    materialization. This helper performs no capture or saved-state inference.
    """
    from dryml.runtime import materialization_admission

    with materialization_admission(operation="global_load_object"):
        with manage_repo(repo=repo) as repo:
            if cdef is None:
                cdef = repo.main_def
                if cdef is None:
                    raise ValueError("When cdef is None, the repo must have a main def, we didn't find one.")
            return repo.load_object(cdef, cache=cache)


def load_state_ref(
        state_ref, repo=None, *, reuse_live: LiveReusePolicy = "matching",
        cache: CachePolicy = "weak", source_store=None, source_stores=None) -> Object:
    """Exactly load a complete SavedNode through the selected Repo lifetime.

    Args:
        state_ref: Immutable complete SavedNode with full LocalState selections.
        repo: Explicit Repo or Store-like source containing the full closure.
        reuse_live: Policy for eligible already-live exact nodes.
        cache: Cache tier for the completed realization.
        source_store: Optional connected Store for selected full keys.
        source_stores: Optional full-StateKey-to-connected-Store overrides.

    Returns:
        A graph restored from the requested saved selections.

    Raises:
        RepoLoadError: If exact authority, including selected snapshot/local
            payload authority, or its local-state closure is missing or corrupt.
        TypeError: If ``source_store`` is not a Store, ``source_stores`` is not a
            mapping, or a mapping key/value is not a StateKey/Store.
        ValueError: If metadata is incomplete, a selected Store is disconnected,
            or a mapping names a StateKey outside the selected closure.

    Side Effects:
        May construct or reuse live Objects. On success, installs ``state_ref``
        as the returned top-level root's last-state receipt only; descendants do
        not receive projected or synthetic receipts. Every independent use freshly
        validates state under native full-key leases. No receipt skips hooks, and
        post-hook failure invalidates affected live nodes for fresh recovery.
    """
    with manage_repo(repo=repo) as sub_repo:
        return sub_repo.load_state_ref(
            state_ref, reuse_live=reuse_live, cache=cache,
            source_store=source_store, source_stores=source_stores,
        )
