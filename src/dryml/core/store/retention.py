"""Store-scoped physical grants, native key leases and affected-intent recovery.

Core supplies protect/confirm/release primitives; callers own graph association
policy. Owner intent locks serialize dispositions across the sequential handshake.
Neither owner death nor derived index loss is evidence authorizing grant removal.
"""

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import wraps
import os
import shutil
import threading

from ... import filesystem
from ...locking import FileLock
from . import authority
from .records import (_NodeRecord, StateKeyRecord, StateAvailabilityRecord,
                      StoreRecordError, _uuid_value, _digest_value, _digest,
                      _require_exact)
from .store import StoreAuthorityError, StoreCapabilityError


def _keys(keys):
    keys = tuple(keys)
    for key in keys:
        StateKeyRecord(key)
    if len(set(keys)) != len(keys):
        raise StoreRecordError("Retention keys must be distinct.")
    return tuple(sorted(keys, key=lambda k: (k.cdef_digest, k.codec_hash)))


@dataclass(frozen=True)
class RetentionResource:
    """Exact affected target, not a discovery query or a portable Store opener.

    domain is the target Store UUID; keys are full StateKeys. staging holds exact
    (allocation ID, owner UUID) pairs. Invalid values raise StoreRecordError.
    Construction performs no IO. Callers explicitly reconnect target handles.
    """

    domain: str
    keys: tuple
    staging: tuple = ()

    def __post_init__(self):
        _uuid_value(self.domain)
        if type(self.keys) is not tuple or _keys(self.keys) != self.keys:
            raise StoreRecordError("Retention resource keys must be sorted immutable values.")
        if type(self.staging) is not tuple or len(set(self.staging)) != len(self.staging):
            raise StoreRecordError("Invalid retention staging resources.")
        for allocation, owner in self.staging:
            if (type(allocation) is not str or len(allocation) != 32
                    or any(c not in "0123456789abcdef" for c in allocation)):
                raise StoreRecordError("Invalid staging allocation ID.")
            _uuid_value(owner)

    def to_data(self):
        """Return detached target/key/staging primitives; performs no IO."""
        return {"domain": self.domain, "keys": [StateKeyRecord(k).to_data() for k in self.keys],
                "staging": [list(pair) for pair in self.staging]}

    @classmethod
    def from_data(cls, data):
        """Validate closed target evidence or raise StoreRecordError, without IO."""
        _require_exact(data, {"domain", "keys", "staging"}, "retention resource")
        return cls(data["domain"], tuple(StateKeyRecord.from_data(k).key for k in data["keys"]),
                   tuple(tuple(pair) for pair in data["staging"]))


@dataclass(frozen=True)
class RetentionIntentRecord(_NodeRecord):
    """Durable affected-resource ledger and irreversible abandonment fence.

    owner_domain is the owner Store UUID, token a stable root/intent UUID,
    graph_digest the caller's saved graph reference. resources name exact Stores,
    keys and staging. disposition is prepared/adopted/abandoned/release-pending/
    released; abandoned remains true through release. released_targets records
    acknowledged targets for restart. Invalid values raise StoreRecordError;
    encoding and construction perform no IO or association policy.
    """

    schema = "dryml-retention-intent"
    fields = frozenset({"owner_domain", "token", "graph_digest", "resources", "disposition",
                        "abandoned", "released_targets"})
    owner_domain: str
    token: str
    graph_digest: str
    resources: tuple
    disposition: str = "prepared"
    abandoned: bool = False
    released_targets: tuple = ()

    def __post_init__(self):
        _uuid_value(self.owner_domain)
        _uuid_value(self.token)
        _digest_value(self.graph_digest, "retained graph")
        if (type(self.resources) is not tuple or any(type(r) is not RetentionResource for r in self.resources)
                or len({r.domain for r in self.resources}) != len(self.resources)):
            raise StoreRecordError("Invalid affected-resource ledger.")
        if self.disposition not in {"prepared", "adopted", "abandoned", "release-pending", "released"}:
            raise StoreRecordError("Invalid retention disposition.")
        if (type(self.abandoned) is not bool or (self.disposition == "abandoned" and not self.abandoned)
                or (self.abandoned and self.disposition in {"prepared", "adopted"})):
            raise StoreRecordError("Invalid abandonment fence.")
        if (type(self.released_targets) is not tuple or len(set(self.released_targets)) != len(self.released_targets)
                or not set(self.released_targets) <= {r.domain for r in self.resources}
                or (self.released_targets and self.disposition not in {"release-pending", "released"})
                or (self.disposition == "released" and set(self.released_targets) != {r.domain for r in self.resources})):
            raise StoreRecordError("Invalid release acknowledgements.")

    def _payload(self):
        return {"owner_domain": self.owner_domain, "token": self.token, "graph_digest": self.graph_digest,
                "resources": [r.to_data() for r in self.resources], "disposition": self.disposition,
                "abandoned": self.abandoned, "released_targets": list(self.released_targets)}

    @classmethod
    def _from_payload(cls, data):
        return cls(data["owner_domain"], data["token"], data["graph_digest"],
                   tuple(RetentionResource.from_data(r) for r in data["resources"]),
                   data["disposition"], data["abandoned"], tuple(data["released_targets"]))


@dataclass(frozen=True)
class RetentionGrantRecord(_NodeRecord):
    """Authoritative per-key protection physically beside the protected data.

    owner_domain/token/graph_digest identify one owner intent; keys lists its
    exact target-local full keys. A provisional or adopted grant protects equally.
    Invalid fields raise StoreRecordError. Construction/encoding has no IO.
    """

    schema = "dryml-retention-grant"
    fields = frozenset({"owner_domain", "token", "graph_digest", "keys"})
    owner_domain: str
    token: str
    graph_digest: str
    keys: tuple

    def __post_init__(self):
        _uuid_value(self.owner_domain)
        _uuid_value(self.token)
        _digest_value(self.graph_digest, "retained graph")
        if type(self.keys) is not tuple or not self.keys or _keys(self.keys) != self.keys:
            raise StoreRecordError("Grant requires sorted full keys.")

    @property
    def address(self):
        """Return the stable owner/root digest used at each protected key (str)."""
        return _digest(self.schema + "/v1", [self.owner_domain, self.token])

    def _payload(self):
        return {"owner_domain": self.owner_domain, "token": self.token, "graph_digest": self.graph_digest,
                "keys": [StateKeyRecord(k).to_data() for k in self.keys]}

    @classmethod
    def _from_payload(cls, data):
        return cls(data["owner_domain"], data["token"], data["graph_digest"],
                   tuple(StateKeyRecord.from_data(k).key for k in data["keys"]))


@dataclass(frozen=True)
class RetentionPendingRecord(_NodeRecord):
    """Exact unsettled intent tokens; derived indexes never authorize release.

    tokens is a distinct immutable UUID tuple. Invalid input raises
    StoreRecordError; construction/encoding performs no IO. Completed/adopted
    history need not be scanned to retry affected prepared or release work.
    """

    schema = "dryml-retention-pending"
    fields = frozenset({"tokens"})
    tokens: tuple = ()

    def __post_init__(self):
        if type(self.tokens) is not tuple or len(set(self.tokens)) != len(self.tokens):
            raise StoreRecordError("Invalid pending retention tokens.")
        for token in self.tokens:
            _uuid_value(token)

    def _payload(self):
        return {"tokens": list(self.tokens)}

    @classmethod
    def _from_payload(cls, data):
        return cls(tuple(data["tokens"]))


@dataclass(frozen=True)
class StateRemovalRecord(_NodeRecord):
    """Withdrawal authority for exactly one availability incarnation.

    key is a full StateKey, incarnation its withdrawn UUID, status withdrawn,
    removed or obsolete. History remains after removal. Invalid input raises
    StoreRecordError. Construction/encoding has no filesystem side effects.
    """

    schema = "dryml-state-removal"
    fields = frozenset({"key", "incarnation", "status"})
    key: object
    incarnation: str
    status: str = "withdrawn"

    def __post_init__(self):
        StateKeyRecord(self.key)
        _uuid_value(self.incarnation)
        if self.status not in {"withdrawn", "removed", "obsolete"}:
            raise StoreRecordError("Invalid removal disposition.")

    def _payload(self):
        return {"key": StateKeyRecord(self.key).to_data(), "incarnation": self.incarnation, "status": self.status}

    @classmethod
    def _from_payload(cls, data):
        return cls(StateKeyRecord.from_data(data["key"]).key, data["incarnation"], data["status"])


@dataclass(frozen=True)
class StateRemovalOutcome:
    """One immediate deletion result, with no queued request for retained/in-use.

    status is retained/in-use/unavailable/removed/obsolete/cleanup-pending;
    incarnation identifies cleanup authority or is None. cleanup_error holds only
    a physical garbage OSError; journal/index/control errors propagate instead.
    Construction has no side effects and promises no federated transaction.
    """

    status: str
    incarnation: str | None = None
    cleanup_error: OSError | None = None


@dataclass(frozen=True)
class StateInventoryEntry:
    """Detached federated inventory row; construction performs no IO.

    domain is a Store authority UUID and descriptor its StateDescriptor. The
    descriptor distinguishes known history from current availability, not a lease.
    No Store handle or private payload is retained; readers own field validation.
    """

    domain: str
    descriptor: object


@dataclass(frozen=True)
class StateCleanupOutcome:
    """One Store/full-key cleanup result, without a global transaction claim.

    domain/key identify the exact target. phase is removal or release. status is a Core removal status or
    failed/unattempted. outcome is optional StateRemovalOutcome (including exact
    incarnation retry evidence); error is an optional authority/IO exception.
    Construction performs no IO or validation. Failed/unattempted outcomes never
    certify withdrawal, grant release or enabled-index completion.
    """

    domain: str
    key: object
    status: str
    outcome: StateRemovalOutcome | None = None
    error: BaseException | None = None
    phase: str = "removal"


class RetentionReleasePending(StoreAuthorityError):
    """An explicit root retirement has disconnected/unconfirmed affected targets.

    intent is the current RetentionIntentRecord; outcomes is the immutable tuple of
    per-target cleanup evidence. Both are exposed as retention_intent and
    cleanup_outcomes. Construction performs no IO and does not authorize deletion.
    """

    def __init__(self, intent, outcomes):
        """Bind pending intent and per-target outcomes without IO; return None."""
        super().__init__("Retained root release remains pending; reconnect its exact targets.")
        self.retention_intent, self.cleanup_outcomes = intent, outcomes


def reclaim_resources(resources, stores, *, blocked=(), before=None):
    """Reclaim only named full keys, continuing independently safe targets.

    resources is RetentionResource values, stores connected handles, blocked
    domains lack prerequisite confirmation, and optional before() rechecks caller
    ownership before each attempt. Returns tuple[StateCleanupOutcome]. Disconnected
    or blocked targets are unattempted. Ordinary deletion failures are reported;
    interruption and before() failures propagate, with no further attempts. No
    grant release, inventory scan or lease upgrade is performed.
    """
    connected = RetentionAuthority._targets(stores)
    result = []
    targets = tuple((resource.domain, key) for resource in resources for key in resource.keys)
    for index, (domain, key) in enumerate(targets):
        target = connected.get(domain)
        if target is None or domain in blocked:
            result.append(StateCleanupOutcome(domain, key, "unattempted"))
            continue
        try:
            if before is not None:
                before()
        except BaseException as error:
            error.cleanup_outcomes = tuple(result) + tuple(
                StateCleanupOutcome(d, k, "unattempted") for d, k in targets[index:])
            raise
        try:
            outcome = target.delete_state(key)
        except Exception as error:
            result.append(StateCleanupOutcome(domain, key, "failed", error=error))
        except BaseException as error:
            # Interrupted withdrawal may already be visible. Do not label that
            # target removed; remaining independent work was never attempted.
            error.cleanup_outcomes = tuple(result) + (StateCleanupOutcome(domain, key, "failed", error=error),) + tuple(
                StateCleanupOutcome(d, k, "unattempted") for d, k in targets[index + 1:])
            raise
        else:
            result.append(StateCleanupOutcome(domain, key, outcome.status, outcome))
    return tuple(result)


_HELD = ContextVar("dryml_state_key_leases", default=())
_WRITERS = ContextVar("dryml_state_writer_fences", default=())


def _execution_owner():
    import asyncio
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return os.getpid(), threading.get_ident(), task


def _held():
    value = _HELD.get()
    return value[1] if value and value[0] == _execution_owner() else ()


def _writers():
    value = _WRITERS.get()
    return value[1] if value and value[0] == _execution_owner() else ()


def _key_address(store, key):
    from ...paths import local_path_key
    return ((store.retention_domain, local_path_key(store.base_dir)), key.cdef_digest, key.codec_hash)


@contextmanager
def writer_scope(store):
    """Track a writer scope to refuse inverse key acquisition, without doing IO."""
    token = _WRITERS.set((_execution_owner(), _writers() + (store.retention_domain,)))
    try:
        yield
    finally:
        _WRITERS.reset(token)


@contextmanager
def state_leases(targets, *, exclusive=False, blocking=True):
    """Acquire native full-key leases in global domain/CDef/hash order.

    targets is an iterable of (Store, StateKey). Yields bool, False only for
    nonblocking contention. Nested same-key shared use reuses protection; growing
    a retained set raises StoreAuthorityError and requires release/reacquire and
    revalidation. In-place upgrades never occur. No writer fence may be held on
    entry. Native unsupported/failure errors propagate, never fake protection.
    Acquiring/releasing must occur on the same thread.
    """
    if _writers():
        raise StoreAuthorityError("Acquire StateKey leases before Store writer fences.")
    ordered = {}
    for store, key in targets:
        store._assert_state_owner_open()
        StateKeyRecord(key)
        ordered[_key_address(store, key)] = (store, key)
    held = dict(_held())
    if held and any(key not in held for key in ordered):
        raise StoreAuthorityError("Growing StateKey set requires release/reacquire in global order and revalidation.")
    stack, addition = ExitStack(), []
    try:
        for store in {id(store): store for store, _ in ordered.values()}.values():
            lifetime = getattr(store, "_state_lease_lifetime", None)
            if lifetime is not None:
                stack.enter_context(lifetime())
        for address, (store, key) in sorted(ordered.items()):
            if address in held:
                if exclusive and not held[address]:
                    if blocking:
                        raise StoreAuthorityError("Never upgrade an owned shared StateKey lease.")
                    yield False
                    return
                continue
            lock = FileLock(store._state_lease_path(key), shared=not exclusive)
            if not lock.acquire(blocking=blocking):
                yield False
                return
            stack.callback(lock.release)
            addition.append((address, exclusive))
        for store, _ in ordered.values():
            store._assert_state_owner_open()
        token = _HELD.set((_execution_owner(), _held() + tuple(addition)))
        try:
            yield True
        finally:
            _HELD.reset(token)
    finally:
        stack.close()


def _variant(store, key):
    return os.path.dirname(store.state_data_path(key))


def _grant_path(store, key, grant):
    return os.path.join(_variant(store, key), "grants", grant.address + ".record")


def grants(store, key):
    """Read exact authoritative per-key grants under detached cut validation.

    Returns tuple of validated RetentionGrantRecords. Missing reverse indexes are
    irrelevant. Malformed or misplaced records fail closed with StoreAuthorityError.
    No owner reconnection, payload IO or grant release occurs.
    """
    with store._metadata_read() as captured:
        relative = store._relative(os.path.join(_variant(store, key), "grants"))
        result = []
        for name in captured.directory(relative):
            if name.startswith(".dryml-"):
                continue
            path = os.path.join(store.base_dir, relative, name)
            grant = store._read_file(path, RetentionGrantRecord)
            if grant is None or key not in grant.keys or path != _grant_path(store, key, grant):
                raise StoreAuthorityError("Missing or mismatched authoritative state grant.")
            result.append(grant)
        return tuple(result)


class RetentionAuthority:
    """Owner-side handshake primitives over a durable Store resource ledger.

    store is the owner DirStore, not necessarily a payload target. All transitions
    are journaled/indexed. Buffered archives refuse durable ownership. Methods
    raise authority/capability/IO errors on ambiguity; only explicit abandonment
    or retirement permits release. No managed association policy runs here.
    """

    def __init__(self, store):
        self.store = store

    def _preflight(self):
        self.store.publication_capabilities.require_durable("retention owner")
        self.store.preflight_publication("retention owner", record_family="dryml-node-graph/v1")

    def _path(self, token):
        _uuid_value(token)
        return os.path.join(self.store.base_dir, "retention", "intents", token + ".record")

    @contextmanager
    def _fence(self, token):
        if _held() or _writers():
            raise StoreAuthorityError("Retention owner intent precedes key leases and Store writers.")
        self._preflight()
        self._path(token)
        with FileLock(os.path.join(self.store.base_dir, ".retention-locks", token + ".lock")):
            yield

    def read(self, token):
        """Return exact RetentionIntentRecord or None, with fresh owner authority.

        token is a UUID. Malformed/misplaced authority raises StoreAuthorityError;
        disconnected/closed owners fail. Reads no target payload or operation history.
        """
        self.store._assert_state_owner_open()
        with self.store._metadata_read():
            value = self.store._read_file(self._path(token), RetentionIntentRecord)
            if value is not None and (value.owner_domain != self.store.retention_domain or value.token != token):
                raise StoreAuthorityError("Retention intent owner/token mismatch.")
            return value

    def pending(self, *, after=None, limit=100):
        """Return a bounded sorted page of unsettled UUIDs, without history scan.

        Prepared/abandoned/release-pending work remains named until settled.
        after is an optional UUID cursor; limit is a nonnegative int (default 100).
        A zero limit performs no source work. In-flight journal resources augment
        the ledger across a partial ledger/head update. This reconciliation page
        is not a fixed-membership query snapshot; retry each token under its fence.
        Corrupt or missing named authority raises StoreAuthorityError; no release
        is inferred from absence. This is an explicit affected-ledger read.
        """
        if type(limit) is not int or limit < 0:
            raise TypeError("Retention page limit must be a nonnegative integer.")
        if after is not None:
            _uuid_value(after)
        if limit == 0:
            return ()
        with self.store._metadata_read():
            record = self.store._read_file(os.path.join(self.store.base_dir, "retention", "pending.record"), RetentionPendingRecord)
            tokens = set(() if record is None else record.tokens)
            for mutation in self.store.pending_mutations():
                for change in mutation.intent["transitions"]:
                    if change["path"].startswith("retention/intents/"):
                        token = change["path"].split("/")[-1][:-7]
                        intent = self.read(token)
                        if intent is None:
                            raise StoreAuthorityError("Pending retention intent needs exact journal recovery.")
                        if intent.disposition not in {"adopted", "released"}:
                            tokens.add(token)
            tokens = tuple(t for t in sorted(tokens) if after is None or t > after)[:limit]
            if any(self.read(token) is None for token in tokens):
                raise StoreAuthorityError("Pending retention intent is missing.")
            return tokens

    def _write(self, value, expected):
        settled = value.disposition in {"adopted", "released"}
        pending_path = os.path.join(self.store.base_dir, "retention", "pending.record")
        def prepare():
            if self.read(value.token) != expected:
                raise StoreAuthorityError("Retention owner disposition changed.")
            pending = self.store._read_file(pending_path, RetentionPendingRecord) or RetentionPendingRecord()
            tokens = pending.tokens
            if (not settled or expected != value) and value.token not in tokens:
                tokens += (value.token,)
            # Name affected resources before changing disposition. Settled tokens
            # leave this head only after disposition AND its index ack complete.
            return [self.store._prepared_file(pending_path, RetentionPendingRecord(tokens), mutable=True),
                    self.store._prepared_file(self._path(value.token), value, mutable=True)], ("retention",), value
        result = self.store._prepare_and_apply(prepare)
        if settled:
            def prune():
                if self.read(value.token) != value:
                    raise StoreAuthorityError("Retention disposition changed before pending-ledger acknowledgement.")
                pending = self.store._read_file(pending_path, RetentionPendingRecord) or RetentionPendingRecord()
                tokens = tuple(t for t in pending.tokens if t != value.token)
                return [self.store._prepared_file(pending_path, RetentionPendingRecord(tokens), mutable=True)], ("retention",), value
            result = self.store._prepare_and_apply(prune)
        return result

    def prepare(self, token, graph_digest, targets, *, staging=()):
        """Persist exact resources before protection; return a prepared intent.

        targets is (Store, iterable[StateKey]) pairs; staging is (Store, owned
        staging handle) pairs. UUID token and digest graph reference are stable
        caller intent. Exact retries are idempotent; token reuse with different
        facts raises StoreAuthorityError. No target grant or association is implied.
        """
        with self._fence(token):
            resources = {}
            for target, keys in targets:
                target.publication_capabilities.require_durable("physical retention grants")
                domain = target.retention_domain
                if domain in resources:
                    raise StoreAuthorityError("Duplicate retention target domain.")
                resources[domain] = RetentionResource(domain, _keys(keys))
            for target, handle in staging:
                domain = target.retention_domain
                resource = resources.get(domain, RetentionResource(domain, ()))
                target.publication_capabilities.require_durable("retention staging resources")
                allocation = target._require_staging_owner(handle)
                owner = target._staging_owner(allocation)
                resources[domain] = replace(resource, staging=resource.staging + ((allocation, owner),))
            value = RetentionIntentRecord(self.store.retention_domain, token, graph_digest,
                                          tuple(resources[k] for k in sorted(resources)))
            old = self.read(token)
            if old is not None:
                if (old.owner_domain, old.token, old.graph_digest, old.resources) != (
                        value.owner_domain, value.token, value.graph_digest, value.resources):
                    raise StoreAuthorityError("Retention token cannot name a new attempt.")
                return self._write(old, old)
            return self._write(value, None)

    @staticmethod
    def _targets(stores):
        result = {}
        for store in stores:
            domain = store.retention_domain
            if domain in result and os.path.realpath(result[domain].base_dir) != os.path.realpath(store.base_dir):
                raise StoreAuthorityError("Ambiguous target authority domain.")
            result[domain] = store
        return result

    def _release_resource(self, intent, resource, target):
        target._assert_state_owner_open()
        target.publication_capabilities.require_durable("release state protection")
        if resource.keys:
            grant = self._grant(intent, resource)
            with state_leases((target, key) for key in resource.keys):
                target.preflight_publication("release state protection", record_family="dryml-node-graph/v1")
                def prepare():
                    changes = []
                    for key in resource.keys:
                        path = _grant_path(target, key, grant)
                        old = target._read_file(path, RetentionGrantRecord)
                        if old is not None and old != grant:
                            raise StoreAuthorityError("Grant release cannot remove another root's authority.")
                        changes.append(target._prepared_file(path, None, mutable=True))
                    return changes, ("retention",), None
                target._prepare_and_apply(prepare)
        return all(target._discard_abandoned_staging(allocation, owner) for allocation, owner in resource.staging)

    @staticmethod
    def _grant(intent, resource):
        return RetentionGrantRecord(intent.owner_domain, intent.token, intent.graph_digest, resource.keys)

    def protect(self, token, stores):
        """Acquire provisional grants on explicitly connected affected targets.

        stores supplies available handles, permitting incremental protection before
        a later target capture. Returns current intent. Only prepared tokens admit
        new grants; abandoned/retired tokens cannot be resurrected. Missing targets
        stay uncertain. Shared leases precede writer CAS/availability revalidation.
        IO/index errors preserve owner intent and any confirmed physical grants.
        """
        with self._fence(token):
            intent = self.read(token)
            if intent is None or intent.disposition != "prepared":
                raise StoreAuthorityError("Only prepared tokens may acquire grants; abandoned/retired tokens are fenced.")
            connected = self._targets(stores)
            for resource in intent.resources:
                target = connected.get(resource.domain)
                if target is None or not resource.keys:
                    continue
                grant = self._grant(intent, resource)
                with state_leases((target, key) for key in resource.keys):
                    target.preflight_publication("protect state", record_family="dryml-node-graph/v1")
                    def prepare():
                        changes = []
                        for key in resource.keys:
                            available = target._read_file(os.path.join(_variant(target, key), "availability.record"), StateAvailabilityRecord)
                            if available is None or available.key != key or available.status != "published" or not os.path.isdir(target.state_data_path(key)):
                                raise StoreAuthorityError("Cannot protect unavailable state.")
                            changes.append(target._prepared_file(_grant_path(target, key, grant), grant))
                        return changes, ("retention",), None
                    target._prepare_and_apply(prepare)
            return intent

    def adopt(self, token, stores):
        """Confirm all named grants before caller association; return adopted intent.

        stores must reconnect every affected target. Adoption is an evidence
        primitive, not a managed control association. Missing/disconnected targets,
        absent grants and abandonment fail closed; no old protection is released.
        """
        with self._fence(token):
            intent = self.read(token)
            if intent is None or intent.abandoned or intent.disposition not in {"prepared", "adopted"}:
                raise StoreAuthorityError("Cannot adopt an abandoned/retired retention intent.")
            connected = self._targets(stores)
            for resource in intent.resources:
                target = connected.get(resource.domain)
                if target is None:
                    raise StoreAuthorityError("Retention target is disconnected; adoption remains uncertain.")
                if resource.keys:
                    grant = self._grant(intent, resource)
                    with state_leases((target, key) for key in resource.keys):
                        with target._metadata_read():
                            for key in resource.keys:
                                if target._read_file(_grant_path(target, key, grant), RetentionGrantRecord) != grant:
                                    raise StoreAuthorityError("Required retention target grant is missing.")
            return self._write(replace(intent, disposition="adopted"), intent)

    def abandon(self, token):
        """Persist an irreversible abandonment tombstone, returning its intent.

        Only prepared/already-abandoned tokens are eligible. Explicit owner-fenced
        evidence is required: death, timeouts and missing indexes never call this
        automatically. This method does not remove grants or active staging.
        """
        with self._fence(token):
            intent = self.read(token)
            if intent is None or intent.disposition not in {"prepared", "abandoned"}:
                raise StoreAuthorityError("Only prepared retention can be abandoned.")
            return self._write(replace(intent, disposition="abandoned", abandoned=True), intent)

    def release(self, token, stores):
        """Authorize retirement, remove exact owned grants, and persist target acks.

        Adopted roots may be explicitly retired; provisional roots require prior
        abandonment. stores reconnects target handles. Disconnected/live-staging
        targets remain release-pending; IO/index errors propagate with that ledger
        intact; independent targets continue after a target failure. The raised
        first error carries retention_failures and retention_intent evidence.
        Owner/control failure or interruption stops dependent work. Exact retries
        skip acknowledged targets and never scan history.
        released is recorded only after all target grants/staging confirm removal.
        No payload deletion or replacement association is performed here.
        """
        with self._fence(token):
            intent = self.read(token)
            if intent is None or intent.disposition not in {"adopted", "abandoned", "release-pending", "released"}:
                raise StoreAuthorityError("Provisional release requires owner-fenced abandonment.")
            if intent.disposition == "released":
                return self._write(intent, intent)
            if intent.disposition != "release-pending":
                intent = self._write(replace(intent, disposition="release-pending"), intent)
            connected = self._targets(stores)
            failures = []
            for resource in intent.resources:
                if resource.domain in intent.released_targets:
                    continue
                target = connected.get(resource.domain)
                if target is None:
                    continue
                try:
                    done = self._release_resource(intent, resource, target)
                except Exception as error:
                    failures.append((resource.domain, error))
                    continue
                if not done:
                    continue
                intent = self._write(replace(intent, released_targets=intent.released_targets + (resource.domain,)), intent)
            if len(intent.released_targets) == len(intent.resources):
                intent = self._write(replace(intent, disposition="released"), intent)
            if failures:
                error = failures[0][1]
                error.retention_intent, error.retention_failures = intent, tuple(failures)
                raise error
            return intent


def _removal_path(store, key, incarnation):
    _uuid_value(incarnation)
    return os.path.join(_variant(store, key), "removals", incarnation + ".record")


def _retired_path(store, key, incarnation):
    _uuid_value(incarnation)
    return os.path.join(_variant(store, key), ".dryml-retired-" + incarnation)


def publication_pending(store, key):
    """Return whether exact pending journal evidence may publish this full key.

    Checks only pending affected transitions, never historical operations or
    reverse counts. Unknown/unreadable evidence raises rather than proves safety.
    """
    prefix = store._relative(_variant(store, key))
    for record in store.pending_mutations():
        for change in record.intent["transitions"]:
            if change["path"] == prefix and change["kind"] == "variant":
                return True
            if change["path"] == prefix + "/data":
                return True
            if change["path"] == prefix + "/availability.record" and change["after"] is not None:
                from .records import _decode_authority_bytes
                value = StateAvailabilityRecord.from_bytes(_decode_authority_bytes(change["after"]))
                if value.status != "withdrawn":
                    return True
    return False


def remove_private_data(path):
    """Remove an already withdrawn/detached directory; preserve metadata siblings.

    path is an exact owned data/remnant path. Missing data is idempotent. OSError
    denotes physical garbage failure, the only successful-with-cleanup exception.
    Callers retain exclusive key protection and verify incarnation first.
    """
    if os.path.exists(path):
        shutil.rmtree(path)


def delete_state(store, key, *, incarnation=None):
    """Try immediate full-key reclamation or retry one exact withdrawal UUID.

    Returns StateRemovalOutcome. Exclusive native acquisition is nonblocking and
    precedes writer fences. Grants/pending publication prohibit withdrawal. The
    lease spans journal/index withdrawal, physical deletion and completion. A newer
    incarnation obsoletes cleanup of the public data path. Metadata is preserved.
    All non-garbage failures propagate with pending evidence, never success.
    """
    store._assert_state_owner_open()
    store.publication_capabilities.require_writable("delete private state")
    StateKeyRecord(key)
    if _key_address(store, key) in dict(_held()):
        return StateRemovalOutcome("in-use", incarnation)
    with store.state_leases((key,), exclusive=True, blocking=False) as acquired:
        if not acquired:
            return StateRemovalOutcome("in-use", incarnation)
        store.preflight_publication("delete private state", record_family="dryml-node-graph/v1")
        def prepare():
            available = store._read_file(os.path.join(_variant(store, key), "availability.record"), StateAvailabilityRecord)
            if available is None:
                if incarnation is not None or os.path.lexists(_variant(store, key)):
                    raise StoreAuthorityError("Removal retry has no availability authority.")
                return [], ("availability",), StateRemovalOutcome("unavailable")
            if available.key != key:
                raise StoreAuthorityError("Availability names another full key.")
            if incarnation is not None:
                old = store._read_file(_removal_path(store, key, incarnation), StateRemovalRecord)
                if old is None or old.key != key or old.incarnation != incarnation:
                    raise StoreAuthorityError("Removal retry requires exact withdrawal evidence.")
                if available.incarnation != incarnation:
                    return [], ("availability",), StateRemovalOutcome("obsolete", incarnation)
            if grants(store, key):
                return [], ("availability",), StateRemovalOutcome("retained", available.incarnation)
            if publication_pending(store, key):
                return [], ("availability",), StateRemovalOutcome("in-use", available.incarnation)
            from .records import StateVariantManifest
            manifest = store._read_file(os.path.join(_variant(store, key), "manifest.record"), StateVariantManifest)
            if manifest is None or manifest.key != key or store._read_creation_record(key.cdef_digest) is None:
                raise StoreAuthorityError("Removal requires exact manifest/header history.")
            removal = store._read_file(_removal_path(store, key, available.incarnation), StateRemovalRecord)
            if removal is None:
                removal = StateRemovalRecord(key, available.incarnation)
            elif removal.key != key or removal.incarnation != available.incarnation:
                raise StoreAuthorityError("Removal intent names another incarnation.")
            changes = [store._prepared_file(_removal_path(store, key, available.incarnation), removal, mutable=True),
                       store._prepared_file(os.path.join(_variant(store, key), "availability.record"),
                                            replace(available, status="withdrawn"), mutable=True)]
            return changes, ("availability",), StateRemovalOutcome("withdrawn", available.incarnation)
        outcome = store._prepare_and_apply(prepare)
        if outcome.status not in {"withdrawn", "obsolete"}:
            return outcome
        try:
            if outcome.status == "withdrawn":
                remove_private_data(store.state_data_path(key))
            remove_private_data(_retired_path(store, key, outcome.incarnation))
        except OSError as error:
            return StateRemovalOutcome("cleanup-pending", outcome.incarnation, error)
        status = "obsolete" if outcome.status == "obsolete" else "removed"
        def complete():
            path = _removal_path(store, key, outcome.incarnation)
            old = store._read_file(path, StateRemovalRecord)
            if old is None or old.key != key or old.incarnation != outcome.incarnation:
                raise StoreAuthorityError("Removal completion lost its exact authority.")
            return [store._prepared_file(path, replace(old, status=status), mutable=True)], ("availability",), None
        store._prepare_and_apply(complete)
        return StateRemovalOutcome(status, outcome.incarnation)


def detach_withdrawn_data(store, key):
    """Detach leftover bytes during a separate exclusive republication phase.

    The caller must hold the exclusive key lease. Checks exact unavailable
    incarnation, grants and pending publication under a writer fence. Moves only
    old private data to an incarnation-named remnant; preserves headers/manifests.
    Refuses ambiguous evidence. Native move/barrier errors propagate.
    """
    address = _key_address(store, key)
    if not dict(_held()).get(address):
        raise StoreAuthorityError("Withdrawn recovery requires an independent exclusive key lease.")
    availability_path = os.path.join(_variant(store, key), "availability.record")
    def prepare():
        available = store._read_file(availability_path, StateAvailabilityRecord)
        if available is None or available.status == "published":
            return [], ("availability",), None
        if available.key != key or publication_pending(store, key) or grants(store, key):
            raise StoreAuthorityError("Pending publication or uncertain grants prevent withdrawn recovery.")
        path = _removal_path(store, key, available.incarnation)
        removal = store._read_file(path, StateRemovalRecord) or StateRemovalRecord(key, available.incarnation)
        if removal.key != key or removal.incarnation != available.incarnation:
            raise StoreAuthorityError("Recovery removal evidence names another incarnation.")
        return [store._prepared_file(path, removal, mutable=True),
                store._prepared_file(availability_path, replace(available, status="withdrawn"), mutable=True)], ("availability",), available.incarnation
    incarnation = store._prepare_and_apply(prepare)
    if incarnation is None:
        return
    with store.writer_lock():
        available = store._read_file(availability_path, StateAvailabilityRecord)
        if (available is None or available.key != key or available.status != "withdrawn"
                or available.incarnation != incarnation or publication_pending(store, key)):
            raise StoreAuthorityError("Pending publication prevents withdrawn recovery.")
        # Inspect authoritative grants directly under this already-exclusive cut.
        directory = os.path.join(_variant(store, key), "grants")
        if os.path.isdir(directory) and any(not n.startswith(".dryml-") for n in os.listdir(directory)):
            raise StoreAuthorityError("Uncertain grants prevent withdrawn recovery.")
        data = store.state_data_path(key)
        if os.path.exists(data):
            store._filesystem(filesystem.publish_directory, data, _retired_path(store, key, available.incarnation))


def settle_interrupted_publication(store, key):
    """In exclusive recovery, incorporate classified interrupted publication cuts.

    A fresh complete owned stage will supersede unavailable partial data. No live
    publisher can hold this key. Classification acknowledges actual authority,
    never the failed publication's intended success; original journal evidence
    remains historical. Enabled indexes incorporate that exact cut before ack.
    Unknown/ambiguous evidence fails closed. No graph or capture hooks run.
    """
    address = _key_address(store, key)
    if not dict(_held()).get(address):
        raise StoreAuthorityError("Interrupted publication recovery requires an exclusive key lease.")
    store.recover_authority()
    index = store.open_query_index()
    if index is not None:
        index.maintain_store()
    else:
        prefix = store._relative(_variant(store, key))
        for record in store.pending_mutations():
            data = record.intent
            if any(row["path"] == prefix or row["path"].startswith(prefix + "/") for row in data["transitions"]):
                if data["status"] == "pending":
                    raise StoreAuthorityError("Publication classification remains pending.")
                store.acknowledge_mutation(data["mutation_id"], data["epoch"])


def _transition_keys(changes):
    from ..state_values import StateKey
    result = set()
    for change in changes:
        parts = change["path"].split("/")
        if parts[0] == "states" and len(parts) >= 4:
            if parts[-1] != "def.pkl":
                result.add(StateKey(parts[2], parts[3]))
    return _keys(result)


def _protect_prepared(function):
    @wraps(function)
    def protected(store, facts):
        if type(facts) is not authority.PreparedMutation:
            return function(store, facts)
        for change in facts.changes:
            if (change.path.startswith("retention/") or "/grants/" in change.path
                    or "/removals/" in change.path or change.path.endswith("/availability.record")):
                raise StoreCapabilityError("Retention/availability transitions require their owning Store lifecycle primitive.")
        changes = [change.to_data() for change in facts.changes]
        keys = _transition_keys(changes)
        with store._replay_staging(changes):
            with store.state_leases(keys):
                return function(store, facts)
    return protected


def _protect_replay(function):
    @wraps(function)
    def protected(store, mutation_id):
        from .records import MutationIntentRecord
        from contextlib import ExitStack
        _uuid_value(mutation_id)
        record = store._read_file(os.path.join(store.base_dir, "mutations", mutation_id + ".record"), MutationIntentRecord)
        if record is None:
            record = store._read_file(os.path.join(store.base_dir, "mutation-prepare.record"), MutationIntentRecord)
            if record is not None and record.intent["mutation_id"] != mutation_id:
                record = None
        changes = () if record is None else record.intent["transitions"]
        if any("/grants/" in change["path"] for change in changes):
            raise StoreCapabilityError("Grant replay requires a freshly connected owner protect/release handshake.")
        tokens = {change["path"].split("/")[-1][:-7] for change in changes
                  if change["path"].startswith("retention/intents/")}
        if len(tokens) > 1:
            raise StoreCapabilityError("Replay cannot nest independent retention owner intents.")
        with ExitStack() as stack:
            for token in tokens:
                stack.enter_context(store.retention._fence(token))
            stack.enter_context(store._replay_staging(changes))
            # Recovery may reinstall unavailable/partial data or withdraw it. A
            # separate nonblocking exclusive phase fences both cases without an
            # upgrade, before any writer/SQL work in the existing replay path.
            keys = _transition_keys(changes)
            if keys and not stack.enter_context(store.state_leases(keys, exclusive=True, blocking=False)):
                raise StoreAuthorityError("Mutation replay is in-use; retry after active leases end.")
            return function(store, mutation_id)
    return protected
