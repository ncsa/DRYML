"""Store-owned prepared byte CAS, durable intent, classification and exact ack.

Graph validation belongs to detached preparation. This module only installs
prepared bytes/complete children and records the actual resulting authority cut;
recovery neither repeats application work nor rolls back published authority.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
import os
import stat
from uuid import uuid4

from ... import filesystem
from .records import (AuthorityRecord, MutationIntentRecord, _json_bytes,
                      _encode_authority_bytes as encode, _decode_authority_bytes as decode,
                      _MAX_RECORD_BYTES, _normalized_file_path, _uuid_value)
from .store import StoreAuthorityError

__all__ = ["PreparedChange", "PreparedMutation", "StoreCutChanged", "StoreMutationError"]


class StoreCutChanged(StoreAuthorityError):
    """Detached facts lost their cut/byte witnesses; explicitly retry preparation."""


class StoreMutationError(StoreAuthorityError, OSError):
    """Failed mutation with inspectable confirmed references and repair evidence.

    mutation_id, cut, confirmed_refs and index_disposition describe actual
    persisted evidence, not rollback or complete publication. recovery_attempt
    identifies the attempted classification/index-maintenance recovery. The original IO or
    classification failure is retained as the exception cause.
    """

    def __init__(self, message, mutation_id, cut, confirmed_refs=()):
        super().__init__(f"{message}; mutation={mutation_id}; cut={cut}; index=repair_pending")
        self.mutation_id = mutation_id
        self.cut = cut
        self.confirmed_refs = tuple(confirmed_refs)
        self.index_disposition = "repair_pending"
        self.recovery_attempt = "authority-classification"


@dataclass(frozen=True)
class PreparedChange:
    """One bounded record CAS or native immutable child install.

    path is Store-relative; before/after are bytes or None; kind is file, variant
    or data. staging names an owned relative complete directory for a child
    install. Values are validated by PreparedMutation before any writer effects.
    """

    path: str
    before: bytes | None
    after: bytes | None
    staging: str | None = None
    kind: str = "file"

    def to_data(self):
        """Return detached exact transition primitives for the persisted intent."""
        return {"path": self.path, "before": encode(self.before), "after": encode(self.after),
                "staging": self.staging, "kind": self.kind}


@dataclass(frozen=True)
class PreparedMutation:
    """Detached immutable facts supplied by Store preparation (and later Repo).

    cut is AuthorityRecord; witnesses are (relative path, bytes-or-None) pairs;
    directories are (relative path, sorted child-name tuple) witnesses; changes
    and scopes are tuples. No callable/graph is accepted. Projection replay uses
    the exact resulting record bytes and full state metadata from these changes.
    owner is the opaque preparing Store handle's token; another handle refuses it.
    Applying stale facts raises StoreCutChanged before effects.
    """

    cut: AuthorityRecord
    witnesses: tuple
    directories: tuple
    changes: tuple[PreparedChange, ...]
    scopes: tuple[str, ...]
    owner: object = field(repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.cut, AuthorityRecord) or any(type(v) is not tuple for v in
                (self.witnesses, self.directories, self.changes, self.scopes)):
            raise TypeError("Prepared authority facts must be immutable tuples.")
        for path, payload in self.witnesses:
            _normalized_file_path(path)
            if payload is not None and type(payload) is not bytes:
                raise TypeError("Witnesses require immutable bytes.")
        for path, names in self.directories:
            _normalized_file_path(path)
            if type(names) is not tuple or any(type(n) is not str for n in names):
                raise TypeError("Directory witnesses require immutable names.")
        if any(type(change) is not PreparedChange for change in self.changes):
            raise TypeError("Prepared changes cannot contain callbacks.")
        # The same closed decoder validates externally supplied prepared facts.
        _intent(self, str(uuid4())).to_bytes()


def _intent(batch, identity):
    return MutationIntentRecord(_json_bytes({
        "mutation_id": identity, "store_uuid": batch.cut.store_uuid, "base_epoch": batch.cut.epoch,
        "scopes": list(batch.scopes), "transitions": [c.to_data() for c in batch.changes],
        "confirmed": [], "projection": [], "epoch": None, "status": "pending"}))


_READ = ContextVar("dryml_store_detached_read", default=None)


def raw(store, relative):
    """Read bounded regular-file bytes or None, without graph decoding."""
    _normalized_file_path(relative)
    path = os.path.join(store.base_dir, relative)
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        observe(store, path, None)
        return None
    if not stat.S_ISREG(mode):
        raise StoreAuthorityError("Authority byte witness is not a regular file.")
    with open(path, "rb") as source:
        result = source.read(_MAX_RECORD_BYTES + 1)
    if len(result) > _MAX_RECORD_BYTES:
        raise StoreAuthorityError("Oversized authority byte witness.")
    observe(store, path, result)
    return result


class _DetachedRead:
    def __init__(self, store, cut):
        self.store, self.cut = store, cut
        self.files, self.directories = {}, {}
        self.node_views = {}
        self.node_view_bytes = 0

    def file(self, path, payload):
        relative = os.path.relpath(path, self.store.base_dir).replace(os.sep, "/")
        if relative in self.files and self.files[relative] != payload:
            raise StoreCutChanged("Authority changed during detached capture.")
        self.files[relative] = payload

    def directory(self, relative):
        path = os.path.join(self.store.base_dir, relative)
        names = tuple(sorted(os.listdir(path))) if os.path.exists(path) else ()
        if relative in self.directories and self.directories[relative] != names:
            raise StoreCutChanged("Authority directory changed during detached capture.")
        self.directories[relative] = names
        return names

    def validate(self):
        self.store._assert_state_owner_open()
        if read_cut(self.store) != self.cut:
            raise StoreCutChanged("Store cut changed; retry detached preparation.")
        for path, value in self.files.items():
            if raw(self.store, path) != value:
                raise StoreCutChanged("Authority byte witness changed; retry preparation.")
        for path, names in self.directories.items():
            current = os.path.join(self.store.base_dir, path)
            if (tuple(sorted(os.listdir(current))) if os.path.exists(current) else ()) != names:
                raise StoreCutChanged("Authority directory witness changed; retry preparation.")

    def batch(self, changes, scopes):
        return PreparedMutation(self.cut, tuple(self.files.items()), tuple(self.directories.items()),
                                tuple(changes), tuple(scopes), self.store._prepared_fact_owner)


def observe(store, path, payload):
    current = _READ.get()
    if current is not None and current.store is store:
        current.file(path, payload)


def detached_owner(store):
    """Return only this operation's detached reader; never an external proof."""
    current = _READ.get()
    return current if current is not None and current.store is store else None


def read_cut(store):
    record = store._read_file(os.path.join(store.base_dir, "authority.record"), AuthorityRecord)
    if record is None or record.store_uuid != store.store_uuid:
        raise StoreAuthorityError("Missing or incompatible Store authority cut.")
    return record


@contextmanager
def detached_read(store):
    current = _READ.get()
    if current is not None and current.store is store:
        yield current
        return
    with store.authority_read_fence():
        current = _DetachedRead(store, read_cut(store))
    token = _READ.set(current)
    try:
        try:
            yield current
        except Exception:
            # A detached decode may encounter an intermediate association while
            # a cooperating writer advances the cut. Retry as instability rather
            # than certifying those transient bytes as stable corruption.
            with store.authority_read_fence():
                current.validate()
            raise
        else:
            # No preparation/expansion is allowed in this short scope.
            with store.authority_read_fence():
                current.validate()
    finally:
        _READ.reset(token)


def _path(store, identity):
    _uuid_value(identity)
    return os.path.join(store.base_dir, "mutations", identity + ".record")


def _observed(store, transition):
    if transition["kind"] == "file":
        return raw(store, transition["path"])
    path = os.path.join(store.base_dir, transition["path"])
    if not os.path.lexists(path):
        return None
    if not stat.S_ISDIR(os.lstat(path).st_mode):
        raise StoreAuthorityError("Installed state child is not a directory.")
    if transition["kind"] == "data":
        return b"directory"
    return variant_evidence(store, transition["path"])


def variant_evidence(store, path):
    """Return exact manifest/incarnation/presence evidence, without payload reads."""
    return _json_bytes({"manifest": encode(raw(store, path + "/manifest.record")),
                        "availability": encode(raw(store, path + "/availability.record")),
                        "data": os.path.isdir(os.path.join(store.base_dir, path, "data"))})


def _classify(store, record, cut):
    data = record.intent
    if data["status"] != "pending":
        if data["epoch"] > cut.epoch:
            cut = replace(cut, epoch=data["epoch"])
            store._atomic_write(os.path.join(store.base_dir, "authority.record"), cut.to_bytes())
        return record, cut
    projection, confirmed, changed = [], [], False
    initial, final = {}, {}
    for transition in data["transitions"]:
        key = (transition["path"], transition["kind"])
        initial.setdefault(key, transition)
        final[key] = transition
    for key, transition in initial.items():
        actual = _observed(store, transition)
        changed |= actual != decode(transition["before"])
        if actual == decode(final[key]["after"]) and actual != decode(transition["before"]):
            confirmed.append(transition["path"])
        projection.append({"path": transition["path"], "kind": transition["kind"], "value": encode(actual)})
    epoch = cut.epoch + int(changed)
    data.update(confirmed=confirmed, projection=projection, epoch=epoch, status="classified")
    classified = MutationIntentRecord(_json_bytes(data))
    # Persist classification before its cut, so a failed cut write is replayable.
    store._atomic_write(_path(store, data["mutation_id"]), classified.to_bytes())
    result = replace(cut, epoch=epoch)
    if result != cut:
        store._atomic_write(os.path.join(store.base_dir, "authority.record"), result.to_bytes())
    return classified, result


def recover(store):
    """Classify only captured pending intents under the writer fence.

    Returns their immutable records. No authoritative records are deleted or
    missing transitions executed. Corrupt/ambiguous evidence fails explicitly;
    no hooks, payload authentication or graph preparation runs during recovery.
    """
    with store.writer_lock():
        cut = read_cut(store)
        # One bounded bridge covers the intent -> pending-registration crash
        # window, avoiding a scan of all historical mutations under the fence.
        bridge_path = os.path.join(store.base_dir, "mutation-prepare.record")
        bridge = store._read_file(bridge_path, MutationIntentRecord)
        if bridge is not None:
            data = bridge.intent
            if data["store_uuid"] != cut.store_uuid:
                raise StoreAuthorityError("Prepared mutation belongs to another Store.")
            identity = data["mutation_id"]
            named = store._read_file(_path(store, identity), MutationIntentRecord)
            if named is None:
                store._atomic_write(_path(store, identity), bridge.to_bytes())
            elif any(named.intent[field] != data[field] for field in
                     ("mutation_id", "store_uuid", "base_epoch", "scopes", "transitions")):
                raise StoreAuthorityError("Prepared/named mutation authority mismatch.")
            if identity not in cut.pending:
                cut = replace(cut, pending=cut.pending + (identity,))
                store._atomic_write(os.path.join(store.base_dir, "authority.record"), cut.to_bytes())
            store._filesystem(filesystem.remove_file, bridge_path,
                              cleanup_directory=store.base_dir, missing_ok=True)
        results = []
        for identity in cut.pending:
            record = store._read_file(_path(store, identity), MutationIntentRecord)
            if record is None or record.intent["mutation_id"] != identity or record.intent["store_uuid"] != cut.store_uuid:
                raise StoreAuthorityError("Missing or mismatched pending mutation intent.")
            record, cut = _classify(store, record, cut)
            results.append(record)
        return tuple(results)


def acknowledge(store, identity, epoch):
    """Acknowledge exactly one classified mutation/cut; never consume later work.

    The caller must have incorporated its replay projection/coverage at epoch.
    Returns the resulting AuthorityRecord. Wrong/missing evidence raises
    StoreAuthorityError before acknowledgement; failures retain pending evidence.
    """
    with store.writer_lock():
        cut = read_cut(store)
        record = store._read_file(_path(store, identity), MutationIntentRecord)
        if (record is None or record.intent["mutation_id"] != identity
                or record.intent["store_uuid"] != cut.store_uuid or type(epoch) is not int
                or record.intent["epoch"] != epoch or record.intent["status"] == "pending"):
            raise StoreAuthorityError("Acknowledgement does not match classified mutation/cut.")
        if identity not in cut.pending:
            return cut
        data = record.intent
        data["status"] = "acknowledged"
        store._atomic_write(_path(store, identity), MutationIntentRecord(_json_bytes(data)).to_bytes())
        updated = replace(cut, pending=tuple(i for i in cut.pending if i != identity))
        store._atomic_write(os.path.join(store.base_dir, "authority.record"), updated.to_bytes())
        return updated


def apply(store, batch, *, _acknowledge_nonpersistent=False):
    """Apply prepared facts with cut/byte CAS and intent-before-effects ordering.

    Returns (mutation UUID or None, AuthorityRecord), with new work still pending
    exact projection acknowledgement. Empty/no-op batches do not churn epochs.
    StoreMutationError preserves partial evidence on IO failure;
    cancellation propagates with mutation/cut evidence attached. No callbacks run.
    """
    if type(batch) is not PreparedMutation:
        raise TypeError("Store mutations require PreparedMutation facts.")
    if batch.owner is not store._prepared_fact_owner:
        raise StoreAuthorityError("Prepared facts belong to a different Store owner.")
    identity = str(uuid4())
    record = _intent(batch, identity)
    payload = record.to_bytes()
    with store.writer_lock():
        witness = _DetachedRead(store, batch.cut)
        witness.files, witness.directories = dict(batch.witnesses), dict(batch.directories)
        witness.validate()
        if store._read_file(_path(store, identity), MutationIntentRecord) is not None:
            raise StoreAuthorityError("Mutation identity collision; existing intent is preserved.")
        if store._read_file(os.path.join(store.base_dir, "mutation-prepare.record"), MutationIntentRecord) is not None:
            raise StoreCutChanged("Intent registration is pending; recover authority and reprepare.")
        for pending_id in batch.cut.pending:
            pending = store._read_file(_path(store, pending_id), MutationIntentRecord)
            if (pending is None or pending.intent["mutation_id"] != pending_id
                    or pending.intent["store_uuid"] != batch.cut.store_uuid):
                raise StoreAuthorityError("Missing or mismatched pending mutation evidence.")
            if pending.intent["status"] == "pending" or pending.intent["epoch"] > batch.cut.epoch:
                raise StoreCutChanged("Unclassified publication is pending; recover authority and reprepare.")
        predicted, initial = {}, {}
        for change in batch.changes:
            key = (change.path, change.kind)
            current = predicted[key] if key in predicted else _observed(store, change.to_data())
            initial.setdefault(key, current)
            if current != change.before:
                raise StoreCutChanged("Prepared mutable CAS changed; retry preparation.")
            predicted[key] = change.after
        # Recovery may deliberately repersist equal bytes after an ambiguous
        # native barrier. Ordinary batches with no net effect need no journal or
        # filesystem/SQL churn, even if they contain intermediate transitions.
        recovery = any(scope.startswith("replay:") for scope in batch.scopes)
        if not batch.changes or (predicted == initial and not recovery):
            return None, batch.cut
        cut = batch.cut
        try:
            bridge = os.path.join(store.base_dir, "mutation-prepare.record")
            store._atomic_write(bridge, payload)
            store._atomic_write(_path(store, identity), payload)
            cut = replace(cut, pending=cut.pending + (identity,))
            store._atomic_write(os.path.join(store.base_dir, "authority.record"), cut.to_bytes())
            store._filesystem(filesystem.remove_file, bridge,
                              cleanup_directory=store.base_dir, missing_ok=True)
            for change in batch.changes:
                target = os.path.join(store.base_dir, change.path)
                if change.kind != "file":
                    store._filesystem(filesystem.publish_directory,
                                      os.path.join(store.base_dir, change.staging), target)
                elif change.after is None:
                    store._filesystem(filesystem.remove_file, target,
                                      cleanup_directory=os.path.dirname(target), missing_ok=True)
                else:
                    store._atomic_write(target, change.after)
            record, cut = _classify(store, record, cut)
            if _acknowledge_nonpersistent:
                cut = acknowledge(store, identity, record.intent["epoch"])
            return identity, cut
        except BaseException as error:
            refs = ()
            try:
                current = read_cut(store)
                cut = current
                persisted = store._read_file(_path(store, identity), MutationIntentRecord)
                if persisted is not None:
                    refs = persisted.intent["confirmed"]
                if identity in current.pending:
                    record, cut = _classify(store, persisted, current)
                    refs = record.intent["confirmed"]
            except BaseException:
                # The original intent remains the recovery authority if even
                # classification/cut publication is unavailable.
                pass
            outcome = StoreMutationError(str(error), identity, cut, refs)
            if not isinstance(error, Exception):
                error.mutation_id, error.cut = identity, cut
                error.confirmed_refs, error.index_disposition = tuple(refs), "repair_pending"
                raise
            raise outcome from error
