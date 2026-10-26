"""Selected resources and nonblocking logical-operation lifetime ownership.

No descendant Store/ObjectId or live-graph claim is an execution-lifetime lock.
Core state-IO reservations remain independent, short mutation protections.
"""

from contextlib import AbstractContextManager
from dataclasses import dataclass
import asyncio
import os
from threading import get_ident
from uuid import uuid4

from dryml.core.repo import Repo
from dryml.core.session import current_repo
from dryml.core.store.dir import DirStore
from dryml.core.store.store import Store
from dryml.locking import FileLock, LockError

from .errors import ManagedConflictError, ManagedStoreError


def _publish_checkpoint(repo, obj, *, states=None, store=None, snapshot_observer=None):
    """Publish through Core's save batch with provisional, non-user retention.

    The returned (SavedNode, StoreReport) names every provisional token, including
    partial failures carried by Core exceptions. Reservations cover only state IO;
    the caller must transfer protection before retiring these publication roots.
    snapshot_observer optionally supplies invocation-scoped environment evidence;
    its result is never ownership, payload-availability, or publication evidence.
    """
    from dryml.core.state import reserve
    from dryml.core.repo_plan import build_routed_save_plan, execute_routed_save_plan
    store = repo._ensure_store(store)
    plan, nodes, object_ids = repo._state_graph_evidence(obj)
    with reserve(obj._identity_node, nodes, object_ids), repo._retain_save_context() as context:
        routed = build_routed_save_plan(repo, plan, context, store=store)
        return execute_routed_save_plan(repo, routed, states=states,
            provisional=True, report_stores=True, snapshot_observer=snapshot_observer)


def _validate_selection(repo, obj, original, selected, *, source_store=None):
    """Freshly preflight original projection, live codecs, and selected availability.

    Returns the Core exact load plan, with no hooks, capture or restoration. The
    supplied graph itself remains the authority; no projected substitute is saved.
    Reader overrides and validation exceptions propagate without cached proof.
    source_store optionally pins fresh publication validation to its explicit
    destination, avoiding unrelated replicas with different capture evidence.
    """
    from dryml.core.node import require_saved
    from dryml.core.repo_plan import build_save_plan
    from dryml.core.materialization import build_exact_state_load_plan
    require_saved(selected)
    live = build_save_plan(repo, obj)
    if not original.object_projection_equal(selected) or not live.node.object_projection_equal(selected):
        raise ManagedConflictError("incompatible_selection", "saved selection differs from original receiver projection")
    for action in live.actions:
        selected.at(action.path).state.validate_for(live.node.at(action.path), codec=action.obj.state_codec)
    return build_exact_state_load_plan(repo, selected, source_store=source_store)


def _cleanup_checkpoints(control, owner, operation_id, *, tokens=None):
    from uuid import UUID
    from dryml.core.store.retention import StateCleanupOutcome, _held
    from .errors import ManagedRecoveryError
    from .runtime import _stores
    owner.require_owner()
    current = control.inspect(operation_id)
    preparation = control.read_preparation(operation_id)
    if preparation is not None and not preparation.settled:
        raise ManagedRecoveryError("pending_reconciliation", "settle checkpoint preparation before cleanup")
    roots = {root.token: root for root in current.intents}
    selected = tuple(root.token for root in current.intents if root.disposition != "released") if tokens is None else tuple(tokens)
    if any(type(token) is not str or token not in roots for token in selected):
        raise ManagedConflictError("unknown_checkpoint", "unknown checkpoint token")
    selected = tuple(dict.fromkeys(selected))
    if any(roots[token].disposition not in {"adopted", "release_pending", "released"} for token in selected):
        raise ManagedRecoveryError("pending_reconciliation", "checkpoint adoption is unresolved")
    if _held():
        return tuple(StateCleanupOutcome(resource.domain, key, "in-use")
            for token in selected for resource in roots[token].resources for key in resource.keys)
    def unattempted(tokens):
        return tuple(StateCleanupOutcome(resource.domain, key, "unattempted")
            for token in tokens for resource in roots[token].resources for key in resource.keys)

    pending = {token for token in selected if roots[token].disposition == "adopted"}
    try:
        if pending:
            current = control._retire_checkpoint_roots(current, pending, "release_pending", owner=owner)
    except BaseException as error:
        error.cleanup_outcomes = unattempted(selected)
        raise
    stores = _stores(control)
    repo = Repo._for_state_io(stores)
    outcomes, released, failures = [], set(), []
    remaining_start = 0
    try:
        for index, token in enumerate(selected):
            remaining_start = index
            owner.require_owner()
            root = roots[token]
            ledger = control.control_store.retention.read(str(UUID(token)))
            from .runtime import _confirm_root
            if ledger is None:
                raise ManagedRecoveryError("missing_grant_intent", "checkpoint lacks retention authority")
            _confirm_root(ledger, root.selection, resources=root.resources)
            try:
                outcomes.extend(repo.release(ledger, owner_store=control.control_store))
            except Exception as error:
                outcomes.extend(getattr(error, "cleanup_outcomes", ()))
                failures.append(error)
            remaining_start = index + 1
            if control.control_store.retention.read(ledger.token).disposition == "released":
                released.add(token)
        changed = {token for token in released if roots[token].disposition != "released"}
        if changed:
            current = control._retire_checkpoint_roots(current, changed, "released", owner=owner)
    except BaseException as error:
        nested = getattr(error, "cleanup_outcomes", ())
        error.cleanup_outcomes = tuple(outcomes) + nested + unattempted(selected[remaining_start + bool(nested):])
        raise
    finally:
        repo.close(flush=False)
    if failures:
        failures[0].cleanup_outcomes = tuple(outcomes)
        raise failures[0]
    return tuple(outcomes)


def _caller():
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return os.getpid(), get_ident(), task


@dataclass(slots=True)
class ResolvedStores:
    """Borrow selected state Repo/control DirStore; release only an owned wrapper.

    state_repo and control_store are open borrowed handles. _wrapper, when set by
    resolve_stores, is a private noncommitting Repo; no resource is serialized.
    Construction has no IO or publication effects and raises no additional errors.
    """

    state_repo: Repo
    control_store: DirStore
    _wrapper: Repo | None = None

    def close(self):
        """Return None after closing only the temporary Repo without flush.

        Caller-owned resources remain open. Wrapper shutdown errors propagate;
        repeated calls have no effect.
        """
        if self._wrapper is not None:
            wrapper, self._wrapper = self._wrapper, None
            wrapper.close(flush=False)


def resolve_stores(obj=None, *, state_repo=None, control_store=None, require_writable=True):
    """Resolve borrowed resources without materializing obj or publishing authority.

    obj is inert receiver metadata (or a live receiver for a caller adapter).
    state_repo accepts Repo, Store or None for the current Repo; control_store is
    an exact DirStore or None for worker/default selection. require_writable is a
    bool requesting format-4 preflight. Returns ResolvedStores. Invalid/missing
    resources raise ManagedStoreError; a temporary Repo wrapper is non-owning.
    """
    wrapper = None
    if state_repo is None:
        repo = current_repo()
    elif isinstance(state_repo, Repo):
        repo = state_repo
    elif isinstance(state_repo, Store):
        repo = wrapper = Repo._for_state_io((state_repo,))
    else:
        raise ManagedStoreError("invalid_state_repo", "state_repo must be a Repo or Store")
    try:
        if repo is None:
            raise ManagedStoreError("explicit_state_repo_required", "supply an existing state Repo")
        if control_store is None:
            from .defaults import _current_control_store_default
            control_store = _current_control_store_default() or repo.default_store
        if type(control_store) is not DirStore:
            raise ManagedStoreError("invalid_control_store", "control_store must be an exact DirStore")
        if require_writable:
            control_store.preflight_publication("managed control", record_family="dryml-node-graph/v1")
        return ResolvedStores(repo, control_store, wrapper)
    except BaseException:
        if wrapper is not None:
            wrapper.close(flush=False)
        raise


class OperationOwnership(AbstractContextManager):
    """Process/thread/task-bound lock for one control domain and operation digest.

    Instances are minted by managed control admission, not caller construction.
    The lock is nonblocking; duplicate acquisition raises ManagedConflictError.
    No live graph, StateKey lease or Store writer fence is retained. Inherited or
    foreign callers cannot use/release this token. Release never unlinks shared
    lock files. Native lock failures raise ManagedStoreError.
    """

    def __init__(self, authority, operation_id, path):
        """Acquire authority/operation_id at path nonblockingly; return None.

        authority is the selected (Store UUID, canonical root) pair, operation_id
        its stable digest and path its pre-created lock address. A contended lock
        raises ManagedConflictError; native failure raises ManagedStoreError.
        The descriptor stays held until this exact caller releases it.
        """
        self.authority = authority
        self.operation_id = operation_id
        self.invocation_id = uuid4().hex
        self._caller = _caller()
        self._active = False
        self._lock = FileLock(path)
        try:
            if not self._lock.acquire(blocking=False):
                raise ManagedConflictError("already_running", "selected logical operation already has an owner")
        except LockError as error:
            raise ManagedStoreError("operation_lock_unavailable", "could not acquire operation ownership") from error
        self._active = True

    @property
    def active(self):
        """Return bool: whether this exact caller still owns the lifetime lock."""
        return self._active and self._caller == _caller()

    def require_owner(self):
        """Return None or raise ManagedConflictError for retired/foreign ownership.

        Performs no IO and cannot release another caller's lock.
        """
        if not self.active:
            raise ManagedConflictError("invalid_operation_owner", "operation ownership is inactive or belongs to another caller")

    def __enter__(self):
        self.require_owner()
        return self

    def __exit__(self, *exc):
        self.release()
        return False

    def __reduce__(self):
        raise TypeError("Operation ownership is caller-local and cannot be serialized.")

    def release(self):
        """Release only this active caller's lock, returning whether it was held.

        Foreign/repeated releases return False; lock IO failures propagate as
        ManagedStoreError. No grants or other invocation resources are released.
        """
        if not self.active:
            return False
        self._active = False
        try:
            self._lock.release()
        except LockError as error:
            raise ManagedStoreError("operation_lock_release_failed", "could not release operation ownership") from error
        return True


class OperationAdmission(AbstractContextManager):
    """Detached admission lifetime, retained across an execution-only handoff.

    identity is AdmissionIdentity, ownership is an active OperationOwnership and
    current is validated metadata or None. run_id is stable for existing current
    and new for absent authority; invocation_id is fresh even for replay. No run
    is persisted by this value. reconciled=True marks a recovery-only admission
    that cannot realize a receiver. cleanup_report holds only Core physical-garbage
    removal outcomes; unresolved grant/control/index failures still propagate.
    U15 owns explicit rerun/start policy. Construction has no effects; callers
    obtain instances from ManagedControlStore.admission.
    """

    def __init__(self, identity, ownership, current, *, reconciled=False):
        """Bind detached facts, active lease and optional snapshot; return None.

        identity is AdmissionIdentity, ownership an OperationOwnership, current a
        ControlSnapshot or None, and reconciled a bool recovery disposition. This
        internal construction seam allocates only the absent operation's run UUID;
        it performs no publication, payload IO or additional lock acquisition.
        """
        self.identity, self.ownership, self.current = identity, ownership, current
        self.run_id = current.run_id if current is not None else uuid4().hex
        self.invocation_id = ownership.invocation_id
        self.reconciled = reconciled
        self.cleanup_report = ()

    @property
    def active(self):
        """Return bool for the exact process/thread/task still owning admission."""
        return self.ownership.active

    def require_owner(self):
        """Return None, raising ManagedConflictError for stale/foreign lifetime use."""
        self.ownership.require_owner()

    def realize(self, realize):
        """Call realize(Node) only under this retained owner and return its result.

        realize is an execution-branch callable supplied by the higher-level
        runtime/transport adapter. Its exceptions propagate. This seam performs no
        admission policy, checkpoint reads or construction itself, and never
        relinquishes/reacquires operation ownership around the callable.
        """
        self.require_owner()
        if self.reconciled:
            raise ManagedConflictError("reconciliation_only", "pending recovery terminates admission without execution")
        result = realize(self.identity.receiver)
        self.require_owner()
        return result

    def release(self):
        """Return bool after releasing only this admission's lifetime token."""
        return self.ownership.release()

    def __enter__(self):
        self.require_owner()
        return self

    def __exit__(self, *exc):
        self.release()
        return False

    def __reduce__(self):
        raise TypeError("Operation admission is caller-local and cannot be serialized.")


__all__ = ["ResolvedStores", "OperationOwnership", "OperationAdmission", "resolve_stores"]
