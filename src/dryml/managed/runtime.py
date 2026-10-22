"""Metadata-first managed admission and owned two-phase start preparation.

Checkpoint/final policy and durable result delivery remain separate owning units.
The execution callback is an internal integration seam, not a completion claim.
"""

from dataclasses import dataclass, replace
from uuid import UUID, uuid4

from .config import ManagedConfig, _ResolvedManagedConfig
from .control import (ControlHistory, ControlIntent, ControlSnapshot, ManagedControlStore,
                      StartPreparation, _same_snapshot_bytes)
from .errors import (ManagedControlError, ManagedConfigError, ManagedConflictError,
                     ManagedRecoveryError, ManagedRerunRequiredError)
from .identity import AdmissionIdentity, _argument_digest_bound, operation_digest
from .model import InterruptRequestResult
from .storage import resolve_stores


def invoke(descriptor, instance, args, managed, kwargs, *, on_raw_result=None):
    """Validate a direct call and enter metadata-first start/replay admission.

    descriptor selects a checked declaration, instance a live Object, args/kwargs
    ordinary caller input, managed optional exact ManagedConfig. Arguments/defaults
    normalize once before ownership or restore. Returns only a recovery disposition
    when reconciliation was required. U16/U17 completion/result policy is not yet
    integrated: execution branches raise completion_policy_not_integrated before
    body entry; replay raises result_policy_not_integrated carrying detached
    admission metadata. on_raw_result is reserved for that later delivery seam.
    Preparation errors preserve old current; hooks follow Core invalidation rules.
    """
    if managed is not None and type(managed) is not ManagedConfig:
        raise ManagedConfigError(message="managed must be an exact ManagedConfig or None")
    config = (managed or ManagedConfig()).snapshot()
    _validate_start_control(config)
    selected_store = descriptor.bind_store_parameter(instance, args, kwargs)
    stores = resolve_stores(state_repo=config.state_repo or selected_store, control_store=config.control_store)
    try:
        boundary = descriptor.signature_plan().prepare_bound(
            descriptor.bind_arguments(instance, args, kwargs), repo=stores.state_repo)
        digest = _argument_digest_bound(tuple(boundary.authority.items()))
        receiver = stores.state_repo.describe(instance)
        identity = AdmissionIdentity(receiver, descriptor.member, digest)
        def unfinished(owner, snapshot, live, resuming):
            raise ManagedControlError("completion_policy_not_integrated", "checkpoint/final/result policy is not integrated")
        result = admit_execution(identity, descriptor, config,
            ManagedControlStore(stores.control_store, stores.state_repo),
            realize=lambda node: instance, execute=unfinished)
        if result.disposition == "replay":
            error = ManagedControlError("result_policy_not_integrated", "completed replay selected; durable result delivery is not integrated")
            error.admission = result
            raise error
        return result
    finally:
        stores.close()


@dataclass(frozen=True, slots=True)
class AdmissionOutcome:
    """Detached replay/reconciliation disposition, never an execution success value.

    disposition is a static phase outcome; run_id/current are optional for absent
    old authority and invocation_id is this caller's fresh UUID. current contains
    historical metadata only. cleanup_report is a tuple of physical garbage-removal
    outcomes, not unresolved grant/index/control work. Construction performs no IO
    or lazy result loading.
    """
    disposition: str
    run_id: str | None
    invocation_id: str
    current: ControlSnapshot | None
    cleanup_report: tuple = ()


def _validate_start_control(config):
    if type(config) is not _ResolvedManagedConfig:
        raise ManagedConfigError(message="admission requires an invocation config snapshot")
    if config.start_from is not None and not config.rerun:
        raise ManagedConfigError("start_requires_rerun", "start_from requires explicit rerun on every invocation")


def admit_execution(identity, descriptor, config, control, *, realize, execute):
    """Select replay/recovery/execution from detached facts under the same owner.

    identity is normalized AdmissionIdentity; descriptor selected ManagedOperation;
    config a ManagedConfig.snapshot(); control selected ManagedControlStore.
    realize(Node) runs only on execution and supplies its live receiver. execute
    (admission, running_snapshot, live, is_resuming) is an internal U16/U17 seam
    retaining that same lifetime owner. Its return does not record completion.
    Replay/recovery returns AdmissionOutcome without realization/body/hooks/capture.
    Invalid controls/arguments, unusable recovery and contention raise typed managed
    errors. Preparation and publication failures preserve exact intent evidence;
    Core owns pre-hook usability and post-hook invalidation. No receipt survives
    this invocation. Caller adapters must not reconstruct receivers before this gate.
    """
    from .descriptor import ManagedOperation
    _validate_start_control(config)
    if (type(identity) is not AdmissionIdentity or type(descriptor) is not ManagedOperation
            or identity.member != descriptor.member or not callable(realize) or not callable(execute)):
        raise ManagedConfigError(message="detached admission requires selected declaration and ordinary argument authority")
    with control.admission(identity, check_arguments=False) as owner:
        preparation = control.read_preparation(identity.operation_id)
        if preparation is not None and not preparation.settled:
            disposition, current = _settle_preparation(control, owner, preparation)
            return AdmissionOutcome(disposition, None if current is None else current.run_id,
                                    owner.invocation_id, current, owner.cleanup_report)
        current = owner.current
        if owner.reconciled:
            disposition = "completion_recorded" if current is not None and current.state == "completed" else "projection_reconciled"
            if current is not None and current.state == "finalizing":
                current = _fail_recovered_finalization(control, owner, current)
                disposition = "finalization_incomplete"
            return AdmissionOutcome(disposition, None if current is None else current.run_id, owner.invocation_id, current)
        if current is not None and current.state == "finalizing":
            current = _fail_recovered_finalization(control, owner, current)
            return AdmissionOutcome("finalization_incomplete", current.run_id, owner.invocation_id, current)
        if current is not None and not config.rerun:
            if current.argument_digest != identity.argument_digest:
                raise ManagedRerunRequiredError("argument_mismatch", "changed ordinary arguments require explicit rerun")
            if current.state == "completed":
                return AdmissionOutcome("replay", current.run_id, owner.invocation_id, current)
            if not descriptor.resumable or current.checkpoint is None:
                raise ManagedRerunRequiredError("rerun_required", "unfinished work requires resumable declaration and latest recovery")
            selected, resuming = current.checkpoint, True
        else:
            selected = config.start_from if config.start_from is not None else (None if current is None else current.initial)
            resuming = False
        if selected is not None and not identity.receiver.object_projection_equal(selected):
            raise ManagedConfigError("incompatible_start", "selected saved start differs from receiver projection")
        # Every failure below retains this intent. Reconciliation is a separate
        # admission and never borrows this invocation's restoration capability.
        prepared = _prepare_run(control, owner, current, selected, resuming)
        sources = {}
        if selected is not None:
            from dryml.core.materialization import build_exact_state_load_plan
            plan = build_exact_state_load_plan(control.state_repo, selected)
            targets = {}
            for action in plan.actions:
                sources[action.state.key] = action.store
                targets.setdefault(action.store, set()).add(action.state.key)
            ledger = control.control_store.retention.prepare(str(UUID(prepared.token)), selected.exact_digest(), targets.items())
            control.control_store.retention.protect(ledger.token, _stores(control))
        # Transported receiver pins are historical, not an additional start.
        # Realization constructs the identity graph; this owner alone restores
        # the selected saved start through prepare_restore below.
        from dryml.core.node import _owned_nodes
        unpinned = identity.receiver.with_realization(states={path: None for path, _ in _owned_nodes(identity.receiver)})
        live = owner.realize(lambda _: realize(unpinned))
        if not control.state_repo.describe(live).object_projection_equal(identity.receiver):
            raise ManagedConfigError("incompatible_receiver", "execution realization differs from admitted projection")
        if selected is not None:
            with control.state_repo.prepare_restore(live, selected, source_stores=sources) as restore:
                restore.apply()
            # Source leases/reservations have ended; grants protect the gap.
            ledger = control.control_store.retention.adopt(str(UUID(prepared.token)), _stores(control))
            root = ControlIntent(prepared.token, prepared.proposed.run_id, owner.invocation_id,
                prepared.proposed.generation, "adopted", ledger.resources, "start", selected)
            proposed = replace(prepared.proposed, intents=prepared.proposed.intents + (root,))
        else:
            proposed = prepared.proposed
        ready = replace(prepared, proposed=proposed, phase="ready")
        prepared = control.write_preparation(ready, owner=owner, expected=prepared)
        control.admit_prepared(prepared, owner=owner)
        _, current = _settle_preparation(control, owner, prepared)
        try:
            owner.require_owner()
            if not _same_snapshot_bytes(control.inspect(identity.operation_id), current) or current.interrupt_request is not None:
                raise ManagedConflictError("interrupted", "admission was fenced before execution")
            return execute(owner, current, live, resuming)
        except Exception:
            observed = control.inspect(identity.operation_id)
            if _same_snapshot_bytes(observed, current) and current.interrupt_request is None:
                control.transition(identity.operation_id, replace(current, generation=current.generation + 1,
                    state="failed", failure_code="execution_error"), owner=owner,
                    expected_generation=current.generation, expected_run_id=current.run_id,
                    expected_invocation_id=current.invocation_id)
            raise


def _stores(control):
    return tuple(dict.fromkeys((*control.state_repo.stores, control.control_store)))


def _prepare_run(control, owner, previous, selected, resuming):
    owner.require_owner()
    identity = owner.identity
    run_id = owner.run_id if previous is None or resuming else uuid4().hex
    generation = (0 if previous is None else previous.generation) + 1
    history = () if previous is None else previous.history + (ControlHistory(previous.run_id,
        previous.invocation_id, previous.generation, previous.state, previous.checkpoint, previous.final),)
    retained = () if previous is None else tuple(v for v in previous.intents if
        v.kind in {"initial", "durable", "root"})
    proposed = ControlSnapshot(identity.receiver, identity.member, identity.argument_digest,
        run_id, owner.invocation_id, generation, "running", checkpoint=selected,
        history=history, intents=retained)
    prepared = StartPreparation(uuid4().hex, previous, proposed)
    return control.write_preparation(prepared, owner=owner,
        expected=control.read_preparation(identity.operation_id))


def _settle_preparation(control, owner, preparation):
    owner.require_owner()
    operation_id = preparation.proposed.operation_id
    current = control.inspect(operation_id)
    admitted = _same_snapshot_bytes(current, preparation.proposed) or _interrupted_successor(preparation.proposed, current)
    old = preparation.previous
    interrupted_old = _interrupted_successor(old, current)
    if not admitted and not _same_snapshot_bytes(current, old) and not interrupted_old:
        raise ManagedRecoveryError("ambiguous_admission", "current matches neither exact preparation fingerprint")
    phase = "admitted" if admitted else "abandoned"
    settled = replace(preparation, phase=phase)
    if settled != preparation:
        settled = control.write_preparation(settled, owner=owner, expected=preparation)
    retention = control.control_store.retention
    token = str(UUID(settled.token))
    ledger = retention.read(token)
    if ledger is not None:
        _confirm_root(ledger, preparation.proposed.checkpoint)
    if admitted:
        if settled.proposed.checkpoint is not None:
            if ledger is None or ledger.disposition != "adopted":
                raise ManagedRecoveryError("unconfirmed_start", "admitted saved start requires confirmed grants")
            retention.adopt(token, _stores(control))
        keep = {v.token for v in settled.proposed.intents}
        for intent in (() if old is None else old.intents):
            if intent.token not in keep and intent.kind in {"start", "latest", "final"}:
                owner.require_owner()
                retired = retention.read(str(UUID(intent.token)))
                if retired is None:
                    raise ManagedRecoveryError("missing_grant_intent", "superseded root lacks physical retention authority")
                _confirm_root(retired, intent.selection, resources=intent.resources)
                released = retention.release(retired.token, _stores(control))
                if released.disposition != "released":
                    raise ManagedRecoveryError("release_pending", "superseded root target remains disconnected")
                _reclaim_resources(control, owner, retired.resources)
    elif ledger is not None:
        # The managed abandonment tombstone above also fences an already-adopted
        # physical grant whose current CAS never committed.
        if ledger.disposition == "prepared":
            retention.abandon(token)
        released = retention.release(token, _stores(control))
        if released.disposition != "released":
            raise ManagedRecoveryError("release_pending", "abandoned start target remains disconnected")
        _reclaim_resources(control, owner, ledger.resources)
    control.write_preparation(replace(settled, settled=True), owner=owner, expected=settled)
    return ("run_admitted" if admitted else "preparation_abandoned"), current


def _interrupted_successor(previous, current):
    return (previous is not None and previous.state in {"running", "finalizing"}
        and previous.interrupt_request is None and current == replace(previous,
            generation=previous.generation + 1, interrupt_request=(previous.run_id, previous.invocation_id)))


def _confirm_root(ledger, selected, *, resources=None):
    if (selected is None or ledger.graph_digest != selected.exact_digest()
            or {key for resource in ledger.resources for key in resource.keys} !=
                {state.key for state in selected.state_spec().values()}
            or resources is not None and ledger.resources != resources):
        raise ManagedRecoveryError("grant_evidence_mismatch", "retention ledger differs from exact saved root evidence")


def _reclaim_resources(control, owner, resources):
    targets = {store.retention_domain: store for store in _stores(control)}
    for resource in resources:
        target = targets.get(resource.domain)
        if target is None:
            raise ManagedRecoveryError("cleanup_disconnected", "retired resource target remains disconnected")
        for key in resource.keys:
            owner.require_owner()
            outcome = target.delete_state(key)
            # Core reports only already-unreferenced physical garbage OSErrors as
            # cleanup_error. Grant/journal/index failures propagate, never success.
            if outcome.cleanup_error is not None:
                owner.cleanup_report += ((resource.domain, key, outcome),)


def _fail_recovered_finalization(control, owner, current):
    # U17 will supply complete result/final acknowledgement evidence. Until then
    # there is no provable completion payload in this format to invent or recapture.
    proposed = replace(current, generation=current.generation + 1, state="failed",
        invocation_id=owner.invocation_id, interrupt_request=None, failure_code="incomplete_finalization")
    return control.transition(current.operation_id, proposed, owner=owner,
        expected_generation=current.generation, expected_run_id=current.run_id,
        expected_invocation_id=current.invocation_id)


def _receiver(instance):
    from dryml.core.node import Node
    from dryml.core.object import Object
    if isinstance(instance, Node): return instance
    if isinstance(instance, Object):
        return instance.definition.with_realization(identities=instance.object_ref.objects)
    raise ManagedConfigError("invalid_receiver", "metadata access requires a Node or Object receiver")


def status(descriptor, instance, *, state_repo=None, control_store=None):
    """Return metadata-only ManagedStatus for descriptor/member and receiver identity.

    instance accepts detached Node or Object; state_repo/control_store are optional
    borrowed selected resources. Invalid inputs raise ManagedConfigError;
    unsupported/corrupt control raises ManagedControlError. No body, realization,
    restore, payload read or lifetime-owner wait occurs; only owned wrappers close.
    """
    receiver = _receiver(instance)
    operation_id = operation_digest(receiver, descriptor.member)
    stores = resolve_stores(receiver, state_repo=state_repo, control_store=control_store, require_writable=False)
    try:
        return ManagedControlStore(stores.control_store, stores.state_repo).status(operation_id)
    finally:
        stores.close()


def request_interrupt(descriptor, instance, *, state_repo=None, control_store=None,
                      expected_run_id=None, expected_invocation_id=None, expected_generation=None):
    """Return InterruptRequestResult from an independent selected-control request.

    descriptor/instance select declared member and Node/live receiver metadata;
    state_repo/control_store are borrowed resources. Optional expected_run_id,
    expected_invocation_id and expected_generation are deliberate stale-caller
    preconditions. Invalid/unsupported authority raises ManagedControlError.
    May journal/publish an accepted request, never runs hooks/opens private state
    or acquires/waits for the operation lifetime lock.
    """
    receiver = _receiver(instance)
    operation_id = operation_digest(receiver, descriptor.member)
    stores = resolve_stores(receiver, state_repo=state_repo, control_store=control_store, require_writable=False)
    try:
        outcome, snapshot = ManagedControlStore(stores.control_store, stores.state_repo).request_interrupt_if_running(
            operation_id, expected_run_id=expected_run_id, expected_invocation_id=expected_invocation_id,
            expected_generation=expected_generation)
        return InterruptRequestResult(outcome, operation_id, None if snapshot is None else snapshot.run_id,
            None if snapshot is None else snapshot.invocation_id, 0 if snapshot is None else snapshot.generation)
    finally:
        stores.close()


def _preobserve_snapshot_environment():
    """Preserve the existing deferred observation seam for later execution integration."""
    from dryml.environments.introspection import inspect_current
    try:
        environment = inspect_current()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        def observer(error=error): raise error
    else:
        def observer(environment=environment): return environment
    return observer


__all__ = ["invoke", "status", "request_interrupt"]
