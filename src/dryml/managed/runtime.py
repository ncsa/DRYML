"""Metadata-first managed execution, durable completion and zero-work replay."""

from dataclasses import dataclass, replace
from uuid import UUID, uuid4
from contextvars import ContextVar

from .config import ManagedConfig, _ResolvedManagedConfig
from .control import (ControlHistory, ControlIntent, ControlSnapshot, ManagedControlStore,
                      StartPreparation, _same_snapshot_bytes)
from .errors import (ManagedConfigError, ManagedConflictError,
                     ManagedRecoveryError, ManagedRerunRequiredError)
from .identity import AdmissionIdentity, _argument_digest_bound, operation_digest
from .model import InterruptRequestResult
from .storage import resolve_stores


_outer_handoff = ContextVar("dryml_managed_outer_handoff", default=None)


def _consume_outer_handoff(descriptor, instance, args, managed, kwargs):
    lease = _outer_handoff.get()
    if lease is None or lease["descriptor"] is not descriptor or lease["instance"] is not instance:
        return None
    from .storage import _caller
    if not lease["active"] or lease["caller"] != _caller() or lease["used"]:
        raise ManagedConflictError("expired_handoff", "managed outer invocation handoff is expired or already consumed")
    if args != lease["args"] or kwargs != lease["kwargs"] or managed is not lease["managed"]:
        raise ManagedConfigError(message="managed outer wrapper changed its admitted arguments")
    lease["owner"].require_owner()
    lease["used"] = True
    return lease["execute"]


def invoke(descriptor, instance, args, managed, kwargs, *, on_raw_result=None, realize=None, outer=None):
    """Validate a direct call and enter metadata-first start/replay admission.

    descriptor selects a checked declaration, instance a live Object or detached Node, args/kwargs
    ordinary caller input, managed optional exact ManagedConfig. Arguments/defaults
    normalize once before ownership or restore. Execution injects ManagedContext
    and returns InvocationResult after durable encoding/finalization/completion.
    Replay returns detached metadata without body/restore/capture/return hooks.
    Recovery returns a detached terminating disposition without execution.
    on_raw_result is an optional execution delivery adapter, never replayed.
    realize optionally constructs a transported receiver only under admission's
    lifetime owner; outer is the captured wrapper executed only on that branch.
    Preparation errors preserve old current; hooks follow Core invalidation rules.
    """
    if managed is not None and type(managed) is not ManagedConfig:
        raise ManagedConfigError(message="managed must be an exact ManagedConfig or None")
    config = (managed or ManagedConfig()).snapshot()
    _validate_start_control(config)
    selected_store = descriptor.bind_store_parameter(instance, args, kwargs)
    stores = resolve_stores(state_repo=config.state_repo if config.state_repo is not None else selected_store,
                            control_store=config.control_store)
    try:
        if selected_store is not None and not any(selected_store is store for store in stores.state_repo.stores):
            from dryml.core.repo import Repo
            from .storage import ResolvedStores
            wrapper = Repo._for_state_io((*stores.state_repo.stores, selected_store))
            control_store = stores.control_store
            stores.close()
            stores = ResolvedStores(wrapper, control_store, wrapper)
        boundary = descriptor.signature_plan().prepare_bound(
            descriptor.bind_arguments(instance, args, kwargs), repo=stores.state_repo)
        digest = _argument_digest_bound(tuple(boundary.authority.items()))
        from dryml.core.node import Node
        receiver = instance if isinstance(instance, Node) else stores.state_repo.describe(instance)
        identity = AdmissionIdentity(receiver, descriptor.member, digest)
        control = ManagedControlStore(stores.control_store, stores.state_repo)
        def execute(owner, snapshot, live, resuming):
            def authored():
                return _execute_authored(owner, snapshot, live, resuming, descriptor, boundary, config,
                    control=control, store=selected_store,
                    invocation_parameters=descriptor.bind_invocation_parameters(instance, args, kwargs))
            if outer is None:
                return authored()
            from .storage import _caller
            lease = dict(descriptor=descriptor, instance=live, args=args, kwargs=kwargs,
                managed=managed, execute=authored, owner=owner, caller=_caller(), active=True, used=False)
            token = _outer_handoff.set(lease)
            try:
                result = outer(live, *args, managed=managed, **kwargs)
                if not lease["used"]:
                    raise ManagedConfigError(message="managed outer wrapper did not invoke admitted declaration")
                return result
            finally:
                lease["active"] = False
                _outer_handoff.reset(token)
        result = admit_execution(identity, descriptor, config,
            control, realize=(lambda node: instance) if realize is None else realize, execute=execute)
        if isinstance(result, AdmissionOutcome) and result.disposition == "replay":
            return _invocation_result(control, result.current, result.invocation_id, result.cleanup_report)
        if on_raw_result is not None and not isinstance(result, AdmissionOutcome):
            return on_raw_result(result)
        return result
    finally:
        stores.close()


def _execute_authored(owner, current, live, resuming, descriptor, boundary, config, *,
                      control, finish=None, store=None, invocation_parameters=None):
    # finish is the single result-owner handoff: normalize/encode first, then call
    # context._finalize_state(), confirm prerequisites and perform completion CAS.
    from .context import _create_context
    from .descriptor import _invoke_target
    context = _create_context(admission=owner, control=control, generation=current.generation,
        obj=live, snapshot=current, callbacks=config.callbacks, is_resuming=resuming, store=store)
    try:
        args, kwargs = boundary.deliver_args()
        raw = _invoke_target(descriptor, live, args, context, kwargs, store=store,
                             invocation_parameters=invocation_parameters)
        context._safe_point()
        result = (_finish_result(context, raw, descriptor) if finish is None else finish(context, raw))
        # Even a result-owner callback cannot turn an accepted request into success.
        context._safe_point(allow_completed=True)
        return result
    except BaseException as error:
        context._record_failure(error)
        if isinstance(error, KeyboardInterrupt):
            from .errors import ManagedInterrupted
            raise ManagedInterrupted() from error
        raise
    finally:
        context._deactivate()


def _invocation_result(control, snapshot, caller, cleanup_report=()):
    from .model import InvocationResult, ExecutionIdentity, ManagedStatus
    from .results import source_descriptor
    from .errors import ManagedResultError
    if snapshot.state != "completed" or snapshot.result_reference is None:
        raise ManagedResultError("missing_result_association", "completion lacks durable ordinary result metadata")
    status = ManagedStatus(snapshot.state, snapshot.operation_id, snapshot.run_id,
        snapshot.invocation_id, snapshot.generation, snapshot.checkpoint, snapshot.final,
        False, None, snapshot.producer_invocation_id)
    execution = ExecutionIdentity(snapshot.operation_id, snapshot.run_id, caller, snapshot.producer_invocation_id)
    return InvocationResult(status, execution, snapshot.final, snapshot.result_reference,
        source_descriptor(control.control_store), cleanup_report)


def _finish_result(context, raw, descriptor):
    from .result_codec import encode_result
    from .results import publish_result, read_result_bytes
    context._failure_code = "result_error"
    value = descriptor.signature_plan().prepare_return(raw, repo=context.state_repo).deliver_return(
        preserve_reference_data=True)
    payload = encode_result(value)
    reference, ack = publish_result(context.control_store, payload, return_ack=True)
    _completion_boundary("result_installed")
    current = context._safe_point()
    current = context._transition(current, state="finalizing", result_reference=reference, result_ack=ack)
    _completion_boundary("finalization_intent")
    context._failure_code = "publication_error"
    final = context._finalize_state()
    _completion_boundary("final_associated")
    current = context._safe_point()
    read_result_bytes(context.control_store, reference)
    _confirm_final(context._control, current)
    _completion_boundary("before_completion")
    current = context._safe_point()
    # The independent CAS journals its OWN invalidation and ack, not the prior
    # state receipt. A post-CAS index failure preserves completed authority.
    current = context._transition(current, state="completed", final=final,
        producer_invocation_id=context.invocation_id)
    _completion_boundary("completion_recorded")
    result = _invocation_result(context._control, current, context.invocation_id, context.cleanup_report)
    hook = getattr(context._obj, "_managed_post_publication", None)
    if hook is not None:
        hook(final, context.state_repo, context._final_report)
    return result


def _confirm_final(control, current):
    roots = [v for v in current.intents if v.kind == "final" and v.disposition == "adopted"]
    if len(roots) != 1 or current.result_reference is None:
        raise ManagedRecoveryError("incomplete_finalization", "final/result association is incomplete")
    root = roots[0]
    if current.result_ack is None:
        raise ManagedRecoveryError("unconfirmed_result", "result publication lacks acknowledgement evidence")
    from .results import result_path
    current.result_ack.confirm(_stores(control), expected_path=control.control_store._relative(
        result_path(control.control_store, current.result_reference.key)))
    for publication in root.publications:
        publication.confirm(_stores(control))
    ledger = control.control_store.retention.read(str(UUID(root.token)))
    if ledger is None or ledger.disposition != "adopted":
        raise ManagedRecoveryError("unconfirmed_final", "final grants require confirmation")
    _confirm_root(ledger, root.selection, resources=root.resources)
    control.control_store.retention.adopt(ledger.token, _stores(control))
    for store in _stores(control):
        if store.authority_cut().pending:
            raise ManagedRecoveryError("pending_finalization", "finalization prerequisites require exact acknowledgement")
    return root.selection


def _completion_boundary(stage):
    """Private deterministic result/finalization crash and race boundary."""


@dataclass(frozen=True, slots=True)
class AdmissionOutcome:
    """Detached replay/reconciliation disposition, never an execution success value.

    disposition is a static phase outcome; run_id/current are optional for absent
    old authority and invocation_id is this caller's fresh UUID. current contains
    historical metadata only. cleanup_report is a tuple of physical garbage-removal
    outcomes, not unresolved grant/index/control work. Construction performs no IO
    or lazy result loading.
    """
    __dryml_execute_adapter__ = "dryml.managed.execute_adapter:ManagedExecuteAdapter"
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
    Cleanup failures attach evidence without replacing the execution error, even
    on Python versions without exception-note support.
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
            if current is not None and current.state == "finalizing":
                disposition, current = _recover_finalization(control, owner, current)
            return AdmissionOutcome(disposition, None if current is None else current.run_id,
                                    owner.invocation_id, current, owner.cleanup_report)
        current = owner.current
        if owner.reconciled:
            disposition = "completion_recorded" if current is not None and current.state == "completed" else "projection_reconciled"
            if current is not None and current.state == "finalizing":
                disposition, current = _recover_finalization(control, owner, current)
            return AdmissionOutcome(disposition, None if current is None else current.run_id, owner.invocation_id, current)
        if current is not None and current.state == "finalizing":
            disposition, current = _recover_finalization(control, owner, current)
            return AdmissionOutcome(disposition, current.run_id, owner.invocation_id, current)
        if current is not None and _live_start_debts(current):
            _retire_live_start_debts(control, owner, current)
            return AdmissionOutcome("projection_reconciled", current.run_id, owner.invocation_id,
                                    control.inspect(identity.operation_id), owner.cleanup_report)
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
        # Receiver pins supply current state only when no start/recovery was
        # selected. They cannot cause a second restore of that selected start.
        from dryml.core.node import _owned_nodes
        unpinned = identity.receiver.with_realization(states={path: None for path, _ in _owned_nodes(identity.receiver)})
        live = owner.realize(lambda _: realize(identity.receiver if selected is None else unpinned))
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
        except BaseException as error:
            observed = control.inspect(identity.operation_id)
            if isinstance(error, Exception) and _same_snapshot_bytes(observed, current) and current.interrupt_request is None:
                control.transition(identity.operation_id, replace(current, generation=current.generation + 1,
                    state="failed", failure_code="execution_error"), owner=owner,
                    expected_generation=current.generation, expected_run_id=current.run_id,
                    expected_invocation_id=current.invocation_id)
            try:
                observed = control.inspect(identity.operation_id)
                if observed is not None:
                    _retire_live_start_debts(control, owner, observed)
            except BaseException as cleanup_error:
                error.cleanup_error = cleanup_error
                error.cleanup_outcomes = getattr(cleanup_error, "cleanup_outcomes", ())
                add_note = getattr(error, "add_note", None)
                if add_note is not None:
                    add_note("Superseded live-start grant cleanup remains pending; reconcile before retry.")
            raise


def _stores(control):
    return tuple(dict.fromkeys((*control.state_repo.stores, control.control_store)))


def _live_start_debts(current):
    return tuple(v.token for v in current.intents if v.kind == "final"
                 and v.disposition == "release_pending" and v.run_id != current.run_id)


def _retire_live_start_debts(control, owner, current):
    from .storage import _cleanup_checkpoints
    tokens = _live_start_debts(current)
    if tokens:
        outcomes = _cleanup_checkpoints(control, owner, current.operation_id, tokens=tokens)
        _record_cleanup(owner, outcomes)
        _require_cleanup(outcomes)


def _prepare_run(control, owner, previous, selected, resuming):
    owner.require_owner()
    identity = owner.identity
    run_id = owner.run_id if previous is None or resuming else uuid4().hex
    generation = (0 if previous is None else previous.generation) + 1
    history = () if previous is None else previous.history + (ControlHistory(previous.run_id,
        previous.invocation_id, previous.generation, previous.state, previous.checkpoint, previous.final,
        previous.result_reference),)
    retained = () if previous is None else tuple(v for v in previous.intents if
        v.kind in {"initial", "durable", "root"})
    if previous is not None and selected is None and not resuming:
        # A live start may still borrow deferred payloads from the former final.
        # Retire its association now, but carry its grant debt until the new final
        # has been published. The ordinary final replacement settles that debt;
        # persisted release_pending evidence also survives interruption/restart.
        retained += tuple(replace(v, disposition="release_pending") for v in previous.intents
                          if v.kind == "final" and v.disposition in {"adopted", "release_pending"})
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
    if (not preparation.settled and preparation.phase in {"prepared", "ready"}
            and preparation.previous is not None and preparation.previous.state == "finalizing"
            and preparation.proposed.state == "finalizing"
            and _same_snapshot_bytes(current, preparation.previous)
            and current.interrupt_request is None and owner.invocation_id != current.invocation_id):
        preparation = _recover_final_association(control, owner, preparation)
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
    selected = next((v.selection for v in preparation.proposed.intents if v.token == preparation.token),
                    preparation.proposed.checkpoint)
    if ledger is not None:
        _confirm_root(ledger, selected)
    if admitted:
        if selected is not None:
            if ledger is None or ledger.disposition != "adopted":
                raise ManagedRecoveryError("unconfirmed_start", "admitted saved start requires confirmed grants")
            retention.adopt(token, _stores(control))
        for root in settled.proposed.intents:
            if root.token == settled.token:
                _retire_publications(control, root)
        keep = {v.token for v in settled.proposed.intents}
        for intent in (() if old is None else old.intents):
            if intent.token not in keep and intent.kind in {"start", "latest", "final"}:
                owner.require_owner()
                retired = retention.read(str(UUID(intent.token)))
                if retired is None:
                    raise ManagedRecoveryError("missing_grant_intent", "superseded root lacks physical retention authority")
                _confirm_root(retired, intent.selection, resources=intent.resources)
                _release_and_reclaim(control, owner, retired)
    elif ledger is not None:
        # The managed abandonment tombstone above also fences an already-adopted
        # physical grant whose current CAS never committed.
        if ledger.disposition == "prepared":
            retention.abandon(token)
        _release_and_reclaim(control, owner, ledger)
    control.write_preparation(replace(settled, settled=True), owner=owner, expected=settled)
    return ("run_admitted" if admitted else "preparation_abandoned"), current


def _recover_final_association(control, owner, preparation):
    from .results import read_result_bytes
    from dryml.core.materialization import build_exact_state_load_plan
    proposed = preparation.proposed
    # A crash can precede the ready sidecar after physical adoption. Only this
    # preparation's final may be promoted, never a former run's retained debt.
    roots = [v for v in proposed.intents if v.kind == "final"
             and v.disposition in {"prepared", "adopted"}
             and (v.token, v.run_id, v.invocation_id, v.generation) ==
                 (preparation.token, proposed.run_id, proposed.invocation_id, proposed.generation)]
    if len(roots) != 1 or proposed.result_reference is None:
        return preparation
    root = roots[0]
    ledger = control.control_store.retention.read(str(UUID(root.token)))
    if ledger is None or ledger.disposition != "adopted":
        return preparation
    _confirm_root(ledger, root.selection, resources=root.resources)
    read_result_bytes(control.control_store, proposed.result_reference)
    build_exact_state_load_plan(control.state_repo, root.selection)
    control.control_store.retention.adopt(ledger.token, _stores(control))
    if preparation.phase == "prepared":
        ready = replace(preparation, phase="ready", proposed=replace(proposed,
            intents=tuple(replace(v, disposition="adopted") if v.token == root.token else v for v in proposed.intents)))
        preparation = control.write_preparation(ready, owner=owner, expected=preparation)
    control.recover_final_association(preparation, owner=owner)
    return preparation


def _retire_publications(control, root):
    for publication in root.publications:
        store = publication.confirm(_stores(control))
        retention = store.retention
        ledger = retention.read(publication.retention_token)
        if ledger is None:
            raise ManagedRecoveryError("missing_publication", "provisional publication ledger is absent")
        if ledger.disposition == "prepared":
            retention.abandon(ledger.token)
        if retention.release(ledger.token, _stores(control)).disposition != "released":
            raise ManagedRecoveryError("release_pending", "provisional publisher cleanup remains pending")


def _associate_state(context, selected, kind, *, report=None, published=None):
    """Use the existing two-phase preparation journal for one root replacement."""
    from .storage import _validate_selection
    current = context._safe_point()
    source_store = context._store if report is not None else None
    plan = _validate_selection(context.state_repo, context._obj, current.receiver, selected,
                               source_store=source_store)
    targets = {}
    for action in plan.actions:
        targets.setdefault(action.store, set()).add(action.state.key)
    retention = context.control_store.retention
    token = uuid4().hex
    ledger = retention.prepare(str(UUID(token)), selected.exact_digest(), targets.items())
    from .results import publication_ack
    publications = () if report is None else tuple(publication_ack(v.store, v.mutation_id,
        retention_token=v.retention_token) for v in report.publications)
    intent = ControlIntent(token, current.run_id, current.invocation_id,
        current.generation + 1, "prepared", ledger.resources, kind, selected, publications)
    replaced = {"latest", "start"} if kind == "latest" else {"final"} if kind == "final" else set()
    keep = tuple(v for v in current.intents if v.kind not in replaced)
    proposed = replace(current, generation=current.generation + 1,
        checkpoint=selected if kind == "latest" else current.checkpoint, intents=keep + (intent,))
    preparation = StartPreparation(token, current, proposed)
    preparation = context._control.write_preparation(preparation, owner=context._admission,
        expected=context._control.read_preparation(context.operation_id))
    retention.protect(ledger.token, _stores(context._control))
    ledger = retention.adopt(ledger.token, _stores(context._control))
    if report is not None:
        # Managed grants now protect all required keys; retire only these named
        # Core provisional publishers, never an independent user save root.
        _retire_publications(context._control, intent)
    if published is not None:
        published()
        # A boundary is not a live availability witness. In particular, a custom
        # reader or a publication-phase failure must still refuse association.
        _validate_selection(context.state_repo, context._obj, current.receiver, selected,
                            source_store=source_store)
    proposed = replace(proposed, intents=keep + (replace(intent, disposition="adopted"),))
    ready = replace(preparation, proposed=proposed, phase="ready")
    preparation = context._control.write_preparation(ready, owner=context._admission, expected=preparation)
    context._control.admit_prepared(preparation, owner=context._admission)
    _, committed = _settle_preparation(context._control, context._admission, preparation)
    context._snapshot, context._generation = committed, committed.generation
    return selected


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


def _release_and_reclaim(control, owner, ledger):
    from dryml.core import Repo
    from dryml.core.store.retention import RetentionReleasePending
    owner.require_owner()
    repo = Repo._for_state_io(_stores(control))
    try:
        try:
            outcomes = repo.release(ledger, owner_store=control.control_store)
        except Exception as error:
            _record_cleanup(owner, getattr(error, "cleanup_outcomes", ()))
            if isinstance(error, RetentionReleasePending):
                failure = ManagedRecoveryError("release_pending", "retired root target remains disconnected")
                failure.cleanup_outcomes = error.cleanup_outcomes
                raise failure from error
            raise
        _record_cleanup(owner, outcomes)
        _require_cleanup(outcomes)
    finally:
        repo.close(flush=False)


def _record_cleanup(owner, outcomes):
    for item in outcomes:
        if item.outcome is not None and item.outcome.cleanup_error is not None:
            owner.cleanup_report += ((item.domain, item.key, item.outcome),)


def _require_cleanup(outcomes):
    failed = next((item for item in outcomes if item.status in {"failed", "unattempted"}), None)
    if failed is not None:
        error = failed.error or ManagedRecoveryError("cleanup_disconnected", "retired resource target remains disconnected")
        error.cleanup_outcomes = outcomes
        raise error


def _recover_finalization(control, owner, current):
    from .results import read_result_bytes
    if current.interrupt_request is not None:
        proposed = replace(current, generation=current.generation + 1, state="interrupted",
            invocation_id=owner.invocation_id, interrupt_request=None)
        return "finalization_interrupted", control.transition(current.operation_id, proposed, owner=owner,
            expected_generation=current.generation, expected_run_id=current.run_id,
            expected_invocation_id=current.invocation_id)
    roots = [v for v in current.intents if v.kind == "final" and v.disposition == "adopted"]
    if current.result_reference is not None and len(roots) == 1:
        # Complete associated evidence is confirmed freshly, without a live receiver
        # or any normalization, return consumer, capture or restore hook.
        read_result_bytes(control.control_store, current.result_reference)
        final = _confirm_final(control, current)
        from dryml.core.materialization import build_exact_state_load_plan
        build_exact_state_load_plan(control.state_repo, final)
        return "completion_recorded", control.complete_recovered(current, final, owner=owner)
    return "finalization_incomplete", _fail_recovered_finalization(control, owner, current)


def _fail_recovered_finalization(control, owner, current):
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
        construction = getattr(instance, "_construction_node", instance.definition)
        return construction.with_realization(identities=instance.object_ref.objects)
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
