"""Managed metadata access; execution waits for the Node lifecycle integration."""

from .control import ManagedControlStore
from .errors import ManagedControlError, ManagedConfigError
from .identity import operation_digest
from .model import InterruptRequestResult
from .storage import resolve_stores


def invoke(descriptor, instance, args, managed, kwargs, *, on_raw_result=None):
    """Refuse the retired StateRef/control-v2 execution path before effects.

    descriptor/instance/args/managed/kwargs are retained adapter inputs and
    on_raw_result an optional execution callback; none is invoked or materialized.
    Always raises ManagedControlError("lifecycle_not_integrated"). U15/U16/U17
    supply start/checkpoint/result policy and U19 the direct transport handoff;
    U14's detached control primitives are independently usable now.
    """
    raise ManagedControlError("lifecycle_not_integrated", "managed Node lifecycle execution awaits start/checkpoint/result integration")


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
