"""Caller-local author checkpoints, immutable initial and exact final selection."""

from dataclasses import replace

from .errors import (ManagedContextError, ManagedConflictError, ManagedControlError,
                     ManagedInterrupted)


def _selection_equal(left, right):
    from .control import _node_equal
    return _node_equal(left, right)


class ManagedContext:
    """Invocation-scoped author lifecycle controls and read-only saved references.

    Runtime-created contexts borrow the live receiver, Repo and control Store.
    checkpoint(states=..., durable=False, initial=False) publishes/associates a
    complete SavedNode before notifying callbacks. initial_state and final_state
    accept complete compatible saved graphs without capture or restore. Initial is
    operation-wide and immutable; final selection protects its exact graph while
    completed-result authority remains owned by the result finalizer. retain/release
    manage independent explicit/durable roots without private-state serialization.
    Snapshot environment evidence is observed lazily once per invocation before
    its first capture and shared by later captures. Replay and explicit final
    selection without capture do not observe the environment. Foreign/retired/recursive use raises
    ManagedContextError; stale ownership raises ManagedConflictError. Accepted
    interruption always prevents completion, even when author code catches it.
    """

    def __init__(self, token, *, admission, control, generation, obj=None,
                 snapshot=None, callbacks=(), is_resuming=False, store=None):
        """Bind runtime-only admission and optional execution inputs, without IO.

        admission/control/generation are the active owner, selected authority and
        integer fence. obj/snapshot/callbacks/is_resuming/store supply execution
        facts. An invalid private token raises ManagedContextError. Returns None.
        """
        if token is not _CONTEXT_TOKEN:
            raise ManagedContextError("context_construction", "managed contexts are runtime-created")
        self._admission, self._control, self._generation = admission, control, generation
        self._obj, self._snapshot = obj, snapshot
        self._callbacks, self._is_resuming, self._store = callbacks, is_resuming, store
        self._active, self._busy, self._interrupted = True, False, False
        self._final = None
        self._final_report = None
        self._failure_code = "method_error"
        self._checkpoint_error = None
        self._snapshot_observer = None

    @property
    def active(self):
        """Return bool for this exact caller's still-active invocation lifetime."""
        return self._active and self._admission.active

    @property
    def operation_id(self):
        """Return stable operation digest (str), without IO."""
        return self._admission.identity.operation_id

    @property
    def run_id(self):
        """Return stable run UUID hex (str), retained across ordinary resume."""
        return self._admission.run_id

    @property
    def invocation_id(self):
        """Return fresh caller invocation UUID hex (str), distinct from producer."""
        return self._admission.invocation_id

    @property
    def generation(self):
        """Return this context's integer fence, not a fresh authority observation."""
        return self._generation

    @property
    def state_repo(self):
        """Return borrowed selected Repo without changing lifetime or routing."""
        return self._control.state_repo

    @property
    def control_store(self):
        """Return borrowed selected control DirStore without opening/closing it."""
        return self._control.control_store

    @property
    def is_resuming(self):
        """Return bool indicating freshly restored reentry in the same run."""
        return self._is_resuming

    @property
    def cleanup_report(self):
        """Return immutable physical-garbage cleanup outcomes accumulated so far.

        Each entry is (Store authority domain, full StateKey, Core removal
        outcome). Unresolved grants, publication, control or index failures are
        raised, never included as successful cleanup. No IO or deletion occurs.
        """
        return self._admission.cleanup_report

    @property
    def initial_state_ref(self):
        """Return optional operation-wide initial SavedNode metadata, without IO."""
        return None if self._snapshot is None else self._snapshot.initial

    @property
    def latest_checkpoint(self):
        """Return optional latest associated recovery SavedNode, without IO."""
        return None if self._snapshot is None else self._snapshot.checkpoint

    @property
    def final_state_ref(self):
        """Return the exact explicitly selected final SavedNode, or None; no IO."""
        return self._final

    def checkpoint(self, *, states=None, durable=False, initial=False):
        """Publish a complete recovery graph, then run each checkpoint callback once.

        states is a partial StateSpec/path-to-LocalState map; explicit compatible
        states skip capture, remaining distinct stateful nodes capture once.
        durable/initial are exact bools adding independent retention reasons.
        Default checkpoints do not designate initial. Returns SavedNode. Core
        validation/publication errors propagate; callbacks run outside locks after
        complete association and propagate errors without undoing latest recovery.
        Interruption/stale ownership fence mutation and completion independently.
        """
        self._require_active()
        if self._busy:
            raise ManagedContextError("recursive_checkpoint", "checkpoint callbacks cannot recursively mutate their context")
        if type(durable) is not bool or type(initial) is not bool:
            raise ManagedContextError("invalid_checkpoint", "durable and initial must be exact bools")
        self._safe_point()
        self._busy = True
        self._failure_code = "publication_error"
        try:
            from .storage import _publish_checkpoint
            from .runtime import _associate_state
            selected, report = _publish_checkpoint(self.state_repo, self._obj, states=states, store=self._store,
                snapshot_observer=self._environment_observer())
            _associate_state(self, selected, "latest", report=report,
                             published=lambda: _checkpoint_boundary("state_published"))
            if initial:
                self._initial_state(selected)
            if durable:
                _associate_state(self, selected, "durable")
            _checkpoint_boundary("checkpoint_associated")
            self._failure_code = "callback_error"
            _checkpoint_boundary("callbacks_started")
            for callback in self._callbacks:
                callback(self._obj, self)
            _checkpoint_boundary("callbacks_finished")
            self._safe_point()
            return selected
        except BaseException as error:
            self._checkpoint_error = (error, self._failure_code)
            raise
        finally:
            self._busy = False
            self._failure_code = "method_error"

    def _prepend_checkpoint_callback(self, callback):
        """Install a consumer callback within the admitted author's active scope.

        This private consumer seam preserves admitted ManagedConfig authority and
        caller callback order. It is never entered on replay. Returns None;
        inactive/recursive contexts fail before changing invocation-local callbacks.
        """
        self._require_mutation()
        if len(self._callbacks) >= 64:
            raise ManagedContextError("invalid_checkpoint", "consumer callback requires room in the callback limit")
        self._callbacks = (callback, *self._callbacks)

    def initial_state(self, state):
        """Designate complete SavedNode state as immutable operation-wide initial.

        Repeating the exact same graph (including retained construction) is
        idempotent. A different selection raises ManagedConflictError. Fresh
        projection/codec/availability validation precedes association; no capture,
        restore or callback occurs. Returns None; Store/control failures propagate.
        May first designate initial during resume.
        """
        self._require_mutation()
        self._initial_state(state)

    def _initial_state(self, state):
        from .storage import _validate_selection
        from .runtime import _associate_state
        current = self._safe_point()
        # Validate even an equal selection: immutable data is not live proof.
        _validate_selection(self.state_repo, self._obj, current.receiver, state)
        if current.initial is not None:
            if not _selection_equal(current.initial, state):
                raise ManagedConflictError("initial_conflict", "operation initial selection is immutable")
            return
        _associate_state(self, state, "initial")

    def final_state(self, state):
        """Select exact SavedNode state for later successful state finalization.

        Validates original receiver projection, current live compatibility and
        availability before protecting and associating the unchanged selection as
        an independent final root. Returns None. There is no implicit save,
        restore, callback or completion. Invalid/incompatible
        metadata or unavailable state raises before changing the selection;
        accepted interruption or stale ownership always fences this operation.
        """
        self._require_mutation()
        from .storage import _validate_selection
        from .runtime import _associate_state
        current = self._safe_point()
        _validate_selection(self.state_repo, self._obj, current.receiver, state)
        if self._final is None or not _selection_equal(self._final, state):
            _associate_state(self, state, "final")
        self._final = state

    def retain(self, state):
        """Add an independent explicit root for a compatible complete SavedNode.

        state is existing saved metadata, freshly validated against the receiver.
        Returns None; no capture, restore, payload serialization or callback occurs.
        Repeated exact retained selections are idempotent. The root survives rerun
        until release() or explicit operation cleanup. Stale/foreign/recursive use,
        incompatible/unavailable state and publication/grant/index failures raise
        before claiming a confirmed retained association.
        """
        self._require_mutation()
        from .storage import _validate_selection
        from .runtime import _associate_state
        current = self._safe_point()
        _validate_selection(self.state_repo, self._obj, current.receiver, state)
        if not any(root.kind == "root" and root.disposition == "adopted"
                   and _selection_equal(root.selection, state) for root in current.intents):
            _associate_state(self, state, "root")

    def release(self, state):
        """Release this operation's explicit/durable roots matching saved state.

        state is SavedNode metadata; exact semantic/retained-construction equality
        selects roots, without loading or serializing private data. Initial, latest,
        final, other-operation and user-save roots survive. Returns per-target
        StateCleanupOutcomes, including retained/in-use and independent failures.
        Same-caller active key scopes return in-use without retirement; retry after
        those scopes end. Unknown selections raise ManagedConflictError. Required
        authority/index failures propagate with cleanup_outcomes; only physical
        garbage failure is cleanup-pending. No history or ordinary result is deleted.
        """
        self._require_mutation()
        from .storage import _cleanup_checkpoints
        current = self._safe_point()
        tokens = tuple(root.token for root in current.intents if root.kind in {"root", "durable"}
                       and root.disposition != "released" and _selection_equal(root.selection, state))
        if not tokens:
            raise ManagedConflictError("unknown_checkpoint", "no explicit or durable root matches selection")
        try:
            return _cleanup_checkpoints(self._control, self._admission, self.operation_id, tokens=tokens)
        finally:
            observed = self._control.inspect(self.operation_id)
            self._snapshot, self._generation = observed, observed.generation

    def interrupt(self, *, cause=None):
        """Checkpoint once then terminate this invocation with ManagedInterrupted.

        cause is optional BaseException chained to the signal; invalid causes
        raise ManagedContextError before capture. Publication/callback/control
        errors propagate instead of claiming an interruption write succeeded.
        A caught interruption remains terminal; the return boundary checks again.
        """
        if cause is not None and not isinstance(cause, BaseException):
            raise ManagedContextError("invalid_cause", "cause must be BaseException or None")
        self.checkpoint()
        self._control.request_interrupt_if_running(self.operation_id,
            expected_run_id=self.run_id, expected_invocation_id=self.invocation_id,
            expected_generation=self.generation)
        try:
            self._safe_point()
        except ManagedInterrupted as error:
            raise error from cause
        raise ManagedControlError("interruption_recording_failed", "request did not establish interruption")

    def _finalize_state(self):
        # Called only after the result owner has normalized and encoded its value.
        self._require_mutation()
        from .storage import _publish_checkpoint, _validate_selection
        from .runtime import _associate_state
        current = self._safe_point()
        if self._final is not None:
            selected = self._final
            _validate_selection(self.state_repo, self._obj, current.receiver, selected)
            intent = next(v for v in current.intents if v.kind == "final")
            from uuid import UUID
            from .runtime import _stores
            self.control_store.retention.adopt(str(UUID(intent.token)), _stores(self._control))
        else:
            _checkpoint_boundary("final_capture_started")
            selected, report = _publish_checkpoint(self.state_repo, self._obj, store=self._store,
                snapshot_observer=self._environment_observer())
            self._final_report = report
            _associate_state(self, selected, "final", report=report,
                             published=lambda: _checkpoint_boundary("final_state_published"))
        self._final = selected
        self._safe_point()
        return selected

    def _environment_observer(self):
        """Retain one invocation's observation, never Store or payload authority."""
        if self._snapshot_observer is None:
            from .runtime import _preobserve_snapshot_environment
            self._snapshot_observer = _preobserve_snapshot_environment()
        return self._snapshot_observer

    def _safe_point(self, *, allow_completed=False):
        self._require_active(terminal_ok=True)
        current = self._control.inspect(self.operation_id)
        current = self._settle_known_preparation(current)
        if (current is not None and (current.run_id, current.invocation_id) == (self.run_id, self.invocation_id)
                and (self._interrupted or current.state == "interrupted" or current.interrupt_request is not None)):
            if current.interrupt_request is not None:
                self._transition(current, state="interrupted", interrupt_request=None)
                _checkpoint_boundary("interrupted_transition")
            self._interrupted = True
            raise ManagedInterrupted()
        if (allow_completed and current is not None and current.state == "completed"
                and current.run_id == self.run_id and current.invocation_id == self.invocation_id
                and current.generation == self.generation):
            return current
        self._control._fence(current, self.operation_id, self._admission,
            self.run_id, self.invocation_id, self.generation)
        if current.state not in {"running", "finalizing"}:
            raise ManagedConflictError("owner_conflict", "invocation is no longer running")
        return current

    def _settle_known_preparation(self, current):
        preparation = self._control.read_preparation(self.operation_id)
        if preparation is None or preparation.settled:
            return current
        if self._control._pending_mutations(self.operation_id):
            raise ManagedControlError("pending_reconciliation", "association acknowledgement requires separate recovery")
        if preparation.proposed.invocation_id != self.invocation_id:
            raise ManagedConflictError("owner_conflict", "preparation belongs to another invocation")
        from .runtime import _settle_preparation
        _, current = _settle_preparation(self._control, self._admission, preparation)
        if current is not None and current.invocation_id == self.invocation_id:
            self._snapshot, self._generation = current, current.generation
        return current

    def _transition(self, current, **fields):
        committed = self._control.transition(self.operation_id,
            replace(current, generation=current.generation + 1, **fields), owner=self._admission,
            expected_generation=current.generation, expected_run_id=self.run_id,
            expected_invocation_id=self.invocation_id)
        self._snapshot, self._generation = committed, committed.generation
        return committed

    def _record_failure(self, error):
        try:
            current = self._control.inspect(self.operation_id)
            if current is None or (current.run_id, current.invocation_id) != (self.run_id, self.invocation_id):
                return
            # Pending Store/index acknowledgement must be recovered, not replaced
            # by a generic failure that hides the exact committed authority cut.
            if self._control._pending_mutations(self.operation_id):
                return
            current = self._settle_known_preparation(current)
            if current.generation != self.generation and current.interrupt_request is None:
                return
            if current.state in {"running", "finalizing"}:
                if current.interrupt_request is not None or isinstance(error, KeyboardInterrupt):
                    self._transition(current, state="interrupted", interrupt_request=None)
                    self._interrupted = True
                else:
                    code = (self._checkpoint_error[1] if self._checkpoint_error is not None
                        and self._checkpoint_error[0] is error else self._failure_code)
                    self._transition(current, state="failed", failure_code=code)
            error.execution = (self.operation_id, self.run_id, self.invocation_id)
            error.checkpoint_state_ref = current.checkpoint
            if current.state == "completed":
                from .runtime import _invocation_result
                error.invocation_result = _invocation_result(self._control, current, self.invocation_id, self.cleanup_report)
                error.__dryml_execute_failure__ = error.invocation_result
            else:
                from .model import FailedInvocationResult, ExecutionIdentity
                error.failed_result = FailedInvocationResult(self._control.status(self.operation_id),
                    ExecutionIdentity(self.operation_id, self.run_id, self.invocation_id, None), current.checkpoint)
                error.__dryml_execute_failure__ = error.failed_result
        except Exception as recovery_error:
            error.recovery_error = recovery_error

    def _require_active(self, *, terminal_ok=False):
        if not self.active or self._interrupted and not terminal_ok:
            raise ManagedContextError("inactive_context", "managed context is inactive or belongs to another caller")

    def _require_mutation(self):
        self._require_active()
        if self._busy:
            raise ManagedContextError("recursive_checkpoint", "callback cannot recursively mutate its context")

    def _deactivate(self):
        self._active = False
        self._snapshot_observer = None


def _checkpoint_boundary(stage):
    """Private deterministic fault-injection seam; retains no authority proof."""


_CONTEXT_TOKEN = object()


def _create_context(**kwargs):
    return ManagedContext(_CONTEXT_TOKEN, **kwargs)


__all__ = ["ManagedContext"]
