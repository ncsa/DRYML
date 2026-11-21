"""Invocation-local accepted input/native coordination, without save authority.

Core owns graph exclusion/publication and invalidation. This helper retains its
stable owner evidence once and coordinates cheap TrainState/frontier facts. Only
requested saves and normal positive exact finalization create detached records.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
import sys
from threading import get_ident
from typing import Any, TYPE_CHECKING

from dryml.core.repo import RepoSaveError, manage_repo
from dryml.core.repo_plan import validate_retained_save_plan
from dryml.data import TraversalOwnershipError, UnsupportedContinuationError
from .train_spec import TrainState

if TYPE_CHECKING:
    from dryml.core import Repo
    from dryml.data import DatasetState, DatasetTraversal, EpochView, PendingElement
    from dryml.data.native import PreparedDataset
    from .experiment import Experiment


class TrainingInputSession:
    """One closeable input lifecycle created by TrainFunction.input_session.

    Args:
        exp: Experiment whose existing graph owns all mutable training state.
        positive: Exact bool controlling whether input is established.
        exact: Exact bool requesting qualified fresh-input recovery; started input
            always retains its lane and cannot fall back after an exact failure.
        repo: Optional Repo owning retained graph evidence and cache invalidation.
        _bridge_factory: Private backend handoff for one inert traversal, accepted
            position and restored window. The returned owner services capture and
            cleanup on its sole source thread; it is never serialized.

    ``prepared`` retains the invocation's source-inert PreparedDataset;
    ``prepared`` and ``traversal`` are None for zero work/data-free calls. Positive exact sessions
    capture before normal closure when their last record is stale or absent.
    Exceptions retain accepted facts; uncertain native mutation invalidates the
    owner graph. No native state copies or extra completion barriers are added.
    All live operations require the entering thread and reject conflicting saves.
    """

    def __init__(self, exp: Experiment, *, positive: bool = True, exact: bool = True,
                 repo: Repo | None = None, _bridge_factory=None) -> None:
        if type(positive) is not bool or type(exact) is not bool:
            raise TypeError("positive and exact must be exact bools.")
        if type(exp.state) is not TrainState:
            raise TypeError("Training input requires an exact TrainState.")
        self.exp, self.positive, self.exact = exp, positive, exact
        self._state = exp.state
        self.traversal: DatasetTraversal[Any] | None = None
        self.prepared: PreparedDataset | None = None
        self._repo_context = manage_repo(repo)
        self._repo = self._plan = self._guard = self._candidate = None
        self._thread = None
        self._entered = self._closed = self._accepted = False
        self._pins = ()
        self._bridge_factory = _bridge_factory

    def __enter__(self) -> TrainingInputSession:
        """Admit owners/retry before preparation, then install one transient provider.

        Returns self. Invalid/stale/concurrent state raises before input demand.
        Failure retires partial invocation resources without changing prior records.
        """
        if self._entered or self._closed:
            raise TraversalOwnershipError("Training input sessions are single-use.")
        self._entered, self._thread = True, get_ident()
        self._repo = self._repo_context.__enter__()
        try:
            self._plan, _, _ = self._repo._state_graph_evidence(self.exp)
            self._pins = tuple((owner, name, getattr(owner, name)) for owner, names in (
                (self.exp, ("model", "train_fn", "train_data", "state")),
                (self.exp.train_fn, ("optimizer",) if hasattr(self.exp.train_fn, "optimizer") else ()),
            ) for name in names)
            state = self.exp.state
            self._state = state
            state.validate()
            if state._input_provider is not None:
                raise TraversalOwnershipError("An input invocation already owns this TrainState.")
            if state.input_started:
                state._require_current_record()
            # Exclusion is checked even on zero work, before any setup effects.
            with self._reserve():
                if not self.positive or self.exp.train_data is None:
                    return self
                prepared = self.exp.train_data.prepare()
                self.prepared = prepared
                started = state.input_started
                lane = state.input_recovery if started else ("exact" if self.exact else "positional")
                if lane == "exact":
                    try:
                        prepared._require_continuation_capability()
                    except UnsupportedContinuationError:
                        if started:
                            raise
                        lane = "positional"
                record = state.dataset_state if lane == "exact" and started else None
                window = None
                if self._bridge_factory is not None:
                    from dryml.data.state import _split_training_window
                    record, window = _split_training_window(record)
                traversal = prepared.open(state=record,
                                          epoch=None if window else state.epoch,
                                          start=None if window else state.next_batch, acceptance="explicit")
                self.traversal = traversal
                if lane == "exact" and not started:
                    try:
                        traversal.enable_replay_recovery_if_needed()
                    except UnsupportedContinuationError:
                        # Static admission failure on fresh input only. A supplied
                        # exact record never reaches this compatibility decision.
                        lane = "positional"
                traversal._hold_boundary_handoff = True
                if self._bridge_factory is not None:
                    self.traversal = self._bridge_factory(traversal, state.input_position if started else traversal.position, window)
                state.input_started = True
                state.input_recovery = lane
                self._install_position()
                state._input_provider = self
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def _check(self):
        if self._closed or not self._entered or get_ident() != self._thread:
            raise TraversalOwnershipError("Training input requires its live owner thread.")
        validate_retained_save_plan(self._plan, self.exp)
        if any(getattr(owner, name) is not value for owner, name, value in self._pins):
            raise RepoSaveError("Training input has changed a retained owner binding.")

    def _reserve(self):
        self._check()
        return self._reserve_checked()

    def _reserve_checked(self):
        """Reserve immediately checked evidence without intervening user/native work."""
        return self._repo._reserve_state_graph_evidence(
            self._plan, self._plan.nodes, self._plan.object_ids)

    def _install_position(self):
        position = self.traversal.position
        state = self.exp.state
        if (state.epoch, state.next_batch) != (position.epoch, position.start):
            raise ValueError("Training accounting and accepted input position differ.")
        state.input_sequence = position.sequence
        state.input_section_complete = position.section_complete
        state.input_root_complete = position.root_complete
        state.input_replay_policy = self.traversal.replay_recovery_policy

    def _begin_bridge_update(self):
        """Hold retained native owners until the consumed token is reported.

        Keras selects input inside its compiled function. Its graph assertions
        validate the token before forward effects; the host lease spans that call
        and the ordinary native reporting/acceptance callback.
        """
        self._check()
        if self._guard is not None:
            raise TraversalOwnershipError("A native step is already in progress.")
        self._guard = self._reserve_checked()
        self._candidate, self._accepted = None, False

    def _end_bridge_update(self, *, failed=False):
        """Release an outstanding native lease, invalidating uncertain mutations."""
        if self._guard is not None:
            if failed and not self._accepted:
                self.invalidate_native()
            self._guard.release()
        self._guard = self._candidate = None

    def _bridge_completed(self, token):
        """Bind completed native evidence to one ledger candidate under the lease."""
        self._check()
        if self._guard is None:
            raise TraversalOwnershipError("Completed token has no native owner guard.")
        self._candidate = self.traversal.candidate(token)
        return self._candidate

    @contextmanager
    def update(self, candidate: PendingElement[Any], *, repairable: bool = False) -> Iterator[None]:
        """Fence one prevalidated candidate through native mutation and acceptance.

        Args:
            candidate: Outstanding PendingElement from this session's traversal.
            repairable: Functional backend guarantees cheap prior-reference repair
                on installation failure; failed repair must invalidate_native().
        Yields:
            None; perform native mutation and record_train_update in this scope.
        Raises:
            TraversalOwnershipError: Wrong token/thread or nested update.
            RepoSaveError: Invalid/changed/reserved native owners.
            BaseException: Native/accounting failures propagate and invalidate all
                affected owners when complete acceptance cannot be established.
        Side Effects:
            Acquires a nonblocking Core lease from retained evidence, released
            before callbacks. No graph reconstruction, SQL, capture, rollback copy,
            or recovery-only native synchronization occurs per update.
            Read-through validation runs before native work and again at acceptance;
            immediate lease acquisition reuses only the preceding local check.
        """
        self._check()
        if type(repairable) is not bool:
            raise TypeError("repairable must be an exact bool.")
        if self.traversal is None or self._guard is not None:
            raise TraversalOwnershipError("Update requires one live input reservation.")
        self.traversal._validate_candidate(candidate)
        if self.exp.state.input_position != self.traversal.position:
            raise ValueError("Training accounting does not match its accepted input cut.")
        self._guard = self._reserve_checked()
        self._candidate, self._accepted = candidate, False
        try:
            yield
            if not self._accepted:
                raise RuntimeError("Native update exited without accepted input accounting.")
        except BaseException:
            if not self._accepted and not repairable:
                self._repo._invalidate_restoration_target(self._plan.nodes)
            raise
        finally:
            self._end_bridge_update()

    def _accept_update(self, candidate, *, examples, loss, complete_epoch, epoch_metrics):
        self._check()
        if (self._guard is None or self._accepted or self._candidate is None
                or candidate._token is not self._candidate._token):
            raise TraversalOwnershipError("Input acceptance requires its native update guard.")
        self.traversal._validate_candidate(candidate)
        if complete_epoch and not candidate.section_complete:
            raise ValueError("Consumer completion disagrees with the input candidate.")
        state = self.exp.state
        state.record_update(examples=examples, loss=loss)
        if candidate.section_complete:
            state.finish_epoch(postlude_pending=True, metrics=epoch_metrics)
        elif epoch_metrics is not None:
            raise ValueError("epoch_metrics require a completed input section.")
        _training_input_boundary("after_accounting")
        self.traversal.accept(candidate)
        _training_input_boundary("after_frontier")
        self._install_position()
        self._accepted = True
        self._guard.release()
        self._guard = None

    def sync_boundary(self) -> None:
        """Install newly discovered EOF/empty-section facts without an optimizer step.

        Call after reserve/next_epoch reports a boundary. Normalizes a single new
        section and its pending postlude under retained Core exclusion. Repeated
        calls are idempotent. Inconsistent facts invalidate the owner graph and
        raise; capture is never performed. Finish the postlude before handoff.
        """
        self._check()
        if self.traversal is None:
            return
        sync = getattr(self.traversal, "_sync_boundary", None)
        if sync is not None:
            sync()
        with self._reserve():
            state, position = self.exp.state, self.traversal.position
            try:
                if position.epoch == state.epoch + 1 and position.start == 0:
                    state.finish_epoch(postlude_pending=True)
                self._install_position()
            except BaseException:
                self._repo._invalidate_restoration_target(self._plan.nodes)
                raise

    def handoff(self) -> None:
        """Release a completed input boundary after its TrainState postlude finishes.

        Returns None without acquiring the next pass. A pending postlude or invalid
        traversal raises ValueError/TraversalOwnershipError without advancement.
        """
        self._check()
        if self.exp.state.pending_epoch_postlude is not None:
            raise ValueError("Finish the prior epoch postlude before input handoff.")
        self.traversal._handoff()

    def epoch_views(self, *, stop: int) -> Iterator[tuple[int, EpochView[Any] | None]]:
        """Yield (epoch, view) including boundary-only restored suffix handoffs.

        ``stop`` is an exclusive logical epoch. A None view reports newly
        discovered EOF of a previously nonempty saved pass. Consumers finish that
        epoch's postlude before resuming this iterator. No next-pass acquisition
        occurs until then. Boundary facts are synchronized without an update;
        exceptions propagate and never downgrade exact recovery.

        Args:
            stop: Exclusive exact-integer epoch, at least the current epoch.
        Yields:
            Logical epoch and borrowed view, or None for a boundary-only handoff.
        Raises:
            TypeError, ValueError: Invalid stop or unfinished prior postlude.
            TraversalOwnershipError: Missing traversal or inactive/wrong-thread use.
            Source and boundary synchronization errors propagate without fallback.
        Side Effects:
            Advances only invocation input and accepted boundary facts; never
            creates an optimizer update or captures a record.
        """
        self._check()
        if self.traversal is None:
            raise TraversalOwnershipError("Epoch views require a positive input session.")
        if type(stop) is not int:
            raise TypeError("stop must be an exact integer.")
        if stop < self.exp.state.epoch:
            raise ValueError("stop must not precede the current epoch.")
        while True:
            if self.traversal._boundary_pending:
                self.handoff()
            if self.exp.state.epoch >= stop:
                return
            epoch = self.exp.state.epoch
            try:
                view = self.traversal.next_epoch()
            except StopIteration:
                self.sync_boundary()
                if self.exp.state.pending_epoch_postlude == epoch:
                    yield epoch, None
                    continue
                return
            yield epoch, view
            self.sync_boundary()

    def invalidate_native(self) -> None:
        """Retire affected owners after a positively identified delayed native failure.

        Returns None and evicts every retained owner through Core's existing
        invalidation boundary. Backends call this when their ordinary completion
        check reports uncertain mutation, even after provisional local acceptance.
        Capture/value or callback errors alone must not call it. Subsequent update,
        retry and save require fresh durable loading; no rollback is claimed.
        Wrong-thread/closed calls raise TraversalOwnershipError.
        """
        self._check()
        self._repo._invalidate_restoration_target(self._plan.nodes)

    def capture(self) -> DatasetState[Any] | None:
        """Return/install a current detached record at an accepted owner-thread cut.

        Reuses an already matching record. Wrong-thread, in-flight native work,
        invalid owners or mismatched progress fail explicitly without source pulls.
        A private native bridge may wait for a bounded owner-thread capture handshake;
        timeout retires provider authority and reports pending cleanup. Dynamic
        capture errors preserve accepted facts and prior records;
        they are save/finalization failures, not fabricated native rollback.
        Core's caller already fences actual payload saving; this adds no save lock.
        """
        self._check()
        state = self.exp.state
        if self._guard is not None or self._candidate is not None and not self._accepted:
            raise TraversalOwnershipError("Cannot capture an in-progress native update.")
        if not state.input_started or state.input_recovery == "positional":
            state._require_current_record()
            return None
        self.traversal._quiescent()
        if (state.input_position != self.traversal.position
                or state.input_replay_policy != self.traversal.replay_recovery_policy):
            raise ValueError("Capture requires matching accepted input/accounting facts; synchronize the boundary first.")
        state.validate()
        try:
            state._require_current_record()
        except ValueError:
            try:
                record = self.traversal.capture_state()
            except TimeoutError:
                if self._state._input_provider is self:
                    self._state._input_provider = None
                raise
            if (record.position != state.input_position
                    or record.replay_recovery_policy != state.input_replay_policy):
                raise ValueError("Capture does not match accepted training input.")
            state.dataset_state = record
            state._require_current_record()
        return state.dataset_state

    def __exit__(self, exc_type, error, traceback) -> bool:
        """Finalize normally, retire the provider on every exit, and close resources.

        Returns False; cleanup preserves an escaping primary exception. Successful
        positive exact completion requires a matching capture even with no updates.
        """
        try:
            if error is None and self.traversal is not None:
                self.capture()
        except BaseException:
            exc_type, error, traceback = sys.exc_info()
            raise
        finally:
            self._end_bridge_update(failed=True)
            if self._state._input_provider is self:
                self._state._input_provider = None
            self._closed = True
            try:
                if self.traversal is not None:
                    self.traversal.__exit__(exc_type, error, traceback)
            finally:
                self._repo_context.__exit__(exc_type, error, traceback)
        return False


def _training_input_boundary(stage):
    """Fault-injection seam between small accounting/frontier installations."""


__all__ = ["TrainingInputSession"]
