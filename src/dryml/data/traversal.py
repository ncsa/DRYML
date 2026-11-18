"""Authoritative local traversal ownership and finite borrowed epoch views.

Sessions retain frozen execution descriptors and cheap progress facts. Physical
pass completion, logical section completion, and terminal traversal EOF are
separate. No capture, encoding, persistence, or trainer policy runs on delivery.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from threading import get_ident
from typing import Generic, Literal, TypeVar, TYPE_CHECKING

from dryml.core.cardinality import Cardinality
from dryml.methods.stream import (
    DatasetExhaustedError, _CursorLifecycle, _SectionExecution, _section_size,
)
from dryml.methods.stream_state import PullKind
from .dataset import DatasetCursor
from .errors import TraversalOwnershipError, UnsupportedContinuationError, ContinuationMismatchError

if TYPE_CHECKING:
    from .native import PreparedDataset
    from .state import DatasetState
    from .recovery import ReplayRecoveryPolicy


T = TypeVar("T")
AcceptanceMode = Literal["delivery", "explicit"]


def _count(value, name):
    if value is None:
        return 0
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact nonnegative int or None.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


@dataclass(frozen=True, slots=True)
class TraversalPosition:
    """Immutable lightweight accepted progress, read without source access.

    Args:
        epoch: Zero-based next logical epoch.
        start: Zero-based next final-output offset within that epoch.
        sequence: Monotonic accepted delivery/boundary sequence.
        section_complete: The preceding transition completed a section.
        root_complete: No further epochs remain in this traversal. Physical EOF
            of a nonempty implicit pass alone does not establish this fact.
    """

    epoch: int
    start: int
    sequence: int
    section_complete: bool
    root_complete: bool

    def __post_init__(self) -> None:
        """Reject malformed exact counts/flags without accessing an execution."""
        for name in ("epoch", "start", "sequence"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact nonnegative int.")
            _count(value, name)
        if type(self.section_complete) is not bool or type(self.root_complete) is not bool:
            raise TypeError("Traversal completion flags must be exact bools.")


@dataclass(frozen=True, slots=True)
class PendingElement(Generic[T]):
    """Ephemeral single-use reservation, with no detached continuation payload.

    ``value`` is the borrowed authored output, ``position`` the proposed accepted
    frontier, and ``section_complete`` a proved final-element fact. Only the
    originating traversal may accept/reject the opaque token. Carrier immutability
    does not freeze mutable value bytes. Pickling this execution authority raises
    TypeError; stale/foreign acceptance raises TraversalOwnershipError.
    """

    value: T
    position: TraversalPosition
    section_complete: bool
    _owner: object = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)

    def __reduce_ex__(self, protocol):
        """Reject serialization of ephemeral live acceptance authority."""
        raise TypeError("PendingElement is ephemeral execution authority.")


class _FallbackExecution(_CursorLifecycle):
    """Lazy boundary-free compatibility opener for an unqualified ordinary tree."""

    def __init__(self, prepared, epoch):
        super().__init__()
        self.prepared, self.epoch = prepared, epoch
        self.source = None
        self.value = None
        self.section_final = self.root_final = self.boundary = False

    def pull(self):
        if self.source is None:
            self.source = self._own(self._call(self.prepared.iterator, epoch=self.epoch))
        try:
            self.value = self._pull(self.source)
            return PullKind.DELIVERY
        except StopIteration:
            return PullKind.ROOT_END


class DatasetTraversal(_CursorLifecycle, Generic[T]):
    """Own one persistent local execution and lend at most one unfinished view.

    Args:
        prepared: PreparedDataset retaining qualified descriptors/selections.
        epoch: Exact nonnegative selected epoch, or None for zero.
        start: Exact nonnegative selected final-output offset, or None for zero.
        state: Optional validated DatasetState; omitted coordinates use its saved
            position. Reattachment repeats compatibility checks without acquisition.
        acceptance: ``"delivery"`` (default) or ``"explicit"`` reservations.

    Sources and positional seeking remain lazy. Explicit sections continue one
    upstream execution; implicit finite passes reopen after a boundary handoff.
    Empty implicit passes terminate. TypeError/ValueError reject invalid controls
    and provably infinite unsectioned roots before acquisition. Source/operator
    failures abort and close this owner; cleanup never masks a primary error.
    Pulls/mutations require the creating thread and reject recursive advancement.
    Independent traversals may interleave. Context exit and close release acquired
    resources once; closing unread sessions acquires nothing.
    """

    def __init__(self, prepared: PreparedDataset, *, epoch: int | None = None, start: int | None = None,
                 state: DatasetState[T] | None = None,
                 acceptance: AcceptanceMode = "delivery") -> None:
        super().__init__()
        if state is not None:
            from .state import DatasetState
            if not isinstance(state, DatasetState):
                raise TypeError("state must be a DatasetState or None.")
            state.validate(prepared)
            position = state.position
            if epoch is not None and _count(epoch, "epoch") != position.epoch or start is not None and _count(start, "start") != position.start:
                raise ContinuationMismatchError("Explicit position differs from DatasetState.")
            epoch, start = position.epoch, position.start
        self._epoch, self._start = _count(epoch, "epoch"), _count(start, "start")
        if acceptance not in ("delivery", "explicit"):
            raise ValueError("acceptance must be 'delivery' or 'explicit'.")
        self._prepared, self._acceptance = prepared, acceptance
        self._root = (prepared.graph._stream_plan._prepared.root
                      if prepared.execution_level == "stream" else None)
        self._explicit = self._root is not None and self._root.sectioned
        self._cardinality = prepared._cardinality
        if self._cardinality is None:
            self._cardinality = prepared.dataset.yield_cardinality()
        if not self._explicit and self._cardinality.is_infinite:
            raise ValueError("Epoch traversal requires boundaries for an infinite root.")
        self._thread = get_ident()
        self._busy = self._aborted = False
        self._sequence = 0
        self._section_complete = self._root_complete = False
        self._execution = None
        self._pass_nonempty = False
        self._initialized = False
        self._active = self._enumerator = self._pending = None
        self._staged = None
        self._staged_boundary = False
        self._boundary_pending = False
        self._hold_boundary_handoff = False
        self._replay_policy = "disabled"
        self._last_capture = self._last_capture_cut = None
        if state is not None:
            from .continuation import rebuild_native
            from .recovery import _restore_execution, _snapshot_root
            data = rebuild_native(state._data())
            self._root = _snapshot_root(self._root, data["snapshots"])
            self._sequence = data["position"]["sequence"]
            self._section_complete = data["position"]["section_complete"]
            self._root_complete = data["position"]["root_complete"]
            self._replay_policy = data["policy"]
            self._pass_nonempty = data["pass_nonempty"]
            self._initialized = data["initialized"]
            self._staged = data["staged"]
            self._staged_boundary = data["staged_boundary"]
            self._boundary_pending = data["boundary_pending"]
            if data["execution"] is not None:
                self._execution = self._own(_restore_execution(self._root, data["execution"], data["snapshots"]))
            self._last_capture, self._last_capture_cut = state, self._capture_cut()

    @property
    def replay_recovery_policy(self) -> ReplayRecoveryPolicy:
        """Read traversal-local disabled/enabled/if_needed policy without source IO."""
        return self._replay_policy

    @property
    def replay_recovery_enabled(self) -> bool:
        """Report explicit enablement or a retained conditional replay occurrence.

        An unresolved conditional factory alone is not active replay. Reading
        neither acquires a source nor extracts producer state.
        """
        from .recovery import _active_replay
        return self._replay_policy == "enabled" or self._replay_policy == "if_needed" and _active_replay(self)

    def _quiescent(self):
        self._check()
        if self._pending is not None:
            raise TraversalOwnershipError("Outstanding public candidate prevents capture/policy change.")

    def enable_replay_recovery(self) -> None:
        """Trust deterministic factory replay using retained original initialization.

        Returns None; no replay, reseeding, producer acquisition, or Dataset/plan
        mutation occurs. Requires an owner-thread quiescent live cut. Missing
        initial evidence raises UnsupportedContinuationError atomically; conflicting
        work/closure raises TraversalOwnershipError. Direct/indexed recovery remains
        independent. Policy and ordinary fetched counts belong to this traversal.
        """
        from .recovery import _check_policy
        self._quiescent()
        _check_policy(self, "enabled")
        self._replay_policy = "enabled"

    def disable_replay_recovery(self) -> None:
        """Disable future replay capture, canceling deferred conditional fallback.

        Returns None without changing sources, positions, direct/indexed capability,
        saved records, or one-time reconstruction of an already selected saved
        position. Owner-thread quiescent live cuts only; otherwise raises
        TraversalOwnershipError. Re-enablement uses complete retained history.
        """
        self._quiescent()
        self._replay_policy = "disabled"

    def enable_replay_recovery_if_needed(self) -> None:
        """Select per-occurrence lazy replay only where direct recovery is absent.

        Generator-free/declared-direct-only trees are source-inert no-ops; unknown
        factory products are classified only at normal acquisition, including
        future Chain/Repeat/pass occurrences. Explicit manual enable is preserved.
        Returns None. Requires a quiescent live owner-thread cut; missing retained
        initialization fails atomically with UnsupportedContinuationError. Competing
        work/closure raises TraversalOwnershipError. No source is probed or reset.
        """
        from .recovery import _check_policy, _ops, _declared_direct
        self._quiescent()
        if self._replay_policy == "enabled":
            return
        generators = [op for op in _ops(self._root) if op.kind == "source" and op.source.kind == "generator"] if self._root else []
        if not generators or all(_declared_direct(op.source.binding.gen_factory) for op in generators):
            return
        _check_policy(self, "if_needed")
        self._replay_policy = "if_needed"

    def _capture_cut(self):
        return (self.position, self._replay_policy, self._boundary_pending,
                self._pass_nonempty, self._initialized)

    def capture_state(self) -> DatasetState[T]:
        """Capture immutable detached next-output state at a quiescent accepted cut.

        Returns a DatasetState without IO, graph identity, or retention grants.
        Only this request validates/freezes dynamic buffers, owns source snapshots,
        and calls active direct-producer hooks; ordinary yields do none of these.
        Outstanding work/in-flight pulls/wrong thread raise TraversalOwnershipError.
        Unsupported sources/values/evidence raise UnsupportedContinuationError,
        leaving accepted progress and earlier records intact. Hook errors propagate.
        A closed/rejected traversal returns only its exactly matching prior capture;
        otherwise it fails without reopening resources or fabricating an old cut.
        """
        if get_ident() != self._thread or self._busy or self._pending is not None:
            raise TraversalOwnershipError("Capture requires a quiescent owner-thread accepted cut.")
        if self._closed or self._aborted:
            if self._last_capture is not None and self._last_capture_cut == self._capture_cut():
                return self._last_capture
            raise UnsupportedContinuationError("Closed traversal has no exactly matching captured record.")
        from .state import _capture
        self._busy = True
        try:
            record = _capture(self)
            self._last_capture, self._last_capture_cut = record, self._capture_cut()
            return record
        finally:
            self._busy = False

    @property
    def position(self) -> TraversalPosition:
        """Return accepted facts without pulling, preparing, or capturing state."""
        return TraversalPosition(self._epoch, self._start, self._sequence,
                                 self._section_complete, self._root_complete)

    def _check(self):
        if get_ident() != self._thread:
            raise TraversalOwnershipError("Traversal mutation requires its owner thread.")
        if self._closed or self._aborted:
            raise TraversalOwnershipError("Traversal is closed or aborted.")
        if self._busy:
            raise TraversalOwnershipError("Recursive traversal advancement is unsupported.")

    def _raw(self):
        if self._execution is None:
            execution = (_SectionExecution(self._root, 0 if self._explicit else self._epoch)
                         if self._root is not None else _FallbackExecution(self._prepared, self._epoch))
            self._execution = self._own(execution)
            self._pass_nonempty = False
        signal = self._execution.pull()
        if signal == PullKind.DELIVERY:
            self._pass_nonempty = True
        return signal

    def _release_execution(self):
        if self._execution is not None:
            execution, self._execution = self._execution, None
            self._release(execution)

    def _seek(self):
        if self._initialized:
            return
        self._initialized = True
        if self._explicit:
            for ordinal in range(self._epoch):
                while True:
                    signal = self._raw()
                    if signal == PullKind.ROOT_END:
                        raise DatasetExhaustedError(self._epoch, ordinal)
                    if signal == PullKind.SECTION_END:
                        self._execution.handoff()
                        break
        for offset in range(self._start):
            if self._raw() != PullKind.DELIVERY:
                raise DatasetExhaustedError(self._start, offset)

    def _complete(self, *, delivery=False, root=False):
        if not delivery:
            self._sequence += 1
        self._epoch += 1
        self._start = 0
        self._section_complete = True
        self._root_complete = root and self._explicit
        self._boundary_pending = True
        if self._active is not None:
            self._active._exhausted = True
            self._active = None
        if root:
            self._release_execution()

    def _handoff(self):
        """Release a consumer-held boundary without pulling/acquiring a next pass."""
        self._check()
        if not self._boundary_pending or self._pending is not None:
            raise TraversalOwnershipError("No quiescent boundary handoff is pending.")
        self._handoff_ready()

    def _handoff_ready(self):
        if self._execution is not None and self._execution.boundary:
            self._execution.handoff()
        self._boundary_pending = False

    def next_epoch(self) -> EpochView[T]:
        """Borrow the next finite view, or raise StopIteration at terminal EOF.

        May stage one output without accepting it to distinguish an empty section
        from root completion. Competing unfinished views/enumerations and recursive
        or foreign-thread advancement raise TraversalOwnershipError. Source and
        positional errors abort the owner and retain their causal identities.
        """
        return self._next_epoch()

    def _next_epoch(self, *, enumerator=None, selected_only=False, limit=None):
        self._check()
        if self._active is not None or self._pending is not None:
            raise TraversalOwnershipError("An unfinished epoch view already borrows this traversal.")
        if self._enumerator is not None and self._enumerator is not enumerator:
            raise TraversalOwnershipError("An epoch enumeration already borrows this traversal.")
        self._busy = True
        try:
            self._seek()
            if self._staged is not None or self._staged_boundary:
                self._active = EpochView(self)
                return self._active
            while not self._root_complete:
                if limit is not None and self._epoch >= limit:
                    raise StopIteration
                if self._boundary_pending:
                    if self._hold_boundary_handoff:
                        raise StopIteration
                    self._handoff_ready()
                signal = self._raw()
                if signal == PullKind.SECTION_END and self._start:
                    self._complete(root=self._execution.after_boundary_root)
                    if selected_only:
                        raise StopIteration
                    continue
                if signal == PullKind.ROOT_END:
                    nonempty = self._pass_nonempty
                    self._release_execution()
                    if self._explicit or not nonempty:
                        self._root_complete = True
                        raise StopIteration
                    self._complete(root=True)
                    if selected_only or self._hold_boundary_handoff:
                        raise StopIteration
                    continue
                self._staged_boundary = signal == PullKind.SECTION_END
                if signal == PullKind.DELIVERY:
                    self._staged = (self._execution.value, self._execution.section_final, self._execution.root_final)
                self._active = EpochView(self)
                return self._active
            raise StopIteration
        except StopIteration:
            raise
        except BaseException as error:
            self._abort(error)
            raise
        finally:
            self._busy = False

    def epochs(self, *, stop: int | None = None) -> Iterator[tuple[int, EpochView[T]]]:
        """Enumerate (ordinal, borrowed view) pairs up to an exclusive stop.

        ``stop`` is None or an exact nonnegative integer at least the current
        epoch. Equal endpoints acquire nothing. Invalid controls raise
        TypeError/ValueError. Only one active enumeration/view is allowed; ending
        an enumeration at stop neither closes the traversal nor certifies EOF.
        Each yielded view must finish before the next pair is requested.
        """
        self._check()
        if stop is not None:
            stop = _count(stop, "stop")
            if stop < self._epoch:
                raise ValueError("stop must not precede the effective epoch.")
        return _Epochs(self, stop)

    def _reserve(self, view):
        if view._exhausted:
            raise StopIteration
        self._check()
        if self._pending is not None:
            raise TraversalOwnershipError("A consumer candidate is already outstanding.")
        if self._active is not view:
            raise TraversalOwnershipError("This view does not borrow the current epoch.")
        self._busy = True
        try:
            if self._staged_boundary:
                self._staged_boundary = False
                self._complete(root=self._execution.after_boundary_root)
                raise StopIteration
            if self._staged is not None:
                value, final, root = self._staged
                self._staged = None
            else:
                signal = self._raw()
                if signal != PullKind.DELIVERY:
                    if signal == PullKind.SECTION_END:
                        self._complete(root=self._execution.after_boundary_root)
                    elif self._explicit:
                        self._complete(root=True)
                    elif self._pass_nonempty:
                        self._complete(root=True)
                    else:
                        self._root_complete = True
                        view._exhausted = True
                        self._active = None
                        self._release_execution()
                    raise StopIteration
                value, final, root = self._execution.value, self._execution.section_final, self._execution.root_final
            final = final or (root and not self._explicit)
            position = TraversalPosition(self._epoch + 1 if final else self._epoch,
                0 if final else self._start + 1, self._sequence + 1,
                final, root and self._explicit)
            candidate = PendingElement(value, position, final, self, object())
            self._pending = (candidate, view, root)
            return candidate
        except StopIteration:
            raise
        except BaseException as error:
            self._abort(error)
            raise
        finally:
            self._busy = False

    def _validate_candidate(self, candidate):
        self._check()
        if (not isinstance(candidate, PendingElement) or self._pending is None
                or candidate._owner is not self or candidate._token is not self._pending[0]._token):
            raise TraversalOwnershipError("Stale or foreign consumer candidate.")

    def accept(self, candidate: PendingElement[T]) -> None:
        """Accept the single outstanding reservation without capturing state.

        Only explicit-mode consumers call this directly. The originating opaque
        token is single-use; aliases share authority. Wrong/stale/foreign tokens,
        competing pulls and foreign threads raise TraversalOwnershipError without
        advancing. Known final boundaries normalize before this method returns.
        """
        self._validate_candidate(candidate)
        self._busy = True
        try:
            reserved, view, root = self._pending
            self._pending = None
            self._start += 1
            self._sequence += 1
            self._section_complete = False
            view._offset += 1
            if reserved.section_complete:
                self._complete(delivery=True, root=root)
        except BaseException as error:
            self._abort(error)
            raise
        finally:
            self._busy = False

    def reject(self, candidate: PendingElement[T]) -> None:
        """Reject an outstanding token, preserve accepted facts and abort/close.

        Stale/foreign tokens raise TraversalOwnershipError without progress.
        Rejection does not capture an unsaved prior cut; cleanup errors propagate.
        """
        self._validate_candidate(candidate)
        self._aborted = True
        self.close()

    def _abort(self, primary):
        self._aborted = True
        self._pending = self._staged = None
        self._close_after_failure(primary)

    def _close_after_failure(self, primary):
        try:
            super()._close_after_failure(primary)
        finally:
            self._aborted = True
            self._pending = self._staged = None

    def close(self) -> None:
        """Invalidate borrowed work and close acquired resources once, without IO.

        Unread close opens nothing. All owned resources get a cleanup attempt;
        the first error propagates, with additional cleanup evidence attached.
        Closing does not certify root EOF. Foreign-thread close raises
        TraversalOwnershipError; repeated owner-thread close is a no-op.
        """
        if get_ident() != self._thread:
            raise TraversalOwnershipError("Traversal close requires its owner thread.")
        if self._busy:
            raise TraversalOwnershipError("Cannot close during a traversal pull.")
        self._pending = self._staged = None
        try:
            super().close()
        finally:
            self._execution = self._active = self._enumerator = None

    def __enter__(self) -> DatasetTraversal[T]:
        """Enter the live owner-thread boundary without acquiring a producer."""
        self._check()
        return self

    def __exit__(self, exc_type, error, traceback) -> bool:
        """Close on the owner thread, preserving a primary consumer failure."""
        if get_ident() != self._thread:
            raise TraversalOwnershipError("Traversal context exit requires its owner thread.")
        return super().__exit__(exc_type, error, traceback)


class EpochView(_CursorLifecycle, Iterator[T], Generic[T]):
    """Finite borrowed view of one logical section; never owns a second stream.

    Args:
        traversal: Owning DatasetTraversal; obtain views through its next_epoch()
            or epochs() methods, which establish exclusive borrowing authority.

    ``epoch`` labels this section; ``position`` is its accepted next output offset. ``cardinality``
    is an exact count from the opening offset or UNKNOWN; ``max_elements`` is a
    separate optional bound. Neither property scans. Delivery-mode next accepts
    before returning; explicit-mode reserve requires an owner accept/reject.
    Exhaustion keeps the explicit upstream alive. Closing an unfinished view
    aborts its owner; closing an exhausted view is a no-op. Source/cleanup errors
    propagate while context exit preserves an escaping primary exception.
    """

    def __init__(self, traversal: DatasetTraversal[T]) -> None:
        super().__init__()
        self._owner = traversal
        self._epoch, self._offset = traversal._epoch, traversal._start
        self._exhausted = False
        root = traversal._root
        size = (_section_size(root.sections, self._epoch) if traversal._explicit else
                traversal._cardinality.require_finite() if traversal._cardinality.is_finite else None)
        if traversal._explicit and root.section_bound == 0:
            size = 0
        self._cardinality = Cardinality.UNKNOWN if size is None else Cardinality.finite(max(0, size - self._offset))
        bound = root.section_bound if traversal._explicit else size
        self._max_elements = None if bound is None else max(0, bound - self._offset)

    @property
    def epoch(self) -> int:
        """Return this view's stable zero-based logical epoch label."""
        return self._epoch

    @property
    def position(self) -> int:
        """Return the accepted next final-output offset within this section."""
        return self._offset

    @property
    def cardinality(self) -> Cardinality:
        """Return exact opening-suffix membership or UNKNOWN; never an upper bound."""
        return self._cardinality

    @property
    def max_elements(self) -> int | None:
        """Return a proved opening-suffix upper bound, or None; performs no pulls."""
        return self._max_elements

    def _read(self):
        if self._exhausted:
            raise StopIteration
        if self._owner._acceptance != "delivery":
            raise TraversalOwnershipError("Explicit acceptance requires reserve(), not next().")
        try:
            candidate = self._owner._reserve(self)
            self._owner.accept(candidate)
            return candidate.value
        except (StopIteration, TraversalOwnershipError):
            raise
        except BaseException as error:
            self._owner._abort(error)
            raise

    def __next__(self) -> T:
        """Deliver/accept one value, or stop permanently after this section ends.

        Wrong acceptance mode or thread raises TraversalOwnershipError without
        invalidating the valid borrow. Real pull failures abort the owner.
        """
        try:
            return self._read()
        except StopIteration:
            self._observed_eof = True
            raise

    def __iter__(self) -> EpochView[T]:
        """Return this finite borrowed iterator without acquisition or advancement."""
        return self

    def reserve(self) -> PendingElement[T]:
        """Reserve one value in explicit mode without advancing accepted progress.

        Returns a lightweight single-use candidate, or raises StopIteration when
        this view ends. Delivery mode, recursive/foreign-thread access and a second
        outstanding reservation raise TraversalOwnershipError. Pull failures abort
        the owner and preserve their primary identity over cleanup failures.
        """
        if self._owner._acceptance != "explicit":
            raise TraversalOwnershipError("reserve() requires explicit acceptance.")
        return self._owner._reserve(self)

    def close(self) -> None:
        """Abort/close the owner if unfinished; no-op after successful exhaustion."""
        if self._exhausted or self._closed:
            return
        owner = self._owner
        if get_ident() != owner._thread or owner._busy:
            raise TraversalOwnershipError("View close requires a quiescent owner thread.")
        self._closed = True
        owner._aborted = True
        owner.close()

    def __exit__(self, exc_type, error, traceback):
        """Release unfinished ownership, preserving a consumer's primary error."""
        if get_ident() != self._owner._thread:
            raise TraversalOwnershipError("View context exit requires its owner thread.")
        if error is None:
            self.close()
        else:
            if isinstance(error, StopIteration):
                failure = RuntimeError("Unexpected StopIteration in epoch consumer.")
                self._owner._abort(failure)
                raise failure from error
            self._owner._abort(error)
        return False


class _Epochs(Iterator):
    def __init__(self, owner, stop):
        self.owner, self.stop, self.done = owner, stop, False

    def __iter__(self):
        return self

    def __next__(self):
        if self.done:
            raise StopIteration
        owner = self.owner
        owner._check()
        if self.stop is not None and owner._epoch >= self.stop:
            self.close()
            raise StopIteration
        if owner._enumerator is not None and owner._enumerator is not self:
            raise TraversalOwnershipError("Another epoch enumeration is active.")
        owner._enumerator = self
        try:
            view = owner._next_epoch(enumerator=self, limit=self.stop)
        except StopIteration:
            self.close()
            raise
        return view.epoch, view

    def close(self):
        if self.owner._enumerator is self:
            self.owner._check()
            self.owner._enumerator = None
        self.done = True


class _SelectedEpochCursor(DatasetCursor):
    def __init__(self, traversal):
        super().__init__(iter(()))
        self._traversal = self._own(traversal)
        self._view = None

    def _read(self):
        if self._view is None:
            self._view = self._traversal._next_epoch(selected_only=True)
        value = self._pull(self._view)
        self._position += 1
        return value


__all__ = ["AcceptanceMode", "DatasetTraversal", "EpochView", "PendingElement", "TraversalPosition"]
