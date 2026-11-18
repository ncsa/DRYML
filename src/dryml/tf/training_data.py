"""TensorFlow-native delivery for prepared DRYML training Dataset batches."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from threading import Condition, Thread, get_ident
from time import monotonic

import numpy as np

from dryml.data.native import BatchPreparation, NativeTrainingCursor, PreparedDataset
from dryml.data.traversal import EpochView, PendingElement, TraversalPosition
from dryml.data.errors import TraversalOwnershipError
from dryml.tf.tensor_spec import output_signature


def as_training_dataset(data: PreparedDataset, preparation):
    """Expose one prepared Dataset pipeline as a native ``tf.data.Dataset``.

    Args:
        data: Dependency-light prepared Dataset pipeline.
        preparation: Retained x/y handoff whose consumer specs form the TensorFlow
            output signature.

    Returns:
        A re-iterable ``tf.data.Dataset`` yielding authored prepared ``(x, y)``
        batches without batching, shuffling, or repeating them again.

    Raises:
        TypeError: If ``data`` is not a :class:`PreparedDataset` or preparation
            does not expose TensorFlow consumer specs.
        DatasetExhaustedError: During iteration if a finite source ends early.

    Side Effects:
        Imports TensorFlow only when called. Each native traversal opens and closes
        an independent DRYML cursor; adapter reads never update training progress.
    """

    if not isinstance(data, PreparedDataset):
        raise TypeError("TensorFlow training input requires a PreparedDataset.")
    try:
        specs = preparation.consumer_specs
    except AttributeError as error:
        raise TypeError("TensorFlow training input requires prepared consumer specs.") from error
    import tensorflow as tf

    def prepared_values():
        """Yield one native traversal while guaranteeing cursor cleanup on cancellation."""

        with data.training_batches(preparation) as cursor:
            yield from cursor

    return tf.data.Dataset.from_generator(
        prepared_values,
        output_signature=output_signature(specs),
    )


def iter_training_batches(data: PreparedDataset | EpochView, preparation: BatchPreparation,
                          *, epoch: int = 0) -> NativeTrainingCursor:
    """Return one closeable prepared TensorFlow batch cursor for explicit loops.

    Args:
        data: PreparedDataset independent pass or borrowed finite EpochView.
        preparation: Retained x/y handoff prepared for TensorFlow.
        epoch: Independent-pass ordinal; must be zero for a borrowed view.

    Returns:
        A closeable iterator yielding native TensorFlow ``(x, y)`` batches.

    Raises:
        TypeError: If ``data`` is not PreparedDataset or EpochView.
        ValueError: Nonzero epoch supplied with a borrowed view.
        DatasetExhaustedError: During iteration if a finite source ends early.

    Side Effects:
        Opens no source until the cursor is advanced and never changes training
        progress, accepted-update state, or checkpoint association.
        Use the cursor as a context manager around updates to preserve consumer
        failures over cleanup errors; closure alone does not prove output EOF.
    """

    if isinstance(data, EpochView):
        if type(epoch) is not int or epoch != 0:
            raise ValueError("A borrowed view already selects its epoch.")
        return data.training_batches(preparation)
    if not isinstance(data, PreparedDataset):
        raise TypeError("TensorFlow training input requires a PreparedDataset or EpochView.")
    return data.training_batches(preparation, epoch=epoch)


__all__ = ["as_training_dataset", "iter_training_batches"]


_INVOCATIONS = count(1)
_ACK_TIMEOUT = 30.0
_WINDOW_CAPACITY = 8
_WINDOW_BYTES = 64 * 1024 * 1024


class _BridgeTimeout(TimeoutError):
    """A retired request, possibly awaiting cooperative source cleanup."""

    def __init__(self, operation, cleanup_pending):
        self.cleanup_pending = cleanup_pending
        super().__init__(f"Keras input {operation} acknowledgment timed out; cleanup_pending={cleanup_pending}.")


@dataclass
class _Segment:
    ordinal: int
    epoch: int
    steps: int | None
    dataset: object = None


def _owned_value(value):
    """Detach reusable host storage; immutable tensors can safely remain shared."""
    if isinstance(value, np.ndarray):
        base = value
        while isinstance(base, np.ndarray):
            if base.flags.writeable:
                return value.copy()
            base = base.base
        return value
    if isinstance(value, dict):
        return {k: _owned_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_owned_value(v) for v in value)
    if hasattr(value, "read_value"):
        return value.read_value()
    return value


def _value_bytes(value):
    if isinstance(value, dict):
        return sum(map(_value_bytes, value.values()))
    if isinstance(value, (tuple, list)):
        return sum(map(_value_bytes, value))
    if isinstance(value, np.ndarray):
        return value.nbytes
    shape, dtype = getattr(value, "shape", None), getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        return int(np.prod(shape)) * int(dtype.size)
    raise TypeError("Keras window requires sized tensor/array leaves.")


class _KerasInputBridge:
    """Private invocation owner, bounded ledger, and exclusive native leases.

    An inert traversal is handed off once to a dedicated source thread. Only that
    thread pulls, converts, captures, hands off boundaries, and closes it. The
    caller owns accepted progress; native generators merely borrow delivery leases.
    No lock spans user code or a source operation. A timeout retires all request
    authority but cannot kill Python: eventual cleanup remains with that owner.
    """

    def __init__(self, traversal, *, preparation=None, capacity=_WINDOW_CAPACITY,
                 byte_limit=_WINDOW_BYTES, timeout=None, position=None, window=None):
        if type(capacity) is not int or not 1 < capacity <= 1024:
            raise ValueError("Keras window capacity must be between 2 and 1024.")
        self.source, self.preparation = traversal, preparation
        self._caller_thread = get_ident()
        self.capacity, self.byte_limit = capacity, byte_limit
        self.timeout = _ACK_TIMEOUT if timeout is None else timeout
        invocation = next(_INVOCATIONS)
        # Keras pythonifies scalar logs as floats. Keep the complete int64 token
        # within the exactly representable integer range of that existing report.
        if invocation >= 2**21:
            raise OverflowError("Keras invocation token authority exhausted.")
        self.token_start = invocation * 2**32
        self._next_token = self.token_start
        self._position = traversal.position if position is None else position
        self.replay_recovery_policy = traversal.replay_recovery_policy
        self._condition = Condition()
        self._ledger = []
        self._bytes = self.high_water_count = self.high_water_bytes = 0
        self._handed = self.token_start - 1
        self._segment = self._view = None
        self._segment_done = False
        self._lease = None
        self._generation = self._request_id = 0
        self._request = self._reply = self._failure = None
        self._retired = self._closed = self._inflight = self._timed_out = False
        self._worker = None
        self._restored = window is not None
        self._resume_producer = False
        self._restore_frontier = self.token_start - 1
        if window:
            if len(window["pending"]) > capacity:
                raise ValueError("Saved Keras window exceeds the configured capacity.")
            from dryml.data.continuation import rebuild_native
            for entry in rebuild_native(window)["pending"]:
                self._append(entry["value"], TraversalPosition(**entry["position"]))
            self._restore_frontier = self._next_token - 1

    @property
    def position(self):
        with self._condition:
            return self._position

    @property
    def pending_count(self):
        with self._condition:
            return len(self._ledger)

    def _start(self):
        with self._condition:
            self._check_live()
            if self._worker is None:
                # The caller relinquishes the inert traversal permanently here.
                self.source._check()
                self._worker = Thread(target=self._run, name="dryml-keras-input", daemon=True)
                self._worker.start()

    def _check_live(self):
        if self._failure is not None:
            raise self._failure
        if self._retired:
            raise TraversalOwnershipError("Keras bridge/segment authority is retired.")

    def _check_caller(self):
        if get_ident() != self._caller_thread:
            raise TraversalOwnershipError("Keras bridge control requires its invocation thread.")

    def _timeout(self, operation):
        self._retired = True
        self._timed_out = True
        self._generation += 1
        self._request = self._reply = None
        self._condition.notify_all()
        return _BridgeTimeout(operation, not self._closed)

    def _ask(self, operation):
        self._check_caller()
        self._start()
        deadline = monotonic() + self.timeout
        with self._condition:
            self._check_live()
            if self._request is not None:
                raise TraversalOwnershipError("A bridge request is already outstanding.")
            self._request_id += 1
            request_id = self._request_id
            self._request = (request_id, operation)
            self._condition.notify_all()
            while self._reply is None or self._reply[0] != request_id:
                self._check_live()
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise self._timeout(operation)
                self._condition.wait(remaining)
            _, result, error = self._reply
            self._reply = None
        if error is not None:
            raise error
        return result

    def _append(self, value, position):
        if self._next_token >= self.token_start + 2**32 - 1:
            raise OverflowError("Keras invocation ordinal exhausted.")
        size = _value_bytes(value)
        if size + self._bytes > self.byte_limit:
            raise ValueError("Keras unacknowledged window exceeds its byte budget.")
        entry = PendingElement(value, position, position.section_complete, self, self._next_token)
        self._next_token += 1
        self._ledger.append(entry)
        self._bytes += size
        self.high_water_count = max(self.high_water_count, len(self._ledger))
        self.high_water_bytes = max(self.high_water_bytes, self._bytes)

    def _open_segment(self):
        with self._condition:
            if self._segment is not None and self._ledger:
                raise TraversalOwnershipError("Prior native segment left unaccepted deliveries.")
            self._generation += 1
            self._lease = None
        if self._restored and not self._ledger and self.source.position != self.position:
            self._restored = False
            self._segment_done = True
            return None
        if self._restored and self._ledger:
            self._restored = False
            if self.source.position.section_complete:
                self._view = None
                steps = len(self._ledger)
                self._segment_done = True
            else:
                self._view = None
                self._resume_producer = True
                count = EpochView(self.source).cardinality
                steps = len(self._ledger) + count.require_finite() if count.is_finite else None
                self._segment_done = False
            self._segment = _Segment(self._generation, self.position.epoch, steps)
            return self._segment
        self._restored = False
        if self.source._boundary_pending:
            self.source._handoff()
        try:
            self._view = self.source.next_epoch()
        except StopIteration:
            self._view = None
            self._segment_done = True
            return None
        cardinality = self._view.cardinality
        if cardinality.is_finite and cardinality.require_finite() == 0:
            try:
                self._view.reserve()
            except StopIteration:
                self._view = None
                self._segment_done = True
                return None
            raise TraversalOwnershipError("An empty view unexpectedly produced input.")
        self._segment = _Segment(self._generation, self._view.epoch,
                                 cardinality.require_finite() if cardinality.is_finite else None)
        self._segment_done = False
        return self._segment

    def _run(self):
        self.source._thread = get_ident()
        try:
            while True:
                with self._condition:
                    while (not self._retired and self._request is None and
                           (self._segment is None or self._segment_done or len(self._ledger) >= self.capacity
                            or self._resume_producer and self._handed < self._restore_frontier)):
                        self._condition.wait()
                    if self._retired:
                        break
                    request = self._request
                    if request is None:
                        self._inflight = True
                if request is not None:
                    request_id, operation = request
                    result = error = None
                    try:
                        if operation == "segment":
                            result = self._open_segment()
                        elif operation == "capture":
                            from dryml.data.state import _training_window
                            with self._condition:
                                position, pending = self._position, tuple(self._ledger)
                            record = self.source.capture_state()
                            result = _training_window(record, position, pending)
                        else:
                            raise ValueError("Unknown bridge request.")
                    except BaseException as failure:
                        error = failure
                    with self._condition:
                        if not self._retired and self._request == request:
                            self._request = None
                            self._reply = (request_id, result, error)
                            self._condition.notify_all()
                    continue
                try:
                    if self._resume_producer:
                        self._resume_producer = False
                        self._view = self.source.next_epoch()
                    candidate = self._view.reserve()
                    value = _owned_value(self.preparation.prepare(*candidate.value))
                    self.source.accept(candidate)
                except StopIteration:
                    with self._condition:
                        self._segment_done = True
                else:
                    with self._condition:
                        if not self._retired:
                            self._append(value, candidate.position)
                            if candidate.section_complete:
                                self._segment_done = True
                finally:
                    with self._condition:
                        self._inflight = False
                        self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._failure = error
                self._retired = True
                self._condition.notify_all()
        finally:
            try:
                self.source.close()
            except BaseException as error:
                with self._condition:
                    if self._failure is None:
                        self._failure = error
                    else:
                        self._failure.add_note(f"Keras source cleanup failed: {error}")
            finally:
                with self._condition:
                    self._closed = True
                    self._condition.notify_all()

    def next_segment(self):
        segment = self._ask("segment")
        if segment is None:
            self._sync_boundary()
            return None
        import tensorflow as tf
        x, y = output_signature(self.preparation.consumer_specs)
        segment.dataset = tf.data.Dataset.from_generator(
            lambda: self._values(segment.ordinal),
            output_signature=dict(x=x, y=y, token=tf.TensorSpec((), tf.int64)),
        ).prefetch(2)
        return segment

    def _values(self, generation):
        lease = object()
        with self._condition:
            self._check_live()
            if generation != self._generation or self._lease is not None:
                raise TraversalOwnershipError("Competing or stale Keras segment-consumer lease.")
            self._lease = lease
        try:
            while True:
                with self._condition:
                    while True:
                        self._check_live()
                        if generation != self._generation or self._lease is not lease:
                            raise TraversalOwnershipError("Stale Keras segment-consumer lease.")
                        entry = next((e for e in self._ledger if e._token > self._handed), None)
                        if entry is not None:
                            self._handed = entry._token
                            self._condition.notify_all()
                            break
                        if self._segment_done:
                            return
                        self._condition.wait()
                x, y = entry.value
                yield dict(x=x, y=y, token=np.int64(entry._token))
        finally:
            with self._condition:
                if self._lease is lease:
                    self._lease = None
                self._condition.notify_all()

    def wait_produced(self, count, *, timeout):
        deadline = monotonic() + timeout
        with self._condition:
            while len(self._ledger) < count:
                self._check_live()
                remaining = deadline - monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def candidate(self, token):
        self._check_caller()
        with self._condition:
            self._check_live()
            if not self._ledger or self._ledger[0]._token != token or token > self._handed:
                raise TraversalOwnershipError("Stale, foreign, or out-of-order consumed token.")
            return self._ledger[0]

    def _validate_candidate(self, candidate):
        if candidate._owner is not self or self.candidate(candidate._token) is not candidate:
            raise TraversalOwnershipError("Stale or foreign Keras candidate.")

    def accept(self, candidate):
        self._validate_candidate(candidate)
        with self._condition:
            self._position = candidate.position
            self._ledger.pop(0)
            self._bytes -= _value_bytes(candidate.value)
            self._condition.notify_all()

    def _sync_boundary(self):
        self._check_caller()
        with self._condition:
            self._check_live()
            if not self._ledger and self._segment_done and not self._inflight:
                self._position = self.source.position

    def _quiescent(self):
        self._check_caller()
        with self._condition:
            self._check_live()

    def capture_state(self):
        return self._ask("capture")

    def close(self):
        self._check_caller()
        with self._condition:
            timed_out = self._timed_out
            self._retired = True
            self._generation += 1
            self._request = self._reply = None
            self._condition.notify_all()
            unstarted = self._worker is None
        if unstarted:
            try:
                self.source.close()
            finally:
                with self._condition:
                    self._closed = True
            return
        with self._condition:
            if timed_out and not self._closed:
                return
            deadline = monotonic() + self.timeout
            while not self._closed:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise self._timeout("close")
                self._condition.wait(remaining)
            if self._failure is not None:
                raise self._failure

    def __exit__(self, exc_type, error, traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if error is None:
                raise
            error.add_note(f"Keras bridge cleanup failed: {cleanup}")
        return False
