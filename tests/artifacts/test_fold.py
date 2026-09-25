"""U5 deferred Fold traversal and managed-lifecycle coverage."""

from __future__ import annotations

import numpy as np
import pytest
from threading import Event, Thread

from dryml.core import Repo
from dryml.core.store.dir import DirStore
from dryml.core.tensor_spec import Dynamic, TensorSpec
from dryml.data import Dataset
from dryml.managed import ManagedConfig, ManagedConflictError, ManagedInterrupted
from dryml.methods import Accumulator, AccumulatorGroup, Method, traits


OBSERVATION_SPEC = TensorSpec("float32", shape=(Dynamic,), batch=Dynamic, backend="numpy")
CARRY_SPEC = TensorSpec("float64", shape=(), backend="numpy")


class CountingDataset(Dataset):
    """Re-iterable batched source that records traversal and iterator cleanup."""

    iterations = 0
    yields = 0
    closed = 0

    def __init__(self, batches, *, spec=OBSERVATION_SPEC, fail_after=None):
        self.batches = tuple(batches)
        self.fail_after = fail_after
        super().__init__(spec=spec)

    @classmethod
    def reset(cls):
        """Clear process-local instrumentation shared by reconstructed sources."""

        cls.iterations = cls.yields = cls.closed = 0

    def __iter__(self):
        """Yield configured batches once per fresh iterator and record closure."""

        type(self).iterations += 1
        try:
            for index, batch in enumerate(self.batches):
                if self.fail_after == index:
                    raise OSError("source failure")
                type(self).yields += 1
                yield batch
        finally:
            type(self).closed += 1


class InitialTotal(Method):
    """Allocate a scalar float64 carry from the first real source batch."""

    def __init__(self):
        self.calls = 0
        self.inferred = 0

    @traits(backend="numpy", batch_mode="batched")
    def numpy(self, observation):
        """Return an independent zero-valued carry without consuming the batch."""

        self.calls += 1
        return np.array(0.0, dtype=np.float64)

    def infer_output_spec(self, observation_spec):
        """Declare a scalar NumPy carry without executing allocation."""

        self.inferred += 1
        return CARRY_SPEC


class Adding(Accumulator):
    """Accumulate the sum of every element in an uneven batch stream."""

    def __init__(self, *, interrupt_after=None, fail=False):
        self.calls = 0
        self.selections = 0
        self.interrupt_after = interrupt_after
        self.fail = fail

    def find_implementation(self, *args, **kwargs):
        """Count Fold's one-time selected carrier binding."""

        self.selections += 1
        return super().find_implementation(*args, **kwargs)

    @traits(backend="numpy", batch_mode="batched")
    def numpy(self, observation, state):
        """Return a fresh scalar carry after one batch transition."""

        self.calls += 1
        if self.interrupt_after == self.calls:
            raise KeyboardInterrupt()
        if self.fail:
            raise ValueError("transition failure")
        return state + observation.sum(dtype=np.float64)

    def infer_output_spec(self, observation_spec, state_spec):
        """Preserve the independent scalar carry specification."""

        return state_spec


class FinishTotal(Method):
    """Convert the final scalar carry to a host scalar result."""

    fail_globally = False

    def __init__(self, *, fail=False):
        self.calls = 0
        self.fail = fail

    @traits(backend="numpy")
    def numpy(self, state):
        """Return the terminal result or raise the configured branch failure."""

        self.calls += 1
        if self.fail or type(self).fail_globally:
            raise ValueError("finalizer failure")
        return state

    def infer_output_spec(self, state_spec):
        """Declare a scalar float64 terminal result without invoking the Method."""

        return TensorSpec("float64", shape=(), backend="numpy")


def _fold(source, *, accumulator=None, finalizer=None):
    """Build one declared Fold with reusable test Method dependencies."""

    from dryml.artifacts import Fold

    return Fold(
        source,
        initial_state=InitialTotal(),
        accumulator=Adding() if accumulator is None else accumulator,
        finalize=FinishTotal() if finalizer is None else finalizer,
    )


def test_fold_declaration_is_inert_and_persists_without_source_or_method_calls(tmp_path):
    """A declared Fold retains source/Methods without traversal or helper construction."""

    CountingDataset.reset()
    source = CountingDataset([np.ones((2, 3), dtype=np.float32)])
    initializer = InitialTotal()
    accumulator = Adding()
    finalizer = FinishTotal()
    from dryml.artifacts import Fold

    fold = Fold(source, initial_state=initializer, accumulator=accumulator, finalize=finalizer)
    repo = Repo(DirStore(tmp_path / "store"))
    state = repo.save_object(fold)
    restored = Repo(DirStore(tmp_path / "store")).load_state_ref(state, reuse_live="never")

    assert (CountingDataset.iterations, CountingDataset.yields, CountingDataset.closed) == (0, 0, 0)
    assert (initializer.calls, accumulator.calls, finalizer.calls) == (0, 0, 0)
    assert fold.ready is False
    assert restored.ready is False
    assert state is not None


def test_fold_consumes_one_iterator_once_and_reuses_selected_dynamic_batch_carriers(tmp_path):
    """The first batch initializes once then transitions exactly once in one traversal."""

    CountingDataset.reset()
    source = CountingDataset([
        np.ones((2, 3), dtype=np.float32),
        np.full((1, 3), 2, dtype=np.float32),
    ])
    accumulator = Adding()
    fold = _fold(source, accumulator=accumulator)
    store = DirStore(tmp_path / "store")

    assert fold.compute(managed=ManagedConfig(state_repo=store)) is None
    assert fold.value() == 12.0
    assert (CountingDataset.iterations, CountingDataset.yields, CountingDataset.closed) == (1, 2, 1)
    assert accumulator.calls == 2
    assert accumulator.selections == 1


def test_fold_group_shares_one_stream_and_keeps_independent_branch_carries(tmp_path):
    """A grouped accumulator receives each observation once with separate carry slots."""

    CountingDataset.reset()
    source = CountingDataset([np.ones((1, 2), dtype=np.float32)] * 3)
    left, right = Adding(), Adding()
    group = AccumulatorGroup({"left": left, "right": right})
    initializer = InitialTotal()
    from dryml.artifacts import Fold

    fold = Fold(
        source,
        initial_state=MethodGroupInitializer(initializer),
        accumulator=group,
        finalize=None,
    )
    store = DirStore(tmp_path / "store")

    fold.compute(managed=ManagedConfig(state_repo=store))
    assert fold.value() == {"left": np.array(6.0), "right": np.array(6.0)}
    assert (CountingDataset.iterations, CountingDataset.yields) == (1, 3)
    assert (left.calls, right.calls) == (3, 3)


def test_fold_failures_and_interruptions_leave_no_partial_payload_then_rerun_fresh(tmp_path):
    """Pre-install failures preserve absence and a rerun gets a new traversal/carry."""

    CountingDataset.reset()
    source = CountingDataset([np.ones((1, 2), dtype=np.float32)] * 3)
    accumulator = Adding(interrupt_after=2)
    fold = _fold(source, accumulator=accumulator)
    store = DirStore(tmp_path / "store")

    with pytest.raises(ManagedInterrupted) as caught:
        fold.compute(managed=ManagedConfig(state_repo=store))
    assert type(caught.value.__cause__) is KeyboardInterrupt
    assert fold.ready is False
    assert CountingDataset.closed == 1

    accumulator.interrupt_after = None
    fold.compute(managed=ManagedConfig(state_repo=store, rerun=True))
    assert fold.value() == 6.0
    assert CountingDataset.iterations == 2


def test_fold_rejects_empty_and_source_failures_without_replacing_old_result(tmp_path):
    """Bad source input and pre-install errors retain a prior complete Value payload."""

    store = DirStore(tmp_path / "store")
    CountingDataset.reset()
    FinishTotal.fail_globally = False
    source = CountingDataset([np.ones((1, 1), dtype=np.float32)])
    fold = _fold(source)
    fold.compute(managed=ManagedConfig(state_repo=store))
    assert fold.value() == 1.0

    FinishTotal.fail_globally = True
    try:
        with pytest.raises(ValueError, match="finalizer failure"):
            fold.compute(managed=ManagedConfig(state_repo=store, rerun=True))
        assert fold.value() == 1.0
    finally:
        FinishTotal.fail_globally = False

    empty = _fold(CountingDataset([]))
    with pytest.raises(ValueError, match="empty"):
        empty.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / "empty")))
    assert empty.ready is False

    broken = _fold(CountingDataset([np.ones((1, 1), dtype=np.float32)], fail_after=0))
    with pytest.raises(OSError, match="source failure"):
        broken.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / "broken")))
    assert broken.ready is False


def test_fold_keeps_complete_live_result_when_final_managed_publication_fails(tmp_path, monkeypatch):
    """Payload installation precedes final state/control publication without false success."""

    CountingDataset.reset()
    source = CountingDataset([np.ones((1, 2), dtype=np.float32)])
    fold = _fold(source)
    store = DirStore(tmp_path / "store")

    def fail_final_save(*args, **kwargs):
        raise OSError("state publication failed")

    monkeypatch.setattr(Repo, "save_object", fail_final_save)
    with pytest.raises(OSError, match="state publication failed"):
        fold.compute(managed=ManagedConfig(state_repo=store))
    assert fold.ready is True
    assert fold.value() == 2.0
    status = fold.compute.status(state_repo=store)
    assert (status.state, status.failure_code, status.final_state_ref) == (
        "failed", "publication_error", None,
    )


def test_fold_rejects_missing_or_ambiguous_specs_and_invalid_role_results(tmp_path):
    """Input, carry, and finalizer contract errors cannot install a partial Value."""

    from dryml.artifacts import Fold

    missing = _fold(CountingDataset([np.ones((1, 1), dtype=np.float32)], spec=None))
    with pytest.raises(ValueError, match="spec"):
        missing.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / "missing")))
    assert missing.ready is False

    ambiguous = _fold(CountingDataset(
        [{"left": np.ones((1, 1), dtype=np.float32), "right": np.ones((1,), dtype=np.float32)}],
        spec={
            "left": TensorSpec("float32", shape=(1,), batch=Dynamic, backend="numpy"),
            "right": TensorSpec("float32", shape=(1,), backend="numpy"),
        },
    ))
    with pytest.raises(ValueError, match="batch"):
        ambiguous.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / "ambiguous")))
    assert ambiguous.ready is False

    zero = _fold(CountingDataset([np.empty((0, 1), dtype=np.float32)]))
    with pytest.raises(ValueError, match="zero size"):
        zero.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / "zero")))
    assert zero.ready is False

    for name, initializer, accumulator, finalizer in (
        ("initial", BadInitial(), Adding(), FinishTotal()),
        ("carry", InitialTotal(), BadCarry(), FinishTotal()),
        ("final", InitialTotal(), Adding(), BadFinalizer()),
    ):
        fold = Fold(
            CountingDataset([np.ones((1, 1), dtype=np.float32)]),
            initial_state=initializer,
            accumulator=accumulator,
            finalize=finalizer,
        )
        with pytest.raises(Exception):
            fold.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / name)))
        assert fold.ready is False


def test_fold_freezes_dynamic_coordinate_shape_from_first_observation(tmp_path):
    """A dynamic source coordinate is concrete carry state after the prototype."""

    from dryml.artifacts import Fold

    fold = Fold(
        CountingDataset([
            np.ones((1, 3), dtype=np.float32),
            np.ones((1, 4), dtype=np.float32),
        ]),
        initial_state=DynamicWidthInitial(),
        accumulator=DynamicWidthAccumulator(),
    )
    fold._install_value_payload({
        "format": "dryml.artifacts.value",
        "version": 1,
        "present": True,
        "result": np.array([9.0]),
    })

    with pytest.raises(Exception):
        fold.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / "store")))

    assert np.array_equal(fold.value(), np.array([9.0]))


def test_fold_validates_first_observation_against_declared_source_spec(tmp_path):
    """An actual prototype cannot weaken a caller's fixed coordinate contract."""

    from dryml.artifacts import Fold

    initializer = InitialTotal()
    fold = Fold(
        CountingDataset(
            [np.ones((1, 4), dtype=np.float32)],
            spec=TensorSpec("float32", shape=(3,), batch=Dynamic, backend="numpy"),
        ),
        initial_state=initializer,
        accumulator=Adding(),
        finalize=FinishTotal(),
    )

    with pytest.raises(ValueError, match="source observation"):
        fold.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / "store")))

    assert initializer.calls == 0
    assert fold.ready is False


@pytest.mark.parametrize("backend", ("numpy", "torch", "tf"))
def test_fold_rejects_zero_sized_native_observation_before_initializer(tmp_path, backend):
    """Native zero-sized inputs fail before allocation or selected Method calls."""

    initializer = NativeInitial()
    fold = _native_fold(backend, (0, 1), initializer=initializer)

    with pytest.raises(ValueError, match="zero size"):
        fold.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / backend)))

    assert initializer.calls == 0
    assert fold.ready is False


@pytest.mark.parametrize("backend", ("torch", "tf"))
def test_fold_normalizes_native_terminal_result_only_after_completion(tmp_path, backend):
    """Native carries remain native until Fold converts the final result to host data."""

    initializer = NativeInitial()
    fold = _native_fold(backend, (1, 3), fill=1, initializer=initializer)

    fold.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / backend)))

    assert fold.value() == 3.0
    assert initializer.calls == 1


@pytest.mark.parametrize("backend", ("torch", "tf"))
def test_fold_rejects_restored_native_tensor_payloads(tmp_path, backend):
    """The result-only persistence boundary never admits a heavyweight tensor."""

    fold = _fold(CountingDataset([np.ones((1, 1), dtype=np.float32)]))

    with pytest.raises(ValueError, match="unsupported"):
        fold._install_value_payload({
            "format": "dryml.artifacts.value",
            "version": 1,
            "present": True,
            "result": _native_tensor(backend, (1,)),
        })


@pytest.mark.parametrize("result", (
    np.array([object()], dtype=object),
    np.array([np.inf], dtype=np.float64),
))
def test_fold_rejects_non_lightweight_or_nonfinite_restored_results(result):
    """Restoration accepts neither object arrays nor non-finite numeric results."""

    fold = _fold(CountingDataset([np.ones((1, 1), dtype=np.float32)]))

    with pytest.raises(ValueError):
        fold._install_value_payload({
            "format": "dryml.artifacts.value",
            "version": 1,
            "present": True,
            "result": result,
        })


def test_fold_iterator_close_failure_is_not_completed(tmp_path):
    """A close error after terminal payload installation remains a failed attempt."""

    ClosingDataset.closed = 0
    source = ClosingDataset([np.ones((1, 1), dtype=np.float32)])
    fold = _fold(source)
    store = DirStore(tmp_path / "store")

    with pytest.raises(OSError, match="iterator close failure"):
        fold.compute(managed=ManagedConfig(state_repo=store))

    assert ClosingDataset.closed == 1
    assert fold.ready is True
    assert fold.compute.status(state_repo=store).state == "failed"


def test_fold_uses_one_managed_owner_and_does_not_poll_interrupt_requests(tmp_path):
    """A concurrent owner conflicts while a request alone cannot stop Fold workload."""

    CountingDataset.reset()
    BlockingAccumulator.entered, BlockingAccumulator.release = Event(), Event()
    fold = _fold(
        CountingDataset([np.ones((1, 1), dtype=np.float32)]),
        accumulator=BlockingAccumulator(),
    )
    store = DirStore(tmp_path / "store")
    outcome = []

    def invoke():
        """Run the only active Fold attempt until the test deliberately releases it."""

        try:
            outcome.append(fold.compute(managed=ManagedConfig(state_repo=store)))
        except BaseException as error:  # pragma: no cover - asserted below.
            outcome.append(error)

    worker = Thread(target=invoke)
    worker.start()
    try:
        assert BlockingAccumulator.entered.wait(10)
        with pytest.raises(ManagedConflictError):
            fold.compute(managed=ManagedConfig(state_repo=store))
        assert fold.compute.request_interrupt(state_repo=store).outcome == "requested"
        assert worker.is_alive()
        BlockingAccumulator.release.set()
        worker.join(10)
        assert not worker.is_alive()
    finally:
        BlockingAccumulator.release.set()
        worker.join(10)

    assert outcome == [None]
    status = fold.compute.status(state_repo=store)
    assert status.state == "completed"
    assert status.attempt_id is not None


class MethodGroupInitializer(Method):
    """Build two independent carries for the mapping-shaped accumulator group."""

    def __init__(self, initializer):
        self.initializer = initializer

    @traits(backend="numpy", batch_mode="batched")
    def numpy(self, observation):
        """Invoke the declared initializer twice without retaining either carry."""

        return {"left": np.array(0.0, dtype=np.float64), "right": np.array(0.0, dtype=np.float64)}

    def infer_output_spec(self, observation_spec):
        """Declare the mapping-shaped independent carry contract."""

        return {"left": CARRY_SPEC, "right": CARRY_SPEC}


class BadInitial(InitialTotal):
    """Return an invalid initial carry to exercise selected output validation."""

    @traits(backend="numpy", batch_mode="batched")
    def numpy(self, observation):
        """Deliberately violate the float64 carry specification."""

        return np.array(0.0, dtype=np.float32)


class BadCarry(Adding):
    """Return an invalid next carry while retaining the declared carry spec."""

    @traits(backend="numpy", batch_mode="batched")
    def numpy(self, observation, state):
        """Deliberately change carry dtype after one transition."""

        return np.array(0.0, dtype=np.float32)


class BadFinalizer(FinishTotal):
    """Return an invalid terminal result after a valid complete carry."""

    @traits(backend="numpy")
    def numpy(self, state):
        """Deliberately violate the inferred float64 final result spec."""

        return np.array(0.0, dtype=np.float32)


class BlockingAccumulator(Adding):
    """Block a Fold transition to expose managed ownership and request behavior."""

    entered = Event()
    release = Event()

    @traits(backend="numpy", batch_mode="batched")
    def numpy(self, observation, state):
        """Wait without a managed checkpoint, then return one valid next carry."""

        type(self).entered.set()
        assert type(self).release.wait(10)
        return state + observation.sum(dtype=np.float64)


class DynamicWidthInitial(Method):
    """Allocate coordinate carry using only the first runtime observation width."""

    @traits(backend="numpy", batch_mode="batched")
    def numpy(self, observation):
        """Return a carry whose coordinate shape follows the allocation prototype."""

        return np.zeros(observation.shape[1:], dtype=np.float32)

    def infer_output_spec(self, observation_spec):
        """Keep the pre-prototype carry coordinate unknown during pure inference."""

        return TensorSpec("float32", shape=(Dynamic,), backend="numpy")


class DynamicWidthAccumulator(Accumulator):
    """Expose a transition that would otherwise let coordinate width drift."""

    @traits(backend="numpy", batch_mode="batched")
    def numpy(self, observation, state):
        """Return a fresh carry sized from the current observation."""

        return np.zeros(observation.shape[1:], dtype=np.float32)

    def infer_output_spec(self, observation_spec, state_spec):
        """Declare that transitions retain the initializer carry slot."""

        return state_spec


class NativeInitial(Method):
    """Create native scalar carry while recording whether initialization ran."""

    def __init__(self):
        self.calls = 0

    @traits(batch_mode="batched")
    def native(self, observation):
        """Create a zero scalar in the observation's selected backend."""

        self.calls += 1
        return _native_sum(observation) * 0

    def infer_output_spec(self, observation_spec):
        """Declare a native scalar carry without allocating it during inference."""

        return TensorSpec("float32", shape=(), backend=observation_spec.backend)


class NativeAccumulator(Accumulator):
    """Advance the native scalar carry without converting observations to host data."""

    @traits(batch_mode="batched")
    def native(self, observation, state):
        """Add the native observation sum to the native carry."""

        return state + _native_sum(observation)

    def infer_output_spec(self, observation_spec, state_spec):
        """Retain the scalar carry specification."""

        return state_spec


class ClosingDataset(Dataset):
    """Dataset whose owned iterator reports a close failure after normal exhaustion."""

    closed = 0

    def __init__(self, values):
        self.values = tuple(values)
        super().__init__(spec=OBSERVATION_SPEC)

    def __iter__(self):
        """Return one explicitly closable iterator owned by this Fold invocation."""

        dataset = self

        class Iterator:
            """Iterate values and fail only when Fold releases the owned resource."""

            def __init__(self):
                self.index = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.index == len(dataset.values):
                    raise StopIteration
                value = dataset.values[self.index]
                self.index += 1
                return value

            def close(self):
                type(dataset).closed += 1
                raise OSError("iterator close failure")

        return Iterator()


def _native_tensor(backend, shape, *, fill=0):
    """Create one CPU-native float32 tensor after explicitly loading its plugin."""

    if backend == "numpy":
        return np.full(shape, fill, dtype=np.float32)
    if backend == "torch":
        import dryml.torch
        import torch

        return torch.full(shape, fill, dtype=torch.float32)
    if backend == "tf":
        import dryml.tf
        import tensorflow as tf

        return tf.fill(shape, tf.cast(fill, tf.float32))
    raise AssertionError(f"unexpected backend {backend!r}")


def _native_sum(value):
    """Sum one native tensor without converting it to a host representation."""

    if type(value).__module__.startswith("tensorflow"):
        import tensorflow as tf

        return tf.reduce_sum(value)
    return value.sum()


class NativeDataset(Dataset):
    """Create one backend-native observation only when Fold starts its traversal."""

    def __init__(self, backend, shape, *, fill=0):
        self.backend = backend
        self.shape = tuple(shape)
        self.fill = fill
        super().__init__(spec=TensorSpec(
            "float32", shape=(Dynamic,), batch=Dynamic, backend=backend,
        ))

    def __iter__(self):
        """Yield the declared CPU-native observation without storing it in the CDef."""

        yield _native_tensor(self.backend, self.shape, fill=self.fill)


def _native_fold(backend, shape, *, fill=0, initializer):
    """Build a Fold over one backend-native batched observation."""

    from dryml.artifacts import Fold

    return Fold(
        NativeDataset(backend, shape, fill=fill),
        initial_state=initializer,
        accumulator=NativeAccumulator(),
    )
