"""Consumed-token, overlapped-window and owner lifecycle proofs for Keras."""

from threading import Event, Thread, get_ident

import numpy as np
import pytest

from dryml.core.tensor_spec import TensorSpec
from dryml.data import ArrayDataset, Epoch, GeneratorDataset, RecoverableGenerator

tf = pytest.importorskip("tensorflow")


class StatefulForward(tf.keras.Model):
    """A wrong token must not reach even non-optimizer forward effects."""

    def __init__(self):
        super().__init__()
        self.forward_calls = tf.Variable(0, trainable=False)
        self.dense = tf.keras.layers.Dense(1, kernel_initializer="zeros")

    def call(self, x):
        self.forward_calls.assign_add(1)
        return self.dense(x)


@pytest.mark.parametrize("eager", (True, False))
def test_token_validation_dominates_stateful_forward(eager):
    from dryml.models.tf.base import _keras_accounting_model

    base = StatefulForward()
    base(tf.ones((1, 1)))
    model = _keras_accounting_model(tf, base, token_start=123)
    model.compile(optimizer=tf.keras.optimizers.SGD(.01), loss="mse", run_eagerly=eager)
    step = model.train_step if eager else tf.function(model.train_step)
    before = int(base.forward_calls.numpy())
    for token in (122, 124, 2**40):
        with pytest.raises(tf.errors.InvalidArgumentError, match="consumed token"):
            step(dict(x=tf.ones((1, 1)), y=tf.ones((1, 1)), token=tf.constant(token, tf.int64)))
        assert int(base.forward_calls.numpy()) == before
        assert int(model.optimizer.iterations.numpy()) == 0
        assert not bool(model.has_completed_step.numpy())
    report = step(dict(x=tf.ones((1, 1)), y=tf.ones((1, 1)), token=tf.constant(123, tf.int64)))
    assert int(report["_dryml_token"]) == 123
    assert int(model.optimizer.iterations.numpy()) == 1
    with pytest.raises(tf.errors.InvalidArgumentError, match="consumed token"):
        step(dict(x=tf.ones((1, 1)), y=tf.ones((1, 1)), token=tf.constant(123, tf.int64)))
    assert int(base.forward_calls.numpy()) == before + 1


class IdentityPreparation:
    consumer_specs = (TensorSpec("float32", shape=(1,), batch=1, backend="tf"),) * 2

    def prepare(self, x, y):
        return x, y


class _LegacyRuntimeError(RuntimeError):
    add_note = None


class _LegacyKeyboardInterrupt(KeyboardInterrupt):
    add_note = None


@pytest.mark.parametrize("error_type", [RuntimeError, _LegacyRuntimeError,
                                       KeyboardInterrupt, _LegacyKeyboardInterrupt])
def test_bridge_source_cleanup_keeps_primary_with_compatible_notes(error_type):
    from types import SimpleNamespace
    from dryml.data import TraversalPosition
    from dryml.tf.training_data import _KerasInputBridge

    primary = error_type("producer failed")
    primary.__notes__ = ["existing diagnostic"]
    events = []

    def reserve():
        events.append("reserve")
        raise primary

    def close():
        events.append("close")
        raise OSError("source close failed")

    source = SimpleNamespace(position=TraversalPosition(0, 0, 0, False, False),
                             replay_recovery_policy="disabled", close=close)
    bridge = _KerasInputBridge(source)
    bridge._segment = object()
    bridge._view = SimpleNamespace(reserve=reserve)
    bridge._run()
    assert bridge._failure is primary and bridge._retired and bridge._closed
    assert not bridge._inflight
    assert events == ["reserve", "close"]
    assert primary.__notes__ == ["existing diagnostic", "Keras source cleanup failed: source close failed"]


@pytest.mark.parametrize("error_type", [RuntimeError, _LegacyRuntimeError,
                                       KeyboardInterrupt, _LegacyKeyboardInterrupt])
def test_bridge_exit_cleanup_keeps_primary_with_compatible_notes(error_type):
    from types import SimpleNamespace
    from dryml.data import TraversalPosition
    from dryml.tf.training_data import _KerasInputBridge

    primary = error_type("consumer failed")
    primary.__notes__ = ["existing diagnostic"]
    events = []

    def close():
        events.append("close")
        raise OSError("bridge close failed")

    source = SimpleNamespace(position=TraversalPosition(0, 0, 0, False, False),
                             replay_recovery_policy="disabled", close=close)
    bridge = _KerasInputBridge(source)
    with pytest.raises(error_type) as caught:
        try:
            raise primary
        except BaseException as error:
            assert bridge.__exit__(type(error), error, error.__traceback__) is False
            raise
    assert caught.value is primary and bridge._closed
    assert events == ["close"]
    assert primary.__notes__ == ["existing diagnostic", "Keras bridge cleanup failed: bridge close failed"]


def test_produces_bc_while_a_update_is_blocked_and_keeps_native_handoffs():
    from dryml.tf.training_data import _KerasInputBridge

    values = np.arange(1, 4, dtype=np.float32).reshape(-1, 1, 1)
    data = ArrayDataset((values, values))
    traversal = data.prepare().open(acceptance="explicit")
    bridge = _KerasInputBridge(traversal, preparation=IdentityPreparation(), capacity=3)
    try:
        segment = bridge.next_segment()
        native = iter(segment.dataset.prefetch(2))
        a = next(native)
        # A is deliberately unacknowledged, exactly as during a blocked update.
        assert bridge.wait_produced(3, timeout=5)
        b = next(native)
        c = next(native)
        assert [int(v["x"][0, 0]) for v in (a, b, c)] == [1, 2, 3]
        assert bridge.high_water_count == 3
        assert bridge.high_water_bytes >= 3 * 2 * 4
        assert bridge.pending_count == 3  # native fetch does not release capacity
        bridge.accept(bridge.candidate(int(a["token"])))
        record = bridge.capture_state()
        assert record.position.start == 1
        assert bridge.pending_count == 2
    finally:
        bridge.close()


def _experiment(*, eager=False, repo=None, data=None, epochs=1, validation_data=None):
    from dryml.models import Experiment
    from dryml.models.tf import BasicTraining, Model, Optimizer
    from tests.models.test_tf_training import ZeroKerasModel

    if data is None:
        from dryml.core.cardinality import Cardinality
        spec = TensorSpec("float32", shape=(1,), batch=1, backend="numpy")
        data = Epoch(GeneratorDataset(ReusablePairs, cardinality=Cardinality.finite(3),
                                      spec=(spec, spec), repo=repo), len=3, repo=repo)
    model = Model(ZeroKerasModel, repo=repo)
    model.obj(tf.ones((1, 1)))
    return Experiment(model, BasicTraining(
        optimizer=Optimizer(tf.keras.optimizers.SGD, learning_rate=.001, momentum=.5, repo=repo),
        compile_kwargs={"run_eagerly": eager}, epochs=epochs, verbose=0, repo=repo,
    ), train_data=data, val_data=validation_data, repo=repo)


_source_events = []
_capture_events = []
_pull_entered = Event()
_pull_release = Event()
_stall_on = None
_capture_failure = False


class ReusablePairs(RecoverableGenerator):
    """A finite, direct source reusing mutable storage and recording owner calls."""

    def __init__(self):
        self.index = 0
        self.buffer = np.zeros((1, 1), np.float32)
        _source_events.append(("open", get_ident()))

    def __next__(self):
        if self.index == 3:
            raise StopIteration
        if self.index + 1 == _stall_on:
            _pull_entered.set()
            assert _pull_release.wait(10), "test failed to release owner pull"
        self.index += 1
        self.buffer.fill(self.index)
        _source_events.append((self.index, get_ident()))
        return self.buffer, self.buffer

    def capture_state(self):
        _capture_events.append((self.index, get_ident()))
        if _capture_failure:
            raise ValueError("dynamic snapshot failed")
        return self.index

    def restore_state(self, state):
        self.index = state

    def close(self):
        _source_events.append(("close", get_ident()))


class ReadonlyForeignPairs(ReusablePairs):
    """Reuse bytearray storage through a readonly NumPy/memoryview chain."""

    def __init__(self):
        super().__init__()
        self.storage = bytearray(np.dtype("float32").itemsize)
        self.buffer = np.frombuffer(memoryview(self.storage).toreadonly(), dtype=np.float32).reshape(1, 1)

    def __next__(self):
        if self.index == 3:
            raise StopIteration
        self.index += 1
        self.storage[:] = np.array([self.index], dtype=np.float32).tobytes()
        return self.buffer, self.buffer


def test_readonly_foreign_window_native_prefetch_and_detached_exact_suffix(tmp_path):
    """Produce A/B/C before native handoff, then restore the unaccepted B/C cut."""
    from dryml.data import DatasetState
    from dryml.data.state import _split_training_window
    from dryml.tf.training_data import _KerasInputBridge

    spec = TensorSpec("float32", shape=(1,), batch=1, backend="numpy")
    prepared = Epoch(GeneratorDataset(ReadonlyForeignPairs, spec=(spec, spec)), len=3).prepare()
    bridge = _KerasInputBridge(prepared.open(acceptance="explicit"),
                               preparation=IdentityPreparation(), capacity=3)
    try:
        segment = bridge.next_segment()
        assert bridge.wait_produced(3, timeout=5)
        native = iter(segment.dataset.prefetch(2))
        first = next(native)
        bridge.accept(bridge.candidate(int(first["token"])))
        record = bridge.capture_state()
        assert record.position.start == 1
        record.save(tmp_path / "foreign-window")
        actual = [first, *native]
    finally:
        bridge.close()
    del bridge, record, native, segment
    record = DatasetState.load(tmp_path / "foreign-window")
    producer, window = _split_training_window(record)
    restored = _KerasInputBridge(prepared.open(state=producer, acceptance="explicit"),
        preparation=IdentityPreparation(), position=record.position, window=window)
    try:
        suffix = []
        for entry in restored.next_segment().dataset.prefetch(2):
            suffix.append((float(entry["x"][0, 0]), float(entry["y"][0, 0])))
            restored.accept(restored.candidate(int(entry["token"])))
    finally:
        restored.close()
    assert ([float(entry["x"][0, 0]) for entry in actual],
            [float(entry["y"][0, 0]) for entry in actual], suffix) == (
                [1, 2, 3], [1, 2, 3], [(2, 2), (3, 3)])


def test_window_ownership_keeps_safe_native_identity_and_reads_variables():
    from dryml.tf.training_data import _owned_value

    tensor = tf.constant([1.0])
    assert _owned_value(tensor) is tensor
    variable = tf.Variable([1.0])
    retained = _owned_value(variable)
    variable.assign([2.0])
    np.testing.assert_array_equal(retained.numpy(), [1.0])
    safe = np.frombuffer(memoryview(np.arange(4).tobytes()), dtype=np.int64)
    assert _owned_value(safe) is safe
    owner = np.arange(4)
    view = owner.view()
    view.flags.writeable = False
    retained = _owned_value(view)
    owner[:] = -1
    np.testing.assert_array_equal(retained, np.arange(4))


@pytest.fixture(autouse=True)
def _reset_source_controls():
    global _stall_on, _capture_failure
    _source_events.clear()
    _capture_events.clear()
    _pull_entered.clear()
    _pull_release.clear()
    _stall_on, _capture_failure = None, False
    yield
    _pull_release.set()
    _stall_on, _capture_failure = None, False


@pytest.mark.parametrize("eager", (True, False))
def test_native_overlap_capture_and_durable_exact_suffix(eager, tmp_path, monkeypatch):
    """B/C are native-queued during A's blocked update; restore bypasses conversion."""
    from dryml.core import Repo
    from dryml.models.utils import TrainingPreparation
    from dryml.tf.training_data import _KerasInputBridge

    repo = Repo(stores=tmp_path)
    exp = _experiment(eager=eager, repo=repo)
    update_entered, update_release, overlap_proved = Event(), Event(), Event()
    bridges, saved = [], []
    original_init = _KerasInputBridge.__init__
    def init(owner, *a, **k):
        original_init(owner, *a, **k)
        bridges.append(owner)
    monkeypatch.setattr(_KerasInputBridge, "__init__", init)
    original_apply = exp.train_fn.optimizer.obj.apply_gradients
    def block():
        update_entered.set()
        assert update_release.wait(10)
        return np.int64(0)
    def apply(*a, **k):
        # Test-only runtime barrier. Production token transport has no PyFunc.
        barrier = tf.py_function(block, [], tf.int64)
        with tf.control_dependencies([barrier]):
            return original_apply(*a, **k)
    monkeypatch.setattr(exp.train_fn.optimizer.obj, "apply_gradients", apply)
    def prove_overlap():
        try:
            assert update_entered.wait(10)
            bridge = bridges[0]
            assert bridge.wait_produced(3, timeout=5)
            with bridge._condition:
                assert bridge._condition.wait_for(lambda: bridge._handed >= bridge.token_start + 2, timeout=5)
            assert exp.state.step == 0
            assert bridge.pending_count == 3
            overlap_proved.set()
        finally:
            update_release.set()
    proof = Thread(target=prove_overlap)
    proof.start()
    def checkpoint():
        assert not _capture_events or exp.state.step > 1
        if exp.state.step == 1:
            assert overlap_proved.is_set()
            saved.append(repo.save(exp, deep_capture=True))
            window = exp.state.dataset_state._data()["adapter"]
            assert len(window["pending"]) == 2
            assert (window["producer_position"]["epoch"], window["producer_position"]["start"]) == (1, 0)
            assert window["producer_position"]["section_complete"] is True
            assert exp.state.dataset_state.position.start == 1
        elif exp.state.step == 2:
            raise RuntimeError("later callback")
    with pytest.raises(RuntimeError, match="later callback"):
        exp.train_fn(exp, callbacks=(checkpoint,))
    proof.join(10)
    assert not proof.is_alive() and overlap_proved.is_set()
    assert not exp.model._restore_failed
    assert exp.state._input_provider is None
    # All three producer values are in the ledger; the known terminal source
    # needs no direct-state hook to restore its two unconsumed deliveries.
    assert _capture_events == []
    caller = get_ident()
    assert all(thread != caller for _, thread in _source_events + _capture_events)
    loaded = repo.load_state_ref(saved[0], reuse_live="never")
    assert loaded.state.step == 1
    prepared = []
    original_prepare = TrainingPreparation.prepare
    monkeypatch.setattr(TrainingPreparation, "prepare", lambda *a, **k:
                        prepared.append(True) or original_prepare(*a, **k))
    suffix = []
    history = loaded.train_fn(loaded, callbacks=(lambda: suffix.append(loaded.state.step),))
    assert suffix == [2, 3] and prepared == []
    assert not any("dryml" in key or "token" in key for key in history.history)
    baseline = _experiment(eager=eager)
    baseline.train_fn(baseline)
    for actual, expected in zip(loaded.model.obj.weights, baseline.model.obj.weights):
        np.testing.assert_allclose(actual.numpy(), expected.numpy())
    for actual, expected in zip(loaded.train_fn.optimizer.obj.variables, baseline.train_fn.optimizer.obj.variables):
        np.testing.assert_allclose(actual.numpy(), expected.numpy())
    assert loaded.state.step == baseline.state.step == 3
    assert loaded.state.examples_seen == loaded.state.loss_denominator == 3
    assert loaded.state.loss_numerator == pytest.approx(baseline.state.loss_numerator)


def test_native_factories_borrow_exclusive_nonrewinding_leases():
    from dryml.data import TraversalOwnershipError
    from dryml.tf.training_data import _KerasInputBridge

    data = Epoch(ArrayDataset((np.ones((4, 1, 1), np.float32),) * 2), len=2)
    bridge = _KerasInputBridge(data.prepare().open(acceptance="explicit"), preparation=IdentityPreparation())
    try:
        segment = bridge.next_segment()
        first = bridge._values(segment.ordinal)
        a = next(first)
        competing = bridge._values(segment.ordinal)
        with pytest.raises(TraversalOwnershipError, match="Competing"):
            next(competing)
        first.close()
        second = bridge._values(segment.ordinal)
        b = next(second)
        assert int(b["token"]) == int(a["token"]) + 1
        bridge.accept(bridge.candidate(int(a["token"])))
        with pytest.raises(TraversalOwnershipError):
            bridge.candidate(int(a["token"]))
        bridge.accept(bridge.candidate(int(b["token"])))
        following = bridge.next_segment()
        with pytest.raises(TraversalOwnershipError, match="Stale"):
            next(second)
        with pytest.raises(TraversalOwnershipError, match="stale"):
            next(bridge._values(segment.ordinal))
        assert following.ordinal != segment.ordinal
    finally:
        bridge.close()


def test_nested_user_dicts_are_not_private_envelope_fields():
    from dryml.tf.training_data import _KerasInputBridge

    values = {key: np.ones((2, 1, 1), np.float32) * i for i, key in enumerate(("x", "y", "token"), 1)}
    spec = IdentityPreparation.consumer_specs[0]
    class NestedPreparation(IdentityPreparation):
        consumer_specs = ({key: spec for key in values},) * 2
    data = ArrayDataset((values, values))
    bridge = _KerasInputBridge(data.prepare().open(acceptance="explicit"), preparation=NestedPreparation())
    try:
        element = next(iter(bridge.next_segment().dataset))
        assert set(element) == {"x", "y", "token"}
        for branch in ("x", "y"):
            assert set(element[branch]) == {"x", "y", "token"}
            assert int(element[branch]["token"][0, 0]) == 3
        assert element["token"].dtype == tf.int64 and element["token"].shape == ()
    finally:
        bridge.close()


def test_active_pull_timeout_retires_provider_and_eventually_closes_on_owner(tmp_path, monkeypatch):
    global _stall_on
    from dryml.core import Repo
    from dryml.tf import training_data

    _stall_on = 2
    monkeypatch.setattr(training_data, "_ACK_TIMEOUT", .1)
    exp = _experiment(eager=True, repo=Repo(stores=tmp_path))
    owners = []
    def capture():
        assert _pull_entered.wait(5)
        owners.append(exp.state._input_provider.traversal)
        exp.state._input_provider.capture()
    try:
        with pytest.raises(TimeoutError, match="cleanup_pending=True") as error:
            exp.train_fn(exp, callbacks=(capture,))
        assert error.value.cleanup_pending
        assert exp.state._input_provider is None and exp.state.dataset_state is None
        owner = owners[0]
        assert owner._retired and not owner._closed
        assert [kind for kind, _ in _source_events].count("close") == 0
        # A replacement invocation has different authority and cannot receive the
        # timed-out request's late reply or produced value.
        monkeypatch.setattr(training_data, "_ACK_TIMEOUT", 30.)
        replacement = _experiment(eager=True)
        replacement.train_data = ArrayDataset((np.ones((1, 1, 1), np.float32),) * 2,
            spec=(TensorSpec("float32", shape=(1,), batch=1, backend="numpy"),) * 2)
        replacement.train_fn(replacement)
        record = replacement.state.dataset_state
    finally:
        _pull_release.set()
    with owner._condition:
        assert owner._condition.wait_for(lambda: owner._closed, timeout=5)
    assert [kind for kind, _ in _source_events].count("close") == 1
    source_threads = {thread for _, thread in _source_events}
    assert len(source_threads) == 1 and get_ident() not in source_threads
    assert owner.pending_count == 0 and owner._reply is None
    assert replacement.state.dataset_state is record
    with pytest.raises(Exception, match="retired"):
        owner.capture_state()


def test_capacity_wait_capture_cancel_and_parked_capture():
    from dryml.tf.training_data import _KerasInputBridge

    values = np.ones((4, 1, 1), np.float32)
    bridge = _KerasInputBridge(ArrayDataset((values, values)).prepare().open(acceptance="explicit"),
                               preparation=IdentityPreparation(), capacity=2, timeout=1)
    segment = bridge.next_segment()
    native = bridge._values(segment.ordinal)
    assert bridge.wait_produced(2, timeout=5)
    assert bridge.capture_state().position.start == 0
    bridge.close()
    assert bridge._closed
    native.close()
    parked = _KerasInputBridge(ArrayDataset((values[:1], values[:1])).prepare().open(acceptance="explicit"),
                               preparation=IdentityPreparation(), timeout=1)
    segment = parked.next_segment()
    envelope = next(parked._values(segment.ordinal))
    parked.accept(parked.candidate(int(envelope["token"])))
    assert parked.capture_state().position.epoch == 1
    parked.close()


def test_logging_never_captures_and_dynamic_capture_failure_preserves_prior(tmp_path):
    global _capture_failure
    from dryml.core import Repo
    from dryml.core.repo import RepoSaveError

    repo = Repo(stores=tmp_path)
    exp = _experiment(eager=True, repo=repo)
    prior = []
    def callback():
        global _capture_failure
        if exp.state.step == 1:
            assert _capture_events == []
            prior.append(repo.save(exp, deep_capture=True))
        elif exp.state.step == 2:
            old = exp.state.dataset_state
            _capture_failure = True
            with pytest.raises(RepoSaveError, match="snapshot-local state preparation failed") as error:
                repo.save(exp, deep_capture=True)
            assert isinstance(error.value.__cause__, ValueError)
            assert "dynamic snapshot failed" in str(error.value.__cause__)
            assert exp.state.dataset_state is old
            _capture_failure = False
    exp.train_fn(exp, callbacks=(callback,))
    assert exp.state.step == 3
    assert repo.load_state_ref(prior[0], reuse_live="never").state.step == 1


@pytest.mark.parametrize("eager", (True, False))
def test_completion_publication_depends_on_actual_optimizer_effects(eager):
    from dryml.models.tf.base import _keras_accounting_model

    base = StatefulForward()
    base(tf.ones((1, 1)))
    model = _keras_accounting_model(tf, base, token_start=7)
    optimizer = tf.keras.optimizers.SGD(.01)
    model.compile(optimizer=optimizer, loss="mse", run_eagerly=eager)
    original = optimizer.apply_gradients
    def fail(*a, **k):
        result = original(*a, **k)
        # Runtime failure follows the mutation in eager and compiled execution.
        with tf.control_dependencies([tf.convert_to_tensor(optimizer.iterations)]):
            check = tf.debugging.assert_equal(optimizer.iterations, tf.constant(0, optimizer.iterations.dtype),
                                              message="after optimizer mutation")
        with tf.control_dependencies([check]):
            return tf.identity(result)
    optimizer.apply_gradients = fail
    step = model.train_step if eager else tf.function(model.train_step)
    with pytest.raises(tf.errors.InvalidArgumentError, match="after optimizer mutation"):
        step(dict(x=tf.ones((1, 1)), y=tf.ones((1, 1)), token=tf.constant(7, tf.int64)))
    assert int(optimizer.iterations.numpy()) == 1
    assert int(model.completed_token.numpy()) == 0
    assert int(model.expected_token.numpy()) == 7
    assert not bool(model.has_completed_step.numpy())


def test_compiled_transport_has_no_python_callback_and_reuses_short_sections(monkeypatch):
    from dryml.models.tf import base
    from dryml.tf.training_data import _KerasInputBridge

    values = np.ones((6, 1, 1), np.float32)
    exp = _experiment(data=Epoch(ArrayDataset((values, values),
        spec=(TensorSpec("float32", shape=(1,), batch=1, backend="numpy"),) * 2), len=2), epochs=3)
    models, bridges, traces, segment_traces, compiles = [], [], [], [], []
    original = base._keras_accounting_model
    def model(*a, **k):
        result = original(*a, **k)
        models.append(result)
        original_step = result._owned_train_step
        def step(*a, **k):
            traces.append(True)
            return original_step(*a, **k)
        result._owned_train_step = step
        original_fit, original_compile = result.fit, result.compile
        def fit(*a, **k):
            history = original_fit(*a, **k)
            segment_traces.append(len(traces))
            return history
        result.fit = fit
        result.compile = lambda *a, **k: compiles.append(True) or original_compile(*a, **k)
        return result
    monkeypatch.setattr(base, "_keras_accounting_model", model)
    original_init = _KerasInputBridge.__init__
    monkeypatch.setattr(_KerasInputBridge, "__init__", lambda owner, *a, **k:
                        original_init(owner, *a, **k) or bridges.append(owner))
    history = exp.train_fn(exp, callbacks=(lambda: _capture_events == [] or pytest.fail("logging captured"),))
    assert len(models) == len(bridges) == 1
    assert len(traces) <= 2  # initial optimizer slot creation can require a second trace
    assert len(compiles) == 1 and len(segment_traces) == 3 and len(set(segment_traces)) == 1
    native = models[0]
    signature = dict(x=tf.TensorSpec((1, 1), tf.float32), y=tf.TensorSpec((1, 1), tf.float32),
                     token=tf.TensorSpec((), tf.int64))
    graph = tf.function(native.train_step).get_concrete_function(signature).graph
    assert not {op.type for op in graph.get_operations()} & {"PyFunc", "EagerPyFunc", "PyFuncStateless"}
    assert set(history.history) == {"loss"}
    print(dict(tensorflow=tf.__version__, keras=__import__("importlib.metadata", fromlist=["version"]).version("keras"),
               capacity=bridges[0].capacity, prefetch=2, high_water_count=bridges[0].high_water_count,
               high_water_bytes=bridges[0].high_water_bytes, models=1, compiles=len(compiles),
               fit_segments=3, segment_trace_counts=segment_traces))


@pytest.mark.parametrize("eager", (True, False))
def test_unknown_short_sections_and_discarded_tails_have_no_phantom_updates(eager):
    from dryml.data import Batch
    spec = TensorSpec("float32", shape=(1,), batch=1, backend="numpy")
    unknown = GeneratorDataset(ReusablePairs, spec=(spec, spec))
    exp = _experiment(eager=eager, data=Epoch(unknown, len=2), epochs=2)
    exp.train_fn(exp)
    assert (exp.state.step, exp.state.epoch, exp.state.examples_seen) == (3, 2, 3)
    assert len([kind for kind, _ in _source_events if type(kind) is int]) == 3
    assert [kind for kind, _ in _source_events].count("open") == 1
    # A dropped final batch changes input boundary evidence, never optimizer work.
    values = np.ones((5, 1), np.float32)
    tail = _experiment(eager=eager, data=Epoch(Batch(ArrayDataset((values, values)), 2, drop_remainder=True), len=3))
    tail.train_fn(tail)
    assert tail.state.step == 2 and tail.state.examples_seen == 4
    assert tail.state.epoch == 1


def test_restored_implicit_eof_parks_before_next_pass_until_postlude(tmp_path, monkeypatch):
    from dryml.data import DatasetState
    from dryml.data.state import _split_training_window
    from dryml.tf.training_data import _KerasInputBridge

    spec = TensorSpec("float32", shape=(1,), batch=1, backend="numpy")
    data = GeneratorDataset(ReusablePairs, spec=(spec, spec))
    prepared = data.prepare()
    bridge = _KerasInputBridge(prepared.open(acceptance="explicit"), preparation=IdentityPreparation())
    try:
        segment = bridge.next_segment()
        # Drain and acknowledge without normalizing the newly observed EOF yet.
        for entry in segment.dataset:
            bridge.accept(bridge.candidate(int(entry["token"])))
        record = bridge.capture_state()
        assert record.position.epoch == 0 and record.position.start == 3
        assert record._data()["adapter"]["producer_position"]["epoch"] == 1
        record.save(tmp_path / "window")
    finally:
        bridge.close()
    producer, window = _split_training_window(DatasetState.load(tmp_path / "window"))
    restored = _KerasInputBridge(prepared.open(state=producer, acceptance="explicit"),
        preparation=IdentityPreparation(), position=record.position, window=window)
    before = [kind for kind, _ in _source_events].count("open")
    try:
        assert restored.next_segment() is None  # boundary handoff, no next pass
        assert restored.position.epoch == 1
        assert [kind for kind, _ in _source_events].count("open") == before
        # A boundary-only saved cut is itself reusable and does not advance twice.
        boundary = restored.capture_state()
        assert boundary.position.sequence == record.position.sequence + 1
        assert restored.next_segment().epoch == 1  # caller has now finished postlude
    finally:
        restored.close()


@pytest.mark.parametrize("eager", (True, False))
def test_uncertain_native_failure_invalidates_all_owners_and_requires_exact_reload(eager, tmp_path, monkeypatch):
    from dryml.core import Repo
    from dryml.core.repo import RepoSaveError

    repo = Repo(stores=tmp_path)
    exp = _experiment(eager=eager, repo=repo)
    saved = []
    def checkpoint():
        if exp.state.step == 1:
            saved.append(repo.save(exp, deep_capture=True))
    optimizer_type = type(exp.train_fn.optimizer.obj)
    original = optimizer_type.apply_gradients
    def fail(owner, *a, **k):
        result = original(owner, *a, **k)
        with tf.control_dependencies([tf.convert_to_tensor(owner.iterations)]):
            check = tf.debugging.assert_less(owner.iterations, tf.constant(2, owner.iterations.dtype),
                                             message="uncertain second mutation")
        with tf.control_dependencies([check]):
            return tf.identity(result)
    monkeypatch.setattr(optimizer_type, "apply_gradients", fail)
    with pytest.raises(tf.errors.InvalidArgumentError, match="uncertain second mutation"):
        exp.train_fn(exp, callbacks=(checkpoint,))
    assert exp.state.step == exp.state.input_sequence == 1
    for owner in (exp, exp.model, exp.train_fn.optimizer):
        assert owner._restore_failed
        with pytest.raises(RepoSaveError, match="invalidated"):
            repo.save(owner, deep_capture=True)
    monkeypatch.setattr(optimizer_type, "apply_gradients", original)
    loaded = repo.load_state_ref(saved[0], reuse_live="never")
    loaded.train_fn(loaded)
    assert loaded.state.step == 3


def test_unsupported_native_modes_reject_before_input_acquisition(monkeypatch):
    exp = _experiment()
    monkeypatch.setattr(type(exp.train_data), "prepare", lambda _: pytest.fail("unsupported mode prepared input"))
    exp.train_fn.compile_kwargs["steps_per_execution"] = 2
    with pytest.raises(ValueError, match="steps_per_execution"):
        exp.train_fn(exp)
    exp.train_fn.compile_kwargs.clear()
    exp.train_fn.optimizer.obj.gradient_accumulation_steps = 2
    with pytest.raises(ValueError, match="gradient accumulation"):
        exp.train_fn(exp)
    exp.train_fn.optimizer.obj.gradient_accumulation_steps = None
    with tf.distribute.OneDeviceStrategy("/cpu:0").scope():
        with pytest.raises(ValueError, match="distributed"):
            exp.train_fn(exp)
    exp.train_fn.callbacks = (tf.keras.callbacks.TerminateOnNaN(),)
    with pytest.raises(ValueError, match="native callbacks"):
        exp.train_fn(exp, callbacks=(lambda: None,))
    assert _source_events == [] and exp.state.step == 0


def test_public_native_logs_and_history_do_not_receive_tokens():
    exp = _experiment(eager=True)
    logs_seen = []
    class Reporting(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            logs_seen.append(dict(logs))
        def on_epoch_end(self, epoch, logs=None):
            logs_seen.append(dict(logs))
        def on_train_end(self, logs=None):
            logs_seen.append(dict(logs))
    exp.train_fn.callbacks = (Reporting(),)
    history = exp.train_fn(exp)
    assert len(logs_seen) == 5
    assert all(set(logs) == {"loss"} for logs in logs_seen)
    assert set(history.history) == {"loss"}


def test_keras_zero_work_is_source_and_setup_inert(monkeypatch):
    exp = _experiment(epochs=0)
    monkeypatch.setattr(type(exp.train_data), "prepare", lambda _: pytest.fail("zero prepared input"))
    monkeypatch.setattr(exp.train_fn, "_make_loss", lambda *a: pytest.fail("zero built loss"))
    monkeypatch.setattr(exp.train_fn, "_training_model", lambda *a, **k: pytest.fail("zero built model"))
    assert exp.train_fn(exp) is None
    assert not exp.state.input_started and exp.state.dataset_state is None and _source_events == []


def test_implicit_eof_exact_reload_finishes_native_validation_before_reopen(tmp_path, monkeypatch):
    from dryml.core import Repo
    from dryml.data import Batch

    repo = Repo(stores=tmp_path)
    spec = TensorSpec("float32", shape=(1,), batch=1, backend="numpy")
    data = GeneratorDataset(ReusablePairs, spec=(spec, spec), repo=repo)
    validation = Batch(ArrayDataset((np.ones((1, 1), np.float32),) * 2, repo=repo), 1, repo=repo)
    exp = _experiment(eager=False, data=data, epochs=2, repo=repo, validation_data=validation)
    saved = []
    class CaptureLast(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            if exp.state.step == 3:
                saved.append(repo.save(exp, deep_capture=True))
                raise RuntimeError("captured implicit last value")
    exp.train_fn.callbacks = (CaptureLast(),)
    with pytest.raises(RuntimeError, match="implicit last value"):
        exp.train_fn(exp)
    loaded = repo.load_state_ref(saved[0], reuse_live="never")
    assert (loaded.state.epoch, loaded.state.next_batch) == (0, 3)
    _source_events.clear()
    evaluations = []
    original = tf.keras.Model.evaluate
    def evaluate(model, *a, **k):
        if not evaluations:
            assert [kind for kind, _ in _source_events].count("open") == 0
            assert loaded.state.pending_epoch_postlude == 0
            assert loaded.state.input_sequence == 4
        evaluations.append(True)
        return original(model, *a, **k)
    monkeypatch.setattr(tf.keras.Model, "evaluate", evaluate)
    loaded.train_fn(loaded)
    assert len(evaluations) == 2
    assert [kind for kind, _ in _source_events].count("open") == 1
    assert (loaded.state.epoch, loaded.state.step, loaded.state.input_sequence) == (2, 6, 8)


def test_empty_full_pass_is_not_a_phantom_update():
    from dryml.data import Batch
    values = np.zeros((0, 1), np.float32)
    exp = _experiment(data=Batch(ArrayDataset((values, values)), 1))
    with pytest.raises(ValueError, match="ended before"):
        exp.train_fn(exp)
    assert exp.state.step == 0 and int(exp.train_fn.optimizer.obj.iterations) == 0


def test_stale_exact_retry_precedes_preparation_and_loss_factory(monkeypatch):
    exp = _experiment(eager=True)
    def interrupt():
        raise RuntimeError("uncaptured callback")
    with pytest.raises(RuntimeError, match="uncaptured callback"):
        exp.train_fn(exp, callbacks=(interrupt,))
    assert not exp.model._restore_failed and exp.state._input_provider is None
    monkeypatch.setattr(type(exp.train_data), "prepare", lambda _: pytest.fail("stale retry prepared"))
    monkeypatch.setattr(exp.train_fn, "_make_loss", lambda *a: pytest.fail("stale retry built loss"))
    with pytest.raises(ValueError, match="current matching record.*durable reload"):
        exp.train_fn(exp)


def test_saved_window_rejects_malformed_positions_stages_and_public_open():
    from dryml.data import ContinuationFormatError, DatasetState, UnsupportedContinuationError
    from dryml.data.continuation import encode_value
    from dryml.tf.training_data import _KerasInputBridge

    values = np.ones((3, 1, 1), np.float32)
    prepared = ArrayDataset((values, values)).prepare()
    bridge = _KerasInputBridge(prepared.open(acceptance="explicit"), preparation=IdentityPreparation())
    try:
        segment = bridge.next_segment()
        entry = next(bridge._values(segment.ordinal))
        assert bridge.wait_produced(3, timeout=5)
        bridge.accept(bridge.candidate(int(entry["token"])))
        record = bridge.capture_state()
    finally:
        bridge.close()
    with pytest.raises(UnsupportedContinuationError, match="window adapter"):
        prepared.open(state=record)
    for key, value in (("stage", "raw"), ("position", dict(epoch=0, start=99, sequence=2,
                                                         section_complete=False, root_complete=False))):
        data = record._data()
        data["adapter"]["pending"][0][key] = value
        with pytest.raises(ContinuationFormatError):
            DatasetState._from_bytes(*encode_value(data))


def test_partial_restored_window_precedes_further_production():
    from dryml.data.state import _split_training_window
    from dryml.tf.training_data import _KerasInputBridge

    values = np.arange(1, 6, dtype=np.float32).reshape(5, 1, 1)
    prepared = Epoch(ArrayDataset((values, values)), len=5).prepare()
    owner = _KerasInputBridge(prepared.open(acceptance="explicit"),
                              preparation=IdentityPreparation(), capacity=3)
    try:
        iterator = owner._values(owner.next_segment().ordinal)
        a = next(iterator)
        assert owner.wait_produced(3, timeout=5)
        owner.accept(owner.candidate(int(a["token"])))
        record = owner.capture_state()
    finally:
        owner.close()
    producer, window = _split_training_window(record)
    converted = []
    class Preparation(IdentityPreparation):
        def prepare(self, x, y):
            assert restored._handed >= restored._restore_frontier
            converted.append(int(x[0, 0]))
            return x, y
    restored = _KerasInputBridge(prepared.open(state=producer, acceptance="explicit"),
        preparation=Preparation(), position=record.position, window=window)
    try:
        segment = restored.next_segment()
        assert converted == []
        received = []
        for entry in segment.dataset:
            received.append(int(entry["x"][0, 0]))
            restored.accept(restored.candidate(int(entry["token"])))
        assert received == [2, 3, 4, 5]
        assert converted == list(range(producer.position.start + 1, 6))
    finally:
        restored.close()
