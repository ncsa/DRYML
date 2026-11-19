"""Deterministic work and exact-frontier proofs for linear Array Batch execution."""

from collections import Counter
from contextlib import contextmanager, nullcontext
from dataclasses import replace

import numpy as np
import pytest

import dryml.data as data
from dryml.core.tensor_spec import TensorSpec
from dryml.data.recovery import _capture_execution
from dryml.methods import ImplementationSelectionError, Method, traits
from dryml.methods.stream import _SectionExecution
from dryml.methods.stream_state import PullKind
from tests.data.test_traversal_preparation import AddTen, Identity


@contextmanager
def _generic(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(_SectionExecution, "_batch_array", _SectionExecution._batch, raising=False)
        yield


def _root(dataset):
    return dataset.prepare().graph._stream_plan._prepared.root


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if type(value) is dict:
        return {key: _plain(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return type(value)(_plain(item) for item in value)
    return value


def _frontier(execution):
    return (_plain(_capture_execution(execution, "disabled")), _plain(execution.value),
            [_frontier(child) for child in execution.children.values()])


@pytest.mark.parametrize("mapped", (False, True))
def test_linear_batch_eliminates_internal_pull_step_delivery_calls(monkeypatch, mapped):
    """Eleven rows need only three Batch steps/deliveries and four public pulls."""
    source = data.ArrayDataset(np.arange(11, dtype=np.int64))
    root = _root(data.Batch(data.Map(source, Identity()) if mapped else source, 4))
    counts = Counter()
    collate = root.selected

    def selected_collate(values):
        counts["collate", "batch"] += 1
        return collate(values)

    root = replace(root, selected=selected_collate)

    def counted(name, function):
        def call(execution, *args, **kwargs):
            counts[name, execution.op.kind] += 1
            return function(execution, *args, **kwargs)
        return call

    for name in ("pull", "_step", "_deliver", "_map"):
        monkeypatch.setattr(_SectionExecution, name, counted(name, getattr(_SectionExecution, name)))

    def forbidden(*args, **kwargs):
        pytest.fail("ordinary Batch delivery captured or encoded state")

    import dryml.data.recovery as recovery
    import dryml.data.continuation as continuation
    import dryml.methods.stream as stream
    from dryml.data.source import _IndexedDatasetCursor
    read = _IndexedDatasetCursor._read
    retain = stream._retain_buffer

    def counted_read(cursor):
        counts["read", "source"] += 1
        return read(cursor)

    def counted_retain(value):
        counts["retain", "batch"] += 1
        return retain(value)

    monkeypatch.setattr(_IndexedDatasetCursor, "_read", counted_read)
    monkeypatch.setattr(stream, "_retain_buffer", counted_retain)
    monkeypatch.setattr(recovery, "_capture_execution", forbidden)
    monkeypatch.setattr(continuation, "encode_value", forbidden)
    with _SectionExecution(root) as execution:
        values = []
        while execution.pull() == PullKind.DELIVERY:
            values.append(execution.value.tolist())
        assert values == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10]]
        source_execution = execution.children[0]
        if mapped:
            assert source_execution.count == 11
            source_execution = source_execution.children[0]
        assert source_execution.count == source_execution.source.position == 11
        assert not source_execution.source.observed_eof
    expected = Counter({("pull", "batch"): 4, ("_step", "batch"): 3,
                        ("_deliver", "batch"): 3, ("read", "source"): 11,
                        ("retain", "batch"): 11, ("collate", "batch"): 3})
    if mapped:
        expected["_map", "map"] = 11
    assert counts == expected


@pytest.mark.parametrize("mapped", (False, True))
def test_linear_batch_publishes_child_metadata_once_per_batch(monkeypatch, mapped):
    """Eleven row frontiers collapse to three publications per child field."""
    source = data.ArrayDataset(np.arange(11, dtype=np.int64))
    root = _root(data.Batch(data.Map(source, Identity()) if mapped else source, 4))
    fields = {"count", "value", "produced", "section_final", "root_final", "terminal"}
    counts = Counter()
    watched = set()
    assign = _SectionExecution.__setattr__

    def counted(execution, name, value):
        if id(execution) in watched and name in fields:
            counts[execution.op.kind, name] += 1
        assign(execution, name, value)

    monkeypatch.setattr(_SectionExecution, "__setattr__", counted)
    for generic in (True, False):
        counts.clear()
        with _generic(monkeypatch) if generic else nullcontext():
            with _SectionExecution(root) as execution:
                child = execution._child()
                children = (child, child._child()) if mapped else (child,)
                watched.update(id(item) for item in children)
                while execution.pull() == PullKind.DELIVERY:
                    pass
                watched.clear()
        for kind in (("source", "map") if mapped else ("source",)):
            expected = 11 if generic else 3
            assert counts[kind, "count"] == expected
            assert counts[kind, "produced"] == expected
            if not generic:
                assert {field: counts[kind, field] for field in fields} == dict.fromkeys(fields, 3)


@pytest.mark.parametrize("mapped", (False, True))
@pytest.mark.parametrize("length,drop", ((0, False), (1, False), (8, False), (11, False), (11, True), (1, True)))
def test_linear_batch_matches_generic_values_and_all_physical_frontiers(monkeypatch, mapped, length, drop):
    source = data.ArrayDataset(np.arange(length, dtype=np.int64))
    root = _root(data.Batch(data.Map(source, AddTen()) if mapped else source, 4, drop_remainder=drop))
    runs = []
    for generic in (True, False):
        with _generic(monkeypatch) if generic else nullcontext():
            with _SectionExecution(root) as execution:
                states = [_frontier(execution)]
                while True:
                    signal = execution.pull()
                    states.append((signal, _frontier(execution)))
                    if signal == PullKind.ROOT_END:
                        break
                runs.append(states)
    assert runs[0] == runs[1]


@pytest.mark.parametrize("mapped", (False, True))
def test_linear_batch_partial_snapshot_restores_across_both_execution_paths(monkeypatch, mapped):
    """A real record resumes at row eight without replaying selected Map calls."""
    from dryml.data.source import _IndexedDatasetCursor

    source = data.ArrayDataset({"x": np.arange(22).reshape(11, 2)})
    prepared = data.Batch(data.Map(source, Identity()) if mapped else source, 4).prepare()
    reads, calls = [], []
    read, apply = _IndexedDatasetCursor._read, _SectionExecution._map

    def counted_read(cursor):
        reads.append(cursor.position)
        return read(cursor)

    def counted_map(execution, value):
        calls.append(True)
        return apply(execution, value)

    monkeypatch.setattr(_IndexedDatasetCursor, "_read", counted_read)
    monkeypatch.setattr(_SectionExecution, "_map", counted_map)
    records = []
    expected = None
    for generic in (True, False):
        with _generic(monkeypatch) if generic else nullcontext():
            with prepared.open() as traversal:
                view = traversal.next_epoch()
                next(view)
                next(view)
                record = traversal.capture_state()
                records.append(record)
                physical = record._data()["execution"]
                while physical["kind"] != "source":
                    physical = physical["children"][0]
                assert physical["source"]["fetched"] == 8
                suffix = _plain(list(view))
                if expected is None:
                    expected = suffix
                assert suffix == expected
                done = traversal.capture_state()
            for state, remaining in ((record, expected), (done, [])):
                with prepared.open(state=state) as resumed:
                    assert _plain([value for _, epoch in resumed.epochs(stop=1) for value in epoch]) == remaining
    assert records[0]._data()["execution"] == records[1]._data()["execution"]
    for record in records:
        for generic in (True, False):
            reads.clear()
            calls.clear()
            with _generic(monkeypatch) if generic else nullcontext():
                with prepared.open(state=record) as resumed:
                    assert _plain(list(resumed.next_epoch())) == expected
            assert reads == [8, 9, 10]
            assert len(calls) == (3 if mapped else 0)


def test_same_array_occurrences_keep_independent_batch_frontiers(monkeypatch):
    source = data.ArrayDataset(np.arange(11, dtype=np.int64))
    prepared = data.Zip((data.Batch(source, 2), data.Batch(data.Map(source, Identity()), 3))).prepare()
    records = []
    for generic in (True, False):
        with _generic(monkeypatch) if generic else nullcontext():
            with prepared.open() as traversal:
                view = traversal.next_epoch()
                next(view)
                next(view)
                state = traversal.capture_state()
                records.append(state)
                left, right = state._data()["execution"]["children"]
                assert left["children"][0]["source"]["fetched"] == 4
                assert right["children"][0]["children"][0]["source"]["fetched"] == 6
                expected = _plain(list(view))
            with prepared.open(state=state) as resumed:
                assert _plain(list(resumed.next_epoch())) == expected
    assert _plain(records[0]._data()["execution"]) == _plain(records[1]._data()["execution"])


@pytest.mark.parametrize("failure", (None, "open", "read", "map", "collate"))
def test_linear_batch_lazy_ownership_and_fault_cleanup(monkeypatch, failure):
    """Every acquired cursor closes once; non-pull StopIteration is never EOF."""
    root = _root(data.Batch(data.Map(data.ArrayDataset(np.arange(11)), Identity()), 4))
    original_source = root.inputs[0].inputs[0]
    for generic in (True, False):
        events = []
        held = []

        def open_source(epoch):
            events.append("open")
            if failure == "open":
                raise StopIteration("acquire failed")
            cursor = original_source.source.open(epoch)
            held.append(cursor)
            original_read, original_close = cursor._item_at, cursor.close

            def read(index):
                events.append(index)
                if failure == "read" and index == 5:
                    raise StopIteration("item access failed")
                return original_read(index)

            def close():
                events.append("close")
                original_close()
                if failure is not None:
                    raise ValueError("cleanup failed")

            cursor._item_at, cursor.close = read, close
            return cursor

        def selected(value):
            if failure == "map" and int(value) == 5:
                raise StopIteration("selected Map failed")
            return root.inputs[0].selected(value)

        def collate(values):
            if failure == "collate" and int(values[0]) == 4:
                raise StopIteration("selected collate failed")
            return root.selected(values)

        source_op = replace(original_source, source=replace(original_source.source, open=open_source))
        map_op = replace(root.inputs[0], inputs=(source_op,), selected=selected)
        op = replace(root, inputs=(map_op,), selected=collate)
        with _generic(monkeypatch) if generic else nullcontext():
            untouched = _SectionExecution(op)
            untouched.close()
            assert events == []
            execution = _SectionExecution(op)
            if failure is None:
                assert execution.pull() == PullKind.DELIVERY
                assert events == ["open", 0, 1, 2, 3]
                execution.close()
            else:
                with pytest.raises(RuntimeError, match="Unexpected StopIteration") as caught:
                    while execution.pull() == PullKind.DELIVERY:
                        pass
                assert isinstance(caught.value.__cause__, StopIteration)
                if failure != "open":
                    assert any("cleanup failed" in note for note in caught.value.__notes__)
            execution.close()
            assert events.count("open") == 1
            assert events.count("close") == (failure != "open")
            assert all(cursor._closed and not cursor.observed_eof for cursor in held)


@pytest.mark.parametrize("failure", ("open", "read", "map", "retain", "collate"))
def test_linear_batch_failure_frontier_matches_generic_before_cleanup(monkeypatch, failure):
    """Failed pulls publish exactly the generic partial facts before owner close."""
    import dryml.methods.stream as stream

    root = _root(data.Batch(data.Map(data.ArrayDataset(np.arange(11)), Identity()), 4))
    original_source = root.inputs[0].inputs[0]
    close_after_failure, retain = _SectionExecution._close_after_failure, stream._retain_buffer
    frontiers = []
    primary = ValueError("selected failure")

    def before_cleanup(execution, error):
        if execution.op.kind == "batch":
            frontiers.append(_frontier(execution))
        return close_after_failure(execution, error)

    def open_source(epoch):
        if failure == "open":
            raise primary
        cursor = original_source.source.open(epoch)
        read = cursor._item_at

        def item_at(index):
            if failure == "read" and index == 5:
                raise primary
            return read(index)

        cursor._item_at = item_at
        return cursor

    def selected(value):
        if failure == "map" and int(value) == 10:
            raise primary
        return root.inputs[0].selected(value)

    def retained(value):
        if failure == "retain" and int(value) == 5:
            raise primary
        return retain(value)

    def collate(values):
        if failure == "collate" and int(values[0]) == 4:
            raise primary
        return root.selected(values)

    source = replace(original_source, source=replace(original_source.source, open=open_source))
    mapper = replace(root.inputs[0], inputs=(source,), selected=selected)
    op = replace(root, inputs=(mapper,), selected=collate)
    monkeypatch.setattr(_SectionExecution, "_close_after_failure", before_cleanup)
    monkeypatch.setattr(stream, "_retain_buffer", retained)
    for generic in (True, False):
        with _generic(monkeypatch) if generic else nullcontext():
            with _SectionExecution(op) as execution:
                with pytest.raises(ValueError) as caught:
                    while execution.pull() == PullKind.DELIVERY:
                        pass
                assert caught.value is primary
                assert execution._closed
                assert all(child._closed for child in execution.children.values())
            assert not execution.children
    assert len(frontiers) == 2
    assert frontiers[0] == frontiers[1]


def test_linear_batch_busy_callbacks_reject_capture_and_reentrant_advance(monkeypatch):
    """Map cannot observe an in-flight frontier through public traversal access."""
    prepared = data.Batch(data.Map(data.ArrayDataset(np.arange(11)), Identity()), 4).prepare()
    apply = _SectionExecution._map
    calls = []

    def selected(execution, value):
        with pytest.raises(data.TraversalOwnershipError):
            traversal.capture_state()
        with pytest.raises(data.TraversalOwnershipError):
            traversal.next_epoch()
        calls.append(int(value))
        return apply(execution, value)

    monkeypatch.setattr(_SectionExecution, "_map", selected)
    with prepared.open() as traversal:
        assert _plain(list(traversal.next_epoch())) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10]]
        assert traversal.capture_state() is not None
    assert calls == list(range(11))


@pytest.mark.parametrize("failure", ("read", "map", "collate"))
def test_linear_batch_failure_aborts_owner_without_fabricating_current_record(monkeypatch, failure):
    from dryml.data.source import _IndexedDatasetCursor

    prepared = data.Batch(data.Map(data.ArrayDataset(np.arange(11)), Identity()), 4).prepare()
    root = prepared.graph._stream_plan._prepared.root
    read, apply, call = _IndexedDatasetCursor._read, _SectionExecution._map, _SectionExecution._call
    primary = ValueError("third batch failed")

    def source_read(cursor):
        if failure == "read" and cursor.position == 10:
            raise primary
        return read(cursor)

    def mapped(execution, value):
        if failure == "map" and int(value) == 10:
            raise primary
        return apply(execution, value)

    def called(function, *args, **kwargs):
        if failure == "collate" and function is root.selected and int(args[0][0]) == 8:
            raise primary
        return call(function, *args, **kwargs)

    monkeypatch.setattr(_IndexedDatasetCursor, "_read", source_read)
    monkeypatch.setattr(_SectionExecution, "_map", mapped)
    monkeypatch.setattr(_SectionExecution, "_call", staticmethod(called))
    with prepared.open() as traversal:
        view = traversal.next_epoch()
        next(view)
        previous = traversal.capture_state()
        next(view)
        with pytest.raises(ValueError) as caught:
            next(view)
        assert caught.value is primary
        assert traversal._aborted and traversal._closed
        assert traversal._last_capture is previous
        with pytest.raises(data.UnsupportedContinuationError, match="matching captured record"):
            traversal.capture_state()
        with pytest.raises(data.TraversalOwnershipError):
            traversal.next_epoch()


def test_linear_batch_keeps_lazy_map_selection_and_per_row_example_validation(monkeypatch):
    root = _root(data.Batch(data.Map(data.ArrayDataset(np.arange(11)), Identity(),
                                     preserves_examples=True), 4))
    mapper = root.inputs[0]
    preserves, finder, examples = mapper.controls
    counts = Counter()

    def selected(**kwargs):
        counts["selection"] += 1
        return finder(**kwargs)

    def example_count(spec, value):
        counts["examples"] += 1
        return examples(spec, value)

    mapper = replace(mapper, selected=None, controls=(preserves, selected, example_count))
    root = replace(root, inputs=(mapper,))
    for generic in (True, False):
        counts.clear()
        with _generic(monkeypatch) if generic else nullcontext():
            with _SectionExecution(root) as execution:
                batches = []
                while execution.pull() == PullKind.DELIVERY:
                    batches.append(_plain(execution.value))
                assert batches == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10]]
                assert execution.children[0].selected is not None
                assert execution.children[0].output_spec == mapper.spec
        assert counts == Counter(selection=1, examples=22)


class InvalidResult(Method):
    @traits(backend="numpy")
    def numpy(self, value):
        return np.asarray([value])

    def infer_output_spec(self, spec):
        return spec


def test_linear_batch_keeps_selected_map_raw_result_validation(monkeypatch):
    method = InvalidResult()
    spec = TensorSpec("int64", shape=(), backend="numpy")
    root = _root(data.Batch(data.Map(data.ArrayDataset(np.arange(7)), method), 4))
    selected = method.find_implementation(spec, output_spec=spec).prepared_invoker()
    root = replace(root, inputs=(replace(root.inputs[0], selected=selected),))
    for generic in (True, False):
        with _generic(monkeypatch) if generic else nullcontext():
            with _SectionExecution(root) as execution:
                with pytest.raises(ImplementationSelectionError):
                    execution.pull()


class ForeignBuffer(Method):
    def __init__(self):
        self.storage = bytearray(16)
        self.view = np.frombuffer(memoryview(self.storage).toreadonly(), dtype=np.int64)

    @traits(backend="numpy")
    def numpy(self, value):
        self.storage[:] = np.full(2, value, dtype=np.int64).tobytes()
        return self.view

    def infer_output_spec(self, spec):
        return TensorSpec("int64", shape=(2,), backend="numpy")


def test_linear_batch_retains_readonly_foreign_map_results_before_next_producer(monkeypatch):
    root = _root(data.Batch(data.Map(data.ArrayDataset(np.arange(7)), ForeignBuffer()), 3))
    for generic in (True, False):
        with _generic(monkeypatch) if generic else nullcontext():
            with _SectionExecution(root) as execution:
                batches = []
                while execution.pull() == PullKind.DELIVERY:
                    batches.append(execution.value)
                assert _plain(batches) == [[[0, 0], [1, 1], [2, 2]],
                                           [[3, 3], [4, 4], [5, 5]], [[6, 6]]]


@pytest.mark.parametrize("validation", ("input", "preserves_examples"))
def test_linear_batch_keeps_map_input_and_example_count_validation(monkeypatch, validation):
    from tests.data.test_traversal_preparation import Shrink

    values = np.arange(21).reshape(7, 3)
    if validation == "input":
        spec = TensorSpec("int64", shape=(), backend="numpy")
        method, error = Identity(), ImplementationSelectionError
    else:
        spec = TensorSpec("int64", shape=(), batch=3, backend="numpy")
        method, error = Shrink(), ValueError
    root = _root(data.Batch(data.Map(data.ArrayDataset(values, spec=spec), method,
                                     preserves_examples=True), 4))
    for generic in (True, False):
        with _generic(monkeypatch) if generic else nullcontext():
            with _SectionExecution(root) as execution:
                with pytest.raises(error):
                    execution.pull()


class BatchedDouble(Method):
    def _double(self, value, module):
        assert tuple(value.shape) == (3, 2)
        assert type(value).__module__.split(".")[0] == module
        return value * 2

    def infer_output_spec(self, spec):
        return spec


class NumpyDouble(BatchedDouble):
    @traits(backend="numpy")
    def native(self, value):
        return self._double(value, "numpy")


class TorchDouble(BatchedDouble):
    @traits(backend="torch")
    def native(self, value):
        return self._double(value, "torch")


class TFDouble(BatchedDouble):
    @traits(backend="tf")
    def native(self, value):
        return self._double(value, "tensorflow")


class JaxDouble(BatchedDouble):
    @traits(backend="jax")
    def native(self, value):
        return self._double(value, "jaxlib")


@pytest.mark.parametrize("backend", ("numpy", "torch", "tf", "jax"))
def test_linear_batch_keeps_native_batched_map_inputs_whole(monkeypatch, backend):
    """An authored (3, 2) source item reaches Map once, without scalarization."""
    values = np.arange(30, dtype=np.float32).reshape(5, 3, 2)
    pytest.importorskip({"tf": "tensorflow"}.get(backend, backend))
    method = {"numpy": NumpyDouble, "torch": TorchDouble, "tf": TFDouble, "jax": JaxDouble}[backend]()
    spec = TensorSpec("float32", shape=(2,), batch=3, backend="numpy")
    root = _root(data.Batch(data.Map(data.ArrayDataset(values, spec=spec), method,
                                     preserves_examples=True), 2))
    original = _SectionExecution._map
    shapes = []

    def mapped(execution, value):
        shapes.append(tuple(value.shape))
        return original(execution, value)

    monkeypatch.setattr(_SectionExecution, "_map", mapped)
    runs = []
    for generic in (True, False):
        shapes.clear()
        with _generic(monkeypatch) if generic else nullcontext():
            with _SectionExecution(root) as execution:
                batches = []
                while execution.pull() == PullKind.DELIVERY:
                    batches.append(np.asarray(execution.value).tolist())
                runs.append(batches)
        assert shapes == [(3, 2)] * 5
    assert runs[0] == runs[1]
    assert runs[1][0] == (np.arange(12, dtype=np.float32).reshape(2, 3, 2) * 2).tolist()


@pytest.mark.parametrize("kind", ("epoch", "shuffle", "zip", "generator", "nested_map"))
def test_non_linear_or_non_array_batch_inputs_retain_generic_execution(monkeypatch, kind):
    source = data.ArrayDataset(np.arange(5, dtype=np.int64))
    if kind == "epoch":
        source = data.Epoch(source, len=3)
    elif kind == "shuffle":
        source = data.Shuffle(source, buffer_size=2, seed=17)
    elif kind == "zip":
        source = data.Zip((source, source))
    elif kind == "generator":
        from tests.data.test_dataset_state import values
        source = data.GeneratorDataset(values, 5, spec=TensorSpec("int64", shape=(), backend="numpy"))
    else:
        source = data.Map(data.Map(source, Identity()), Identity())

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported Batch input selected the linear Array kernel")

    monkeypatch.setattr(_SectionExecution, "_batch_array", forbidden)
    with _SectionExecution(_root(data.Batch(source, 2))) as execution:
        deliveries = 0
        while True:
            signal = execution.pull()
            if signal == PullKind.ROOT_END:
                break
            if signal == PullKind.SECTION_END:
                execution.handoff()
            else:
                deliveries += 1
        assert deliveries == 3
