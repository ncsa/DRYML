"""U3 happy, edge, ownership, and on-demand continuation proofs."""

import gc
import itertools
from dataclasses import replace

import numpy as np
import pytest

import dryml.data as data
from dryml.core.object import Object
from dryml.core.tensor_spec import TensorSpec


_value_opens = []


def values(size=17):
    _value_opens.append(size)
    yield from map(np.int64, range(size))


def tree_list(values):
    def plain(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if type(v) is dict:
            return {k: plain(x) for k, x in v.items()}
        if type(v) in (list, tuple):
            return type(v)(plain(x) for x in v)
        return v
    return [plain(v) for v in values]


@pytest.mark.parametrize("operator", ("shuffle", "unbatch"))
def test_ae4_ae5_buffered_suffix_survives_destroyed_traversal(tmp_path, operator):
    source = data.ArrayDataset(np.arange(23))
    dataset = data.Shuffle(source, buffer_size=7, seed=42) if operator == "shuffle" else data.Unbatch(data.Batch(source, 6))
    traversal = dataset.open()
    view = traversal.next_epoch()
    for _ in range(3):
        next(view)
    state = traversal.capture_state()
    expected = list(view)
    state.save(tmp_path / "cut")
    traversal.close()
    del traversal, view, state
    gc.collect()
    state = data.DatasetState.load(tmp_path / "cut")
    with dataset.open(state=state) as resumed:
        assert list(resumed.next_epoch()) == expected


def test_immutable_record_has_no_graph_identity_and_two_opens_are_independent():
    dataset = data.Shuffle(data.ArrayDataset(np.arange(20)), buffer_size=5, seed=17)
    with dataset.open() as traversal:
        view = traversal.next_epoch()
        next(view)
        state = traversal.capture_state()
        expected = list(view)
    assert not isinstance(state, Object)
    assert not hasattr(state, "object_id") and not hasattr(state, "dataset")
    before = (state._metadata, state._payloads, state.position)
    with dataset.open(state=state) as first, dataset.open(state=state) as second:
        left, right = first.next_epoch(), second.next_epoch()
        assert next(left) == expected[0]
        assert list(right) == expected
        assert list(left) == expected[1:]
    assert (state._metadata, state._payloads, state.position) == before


def test_retained_map_selection_and_downstream_transitions_are_not_replayed(monkeypatch):
    from tests.data.test_traversal_preparation import Identity
    from dryml.methods import traits
    original = Identity.numpy
    calls = []
    @traits(backend="numpy")
    def counted(self, value):
        calls.append(int(value))
        return original(self, value)
    monkeypatch.setattr(Identity, "numpy", counted)
    method = Identity()
    dataset = data.Map(data.Shuffle(data.ArrayDataset(np.arange(17)), buffer_size=5, seed=7), method)
    with dataset.open() as traversal:
        view = traversal.next_epoch()
        for _ in range(4):
            next(view)
        state = traversal.capture_state()
        expected = list(view)
    calls.clear()
    with dataset.open(state=state) as resumed:
        assert list(resumed.next_epoch()) == expected
    assert calls == expected


def test_retained_method_alternative_is_part_of_target_evidence():
    from dataclasses import replace
    from tests.data.test_traversal_preparation import Identity
    dataset = data.Map(data.ArrayDataset(np.arange(5)), Identity())
    prepared = dataset.prepare()
    with prepared.open() as traversal:
        next(traversal.next_epoch())
        state = traversal.capture_state()
    other = dataset.prepare()
    descriptor = other.graph._stream_plan._prepared
    selected = replace(descriptor.root.selected, name="different-alternative")
    other.graph._stream_plan._prepared = replace(descriptor, root=replace(descriptor.root, selected=selected))
    with pytest.raises(data.ContinuationMismatchError):
        other.open(state=state)


def test_pending_candidate_blocks_capture_but_staged_lookahead_round_trips():
    dataset = data.ArrayDataset(np.arange(4))
    with dataset.open(acceptance="explicit") as traversal:
        view = traversal.next_epoch()  # stage zero without exposing a candidate
        state = traversal.capture_state()
        candidate = view.reserve()
        with pytest.raises(data.TraversalOwnershipError):
            traversal.capture_state()
        for operation in (traversal.enable_replay_recovery, traversal.disable_replay_recovery,
                          traversal.enable_replay_recovery_if_needed):
            with pytest.raises(data.TraversalOwnershipError):
                operation()
        traversal.accept(replace(candidate, value="converted"))
        with pytest.raises(data.TraversalOwnershipError):
            traversal.accept(candidate)
    with dataset.open(state=state) as resumed:
        assert list(resumed.next_epoch()) == [0, 1, 2, 3]


def test_closed_and_rejected_return_only_matching_prior_capture():
    dataset = data.ArrayDataset(np.arange(5))
    with dataset.open(acceptance="explicit") as traversal:
        view = traversal.next_epoch()
        state = traversal.capture_state()
        traversal.reject(view.reserve())
        assert traversal.capture_state() is state
        with pytest.raises(data.TraversalOwnershipError):
            traversal.disable_replay_recovery()
    traversal = dataset.open()
    view = traversal.next_epoch()
    traversal.capture_state()
    next(view)
    traversal.close()
    with pytest.raises(data.UnsupportedContinuationError):
        traversal.capture_state()


def test_known_section_boundary_does_not_invent_unknown_terminal_eof():
    dataset = data.Epoch(data.GeneratorDataset(values, 6, spec=TensorSpec("int64", shape=(), backend="numpy")), len=3)
    with dataset.open() as traversal:
        traversal.enable_replay_recovery()
        assert list(traversal.next_epoch()) == [0, 1, 2]
        state = traversal.capture_state()
        assert state.position.epoch == 1 and not state.position.root_complete
        expected = [(i, list(v)) for i, v in traversal.epochs()]
    with dataset.open(state=state) as resumed:
        assert [(i, list(v)) for i, v in resumed.epochs()] == expected == [(1, [3, 4, 5])]


def test_exact_end_unknown_implicit_eof_preserves_nonempty_handoff_once():
    _value_opens.clear()
    dataset = data.GeneratorDataset(values, 3, spec=TensorSpec("int64", shape=(), backend="numpy"))
    with dataset.open() as traversal:
        traversal.enable_replay_recovery()
        view = traversal.next_epoch()
        assert [next(view) for _ in range(3)] == [0, 1, 2]
        state = traversal.capture_state()
    with dataset.open(state=state) as resumed:
        resumed._hold_boundary_handoff = True
        with pytest.raises(StopIteration):
            resumed.next_epoch()
        assert resumed.position == data.TraversalPosition(1, 0, 4, True, False)
        assert resumed._boundary_pending
        assert len(_value_opens) == 2
        updated = resumed.capture_state()
        with pytest.raises(StopIteration):
            resumed.next_epoch()
        assert resumed.position == updated.position
        assert len(_value_opens) == 2
    with dataset.open(state=updated) as resumed:
        resumed._hold_boundary_handoff = True
        with pytest.raises(StopIteration):
            resumed.next_epoch()
        assert resumed.position == updated.position
        resumed._handoff()
        assert list(resumed.next_epoch()) == [0, 1, 2]
        assert len(_value_opens) == 3


def reused_arrays():
    array = np.zeros(2, dtype=np.int64)
    for i in range(8):
        array[:] = i
        yield array


def test_reused_writable_arrays_and_unbatch_views_do_not_mutate_buffers():
    spec = TensorSpec("int64", shape=(2,), backend="numpy")
    dataset = data.Shuffle(data.GeneratorDataset(reused_arrays, spec=spec), buffer_size=4, seed=7)
    with dataset.open() as traversal:
        traversal.enable_replay_recovery()
        view = traversal.next_epoch()
        next(view)
        state = traversal.capture_state()
        expected = tree_list(view)
    with dataset.open(state=state) as resumed:
        assert tree_list(resumed.next_epoch()) == expected
    dataset = data.Unbatch(data.ArrayDataset(np.arange(12).reshape(3, 4), spec=TensorSpec("int64", shape=(), batch=4, backend="numpy")))
    with dataset.open() as traversal:
        view = traversal.next_epoch()
        first = next(view)
        assert first == 0
        state = traversal.capture_state()
        assert list(view) == list(range(1, 12))
    with dataset.open(state=state) as resumed:
        assert list(resumed.next_epoch()) == list(range(1, 12))


def test_capture_owns_array_source_bytes_but_prepare_open_do_not_copy(monkeypatch):
    import dryml.data.source as sources
    dataset = data.ArrayDataset(np.arange(4096))
    prepared = dataset.prepare()
    assert prepared.graph._stream_plan._prepared.root.source.binding[0] is dataset.arrays
    with prepared.open() as traversal:
        view = traversal.next_epoch()
        next(view)
        state = traversal.capture_state()
        dataset.arrays[:] = -1
        assert state._data()["snapshots"]["()"][1] == 1
    equivalent = data.ArrayDataset(np.arange(4096))
    with equivalent.open(state=state) as resumed:
        assert next(resumed.next_epoch()) == 1


def test_initial_record_and_future_chain_sources_use_record_owned_snapshots():
    left, right = data.ArrayDataset(np.arange(3)), data.ArrayDataset(np.arange(3) + 10)
    dataset = data.Chain(left, right)
    with dataset.open() as traversal:
        state = traversal.capture_state()
    with dataset.open(state=state) as resumed:
        left.arrays[:] = -1
        right.arrays[:] = -2
        assert list(resumed.next_epoch()) == [0, 1, 2, 10, 11, 12]


def readonly_reused_arrays():
    array = np.zeros(2, dtype=np.int64)
    view = array.view()
    view.flags.writeable = False
    for i in range(8):
        array[:] = i
        yield view


def test_readonly_views_of_writable_producer_storage_are_not_treated_as_owned():
    source = data.GeneratorDataset(readonly_reused_arrays, spec=TensorSpec("int64", shape=(2,), backend="numpy"))
    dataset = data.Shuffle(source, buffer_size=4, seed=7)
    with dataset.open() as traversal:
        view = traversal.next_epoch()
        result = tree_list(view)
    assert sorted(result) == [[i, i] for i in range(8)]


def test_unbatch_emitted_writable_views_cannot_change_retained_rows_or_record():
    original = np.arange(24).reshape(3, 4, 2)
    spec = TensorSpec("int64", shape=(2,), batch=4, backend="numpy")
    source = data.ArrayDataset(original.copy(), spec=spec)
    dataset = data.Unbatch(source)
    with dataset.open() as traversal:
        view = traversal.next_epoch()
        first = next(view)
        record = traversal.capture_state()
        first[:] = -1
        source.arrays[0] = -2
        assert tree_list(view)[:3] == [[2, 3], [4, 5], [6, 7]]
    equivalent = data.Unbatch(data.ArrayDataset(original.copy(), spec=spec))
    with equivalent.open(state=record) as resumed:
        assert tree_list(resumed.next_epoch()) == original.reshape(-1, 2)[1:].tolist()


def test_large_array_cdef_literals_are_inventoried_payloads_not_manifest_strings(tmp_path):
    dataset = data.ArrayDataset(np.zeros(1_600_000, dtype=np.int64))
    with dataset.open() as traversal:
        record = traversal.capture_state()
    assert len(record._metadata) < 100_000
    path = record.save(tmp_path / "large")
    loaded = data.DatasetState.load(path)
    with dataset.open(state=loaded) as resumed:
        assert next(resumed.next_epoch()) == 0


def opaque_values():
    yield object()
    yield object()


def test_unsupported_dynamic_buffer_fails_capture_without_disabling_live_iteration():
    source = data.GeneratorDataset(opaque_values, spec=TensorSpec("object", layout="python"))
    dataset = data.Shuffle(source, buffer_size=2, seed=0)
    with dataset.open() as traversal:
        traversal.enable_replay_recovery()
        view = traversal.next_epoch()
        next(view)
        position = traversal.position
        with pytest.raises(data.UnsupportedContinuationError):
            traversal.capture_state()
        assert traversal.position == position
        assert len(list(view)) == 1
