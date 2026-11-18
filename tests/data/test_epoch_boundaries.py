"""Boundary algebra must agree for ordinary flattened and retained execution."""

import numpy as np
import pytest

from dryml.core.cardinality import Cardinality
from dryml.data import ArrayDataset, Batch, Chain, DatasetExhaustedError, Repeat, Shuffle, Skip, Take, Unbatch, Zip
import dryml.data as data
from tests.data.test_dataset_traversal import source, _events


def values(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def sections(dataset):
    with dataset.open() as traversal:
        return [(ordinal, [values(value) for value in view]) for ordinal, view in traversal.epochs()]


def test_batch_drops_partial_groups_without_losing_empty_sections():
    dataset = Batch(data.Epoch(Take(source(), 6), len=3), 4, drop_remainder=True)
    assert sections(dataset) == [(0, []), (1, [])]
    assert list(dataset.iterator()) == list(dataset.prepare().iterator()) == []


def test_epoch_batch_order_is_intentionally_observable():
    before = data.Epoch(Batch(Take(source(), 12), 2), len=3)
    after = Batch(data.Epoch(Take(source(), 12), len=3), 2)
    assert sections(before) == [(0, [[0, 1], [2, 3], [4, 5]]), (1, [[6, 7], [8, 9], [10, 11]])]
    assert sections(after) == [(i, [[3*i, 3*i+1], [3*i+2]]) for i in range(4)]
    for dataset in (before, after):
        expected = [item for _, group in sections(dataset) for item in group]
        assert [values(value) for value in dataset.iterator()] == expected
        assert [values(value) for value in dataset.prepare().iterator()] == expected


def test_global_take_skip_preserve_crossed_markers_and_strict_shortfall():
    dataset = data.Epoch(ArrayDataset(np.arange(8)), len=3)
    assert sections(Take(dataset, 5)) == [(0, [0, 1, 2]), (1, [3, 4])]
    assert sections(Skip(dataset, 4)) == [(0, []), (1, [4, 5]), (2, [6, 7])]
    assert sections(Skip(dataset, 99)) == [(0, []), (1, []), (2, [])]
    with pytest.raises(DatasetExhaustedError) as caught:
        sections(Take(dataset, 9))
    assert (caught.value.requested, caught.value.yielded) == (9, 8)
    _events.clear()
    assert sections(Take(data.Epoch(source(), len=3), 0)) == []
    assert _events == []


def test_zip_discards_shortest_section_tails_then_realigns():
    left = data.Epoch(ArrayDataset(np.arange(6)), len=2)
    right = data.Epoch(ArrayDataset(np.arange(9) + 10), len=3)
    dataset = Zip(left, right)
    assert sections(dataset) == [(0, [(0, 10), (1, 11)]), (1, [(2, 13), (3, 14)]), (2, [(4, 16), (5, 17)])]
    assert list(dataset.iterator()) == list(dataset.prepare().iterator()) == [
        (0, 10), (1, 11), (2, 13), (3, 14), (4, 16), (5, 17)]


def test_chain_renumbers_sections_and_skips_empty_first_child_root():
    empty = data.Epoch(ArrayDataset(np.arange(0)), len=2)
    first = data.Epoch(ArrayDataset(np.arange(3)), len=2)
    second = data.Epoch(ArrayDataset(np.arange(2) + 10), len=1)
    assert sections(Chain(empty, first, second)) == [(0, [0, 1]), (1, [2]), (2, [10]), (3, [11])]
    assert sections(Repeat(first, 2)) == [(0, [0, 1]), (1, [2]), (2, [0, 1]), (3, [2])]
    assert list(Repeat(ArrayDataset(np.arange(2)), 2).prepare().iterator()) == [0, 1, 0, 1]


@pytest.mark.parametrize("kind", ["nested", "chain", "zip", "custom"])
def test_unsupported_section_compositions_reject_before_acquisition(kind):
    from dryml.data import Dataset

    class Legacy(Dataset):
        def __init__(self, src):
            self.src = src
            super().__init__(spec=src.spec)

        def __iter__(self):
            return iter(self.src)

    src = source()
    sectioned = data.Epoch(src, len=2)
    dataset = {"nested": lambda: data.Epoch(sectioned, len=2),
               "chain": lambda: Chain(src, sectioned),
               "zip": lambda: Zip(src, sectioned),
               "custom": lambda: Legacy(sectioned)}[kind]()
    _events.clear()
    with pytest.raises((ValueError, NotImplementedError)):
        dataset.prepare()
    assert _events == []


def test_section_sensitive_counts_are_exact_only_when_provable():
    dataset = data.Epoch(ArrayDataset(np.arange(6)), len=3)
    assert Batch(dataset, 4, drop_remainder=True).yield_cardinality() == Cardinality.finite(0)
    assert Batch(dataset, 2).yield_cardinality() == Cardinality.finite(4)
    uneven = data.Epoch(ArrayDataset(np.arange(6)), len=4)
    assert Batch(uneven, 3).yield_cardinality() == Cardinality.finite(3)
    paired = Zip(dataset, data.Epoch(ArrayDataset(np.arange(8)), len=4))
    assert paired.yield_cardinality() == Cardinality.finite(6)
    unknown = Batch(data.Epoch(source(2), len=3), 2)
    assert unknown.yield_cardinality() == Cardinality.UNKNOWN
    with unknown.open() as traversal:
        view = traversal.next_epoch()
        assert view.cardinality == Cardinality.UNKNOWN
        assert view.max_elements == 2
        assert [values(value) for value in view] == [[0, 1]]


def test_shuffle_is_section_local_and_unbatch_finishes_splits_before_boundary():
    dataset = data.Epoch(ArrayDataset(np.arange(12)), len=3)
    shuffled = Shuffle(dataset, 2, seed=17)
    first = sections(shuffled)
    assert first == sections(shuffled)
    assert [sorted(group) for _, group in first] == [list(range(i, i+3)) for i in range(0, 12, 3)]
    assert list(shuffled.iterator()) == list(shuffled.prepare().iterator())
    assert sections(Unbatch(Batch(dataset, 2))) == [(i, list(range(3*i, 3*i+3))) for i in range(4)]


def test_zip_does_not_invent_an_empty_section_when_another_branch_has_no_section():
    transformed_empty = Batch(data.Epoch(ArrayDataset(np.arange(3)), len=3), 4, drop_remainder=True)
    absent = data.Epoch(ArrayDataset(np.arange(0)), len=3)
    assert sections(Zip(transformed_empty, absent)) == []
    assert sections(Zip(absent, transformed_empty)) == []


def test_opaque_operator_below_epoch_rejects_instead_of_recursing_through_fallback():
    from dryml.data import Dataset

    class Legacy(Dataset):
        def __init__(self, src):
            self.src = src
            super().__init__(spec=src.spec)

        def __iter__(self):
            return iter(self.src)

    with pytest.raises(NotImplementedError, match="section"):
        data.Epoch(Legacy(source(2)), len=1).prepare()


def test_thousand_repeated_section_occurrences_keep_live_ownership_bounded():
    _events.clear()
    with Repeat(data.Epoch(source(1), len=1), 1000).open() as traversal:
        for ordinal, view in traversal.epochs():
            assert list(view) == [0]
            root = traversal._execution
            assert len(root._resources) <= 1
            assert len(root.children) <= 1
            assert _events.count("close") >= ordinal
    assert _events.count("open") == _events.count("close") == 1000


def test_zip_known_boundary_defers_tail_drain_and_close_never_drains():
    _events.clear()
    dataset = Zip(data.Epoch(source(), len=2), data.Epoch(source(), len=3))
    with dataset.open() as traversal:
        assert list(traversal.next_epoch()) == [(0, 0), (1, 1)]
        assert _events.count("pull") == 4
        assert list(traversal.next_epoch()) == [(2, 3), (3, 4)]
        assert _events.count("pull") == 9
    assert _events.count("pull") == 9
    assert _events.count("close") == 2


def test_skip_repeat_shuffle_are_qualified_and_keep_prepared_controls():
    src = ArrayDataset(np.arange(6))
    skipped = Skip(src, 2)
    repeated = Repeat(Take(src, 2), 2)
    shuffled = Shuffle(src, 3, seed=11)
    for dataset in (skipped, repeated, shuffled):
        prepared = dataset.prepare()
        assert prepared.execution_level == "stream"
        expected = list(dataset.iterator())
        if dataset is skipped:
            dataset.n = 4
        elif dataset is repeated:
            dataset.count = 3
        else:
            dataset.buffer_size, dataset.seed = 1, 9
        assert list(prepared.iterator()) == expected
        with prepared.open() as traversal:
            assert list(traversal.next_epoch()) == expected


def test_session_map_keeps_vectorized_batch_calls():
    from tests.data.test_traversal_preparation import BatchedOnly

    method = BatchedOnly()
    method.shapes = []
    dataset = data.Map(Batch(data.Epoch(ArrayDataset(np.arange(6)), len=3), 2), method)
    assert sections(dataset) == [(0, [[0, 1], [2]]), (1, [[3, 4], [5]])]
    assert method.shapes == [(2,), (1,), (2,), (1,)]


@pytest.mark.parametrize("backend", ["numpy", "torch", "tf", "jax"])
def test_local_section_kernel_keeps_native_batch_split_values(backend):
    from dryml.core.tensor_spec import TensorSpec
    from tests.data.test_native_stream_operators import Values

    if backend == "torch":
        array = pytest.importorskip("torch").as_tensor
    elif backend == "tf":
        array = pytest.importorskip("tensorflow").convert_to_tensor
    elif backend == "jax":
        array = pytest.importorskip("jax.numpy").asarray
    else:
        array = np.asarray
    src = Values(TensorSpec("int32", shape=(1,), backend=backend))
    src.values = tuple(array(np.asarray([i], dtype=np.int32)) for i in range(5))
    dataset = Unbatch(Batch(data.Epoch(src, len=3), 2))
    with dataset.open() as traversal:
        groups = [list(view) for _, view in traversal.epochs()]
    assert [len(group) for group in groups] == [3, 2]
    flat = [value for group in groups for value in group]
    assert all(type(value) is type(src.values[0]) for value in flat)
    assert [int(np.asarray(value)[0]) for value in flat] == list(range(5))


def test_zero_section_bound_proves_empty_membership_without_inventing_root_eof():
    _events.clear()
    dataset = Batch(data.Epoch(source(), len=3), 4, drop_remainder=True)
    assert dataset.yield_cardinality() == Cardinality.finite(0)
    assert Zip(dataset, data.Epoch(source(), len=2)).yield_cardinality() == Cardinality.finite(0)
    assert _events == []
    with dataset.open() as traversal:
        for ordinal, view in traversal.epochs(stop=2):
            assert view.cardinality == Cardinality.finite(0)
            assert view.max_elements == 0
            assert list(view) == []
            assert not traversal.position.root_complete
    assert _events.count("open") == 1
    assert _events.count("pull") == 6
