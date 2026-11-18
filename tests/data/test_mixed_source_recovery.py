"""Mixed occurrence recovery must not replay the combination root."""

import numpy as np
import pytest

import dryml.data as data
from dryml.core.tensor_spec import TensorSpec


_fetched = []


def ordinary(size=12):
    for i in range(size):
        _fetched.append(i)
        yield np.int64(i)


def test_indexed_replay_zip_suffix_has_only_source_prefix_work(monkeypatch):
    import dryml.data.source as sources
    dataset = data.Epoch(data.Zip(data.ArrayDataset(np.arange(12)), data.GeneratorDataset(
        ordinary, spec=TensorSpec("int64", shape=(), backend="numpy"))), len=4)
    with dataset.open() as traversal:
        traversal.enable_replay_recovery_if_needed()
        assert list(traversal.next_epoch()) == [(i, i) for i in range(4)]
        view = traversal.next_epoch()
        assert next(view) == (4, 4)
        record = traversal.capture_state()
        assert (record.position.epoch, record.position.start) == (1, 1)
        expected = list(view) + [value for _, epoch in traversal.epochs() for value in epoch]
    _fetched.clear()
    reads = []
    original = sources._tree_index
    monkeypatch.setattr(sources, "_tree_index", lambda x, i: (reads.append(i), original(x, i))[1])
    with dataset.open(state=record) as resumed:
        assert [value for _, epoch in resumed.epochs() for value in epoch] == expected
    assert _fetched == list(range(12))
    assert reads == list(range(5, 12))


@pytest.mark.parametrize("modes", [("indexed", "direct"), ("indexed", "replay", "direct")])
def test_named_mixed_direct_replay_indexed_branches_restore_independently(modes):
    from tests.data.test_generator_recovery import Direct, _events
    sources = {mode: data.ArrayDataset(np.arange(12)) if mode == "indexed" else
               data.GeneratorDataset(Direct, 12, spec=TensorSpec("int64", shape=(), backend="numpy")) if mode == "direct" else
               data.GeneratorDataset(ordinary, spec=TensorSpec("int64", shape=(), backend="numpy")) for mode in modes}
    dataset = data.Zip(sources)
    with dataset.open() as traversal:
        traversal.enable_replay_recovery_if_needed()
        view = traversal.next_epoch()
        for _ in range(5):
            next(view)
        state = traversal.capture_state()
        expected = list(view)
    _events.clear()
    _fetched.clear()
    with dataset.open(state=state) as resumed:
        assert list(resumed.next_epoch()) == expected
    assert _events.count("yield") == 7
    assert _events.count("restore") == 1
    assert _fetched == (list(range(12)) if "replay" in modes else [])


@pytest.mark.parametrize("sequence", [list, tuple])
def test_nested_named_mixed_structure_and_reused_indexed_source_round_trip(sequence):
    from tests.data.test_generator_recovery import Direct
    spec = TensorSpec("int64", shape=(), backend="numpy")
    indexed = data.ArrayDataset(np.arange(12))
    generated = data.GeneratorDataset(ordinary, spec=spec)
    direct = data.GeneratorDataset(Direct, 12, spec=spec)
    dataset = data.Zip({"indexed": indexed, "nested": sequence((generated, {"direct": direct, "again": indexed}))})
    with dataset.open() as traversal:
        traversal.enable_replay_recovery_if_needed()
        view = traversal.next_epoch()
        for _ in range(5):
            next(view)
        state = traversal.capture_state()
        expected = list(view)
    with dataset.open(state=state) as resumed:
        actual = list(resumed.next_epoch())
    assert actual == expected
    assert type(actual[0]["nested"]) is sequence
    assert actual[0]["indexed"] == actual[0]["nested"][1]["again"] == 5


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("sectioned", [False, True])
def test_divergent_shuffle_counts_shortest_branch_and_section_tail_phases(reverse, sectioned):
    spec = TensorSpec("int64", shape=(), backend="numpy")
    generator = data.Shuffle(data.GeneratorDataset(ordinary, size=21, spec=spec), buffer_size=6, seed=11)
    indexed = data.ArrayDataset(np.arange(13))
    if sectioned:
        generator, indexed = data.Epoch(generator, len=7), data.Epoch(indexed, len=5)
    dataset = data.Zip(*((generator, indexed) if reverse else (indexed, generator)))
    with dataset.open() as traversal:
        traversal.enable_replay_recovery_if_needed()
        view = traversal.next_epoch()
        for _ in range(5 if sectioned else 3):
            next(view)
        state = traversal.capture_state()
        if sectioned:
            assert state._data()["execution"]["phase"] == "drain"
        expected = list(view) + [value for _, epoch in traversal.epochs(stop=1 if not sectioned else None) for value in epoch]
    _fetched.clear()
    with dataset.open(state=state) as resumed:
        view = resumed.next_epoch()
        actual = list(view) + [value for _, epoch in resumed.epochs(stop=1 if not sectioned else None) for value in epoch]
        assert actual == expected
    assert _fetched == list(range(len(_fetched)))


def test_shared_dataset_slots_have_independent_frontiers_and_direct_failure_never_replays(monkeypatch):
    from tests.data.test_generator_recovery import Direct, _events
    source = data.GeneratorDataset(Direct, 12, spec=TensorSpec("int64", shape=(), backend="numpy"))
    dataset = data.Zip(source, source, data.GeneratorDataset(ordinary, spec=source.spec))
    with dataset.open() as traversal:
        traversal.enable_replay_recovery_if_needed()
        view = traversal.next_epoch()
        for _ in range(3):
            next(view)
        state = traversal.capture_state()
        expected = list(view)
    _events.clear()
    with dataset.open(state=state) as resumed:
        assert list(resumed.next_epoch()) == expected
    assert _events.count("open") == _events.count("restore") == 2
    _events.clear()
    _fetched.clear()
    original_restore = Direct.restore_state
    restores = []
    def incompatible(self, state):
        restores.append(1)
        if len(restores) == 2:
            raise ValueError("incompatible direct state")
        original_restore(self, state)
    monkeypatch.setattr(Direct, "restore_state", incompatible)
    with dataset.open(state=state) as resumed:
        with pytest.raises(ValueError, match="incompatible direct state"):
            resumed.next_epoch()
    assert _events.count("yield") == 1
    assert _events.count("open") == _events.count("close") == 2
    assert _fetched == []
