"""U3 direct/replay selection, initial evidence, lazy policy and notebook proofs."""

import copy
import json
import linecache
import subprocess
import sys

import numpy as np
import pytest

import dryml.data as data
from dryml.core.tensor_spec import TensorSpec
from dryml.data.source import _epoch_seed


SPEC = TensorSpec("int64", shape=(), backend="numpy")
_events = []


class Direct(data.RecoverableGenerator):
    def __init__(self, size=30, *, seed=7):
        _events.append("open")
        self.size, self.index = size, 0
        self.rng = np.random.default_rng(seed)

    def __next__(self):
        if self.index == self.size:
            raise StopIteration
        self.index += 1
        _events.append("yield")
        return np.int64(self.rng.integers(0, 100000))

    def capture_state(self):
        _events.append("capture")
        return {"index": self.index, "rng": copy.deepcopy(self.rng.bit_generator.state)}

    def restore_state(self, state):
        _events.append("restore")
        self.index = state["index"]
        self.rng.bit_generator.state = state["rng"]

    def close(self):
        _events.append("close")


def ordinary(config=None, size=30, *, seed=7):
    _events.append(("open", seed))
    if config is not None:
        size = config["size"]
        config["size"] = -99
    rng = np.random.default_rng(seed)
    for _ in range(size):
        _events.append("yield")
        yield np.int64(rng.integers(0, 100000))


def returning_inner(size=30, *, seed=7):
    def inner():
        yield from ordinary(size=size, seed=seed)
    return inner


def lazy_direct(size=30):
    return Direct(size)


@pytest.fixture(autouse=True)
def events():
    _events.clear()


@pytest.mark.parametrize("factory,direct", [(Direct, True), (ordinary, False), (returning_inner, False), (lazy_direct, True)])
def test_ae6_ae7_exact_prefix_once_or_zero_and_lazy_actual_iterator(factory, direct):
    dataset = data.GeneratorDataset(factory, spec=SPEC)
    with dataset.open() as traversal:
        traversal.enable_replay_recovery_if_needed()
        assert _events == []
        view = traversal.next_epoch()
        for _ in range(6):
            next(view)
        assert _events.count("capture") == 0
        record = traversal.capture_state()
        assert _events.count("capture") == int(direct)
        expected = list(view)
    _events.clear()
    record.validate(dataset)
    assert _events == []
    with dataset.open(state=record) as resumed:
        resumed.disable_replay_recovery()  # selected saved reconstruction still required
        assert _events == []
        view = resumed.next_epoch()
        assert _events.count("yield") == (1 if direct else 7)
        assert list(view) == expected
    assert _events.count("yield") == (24 if direct else 30)
    assert _events.count("restore") == int(direct)


def test_default_policy_late_toggle_retained_config_actual_seed_and_independent_traversals():
    dataset = data.GeneratorDataset(ordinary, {"size": 12}, spec=SPEC, seed=19, seed_aware=True)
    with dataset.open(epoch=3) as traversal, dataset.open() as sibling:
        assert traversal.replay_recovery_policy == "disabled"
        view = traversal.next_epoch()
        for _ in range(5):
            next(view)
        before = list(_events)
        with pytest.raises(data.UnsupportedContinuationError):
            traversal.capture_state()
        traversal.enable_replay_recovery()
        traversal.enable_replay_recovery_if_needed()
        assert traversal.replay_recovery_policy == "enabled"
        assert sibling.replay_recovery_policy == "disabled"
        assert _events == before
        state = traversal.capture_state()
        source = state._data()["execution"]["source"]
        from dryml.data.recovery import _initial_value
        assert _initial_value(source["initial"])[0] == ({"size": 12},)
        assert source["seed"] == _epoch_seed(19, 3, 1)
        assert source["fetched"] == 5
        traversal.disable_replay_recovery()
        assert state.replay_recovery_enabled
        traversal.enable_replay_recovery()
        expected = list(view)
    _events.clear()
    with dataset.open(state=state) as resumed:
        assert list(resumed.next_epoch()) == expected


def test_initial_pending_conditional_record_and_unread_close_are_inert(tmp_path):
    dataset = data.GeneratorDataset(ordinary, spec=SPEC)
    with dataset.open() as traversal:
        traversal.enable_replay_recovery_if_needed()
        assert traversal.replay_recovery_policy == "if_needed"
        assert not traversal.replay_recovery_enabled
        state = traversal.capture_state()
        state.save(tmp_path / "initial")
    assert _events == []
    loaded = data.DatasetState.load(tmp_path / "initial")
    assert not loaded.replay_recovery_enabled
    with dataset.open(state=loaded):
        pass
    assert _events == []


def test_declared_direct_and_indexed_helper_noop_and_disable_preserves_direct_capture():
    dataset = data.Zip(data.ArrayDataset(np.arange(30)), data.GeneratorDataset(Direct, spec=SPEC))
    with dataset.open() as traversal:
        traversal.enable_replay_recovery_if_needed()
        traversal.enable_replay_recovery_if_needed()
        assert traversal.replay_recovery_policy == "disabled"
        assert _events == []
        initial = traversal.capture_state()
        assert _events == []
        next(traversal.next_epoch())
        traversal.enable_replay_recovery()
        traversal.disable_replay_recovery()
        assert traversal.capture_state().replay_recovery_policy == "disabled"
        assert _events.count("capture") == 1


def test_recovery_policy_and_capture_reject_recursive_acquisition_without_mutation(monkeypatch):
    import dryml.data.source as source_module
    original = source_module._fresh_iterable
    dataset = data.GeneratorDataset(ordinary, spec=SPEC)
    with dataset.open() as traversal:
        def factory(*args, **kwargs):
            for operation in (traversal.capture_state, traversal.enable_replay_recovery,
                              traversal.enable_replay_recovery_if_needed, traversal.disable_replay_recovery):
                with pytest.raises(data.TraversalOwnershipError):
                    operation()
            assert traversal.replay_recovery_policy == "disabled"
            return original(*args, **kwargs)
        monkeypatch.setattr(source_module, "_fresh_iterable", factory)
        assert len(list(traversal.next_epoch())) == 30


class Duck:
    def __iter__(self):
        return iter(range(9))

    def capture_state(self):
        pytest.fail("duck hook called")

    def restore_state(self, state):
        pytest.fail("duck hook called")


class LyingDirect(Direct):
    def __iter__(self):
        return iter(range(9))


class InheritedDirect(Direct):
    pass


class RegisteredDirect:
    def __init__(self):
        self.position = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.position == 9:
            raise StopIteration
        value = self.position
        self.position += 1
        return np.int64(value)
    def capture_state(self):
        return self.position
    def restore_state(self, state):
        self.position = state


class IncompleteRegistered:
    def __iter__(self):
        return self
    def __next__(self):
        return np.int64(0)


@pytest.mark.parametrize("producer", [InheritedDirect, RegisteredDirect])
def test_inherited_and_explicitly_registered_abc_capability_is_direct(producer):
    data.RecoverableGenerator.register(RegisteredDirect)
    dataset = data.GeneratorDataset(producer, spec=SPEC)
    with dataset.open() as traversal:
        traversal.enable_replay_recovery_if_needed()
        assert _events == []
        view = traversal.next_epoch()
        for _ in range(3):
            next(view)
        state = traversal.capture_state()
        assert state._data()["execution"]["source"]["strategy"] == "direct"
        expected = list(view)
    with dataset.open(state=state) as resumed:
        assert list(resumed.next_epoch()) == expected


def test_registered_incomplete_protocol_and_changed_lazy_product_never_fall_back(monkeypatch):
    import dryml.data.source as source_module
    data.RecoverableGenerator.register(IncompleteRegistered)
    with data.GeneratorDataset(IncompleteRegistered, spec=SPEC).open() as traversal:
        with pytest.raises(data.UnsupportedContinuationError):
            traversal.next_epoch()
    dataset = data.GeneratorDataset(lazy_direct, spec=SPEC)
    with dataset.open() as traversal:
        traversal.enable_replay_recovery_if_needed()
        next(traversal.next_epoch())
        record = traversal.capture_state()
    monkeypatch.setattr(source_module, "_fresh_iterable", lambda *a, **k: iter(range(30)))
    with dataset.open(state=record) as resumed:
        with pytest.raises(data.ContinuationMismatchError, match="strategy"):
            resumed.next_epoch()


def test_no_duck_typing_and_declared_direct_must_be_actual_iterator():
    dataset = data.GeneratorDataset(Duck, spec=SPEC)
    with dataset.open() as traversal:
        view = traversal.next_epoch()
        next(view)
        with pytest.raises(data.UnsupportedContinuationError):
            traversal.capture_state()
        traversal.enable_replay_recovery_if_needed()
        assert traversal.capture_state()._data()["execution"]["source"]["strategy"] == "replay"
    with data.GeneratorDataset(LyingDirect, spec=SPEC).open() as traversal:
        with pytest.raises(data.UnsupportedContinuationError, match="incompatible iterator"):
            traversal.next_epoch()


def large_argument(payload):
    yield from map(np.int64, payload)


def array_configuration(config, *, alias):
    assert config is alias
    start = int(config[0])
    config[0] = -999
    yield from map(np.int64, range(start, start + 8))


def test_small_array_configuration_is_owned_once_and_factory_cannot_mutate_initial_evidence():
    dataset = data.GeneratorDataset(array_configuration, np.array([10]), alias=np.array([10]), spec=SPEC)
    dataset.factory_kwargs["alias"] = dataset.factory_args[0]
    prepared = dataset.prepare()
    dataset.factory_args[0][0] = 20
    with prepared.open() as traversal:
        traversal.enable_replay_recovery()
        view = traversal.next_epoch()
        assert next(view) == 10
        record = traversal.capture_state()
        expected = list(view)
    with prepared.open(state=record) as resumed:
        assert list(resumed.next_epoch()) == expected == list(range(11, 18))


def test_payload_split_into_small_arrays_is_still_borrowed_as_large_configuration():
    dataset = data.GeneratorDataset(large_argument, [np.arange(128) for _ in range(4)],
                                   spec=TensorSpec("int64", shape=(128,), backend="numpy"))
    payload = dataset.factory_args[0]
    prepared = dataset.prepare()
    binding = prepared.graph._stream_plan._prepared.root.source.binding
    assert binding.args[0] is payload
    with prepared.open() as traversal:
        with pytest.raises(data.UnsupportedContinuationError):
            traversal.enable_replay_recovery()
        assert np.array_equal(next(traversal.next_epoch()), payload[0])


def test_late_conditional_missing_large_initial_evidence_is_atomic():
    dataset = data.Zip(data.GeneratorDataset(ordinary, spec=SPEC),
                       data.GeneratorDataset(large_argument, list(range(1024)), spec=SPEC))
    with dataset.open() as unread:
        with pytest.raises(data.UnsupportedContinuationError):
            unread.enable_replay_recovery()
        assert _events == [] and unread.position.start == 0
    with dataset.open() as traversal:
        view = traversal.next_epoch()
        next(view)
        position = traversal.position
        with pytest.raises(data.UnsupportedContinuationError, match="initialization"):
            traversal.enable_replay_recovery_if_needed()
        assert traversal.replay_recovery_policy == "disabled"
        assert traversal.position == position
        assert len(list(view)) == 29


@pytest.mark.parametrize("composition", ("chain", "repeat", "implicit"))
def test_conditional_helper_covers_future_physical_occurrences(composition):
    ordinary_source = data.GeneratorDataset(ordinary, size=3, spec=SPEC)
    direct_source = data.GeneratorDataset(Direct, 3, spec=SPEC)
    dataset = (data.Chain(direct_source, ordinary_source) if composition == "chain" else
               data.Repeat(ordinary_source, 2) if composition == "repeat" else ordinary_source)
    with dataset.open() as traversal:
        traversal.enable_replay_recovery_if_needed()
        assert _events == []
        view = traversal.next_epoch()
        count = 4 if composition != "implicit" else 2
        for _ in range(count):
            next(view)
        state = traversal.capture_state()
        expected = list(view)
    with dataset.open(state=state) as resumed:
        assert list(resumed.next_epoch()) == expected


def test_many_yields_no_capture_tree_codec_hooks_until_requested(monkeypatch):
    import dryml.data.state as state_module
    import dryml.data.recovery as recovery
    from dryml.data.continuation import encode_value
    original = recovery._capture_execution
    calls = []
    monkeypatch.setattr(recovery, "_capture_execution", lambda *a: (calls.append(1), original(*a))[1])
    # state imported the same adapter by value.
    monkeypatch.setattr(state_module, "_capture_execution", recovery._capture_execution)
    dataset = data.Zip(data.ArrayDataset(np.arange(500)), data.GeneratorDataset(Direct, 500, spec=SPEC))
    with dataset.open() as traversal:
        traversal.enable_replay_recovery()
        view = traversal.next_epoch()
        with monkeypatch.context() as patch:
            import dryml.data.continuation as codec
            def forbidden(*args, **kwargs):
                pytest.fail("steady yield captured/encoded/inspected a continuation tree")
            patch.setattr(codec, "encode_value", forbidden)
            patch.setattr(state_module, "encode_value", forbidden)
            patch.setattr(state_module, "_evidence", forbidden)
            for _ in range(140):
                next(view)
        assert not calls and "capture" not in _events
        state = traversal.capture_state()
        before = state._metadata, state._payloads
        for _ in range(20):
            next(view)
        assert _events.count("capture") == 1
        assert (state._metadata, state._payloads) == before


CELL = '''class NotebookProducer(RecoverableGenerator):
    def __init__(self, *, seed=7):
        self.index = 0
        self.rng = np.random.default_rng(seed)
    def __next__(self):
        if self.index == 13:
            raise StopIteration
        self.index += 1
        return np.int64(self.rng.integers(0, 100000))
    def capture_state(self):
        return {"index": self.index, "rng": self.rng.bit_generator.state}
    def restore_state(self, state):
        self.index = state["index"]
        self.rng.bit_generator.state = state["rng"]
'''


def test_same_notebook_definition_new_identity_compatible_changed_body_rejects_before_acquisition():
    from dryml.core.symbol import SourceSpec
    spec = SourceSpec.from_source(CELL, kind="class", name="NotebookProducer",
                                  imports={"np": "numpy", "RecoverableGenerator": "dryml.data:RecoverableGenerator"})
    first, second = spec.resolve(), spec.resolve()
    assert first is not second
    dataset = data.GeneratorDataset(first, spec=SPEC)
    with dataset.open() as traversal:
        next(traversal.next_epoch())
        state = traversal.capture_state()
    equivalent = data.GeneratorDataset(second, spec=SPEC)
    state.validate(equivalent)
    changed = SourceSpec.from_source(CELL.replace("100000", "100001"), kind="class", name="NotebookProducer", imports=spec.imports).resolve()
    with pytest.raises(data.ContinuationMismatchError):
        data.GeneratorDataset(changed, spec=SPEC).open(state=state)


def notebook_factory(*, seed):
    rng = np.random.default_rng(seed)
    def inner():
        for _ in range(15):
            yield np.int64(rng.integers(0, 100000))
    return inner


@pytest.mark.parametrize("shuffle", [False, True])
def test_notebook_style_local_rng_factory_reconstruction_with_buffer(shuffle):
    from dryml.core.symbol import SourceSpec
    # Force source reconstruction rather than the importable fixture's identity.
    factory_spec = SourceSpec.from_function(notebook_factory)
    factory = factory_spec.resolve()
    assert factory is not notebook_factory
    dataset = data.GeneratorDataset(factory, spec=SPEC, seed=9, seed_aware=True)
    if shuffle:
        dataset = data.Shuffle(dataset, buffer_size=4, seed=17)
    dataset = data.Epoch(dataset, len=4)
    with dataset.open() as traversal:
        traversal.enable_replay_recovery()
        assert len(list(traversal.next_epoch())) == 4
        view = traversal.next_epoch()
        next(view)
        next(view)
        state = traversal.capture_state()
        assert (state.position.epoch, state.position.start) == (1, 2)
        expected = list(view) + [x for _, v in traversal.epochs() for x in v]
    reconstructed = factory_spec.resolve()
    assert reconstructed is not factory
    target = data.GeneratorDataset(reconstructed, spec=SPEC, seed=9, seed_aware=True)
    if shuffle:
        target = data.Shuffle(target, buffer_size=4, seed=17)
    target = data.Epoch(target, len=4)
    with target.open(state=state) as resumed:
        assert [x for _, v in resumed.epochs() for x in v] == expected


def test_notebook_direct_class_clean_process_epoch_one_offset_three(tmp_path):
    from dryml.core.symbol import SourceSpec
    filename = "<ipython-input-91-u3>"
    linecache.cache[filename] = (len(CELL), None, CELL.splitlines(True), filename)
    namespace = {"__name__": "__main__", "np": np, "RecoverableGenerator": data.RecoverableGenerator}
    exec(compile(CELL, filename, "exec"), namespace)
    producer = namespace.pop("NotebookProducer")
    spec = SourceSpec.from_class(producer)
    dataset = data.Epoch(data.GeneratorDataset(producer, spec=SPEC), len=4)
    with dataset.open() as traversal:
        assert list(traversal.next_epoch())
        view = traversal.next_epoch()
        assert len([next(view) for _ in range(3)]) == 3
        state = traversal.capture_state()
        assert (state.position.epoch, state.position.start) == (1, 3)
        state.save(tmp_path / "record")
        expected = list(view) + [v for _, epoch in traversal.epochs() for v in epoch]
    script = '''
import json, sys
import numpy as np
from dryml.core.symbol import SourceSpec
from dryml.core.tensor_spec import TensorSpec
from dryml.data import DatasetState, Epoch, GeneratorDataset, RecoverableGenerator
state = DatasetState.load(sys.argv[1])
source = json.loads(sys.argv[2])
producer = SourceSpec.from_source(source, kind="class", name="NotebookProducer", imports={"np":"numpy", "RecoverableGenerator":"dryml.data:RecoverableGenerator"}).resolve()
assert issubclass(producer, RecoverableGenerator)
assert producer.__module__ != "__main__"
calls = []
old = producer.__next__
def counted(self):
    calls.append(self.index)
    return old(self)
# Install only after constructing/validating the target so source anchors remain genuine.
dataset = Epoch(GeneratorDataset(producer, spec=TensorSpec("int64", shape=(), backend="numpy")), len=4)
with dataset.open(state=state) as traversal:
    first = traversal.next_epoch()
    execution = traversal._execution.children[0]
    assert execution.source._values.index == 8  # seven restored, one staged; no historical yields
    result = list(first) + [v for _, epoch in traversal.epochs() for v in epoch]
print(json.dumps([int(v) for v in result]))
'''
    del namespace, producer, dataset, traversal, view
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path / "record"), json.dumps(spec.source)],
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected
