"""Frozen preparation contracts, without the later public traversal APIs."""

import numpy as np
import pytest

from dryml.core.tensor_spec import TensorSpec
from dryml.data import ArrayDataset, Batch, Chain, GeneratorDataset, Map, Take, Unbatch, Zip
from dryml.methods import Method, traits


class Identity(Method):
    @traits(backend="numpy")
    def numpy(self, value):
        return value

    def infer_output_spec(self, spec):
        return spec


class Shrink(Identity):
    @traits(backend="numpy")
    def numpy(self, value):
        return value[:1]


class AddTen(Identity):
    @traits(backend="numpy")
    def numpy(self, value):
        return value + 10


_opens = []


def configured_factory(*, config):
    _opens.append(config)
    return map(np.int64, config["values"])


def source():
    return ArrayDataset(np.arange(5, dtype=np.int64))


def test_flattened_lazy_compatibility_before_freezing():
    """Prepared Chain/Zip/Batch/Unbatch keep ordinary outputs and lazy acquisition."""
    _opens.clear()
    src = GeneratorDataset(configured_factory, config={"values": [1, 2, 3]},
                           spec=TensorSpec("int64", shape=(), backend="numpy"))
    dataset = Chain(Unbatch(Batch(Map(src, Identity()), 2)), Take(src, 1))
    prepared = dataset.prepare()
    first, second = prepared.iterator(), prepared.iterator()
    assert _opens == []
    assert next(first) == next(second) == 1
    assert len(_opens) == 2
    first.close()
    assert list(second) == [2, 3, 1]
    assert list(dataset.iterator()) == [1, 2, 3, 1]
    paired = Zip({"a": src, "nested": [src, (src,)]}).prepare()
    assert list(paired.iterator()) == [
        {"a": value, "nested": [value, (value,)]} for value in (1, 2, 3)
    ]


@pytest.mark.parametrize("control", ("batch_size", "drop_remainder", "take", "map_method", "map_preservation"))
def test_prepared_controls_survive_recipe_mutation(control):
    """Existing descriptors remain authoritative; fresh preparation sees edits."""
    if control in {"batch_size", "drop_remainder"}:
        dataset = Batch(source(), 2)
        prepared = dataset.prepare()
        setattr(dataset, control, 3 if control == "batch_size" else True)
        assert [value.tolist() for value in prepared.iterator()] == [[0, 1], [2, 3], [4]]
        expected = [[0, 1, 2], [3, 4]] if control == "batch_size" else [[0, 1], [2, 3]]
        assert [value.tolist() for value in dataset.prepare().iterator()] == expected
    elif control == "take":
        dataset = Take(source(), 2)
        prepared = dataset.prepare()
        dataset.n = 3
        assert list(prepared.iterator()) == [0, 1]
        assert list(dataset.prepare().iterator()) == [0, 1, 2]
    else:
        dataset = Map(Batch(source(), 2), Shrink())
        prepared = dataset.prepare()
        if control == "map_method":
            dataset.method = Identity()
        else:
            dataset.preserves_examples = True
        assert [value.tolist() for value in prepared.iterator()] == [[0], [2], [4]]
        if control == "map_method":
            assert [value.tolist() for value in dataset.prepare().iterator()] == [[0, 1], [2, 3], [4]]
        else:
            with pytest.raises(ValueError, match="preserves_examples"):
                list(dataset.prepare().iterator())


def test_nested_zip_topology_is_detached_from_recipe_containers():
    src = source()
    dataset = Zip({"left": src, "nested": [src, (src,)]})
    prepared = dataset.prepare()
    dataset.sources["nested"].append(src)
    dataset.sources["renamed"] = dataset.sources.pop("left")
    assert next(prepared.iterator()) == {"left": 0, "nested": [0, (0,)]}
    assert next(dataset.prepare().iterator()) == {"nested": [0, (0,), 0], "renamed": 0}


def test_small_generator_configuration_is_frozen_and_per_cursor_owned():
    _opens.clear()
    dataset = GeneratorDataset(configured_factory, config={"values": [1, 2]},
                               spec=TensorSpec("int64", shape=(), backend="numpy"))
    prepared = dataset.prepare()
    dataset.factory_kwargs["config"]["values"][0] = 9
    assert _opens == []
    assert list(prepared.iterator()) == [1, 2]
    _opens[-1]["values"][0] = 8
    assert list(prepared.iterator()) == [1, 2]
    assert list(dataset.prepare().iterator()) == [9, 2]


def test_unknown_backend_discovery_uses_the_retained_map_method():
    src = GeneratorDataset(configured_factory, config={"values": [1, 2]},
                           spec=TensorSpec("int64", shape=(), backend=None))
    dataset = Map(src, Identity())
    prepared = dataset.prepare()
    dataset.method = AddTen()
    assert list(prepared.iterator()) == [1, 2]
    assert list(prepared.iterator()) == [1, 2]
    assert list(dataset.prepare().iterator()) == [11, 12]


def test_declared_selections_are_retained_across_independent_cursors(monkeypatch):
    import dryml.data.structural as structural

    method = Identity()
    dataset = Unbatch(Batch(Map(source(), method), 2))
    counts = {"map": 0, "collate": 0, "split": 0}

    def count(name, function):
        def wrapped(*args, **kwargs):
            counts[name] += 1
            return function(*args, **kwargs)
        return wrapped

    monkeypatch.setattr(method, "find_implementation", count("map", method.find_implementation))
    monkeypatch.setattr(structural, "collate_for_spec", count("collate", structural.collate_for_spec))
    monkeypatch.setattr(structural, "split_for_spec", count("split", structural.split_for_spec))
    prepared = dataset.prepare()
    assert counts == {"map": 1, "collate": 1, "split": 1}
    first, second = prepared.iterator(), prepared.iterator()
    assert next(first) == next(second) == 0
    assert next(first) == 1
    assert list(first) == [2, 3, 4]
    assert list(second) == [1, 2, 3, 4]
    assert counts == {"map": 1, "collate": 1, "split": 1}


def seeded_factory(*, seed):
    return iter((np.int64(seed % 1_000_000),))


def test_take_seed_coordinates_and_generator_controls_remain_frozen():
    from dryml.data.source import _epoch_seed

    src = GeneratorDataset(seeded_factory, spec=TensorSpec("int64", shape=(), backend="numpy"),
                           seed=17, seed_aware=True, cardinality=1)
    dataset = Take(src, 1, epoch=2)
    prepared = dataset.prepare()
    src.seed = 99
    src.epoch_seed_version = 3
    dataset.epoch = 7
    dataset.fixed_prefix = True
    with prepared.iterator(epoch=4) as cursor:
        assert int(next(cursor)) == _epoch_seed(17, 6, 1) % 1_000_000
    with dataset.prepare().iterator(epoch=4) as cursor:
        assert int(next(cursor)) == _epoch_seed(99, 7, 3) % 1_000_000


def test_array_payload_is_not_copied_while_container_bindings_are_frozen(monkeypatch):
    import copy

    dataset = ArrayDataset({"x": np.arange(4096, dtype=np.int64)})
    payload = dataset.arrays["x"]
    monkeypatch.setattr(copy, "deepcopy", lambda *a, **k: pytest.fail("source-sized copy"))
    prepared = dataset.prepare()
    root = prepared.graph._stream_plan._prepared.root
    assert root.source.binding[0].unpack()["x"] is payload
    dataset.arrays["x"] = np.arange(4096, dtype=np.int64) + 99
    assert next(prepared.iterator()) == {"x": 0}
    assert next(dataset.prepare().iterator()) == {"x": 99}


def payload_factory(payload):
    return iter(payload)


def test_large_factory_payload_binding_is_not_cloned():
    dataset = GeneratorDataset(payload_factory, list(range(4096)),
                               spec=TensorSpec("object", layout="python"))
    payload = dataset.factory_args[0]
    prepared = dataset.prepare()
    binding = prepared.graph._stream_plan._prepared.root.source.binding
    assert binding.factory_args[0] is payload
    assert next(prepared.iterator()) == 0


class CaptureTrap:
    def __init__(self):
        self.values = iter(np.arange(4, dtype=np.int64))

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.values)

    def capture_state(self):
        pytest.fail("ordinary pulls extracted producer state")

    def __reduce_ex__(self, protocol):
        pytest.fail("ordinary pulls serialized a producer")


def test_ordinary_pulls_never_extract_or_encode_state(monkeypatch):
    import dill
    import pickle

    src = GeneratorDataset(CaptureTrap, spec=TensorSpec("int64", shape=(), backend="numpy"))
    dataset = Batch(Map(src, Identity()), 2)

    def forbidden(*args, **kwargs):
        pytest.fail("ordinary pulls encoded state")

    monkeypatch.setattr(pickle, "dumps", forbidden)
    monkeypatch.setattr(dill, "dumps", forbidden)
    prepared = dataset.prepare()
    with prepared.iterator() as cursor:
        assert [value.tolist() for value in cursor] == [[0, 1], [2, 3]]


class BatchedOnly(Method):
    @traits(backend="numpy", batch_mode="batched")
    def numpy(self, value):
        self.shapes.append(value.shape)
        return value

    def infer_output_spec(self, spec):
        return spec


def test_prepared_native_batched_invoker_is_not_scalarized():
    method = BatchedOnly()
    method.shapes = []
    prepared = Map(Batch(source(), 2), method).prepare()
    with prepared.iterator() as cursor:
        assert [value.tolist() for value in cursor] == [[0, 1], [2, 3], [4]]
    assert method.shapes == [(2,), (2,), (1,)]


def test_prepared_output_metadata_and_native_count_do_not_reread_recipe():
    class Handoff:
        def prepare(self, x, y):
            return x, y

    src = ArrayDataset((np.arange(4), np.arange(4)))
    dataset = Take(Batch(src, 2), 2)
    prepared = dataset.prepare()
    captured_spec = prepared.spec
    dataset.n = 3
    dataset._spec = TensorSpec("object", layout="python")
    assert prepared.spec == captured_spec
    assert prepared.examples_in((np.zeros(2), np.zeros(2))) == 2
    with prepared.training_batches(Handoff()) as cursor:
        assert len(list(cursor)) == 2
        assert cursor.observed_eof


def alias_factory(config, *, same):
    assert config is same
    assert config["left"] is config["right"]
    config["left"].append(7)
    return iter(config["right"])


def test_factory_config_aliases_survive_freezing_with_independent_cursor_copies():
    shared = [1]
    config = {"left": shared, "right": shared}
    dataset = GeneratorDataset(alias_factory, config, same=config,
                               spec=TensorSpec("object", layout="python"))
    config = dataset.factory_args[0]
    config["right"] = config["left"]
    dataset.factory_kwargs["same"] = config
    prepared = dataset.prepare()
    assert list(prepared.iterator()) == [1, 7]
    assert list(prepared.iterator()) == [1, 7]
    assert config["left"] == [1]


def test_failed_source_binding_replacement_preserves_prior_execution():
    _opens.clear()
    dataset = GeneratorDataset(configured_factory, config={"values": [1, 2]},
                               spec=TensorSpec("int64", shape=(), backend="numpy"))
    graph = dataset.method_graph()
    graph.learn()
    nodes = graph.nodes
    cursor = graph.iterator()
    cyclic = {}
    cyclic["self"] = cyclic
    dataset.factory_kwargs["config"] = cyclic
    with pytest.raises(ValueError, match="container cycle"):
        graph.learn()
    assert _opens == []
    assert graph.nodes is nodes
    assert list(cursor) == [1, 2]
    assert list(graph.iterator()) == [1, 2]


def variable_factory():
    return iter((np.arange(2, dtype=np.int64), np.arange(3, dtype=np.int64)))


def test_ragged_selection_is_retained_after_batch_policy_changes():
    from dryml.core.tensor_spec import Dynamic

    src = GeneratorDataset(variable_factory, spec=TensorSpec("int64", shape=(Dynamic,), backend="numpy"))
    dataset = Batch(src, 2, ragged=True)
    prepared = dataset.prepare()
    dataset.ragged = False
    with prepared.iterator() as cursor:
        assert next(cursor).nested_row_splits[0].tolist() == [0, 2, 5]
    with dataset.prepare().iterator() as cursor:
        with pytest.raises(ValueError, match="same shape"):
            next(cursor)


def test_npy_source_binding_freezes_file_order_and_loading_policy(tmp_path, monkeypatch):
    from dryml.data import NpyFileDataset

    for index in range(2):
        np.save(tmp_path / f"{index}.npy", np.asarray(index))
    dataset = NpyFileDataset(str(tmp_path), spec=TensorSpec("int64", shape=(), backend="numpy"))
    prepared = dataset.prepare()
    dataset.files = tuple(reversed(dataset.files))
    dataset.allow_pickle = True
    calls = []
    load = np.load

    def tracked(path, *, allow_pickle):
        calls.append(allow_pickle)
        return load(path, allow_pickle=allow_pickle)

    monkeypatch.setattr(np, "load", tracked)
    assert [int(value) for value in prepared.iterator()] == [0, 1]
    assert calls == [False, False]
    assert [int(value) for value in dataset.prepare().iterator()] == [1, 0]
    assert calls == [False, False, True, True]


def test_tfds_binding_retains_native_handle_and_mode_without_opening():
    from dryml.data import TFDSAdapter

    class Native:
        opens = 0

        def as_numpy_iterator(self):
            self.opens += 1
            return iter((1, 2))

        def __iter__(self):
            self.opens += 1
            return iter((9, 10))

        def cardinality(self):
            return self

        def numpy(self):
            return 2

    native = Native()
    dataset = object.__new__(TFDSAdapter)
    dataset.dataset = native
    dataset.as_numpy = True
    dataset._spec = TensorSpec("int64", shape=(), backend="numpy")
    prepared = dataset.prepare()
    dataset.as_numpy = False
    assert native.opens == 0
    assert list(prepared.iterator()) == [1, 2]
    assert list(dataset.prepare().iterator()) == [9, 10]
    dataset.dataset = Native()
    assert list(prepared.iterator()) == [1, 2]
    assert native.opens == 3


@pytest.mark.parametrize("iterable", (False, True))
def test_torch_adapter_binding_retains_native_source_and_lazy_acquisition(iterable):
    torch = pytest.importorskip("torch")
    from dryml.data import TorchDatasetAdapter

    class Native(torch.utils.data.IterableDataset if iterable else torch.utils.data.Dataset):
        opens = 0

        def __len__(self):
            return 2

        def __getitem__(self, index):
            self.opens += 1
            return index

        def __iter__(self):
            self.opens += 1
            return iter((0, 1))

    native = Native()
    dataset = object.__new__(TorchDatasetAdapter)
    dataset.dataset = native
    dataset._spec = TensorSpec("int64", shape=(), backend="numpy")
    prepared = dataset.prepare()
    dataset.dataset = Native()
    first, second = prepared.iterator(), prepared.iterator()
    assert native.opens == 0
    assert next(first) == next(second) == 0
    first.close()
    assert list(second) == [1]
    assert native.opens == (2 if iterable else 3)
    assert dataset.dataset.opens == 0


def test_unqualified_fallback_does_not_require_new_spec_or_count_validation():
    from dryml.data import Dataset

    class Legacy(Dataset):
        def __init__(self, src):
            self.src = src
            self.count_reads = 0
            super().__init__(spec="legacy declaration")

        def __iter__(self):
            return iter(self.src)

        def __len__(self):
            self.count_reads += 1
            return 5

    dataset = Legacy(source())
    prepared = dataset.prepare()
    assert prepared.execution_level == "eager"
    assert prepared.spec == "legacy declaration"
    assert dataset.count_reads == 0
    with prepared.iterator() as cursor:
        assert list(cursor) == [0, 1, 2, 3, 4]


def test_bounded_source_membership_is_read_once_per_execution(monkeypatch):
    """Frozen source bounds belong to an occurrence, not each row's bookkeeping."""
    from dryml.core.cardinality import Cardinality
    from dryml.methods.stream import _SectionExecution
    from dryml.methods.stream_state import PullKind

    prepared = source().prepare()
    root = prepared.graph._stream_plan._prepared.root
    reads = []
    require_finite = Cardinality.require_finite

    def counted(value):
        reads.append(value)
        return require_finite(value)

    monkeypatch.setattr(Cardinality, "require_finite", counted)
    for _ in range(2):
        with _SectionExecution(root) as execution:
            for value in range(5):
                assert execution.pull() == PullKind.DELIVERY
                assert execution.value == value
                assert execution.root_final is (value == 4)
            assert execution.pull() == PullKind.ROOT_END
    assert reads == [root.cardinality, root.cardinality]


def test_section_analysis_is_once_per_recipe_per_learn_and_never_stale(monkeypatch):
    """Nested build occurrences reuse one preflight, then discard its memo."""
    import dryml.methods.stream as stream
    from dryml.data import Epoch

    src = Epoch(source(), len=2)
    first = Chain(src, src)
    second = Chain(first, src)
    dataset = Chain(second, src)
    graph = dataset.method_graph()
    visits = []
    freeze = stream._freeze_zip_tree

    def counted(tree, leaves):
        if tree is first.sources or tree is second.sources or tree is dataset.sources:
            visits.append(id(tree))
        return freeze(tree, leaves)

    monkeypatch.setattr(stream, "_freeze_zip_tree", counted)
    graph.learn()
    retained = graph._stream_plan._prepared.root
    assert retained.sections == ((2, 2), (1, 1)) * 4
    assert len(visits) == 3
    src.length = 3
    graph.learn()
    assert len(visits) == 6
    assert graph._stream_plan._prepared.root.sections == ((3, 1), (2, 1)) * 4
    assert retained.sections == ((2, 2), (1, 1)) * 4
