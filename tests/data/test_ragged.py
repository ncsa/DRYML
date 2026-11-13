"""Packed NumPy and native TensorFlow ragged values through ordinary/prepared IR."""

from dataclasses import replace
from collections import namedtuple
import importlib.util
import gc
import pickle
import subprocess
import sys
import weakref

import numpy as np
import pytest

from dryml.core.backend import Backend, discover_backend
from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import (
    Dynamic, Layout, SpecHint, TensorSpec, as_tensor_spec, batch_spec_tree,
    fake_from_spec_tree, unbatch_spec_tree,
)
from dryml.data import ArgMax, Batch, Cast, Flatten, GeneratorDataset, Map, Pipe, Project, Scale, Select, Unbatch
from dryml.methods import ImplementationSelectionError, Method, traits
from dryml.models import Model
from dryml.numpy import RaggedTensor

Record = namedtuple("Record", "x label")


def examples():
    return iter(np.arange(length * 5, dtype="float32").reshape(length, 5)
                for length in (3, 0, 2, 5, 1))


def interior_examples():
    return iter(np.arange(10 * length, dtype="float32").reshape(10, length) for length in (2, 4, 0))


def nested_examples(splits="int64"):
    arrays = list(examples())
    return iter((RaggedTensor.stack(arrays[:2], row_splits_dtype=splits),
                 RaggedTensor.stack(arrays[2:], row_splits_dtype=splits)))


def structured_examples():
    return iter({"x": value, "label": np.asarray(len(value), dtype="int64"), "object": object()}
                for value in examples())


def named_examples():
    return iter(Record(value, np.asarray(len(value), dtype="int64")) for value in examples())


class RaggedIdentity(Method):
    """Selected NumPy identity with observable admission before invocation."""

    def __init__(self):
        self.calls = 0

    @traits(backend="numpy")
    def numpy(self, value):
        self.calls += 1
        return value

    def infer_output_spec(self, spec):
        return spec


class RowSum(Model):
    """Return dense per-example sums, including zeros for empty rows."""

    def __init__(self):
        super().__init__(output_spec=TensorSpec("float64", shape=(5,), backend="numpy"))

    @traits(backend="numpy", batch_mode="batched")
    def numpy(self, value):
        return np.stack([value[i].sum(axis=0) for i in range(len(value))]).astype("float64")


class TensorFlowRowSum(Model):
    """Authored TensorFlow model reducing ragged rows to dense feature vectors."""

    @traits(backend="tf", batch_mode="batched")
    def tf_call(self, value):
        import tensorflow as tf
        return tf.reduce_sum(value, axis=1)


def ragged_spec(*, rank=1, shape=(Dynamic, 5), batch=Dynamic, backend="numpy", splits="int64"):
    return TensorSpec("float32", shape=shape, batch=batch, backend=backend, layout="ragged",
                      ragged_rank=rank, row_splits_dtype=splits)


def test_packed_reference_owns_validated_read_only_buffers_and_preserves_empty_rows():
    values = np.arange(15, dtype="float32").reshape(3, 5)
    splits = np.array([0, 2, 2, 3], dtype="int64")
    tensor = RaggedTensor(values, (splits,))
    values[:] = -1
    splits[:] = 0
    assert tensor.shape == (3, None, 5)
    assert tensor[1].shape == (0, 5)
    assert tensor[-1].shape == (1, 5)
    np.testing.assert_array_equal(tensor.flat_values, np.arange(15).reshape(3, 5))
    with pytest.raises(ValueError):
        tensor.flat_values[0] = 2
    with pytest.raises(ValueError):
        tensor.nested_row_splits[0][0] = 2
    with pytest.raises(IndexError):
        tensor[3]
    with pytest.raises(TypeError):
        tensor[:1]
    with pytest.raises(ValueError):
        tensor.with_flat_values(np.zeros((4, 5)))
    restored = pickle.loads(pickle.dumps(tensor))
    assert restored.shape == tensor.shape
    assert not restored.flat_values.flags.writeable
    assert not restored.nested_row_splits[0].flags.writeable
    with pytest.raises(TypeError, match="implicit dense conversion"):
        np.asarray(RaggedTensor.stack([np.ones((2, 5))] * 2))


@pytest.mark.parametrize("splits", ([1, 3], [0, 2, 1, 3], [0, 4], []))
def test_bad_partition_offsets_fail_before_publication(splits):
    with pytest.raises(ValueError):
        RaggedTensor(np.ones((3, 5)), (np.asarray(splits, dtype="int64"),))


def test_nested_partition_linkage_types_and_dense_tail_conflicts_are_rejected():
    with pytest.raises(ValueError):
        RaggedTensor(np.ones((3, 5)), (np.array([0, 2]), np.array([0, 3])))
    with pytest.raises(TypeError):
        RaggedTensor(np.ones((3, 5)), (np.array([0., 3.]),))
    with pytest.raises(TypeError):
        RaggedTensor(np.ones((3, 5), dtype=object), (np.array([0, 3]),))
    with pytest.raises(ValueError):
        RaggedTensor.stack([np.ones((2, 5)), np.ones((3, 6))])
    with pytest.raises(ValueError):
        RaggedTensor.stack([])


@pytest.mark.parametrize("splits", ("int32", "int64"))
def test_numpy_specs_discovery_and_known_spec_calls_preserve_ragged_facts(splits):
    tensor = RaggedTensor.stack(list(examples())[:2], row_splits_dtype=splits)
    spec = as_tensor_spec(tensor, batched=True)
    assert spec.equal_exact(ragged_spec(batch=2, splits=splits))
    assert discover_backend(tensor) is Backend.numpy
    method = RaggedIdentity()
    method.learn(ragged_spec(splits=splits), output_spec=ragged_spec(splits=splits))
    assert method(tensor) is tensor
    second = RaggedTensor.stack(list(examples())[2:], row_splits_dtype=splits)
    assert method(second) is second
    for invalid in (np.ones((2, 3, 5), dtype="float32"),
                    replace(ragged_spec(splits=splits), ragged_rank=2),
                    RaggedTensor.stack([np.ones((1, 4), dtype="float32")]),
                    RaggedTensor.stack(list(examples())[:2], row_splits_dtype="int64" if splits == "int32" else "int32")):
        with pytest.raises(ImplementationSelectionError):
            method(invalid)
    assert method.calls == 2


def test_preparation_does_not_retain_ragged_payload_or_partitions():
    tensor = RaggedTensor.stack(list(examples())[:2])
    payload = weakref.ref(tensor.flat_values.base)
    splits = weakref.ref(tensor.nested_row_splits[0].base)
    method = RaggedIdentity()
    method.learn(ragged_spec(), output_spec=ragged_spec())
    assert method(tensor) is tensor
    del tensor
    gc.collect()
    assert payload() is None and splits() is None


def test_batch_rank_metadata_has_a_truthful_unbatch_inverse():
    source = ragged_spec(rank=1, shape=(Dynamic, Dynamic, 5), batch=None)
    batched = batch_spec_tree(source)
    assert batched.ragged_rank == 2
    assert batched.full_shape == (Dynamic, Dynamic, Dynamic, 5)
    assert unbatch_spec_tree(batched).equal_exact(source)
    outer_only = ragged_spec()
    dense = unbatch_spec_tree(outer_only)
    assert dense.layout is Layout.DENSE
    assert dense.ragged_rank is None and dense.row_splits_dtype is None
    assert dense.shape == (Dynamic, 5) and not dense.batched
    assert outer_only.with_batch(3).ragged_rank == 1


@pytest.mark.parametrize("changes", ({"ragged_rank": 0}, {"ragged_rank": True},
                                    {"ragged_rank": 3}, {"row_splits_dtype": "float32"}))
def test_invalid_ragged_spec_metadata_is_rejected(changes):
    with pytest.raises(ValueError):
        replace(ragged_spec(), **changes)


def test_dense_ragged_batch_map_model_and_unbatch_share_ordinary_and_prepared_semantics():
    source = GeneratorDataset(examples, cardinality=5,
                              spec=TensorSpec("float32", shape=(Dynamic, 5), backend="numpy"))
    batches = Batch(source, 2, ragged=True)
    assert batches.spec.equal_exact(ragged_spec())
    mapped = Map(batches, Pipe(Cast("float64"), Scale(mean=1, std=2, dtype="float64")), preserves_examples=True)
    assert mapped.example_cardinality() == Cardinality.finite(5)
    graph = mapped.method_graph()
    graph.learn()
    for opener in (mapped.iterator, graph.iterator):
        with opener() as cursor:
            values = list(cursor)
        assert [len(value) for value in values] == [2, 2, 1]
        assert all(value.ragged_rank == 1 and value.dtype == np.dtype("float64") for value in values)
        rows = [value[i] for value in values for i in range(len(value))]
        for row, original in zip(rows, examples(), strict=True):
            np.testing.assert_array_equal(row, (original.astype("float64") - 1) / 2)

    unbatched = Unbatch(mapped)
    prepared = unbatched.method_graph()
    prepared.learn()
    for opener in (unbatched.iterator, prepared.iterator):
        with opener() as cursor:
            assert [value.shape for value in cursor] == [(3, 5), (0, 5), (2, 5), (5, 5), (1, 5)]
    model = Map(batches, RowSum(), preserves_examples=True)
    prepared_model = model.method_graph()
    prepared_model.learn()
    with prepared_model.iterator() as cursor:
        result = np.concatenate(list(cursor))
    np.testing.assert_array_equal(result, np.stack([value.sum(axis=0) for value in examples()]))


def test_fixed_axes_before_ragged_axes_and_nested_ragged_samples_keep_partitions():
    source = GeneratorDataset(interior_examples, cardinality=3,
                              spec=TensorSpec("float32", shape=(10, Dynamic), backend="numpy"))
    dataset = Batch(source, 2, ragged=True)
    assert dataset.spec.equal_exact(ragged_spec(rank=2, shape=(10, Dynamic)))
    graph = dataset.method_graph()
    graph.learn()
    with graph.iterator() as cursor:
        packed = list(cursor)
    assert [value.shape for value in packed] == [(2, 10, None), (1, 10, 0)]
    unpacked = Unbatch(dataset)
    assert unpacked.spec.ragged_rank == 1
    for result, original in zip(unpacked, interior_examples(), strict=True):
        assert isinstance(result, RaggedTensor)
        for row in range(10):
            np.testing.assert_array_equal(result[row], original[row])

    samples = GeneratorDataset(nested_examples, cardinality=2,
                               spec=ragged_spec(shape=(Dynamic, Dynamic, 5), batch=None))
    grouped = Batch(samples, 2)
    assert grouped.spec.ragged_rank == 2
    roundtrip = Unbatch(grouped)
    for result, original in zip(roundtrip, nested_examples(), strict=True):
        np.testing.assert_array_equal(result.flat_values, original.flat_values)
        np.testing.assert_array_equal(result.nested_row_splits[0], original.nested_row_splits[0])


@pytest.mark.parametrize("splits", ("int32", "int64"))
def test_unspecified_partition_dtype_uses_actual_common_dtype_without_narrowing(splits):
    source = GeneratorDataset(nested_examples, splits=splits, cardinality=2,
                              spec=replace(ragged_spec(shape=(Dynamic, Dynamic, 5), batch=None), row_splits_dtype=None))
    assert next(iter(Batch(source, 2))).row_splits_dtype == np.dtype(splits)


def test_all_dynamic_axes_and_all_empty_rows_keep_rank_and_dense_tail():
    value = RaggedTensor.stack([np.ones((2, 3), dtype="float32"), np.ones((4, 1), dtype="float32")], ragged_rank=2)
    assert as_tensor_spec(value, batched=True).equal_exact(ragged_spec(rank=2, shape=(Dynamic, Dynamic), batch=2))
    empty = RaggedTensor.stack([np.empty((0, 5), dtype="float32")] * 2)
    assert empty.shape == (2, 0, 5)
    assert len(empty) == 2
    assert [empty[i].shape for i in range(2)] == [(0, 5), (0, 5)]


def test_mixed_trees_keep_scalar_labels_dense_and_opaque_values_uninspected():
    source = GeneratorDataset(structured_examples, cardinality=5, spec={
        "x": TensorSpec("float32", shape=(Dynamic, 5), backend="numpy"),
        "label": TensorSpec("int64", shape=(), backend="numpy"),
        "object": TensorSpec("object", layout="python"),
    })
    dataset = Batch(source, 2, ragged=True)
    assert dataset.spec["label"].layout is Layout.DENSE
    mapped = Map(dataset, Project({"x": Pipe(Select("x"), Cast("float64")),
                                   "label": Select("label"), "object": Select("object")}), preserves_examples=True)
    graph = Unbatch(mapped).method_graph()
    graph.learn()
    with graph.iterator() as cursor:
        values = list(cursor)
    assert [int(value["label"]) for value in values] == [3, 0, 2, 5, 1]
    assert all(value["x"].dtype == np.dtype("float64") and type(value["object"]) is object for value in values)


def test_ragged_namedtuple_trees_preserve_runtime_structure_in_prepared_roundtrip():
    source = GeneratorDataset(named_examples, cardinality=5, spec=Record(
        TensorSpec("float32", shape=(Dynamic, 5), backend="numpy"),
        TensorSpec("int64", shape=(), backend="numpy")))
    dataset = Unbatch(Batch(source, 2, ragged=True))
    graph = dataset.method_graph()
    graph.learn()
    for opener in (dataset.iterator, graph.iterator):
        with opener() as cursor:
            values = list(cursor)
        assert all(type(value) is Record for value in values)
        assert [int(value.label) for value in values] == [3, 0, 2, 5, 1]


def test_unsupported_ragged_transforms_and_implicit_dense_conversion_fail_explicitly():
    spec = ragged_spec()
    for method in (Flatten(), ArgMax()):
        with pytest.raises(TypeError, match="ragged"):
            method.infer_output_spec(spec)
    with pytest.raises(ValueError, match="ragged"):
        fake_from_spec_tree(spec)
    from dryml.methods.conversion import make_edge
    with pytest.raises(TypeError):
        make_edge(spec, Backend.torch)
    with pytest.raises(TypeError):
        Batch(GeneratorDataset(examples, spec=TensorSpec("float32", shape=(Dynamic, 5), backend="torch")), 2, ragged=True)


def _tf_examples():
    import tensorflow as tf
    return iter(tf.convert_to_tensor(value) for value in examples())


def _tf_interior_examples():
    import tensorflow as tf
    return iter(tf.convert_to_tensor(value) for value in interior_examples())


def _tf_nested_examples(splits="int64"):
    import tensorflow as tf
    return iter(tf.RaggedTensor.from_nested_row_splits(value.flat_values, value.nested_row_splits)
                for value in nested_examples(splits))


def _check_tensorflow():
    import tensorflow as tf
    import dryml.tf

    spec = TensorSpec("float32", shape=(Dynamic, 5), backend="tf")
    source = GeneratorDataset(_tf_examples, cardinality=5, spec=spec)
    dataset = Map(Batch(source, 2, ragged=True), Pipe(Cast("float64"), Scale(dtype="float64")), preserves_examples=True)
    assert dataset.spec.equal_exact(replace(ragged_spec(backend="tf"), dtype="float64"))
    graph = dataset.method_graph()
    graph.learn()
    for opener in (dataset.iterator, graph.iterator):
        with opener() as cursor:
            values = list(cursor)
        assert [int(value.nrows()) for value in values] == [2, 2, 1]
        assert all(value.ragged_rank == 1 for value in values)
    unbatched = Unbatch(dataset)
    unbatch_graph = unbatched.method_graph()
    unbatch_graph.learn()
    with unbatch_graph.iterator() as cursor:
        results = list(cursor)
    for value, expected in zip(results, examples(), strict=True):
        np.testing.assert_array_equal(value.numpy(), expected)

    native = tf.data.Dataset.from_generator(dataset.iterator, output_signature=dryml.tf.output_signature(dataset.spec))
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    assert sum(int(value.nrows()) for value in native.with_options(options)) == 5
    model = Map(dataset, TensorFlowRowSum(output_spec=TensorSpec("float64", shape=(5,), backend="tf")),
                preserves_examples=True)
    model_graph = model.method_graph()
    model_graph.learn()
    with model_graph.iterator() as cursor:
        sums = np.concatenate([value.numpy() for value in cursor])
    np.testing.assert_array_equal(sums, np.stack([value.sum(axis=0) for value in examples()]))

    tensor = tf.RaggedTensor.from_row_splits(tf.ones((3, 5)), [0, 3, 3])
    declared = ragged_spec(backend="tf")
    assert isinstance(declared.tf(include_batch=False), tf.TensorSpec)
    method = Cast("float64")
    method.learn(declared, output_spec=declared.with_dtype("float64"))
    compiled = tf.function(method, input_signature=[declared.tf()])
    assert compiled(tensor).ragged_rank == 1
    np.testing.assert_array_equal(compiled(tensor).flat_values.numpy(), np.ones((3, 5)))

    interior = GeneratorDataset(_tf_interior_examples, cardinality=3,
                                spec=TensorSpec("float32", shape=(10, Dynamic), backend="tf"))
    packed = Batch(interior, 2, ragged=True)
    selected = Select()
    selected.learn(packed.spec, output_spec=packed.spec)
    compiled_interior = tf.function(selected, input_signature=[packed.spec.tf()])
    for value in packed:
        assert value.ragged_rank == 2
        assert selected(value) is value
        assert compiled_interior(value).ragged_rank == 2
    for value, expected in zip(Unbatch(packed), interior_examples(), strict=True):
        np.testing.assert_array_equal(value.to_tensor().numpy(), expected)
    samples = GeneratorDataset(_tf_nested_examples, cardinality=2,
                               spec=ragged_spec(shape=(Dynamic, Dynamic, 5), batch=None, backend="tf"))
    grouped = Batch(samples, 2)
    roundtrip = Unbatch(grouped)
    prepared = roundtrip.method_graph()
    prepared.learn()
    with prepared.iterator() as cursor:
        results = list(cursor)
    for value, expected in zip(results, nested_examples(), strict=True):
        np.testing.assert_array_equal(value.flat_values.numpy(), expected.flat_values)
        np.testing.assert_array_equal(value.nested_row_splits[0].numpy(), expected.nested_row_splits[0])
    samples32 = GeneratorDataset(_tf_nested_examples, splits="int32", cardinality=2,
                                 spec=replace(samples.spec, row_splits_dtype=None))
    assert next(iter(Batch(samples32, 2))).row_splits.dtype == tf.int32

    tensor32 = tf.RaggedTensor.from_row_splits(tf.ones((3, 5)), tf.constant([0, 3, 3], dtype=tf.int32))
    assert as_tensor_spec(tensor32, batched=True).row_splits_dtype.name == "int32"
    try:
        method(tensor32)
    except ImplementationSelectionError:
        pass
    else:
        raise AssertionError("ragged partition dtype drift was not checked")


def test_tensorflow_ragged_roundtrip_in_fresh_process():
    if importlib.util.find_spec("tensorflow") is None:
        pytest.skip("optional TensorFlow backend is not installed")
    result = subprocess.run([sys.executable, "-c", "from tests.data.test_ragged import _check_tensorflow; _check_tensorflow()"],
                            capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
