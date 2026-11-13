"""Padding geometry, masks, spec inference and ordinary/prepared execution."""

import importlib.util
import subprocess
import sys

import numpy as np
import pytest

from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import Dynamic, TensorSpec
from dryml.data import Batch, GeneratorDataset, Map, Padding, Pipe, Project, Select, Unbatch
from dryml.methods import traits
from dryml.models import Model
from dryml.numpy import RaggedTensor


def dense_spec(shape=(Dynamic, 5), *, batch=Dynamic, backend="numpy"):
    return TensorSpec("float32", shape=shape, batch=batch, backend=backend)


def ragged_spec(shape=(Dynamic, 5), *, rank=1, batch=Dynamic, backend="numpy"):
    return TensorSpec("float32", shape=shape, batch=batch, backend=backend,
                      layout="ragged", ragged_rank=rank, row_splits_dtype="int64")


def prepared(method, spec):
    method.learn(spec, output_spec=method.infer_output_spec(spec))
    return method


def examples():
    return iter(np.arange(length * 5, dtype="float32").reshape(length, 5) for length in (0, 2, 4))


def tf_examples():
    import tensorflow as tf
    return iter(tf.convert_to_tensor(value) for value in examples())


_closes = []


def closeable_examples():
    try:
        for length in (2, 5):
            yield np.ones((length, 5), dtype="float32")
    finally:
        _closes.append(True)


class MaskedTorchSum(Model):
    """Consume padded values and masks after an ordinary dense backend handoff."""

    @traits(backend="torch", batch_mode="batched")
    def _torch(self, value):
        return (value["values"] * value["mask"].unsqueeze(-1)).sum(dim=1)


@pytest.mark.parametrize("options", ({"axis": True}, {"axis": ()}, {"axis": (0, 0)},
    {"length": -1}, {"length": True}, {"length": (2, 3)}, {"side": "middle"},
    {"return_mask": 1}, {"truncate": 1}, {"value": []}))
def test_invalid_padding_options_fail_at_declaration(options):
    with pytest.raises((TypeError, ValueError)):
        Padding(**options)


def test_output_spec_fixes_targets_clears_partitions_and_names_mask_prefix():
    spec = TensorSpec("float32", shape=(Dynamic, 5), batch=Dynamic, backend="numpy",
                      layout="ragged", ragged_rank=1, row_splits_dtype="int32",
                      axis_names=("time", "features"), batch_axis_name="examples")
    result = Padding(7, return_mask=True).infer_output_spec(spec)
    assert result["values"].equal_exact(TensorSpec("float32", shape=(7, 5), batch=Dynamic,
        backend="numpy", axis_names=("time", "features"), batch_axis_name="examples"))
    assert result["mask"].equal_exact(TensorSpec("bool", shape=(7,), batch=Dynamic,
        backend="numpy", axis_names=("time",), batch_axis_name="examples"))
    assert Padding().infer_output_spec(spec).shape == (Dynamic, 5)
    assert Padding().infer_output_spec(dense_spec((3, 5))).shape == (3, 5)


def test_inference_rejects_ambiguous_axes_layouts_and_unresolved_ragged_dimensions():
    for method, spec in (
        (Padding(axis=2), dense_spec()),
        (Padding(axis=(0, -2)), dense_spec()),
        (Padding(3), dense_spec((4, 5))),
        (Padding(), TensorSpec("object", layout="python")),
        (Padding(), TensorSpec("float32", shape=None)),
        (Padding(), ragged_spec((Dynamic, Dynamic), rank=2)),
        (Padding(), {"x": dense_spec()}),
    ):
        with pytest.raises((TypeError, ValueError)):
            method.infer_output_spec(spec)


@pytest.mark.parametrize("side", ("left", "right"))
def test_dense_padding_and_mask_preserve_real_zero_values_and_batch_axis(side):
    value = np.zeros((2, 3, 5), dtype="float32")
    method = prepared(Padding(5, value=0, side=side, return_mask=True), dense_spec())
    result = method(value)
    assert result["values"].shape == (2, 5, 5)
    assert result["values"].dtype == value.dtype
    expected = np.array([False, False, True, True, True] if side == "left" else [True, True, True, False, False])
    np.testing.assert_array_equal(result["mask"], np.tile(expected, (2, 1)))
    np.testing.assert_array_equal(value, np.zeros((2, 3, 5)))


def test_fixed_targets_reject_oversize_unless_truncation_is_explicit():
    value = np.arange(25, dtype="float32").reshape(1, 5, 5)
    reject = prepared(Padding(3, return_mask=True), dense_spec())
    with pytest.raises(ValueError, match="truncate"):
        reject(value)
    for side, indices in (("right", slice(0, 3)), ("left", slice(2, 5))):
        result = prepared(Padding(3, side=side, truncate=True, return_mask=True), dense_spec())(value)
        np.testing.assert_array_equal(result["values"], value[:, indices])
        assert result["mask"].all()


def test_multiple_axes_negative_axis_and_unbatched_samples_use_sample_coordinates():
    value = np.ones((2, 3, 4, 5), dtype="float32")
    method = prepared(Padding((6, 7), axis=(0, 1), side="left", return_mask=True),
                      dense_spec((Dynamic, Dynamic, 5)))
    result = method(value)
    assert result["values"].shape == (2, 6, 7, 5)
    assert result["mask"].shape == (2, 6, 7)
    assert result["mask"][:, 3:, 3:].all()
    assert result["mask"].sum() == 2 * 3 * 4
    element = prepared(Padding(7, axis=-1, return_mask=True), dense_spec((3, 5), batch=None))
    assert element(np.ones((3, 5), dtype="float32"))["mask"].shape == (3, 7)


def test_masks_preserve_boolean_data_and_values_only_mode_keeps_tensor_contract():
    spec = TensorSpec("bool", shape=(Dynamic,), backend="numpy")
    result = prepared(Padding(4, value=False, return_mask=True), spec)(np.array([False, True]))
    np.testing.assert_array_equal(result["values"], [False, True, False, False])
    np.testing.assert_array_equal(result["mask"], [True, True, False, False])
    only = prepared(Padding(4), spec)(np.array([False, True]))
    assert isinstance(only, np.ndarray) and only.dtype == np.dtype("bool")
    nan_data = prepared(Padding(4, return_mask=True), dense_spec((Dynamic,), batch=None))(np.array([np.nan, 0], dtype="float32"))
    np.testing.assert_array_equal(nan_data["mask"], [True, True, False, False])


@pytest.mark.parametrize("side", ("left", "right"))
def test_ragged_padding_uses_row_lengths_and_emits_masks_without_feature_axes(side):
    rows = list(examples())
    value = RaggedTensor.stack(rows)
    method = prepared(Padding(5, side=side, value=-1, return_mask=True), ragged_spec())
    result = method(value)
    assert result["values"].shape == (3, 5, 5)
    assert result["mask"].shape == (3, 5)
    for index, row in enumerate(rows):
        start = 5 - len(row) if side == "left" else 0
        np.testing.assert_array_equal(result["values"][index, start:start + len(row)], row)
        expected = np.zeros(5, dtype=bool)
        expected[start:start + len(row)] = True
        np.testing.assert_array_equal(result["mask"][index], expected)
        assert (result["values"][index, ~expected] == -1).all()


def test_batch_maxima_empty_rows_zero_target_and_zero_feature_axes_have_geometry_masks():
    value = RaggedTensor.stack(list(examples()))
    result = prepared(Padding(return_mask=True), ragged_spec())(value)
    assert result["values"].shape == (3, 4, 5)
    assert result["mask"].sum() == 6
    empty = RaggedTensor.stack([np.empty((0, 5), dtype="float32")] * 2)
    result = prepared(Padding(return_mask=True), ragged_spec())(empty)
    assert result["values"].shape == (2, 0, 5)
    assert result["mask"].shape == (2, 0)
    result = prepared(Padding(0, truncate=True, return_mask=True), ragged_spec())(value)
    assert result["values"].shape == (3, 0, 5)
    for feature_size in (0, 5):
        uniform_features = RaggedTensor.stack([np.ones((2, feature_size), dtype="float32"),
                                               np.ones((1, feature_size), dtype="float32")], ragged_rank=2)
        result = prepared(Padding(3, return_mask=True), ragged_spec((Dynamic, feature_size), rank=2))(uniform_features)
        assert result["values"].shape == (2, 3, feature_size)
        np.testing.assert_array_equal(result["mask"], [[True, True, False], [True, False, False]])
    result = prepared(Padding(3, return_mask=True), dense_spec((Dynamic, 0)))(np.empty((2, 1, 0), dtype="float32"))
    np.testing.assert_array_equal(result["mask"], [[True, False, False]] * 2)


def nested_value():
    return RaggedTensor(np.arange(30, dtype="float32").reshape(6, 5),
                        (np.array([0, 2, 3]), np.array([0, 1, 4, 6])))


def test_nested_ragged_axes_left_alignment_and_explicit_truncation():
    value = nested_value()
    spec = ragged_spec((Dynamic, Dynamic, 5), rank=2)
    result = prepared(Padding((3, 4), axis=(0, 1), side="left", return_mask=True), spec)(value)
    expected = np.zeros((2, 3, 4), dtype=bool)
    expected[0, 1, 3] = True
    expected[0, 2, 1:] = True
    expected[1, 2, 2:] = True
    np.testing.assert_array_equal(result["mask"], expected)
    np.testing.assert_array_equal(result["values"][0, 2, 1:], value[0][1])
    result = prepared(Padding((1, 2), axis=(0, 1), side="left", truncate=True, return_mask=True), spec)(value)
    np.testing.assert_array_equal(result["values"][0, 0], value[0][1][-2:])
    np.testing.assert_array_equal(result["values"][1, 0], value[1][0])
    assert result["mask"].all()
    scalar_rows = RaggedTensor.stack([np.ones((2, 3), dtype="float32"), np.ones((3, 1), dtype="float32")], ragged_rank=2)
    result = prepared(Padding((3, 4), axis=(0, 1), return_mask=True), ragged_spec((Dynamic, Dynamic), rank=2))(scalar_rows)
    assert result["mask"].shape == result["values"].shape == (2, 3, 4)
    assert result["mask"].sum() == 9
    assert (result["values"][result["mask"]] == 1).all()


def test_padding_ragged_axes_and_dense_tail_axes_together_extends_mask_prefix():
    value = RaggedTensor.stack(list(examples()))
    result = prepared(Padding((None, 7), axis=(0, 1), return_mask=True), ragged_spec())(value)
    assert result["values"].shape == result["mask"].shape == (3, 4, 7)
    assert result["mask"][1, :2, :5].all()
    assert result["mask"].sum() == 6 * 5


def test_padding_pipeline_and_prepared_ports_keep_examples_and_structured_masks(monkeypatch):
    import dryml.data.source as source_module
    source = GeneratorDataset(examples, cardinality=3, spec=dense_spec(batch=None))
    dataset = Map(Batch(source, 2, ragged=True), Padding(5, return_mask=True), preserves_examples=True)
    graph = dataset.method_graph()
    def forbidden(*args, **kwargs):
        pytest.fail("Padding preparation opened the source")
    with monkeypatch.context() as patch:
        patch.setattr(source_module, "_fresh_iterable", forbidden)
        graph.learn()
        assert dataset.example_cardinality() == Cardinality.finite(3)
        assert dataset.spec["mask"].shape == (5,)
    for opener in (dataset.iterator, graph.iterator):
        with opener() as cursor:
            batches = list(cursor)
        assert [value["values"].shape[0] for value in batches] == [2, 1]
        assert sum(int(value["mask"].sum()) for value in batches) == 6
    unbatched = Unbatch(dataset)
    assert [value["mask"].sum() for value in unbatched] == [0, 2, 4]
    method = prepared(Pipe(Padding(5, return_mask=True), Project({"x": Select("values"), "valid": Select("mask")})),
                      ragged_spec())
    assert set(method(RaggedTensor.stack(list(examples())))) == {"x", "valid"}


def test_padding_failure_closes_ordinary_and_prepared_source_without_partial_output():
    _closes.clear()
    source = GeneratorDataset(closeable_examples, cardinality=2, spec=dense_spec(batch=None))
    dataset = Map(Batch(source, 2, ragged=True), Padding(3), preserves_examples=True)
    graph = dataset.method_graph()
    graph.learn()
    for opener in (dataset.iterator, graph.iterator):
        with opener() as cursor, pytest.raises(ValueError, match="truncate"):
            next(cursor)
    assert len(_closes) == 2


@pytest.mark.parametrize("backend,module", (("tf", "tensorflow"), ("torch", "torch"), ("jax", "jax")))
def test_native_padding_and_masks_in_fresh_process(backend, module):
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"optional {module} backend is not installed")
    result = subprocess.run([sys.executable, "-c", "from tests.data.test_padding import _check_native; import sys; _check_native(sys.argv[1])", backend],
                            capture_output=True, text=True, timeout=90, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def _check_native(backend):
    if backend == "tf":
        import tensorflow as tf
        make, host = tf.convert_to_tensor, lambda value: value.numpy()
    elif backend == "torch":
        import torch
        make, host = torch.from_numpy, lambda value: value.detach().numpy()
    else:
        import jax
        import jax.numpy as jnp
        make, host = jnp.asarray, np.asarray
    array = np.arange(30, dtype="float32").reshape(2, 3, 5)
    spec = dense_spec(backend=backend)
    for side in ("left", "right"):
        method = prepared(Padding(5, side=side, return_mask=True), spec)
        result = method(make(array))
        expected = prepared(Padding(5, side=side, return_mask=True), dense_spec())(array)
        for key in expected:
            np.testing.assert_array_equal(host(result[key]), expected[key])
    cropped = prepared(Padding(2, side="left", truncate=True, return_mask=True), spec)(make(array))
    np.testing.assert_array_equal(host(cropped["values"]), array[:, -2:])
    assert host(cropped["mask"]).all()
    empty = prepared(Padding(4, return_mask=True), dense_spec((Dynamic, 0), backend=backend))(make(np.empty((2, 1, 0), dtype="float32")))
    np.testing.assert_array_equal(host(empty["mask"]), [[True, False, False, False]] * 2)
    boolean = TensorSpec("bool", shape=(Dynamic,), batch=Dynamic, backend=backend)
    result = prepared(Padding(4, return_mask=True), boolean)(make(np.array([[False, True]])))
    np.testing.assert_array_equal(host(result["values"]), [[False, True, False, False]])
    np.testing.assert_array_equal(host(result["mask"]), [[True, True, False, False]])
    zero = prepared(Padding(0, truncate=True, return_mask=True), spec)(make(array))
    assert tuple(zero["values"].shape) == (2, 0, 5)
    reject = prepared(Padding(2, return_mask=True), spec)
    errors = (ValueError, tf.errors.InvalidArgumentError) if backend == "tf" else (ValueError,)
    try:
        reject(make(array))
    except errors:
        pass
    else:
        raise AssertionError("oversized inputs were silently truncated")
    if backend == "torch":
        value = make(array).requires_grad_()
        result = prepared(Padding(5, return_mask=True), spec)(value)
        result["values"].sum().backward()
        np.testing.assert_array_equal(value.grad.numpy(), np.ones_like(array))
        source = GeneratorDataset(examples, cardinality=3, spec=dense_spec(batch=None))
        model = MaskedTorchSum(output_spec=TensorSpec("float32", shape=(5,), backend="torch"))
        dataset = Map(Batch(source, 2, ragged=True), Padding(5, value=10, return_mask=True), model,
                      preserves_examples=True)
        graph = dataset.method_graph()
        graph.learn()
        for opener in (dataset.iterator, graph.iterator):
            with opener() as cursor:
                sums = np.concatenate([host(value) for value in cursor])
            np.testing.assert_array_equal(sums, np.stack([value.sum(axis=0) for value in examples()]))
    elif backend == "jax":
        method = prepared(Padding(5, return_mask=True), spec)
        np.testing.assert_array_equal(host(jax.jit(method)(make(array))["mask"]), [[True, True, True, False, False]] * 2)
        gradient = jax.grad(lambda value: method(value)["values"].sum())(make(array))
        np.testing.assert_array_equal(host(gradient), np.ones_like(array))
    else:
        method = prepared(Padding(5, return_mask=True), spec)
        compiled = tf.function(method, input_signature=[spec.tf()])
        assert tuple(compiled(make(array))["values"].shape) == (2, 5, 5)
        source = GeneratorDataset(tf_examples, cardinality=3, spec=dense_spec(batch=None, backend="tf"))
        dataset = Map(Batch(source, 2, ragged=True), Padding(return_mask=True), preserves_examples=True)
        graph = dataset.method_graph()
        graph.learn()
        for opener in (dataset.iterator, graph.iterator):
            with opener() as cursor:
                assert [tuple(value["mask"].shape) for value in cursor] == [(2, 2), (1, 4)]
        for reference, declared in ((RaggedTensor.stack(list(examples())), ragged_spec()),
                                    (nested_value(), ragged_spec((Dynamic, Dynamic, 5), rank=2)),
                                    (RaggedTensor.stack([np.ones((2, 3), dtype="float32"), np.ones((3, 1), dtype="float32")], ragged_rank=2),
                                     ragged_spec((Dynamic, Dynamic), rank=2))):
            value = tf.RaggedTensor.from_nested_row_splits(reference.flat_values, reference.nested_row_splits)
            axes = tuple(range(declared.ragged_rank))
            lengths = (5,) if len(axes) == 1 else (3, 4)
            tf_spec = ragged_spec(declared.shape, rank=declared.ragged_rank, backend="tf")
            method = prepared(Padding(lengths, axis=axes, side="left", return_mask=True), tf_spec)
            expected = prepared(Padding(lengths, axis=axes, side="left", return_mask=True), declared)(reference)
            compiled = tf.function(method, input_signature=[tf_spec.tf()])
            for output in (method(value), compiled(value)):
                for key in expected:
                    np.testing.assert_array_equal(host(output[key]), expected[key])
        for reference, declared, axes, lengths in (
            (RaggedTensor.stack(list(examples())), ragged_spec(), (0, 1), (None, 7)),
            (RaggedTensor.stack([np.empty((0, 5), dtype="float32")] * 2), ragged_spec(), (0,), (None,)),
            (RaggedTensor.stack([np.ones((2, 0), dtype="float32")] * 2, ragged_rank=2),
             ragged_spec((Dynamic, 0), rank=2), (0,), (4,)),
        ):
            value = tf.RaggedTensor.from_nested_row_splits(reference.flat_values, reference.nested_row_splits)
            tf_spec = ragged_spec(declared.shape, rank=declared.ragged_rank, backend="tf")
            method = prepared(Padding(lengths, axis=axes, side="left", return_mask=True), tf_spec)
            expected = prepared(Padding(lengths, axis=axes, side="left", return_mask=True), declared)(reference)
            compiled = tf.function(method, input_signature=[tf_spec.tf()])
            for key, output in compiled(value).items():
                np.testing.assert_array_equal(host(output), expected[key])
        flat = tf.Variable(np.arange(30, dtype="float32").reshape(6, 5))
        method = prepared(Padding(3, side="left", truncate=True, return_mask=True), ragged_spec(backend="tf"))
        with tf.GradientTape() as tape:
            value = tf.RaggedTensor.from_row_splits(flat, tf.constant([0, 2, 6], tf.int64))
            loss = tf.reduce_sum(method(value)["values"])
        expected = np.ones((6, 5), dtype="float32")
        expected[2] = 0
        np.testing.assert_array_equal(tape.gradient(loss, flat).numpy(), expected)
