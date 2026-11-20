"""TF metadata semantics and user hooks; Padding staging is checked in test_padding."""

from collections import namedtuple

import numpy as np
import pytest

from dryml.core.tensor_spec import Dynamic, TensorSpec
from dryml.methods import Method, traits


def _tensor_hook(value):
    import tensorflow as tf
    if tf.reduce_sum(value) > 0:
        return value * value
    return -value


class TensorBranch(Method):
    """Keep user tensor control flow and a called hook inside AutoGraph."""

    @traits(backend="tf", batch_mode="batched")
    def tf(self, value):
        return _tensor_hook(value)


def test_staging_preserves_user_autograph_and_method_validation():
    tf = pytest.importorskip("tensorflow")
    import dryml.tf
    from tensorflow.python.autograph.impl.api import StagingError

    spec = TensorSpec("float32", shape=(Dynamic, 5), batch=Dynamic, backend="tf")
    method = TensorBranch()
    method.learn(spec, output_spec=spec)
    compiled = tf.function(method, input_signature=[spec.tf()])
    for shape, sign in (((2, 3, 5), 1), ((1, 1, 5), -1)):
        value = tf.constant(np.full(shape, sign * 2, dtype="float32"))
        with tf.GradientTape() as tape:
            tape.watch(value)
            result = compiled(value)
            loss = tf.reduce_sum(result)
        np.testing.assert_array_equal(result, np.full(shape, 4 if sign > 0 else 2))
        np.testing.assert_array_equal(tape.gradient(loss, value), np.full(shape, 4 if sign > 0 else -1))
    assert compiled.experimental_get_tracing_count() == 1
    for signature in (tf.TensorSpec((None, None, 4), tf.float32),
                      tf.TensorSpec((None, None, 5), tf.float64)):
        with pytest.raises(StagingError, match="_validate_input_specs"):
            tf.function(method, input_signature=[signature]).get_concrete_function()
    invalid = TensorBranch()
    invalid.learn(spec, output_spec=spec.with_dtype("int32"))
    with pytest.raises(StagingError, match="_validate_raw_result"):
        tf.function(invalid, input_signature=[spec.tf()]).get_concrete_function()


def test_metadata_keeps_tree_layout_host_reads_and_failure_order(monkeypatch):
    tf = pytest.importorskip("tensorflow")
    from dryml.tf import as_tensor_spec
    from tensorflow.python.autograph.impl import api

    uniform = tf.RaggedTensor.from_nested_row_splits(
        tf.ones((4, 5)), (tf.constant([0, 2], tf.int32), tf.constant([0, 2, 4], tf.int32)))
    reads = []
    eager_type = type(uniform.row_splits)
    original_numpy = eager_type.numpy

    def numpy(value):
        reads.append(id(value))
        return original_numpy(value)

    monkeypatch.setattr(eager_type, "numpy", numpy)
    eager = as_tensor_spec(uniform, batched=True, batch_axis_name="examples")
    assert eager.equal_exact(TensorSpec("float32", shape=(2, 2, 5), batch=1,
        layout="ragged", ragged_rank=2, row_splits_dtype="int32", backend="tf",
        batch_axis_name="examples"))
    assert reads == [id(splits) for splits in uniform.nested_row_splits]
    reads.clear()
    Record = namedtuple("Record", "dense sparse ragged")
    native = Record(tf.TensorSpec((None, 5), tf.float64),
                    tf.SparseTensorSpec((None, 5), tf.float32), tf.type_spec_from_value(uniform))
    expected = as_tensor_spec({"x": [native, ()]}, batched=True)
    assert expected["x"][0].dense.equal_exact(TensorSpec("float64", shape=(5,), batch=Dynamic, backend="tf"))
    assert expected["x"][0].sparse.equal_exact(TensorSpec("float32", shape=(5,), batch=Dynamic,
        backend="tf", layout="sparse", sparse_format="tf_sparse"))
    assert expected["x"][0].ragged.shape == (Dynamic, Dynamic, 5)

    @tf.function(input_signature=[native])
    def traced(values):
        inferred = as_tensor_spec(values, batched=True)
        assert all(left.equal_exact(right) for left, right in zip(inferred, expected["x"][0]))
        actual = as_tensor_spec({"x": [native, ()]}, batched=True)
        assert type(actual["x"][0]) is Record
        assert actual == expected
        assert as_tensor_spec(tf.TensorSpec(None, tf.float32)).shape is None
        assert as_tensor_spec(tf.TensorSpec((), tf.float32)).shape == ()
        return values.ragged.flat_values

    values = Record(tf.ones((1, 5), tf.float64),
                    tf.SparseTensor([[0, 1]], tf.constant([1.]), [1, 5]), uniform)
    reads.clear()
    metadata_requests = []
    original_convert = api._convert_actual

    def counted(entity, program_ctx):
        if ((entity.__module__ == "dryml.tf.tensor_spec" and entity.__name__ != "as_tensor_spec")
                or entity.__module__ == "dryml.tf.dtype"):
            metadata_requests.append((entity.__module__, entity.__qualname__))
        return original_convert(entity, program_ctx)

    with monkeypatch.context() as patch:
        patch.setattr(api, "_convert_actual", counted)
        concrete = traced.get_concrete_function()
    assert metadata_requests == []
    assert reads == []
    concrete(values)
    for bad, error, message in (
        (tf.TensorSpec(None, tf.resource), ValueError, "unknown rank"),
        (tf.TensorSpec((), tf.resource), ValueError, "rank-0"),
        (tf.TensorSpec((2,), tf.resource), ValueError, "Unrecognized dtype string 'resource'"),
        (tf.RaggedTensorSpec((2, 3), tf.float32, ragged_rank=0), ValueError, "ragged_rank"),
        (tf.RaggedTensorSpec((2, None), tf.float32, ragged_rank=1,
                            row_splits_dtype=tf.float32), ValueError, "row_splits_dtype"),
        (tf.data.DatasetSpec(tf.TensorSpec((), tf.float32)), TypeError, "Unsupported TensorFlow spec/value type"),
        (object(), TypeError, "Could not build"),
    ):
        with pytest.raises(error, match=message):
            as_tensor_spec({"first": bad, "later": object()}, batched=True)

        @tf.function
        def rejected():
            as_tensor_spec({"first": bad, "later": object()}, batched=True)
            return tf.constant(1)

        with pytest.raises(error, match=message):
            rejected()


def test_custom_typespec_retains_ordinary_conversion_path(monkeypatch):
    tf = pytest.importorskip("tensorflow")
    from dryml.tf import as_tensor_spec
    from tensorflow.python.autograph.impl import api
    from tensorflow.python.autograph.core import ag_ctx

    class CustomSpec(tf.TensorSpec):
        pass

    conversion_contexts = []
    shape_contexts = []

    class CustomShape(tf.TensorShape):
        def as_list(self):
            shape_contexts.append(ag_ctx.control_status_ctx().status)
            return super().as_list()

    class Convertible:
        def __tf_tensor__(self, dtype=None, name=None):
            conversion_contexts.append(ag_ctx.control_status_ctx().status)
            return tf.ones((2, 5), dtype=dtype or tf.float32, name=name)

    requests = []
    original = api._convert_actual

    def counted(entity, program_ctx):
        requests.append((entity.__module__, entity.__name__))
        return original(entity, program_ctx)

    monkeypatch.setattr(api, "_convert_actual", counted)

    @tf.function
    def traced():
        for native in (CustomSpec((None, 5), tf.float32),
                       tf.SparseTensorSpec(CustomShape((None, 5)), tf.float32)):
            spec = as_tensor_spec(native, batched=True)
            assert spec.shape == (5,) and spec.batch is Dynamic
        assert as_tensor_spec(Convertible(), batched=True).batch == 2
        return tf.constant(1)

    assert int(traced()) == 1
    assert conversion_contexts == [ag_ctx.Status.ENABLED]
    assert shape_contexts == [ag_ctx.Status.ENABLED]
    assert requests.count(("dryml.tf.tensor_spec", "_split_batch")) == 2
