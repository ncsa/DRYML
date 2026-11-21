"""Dynamic dense shape contracts across reusable Methods and Dataset graphs."""

from dataclasses import replace
import importlib.util
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import Dynamic, TensorSpec, SpecHint, detect_spec_tree, merge_tensor_specs
from dryml.data import ArgMax, Batch, Cast, Diff, Flatten, GeneratorDataset, Map, Pipe, Project, Scale, Select, Unbatch
from dryml.methods import ImplementationSelectionError, Method, PreparedCallMismatchError, traits
from dryml.methods.signature import spec_from_node
from dryml.models import Model


class DenseIdentity(Method):
    """Record target execution while preserving a declared dense tensor."""

    def __init__(self):
        self.calls = 0

    @traits(backend="numpy")
    def numpy(self, value):
        self.calls += 1
        return value

    def infer_output_spec(self, input_spec):
        return input_spec


class SumFeatures(Model):
    """Use an explicit dynamic output contract for a variable-length model."""

    def __init__(self):
        super().__init__(output_spec=TensorSpec("float32", shape=(Dynamic,), backend="numpy"))

    @traits(backend="numpy")
    def numpy(self, value):
        return value.sum(axis=-1)


class NativeSumFeatures(Model):
    """Model fixture exercising authored native dynamic-output contracts."""

    @traits(backend="torch")
    def torch(self, value):
        return value.sum(dim=-1)

    @traits(backend="tf")
    def tf(self, value):
        import tensorflow as tf
        return tf.reduce_sum(value, axis=-1)

    @traits(backend="jax")
    def jax(self, value):
        return value.sum(axis=-1)


def variable_batches():
    return iter(np.arange(np.prod(shape), dtype="float32").reshape(shape)
                for shape in ((2, 3, 5), (1, 7, 5), (3, 2, 5)))


def variable_examples():
    return iter(np.ones((length, 5), dtype="float32") for length in (3, 3, 7, 7, 2))


@pytest.mark.parametrize("shape,full_shapes", (
    ((10, Dynamic), ((2, 10, 3), (1, 10, 7))),
    ((Dynamic, 5), ((2, 3, 5), (1, 7, 5))),
    ((10, 5), ((2, 10, 5), (1, 10, 5))),
    ((Dynamic, Dynamic), ((2, 3, 5), (1, 7, 9))),
))
def test_known_spec_reuses_dynamic_dimensions_and_batches_without_reselection(shape, full_shapes, monkeypatch):
    method = DenseIdentity()
    spec = TensorSpec("float32", shape=shape, batch=Dynamic, backend="numpy")
    method.learn(spec, output_spec=spec)
    graph = method.method_graph()
    nodes = graph.method_nodes

    def forbidden(*args, **kwargs):
        pytest.fail("a changed dynamic size triggered Method reselection")

    monkeypatch.setattr(method, "find_implementation", forbidden)
    monkeypatch.setattr("dryml.core.signatures.compile_signature", forbidden)
    for full_shape in full_shapes:
        value = np.ones(full_shape, dtype="float32")
        assert method(value) is value
    assert method.calls == len(full_shapes)
    assert graph.method_nodes[0].selected is nodes[0].selected
    assert spec_from_node(nodes[0].inputs[0].spec).equal_exact(spec)
    assert spec_from_node(nodes[0].outputs[0].spec).equal_exact(spec)


def test_dynamic_contract_rejects_fixed_axis_rank_dtype_and_layout_drift_before_body():
    method = DenseIdentity()
    spec = TensorSpec("float32", shape=(Dynamic, 5), batch=Dynamic, backend="numpy")
    method.learn(spec, output_spec=spec)
    for value in (np.ones((2, 3, 4), dtype="float32"), np.ones((2, 3, 5, 1), dtype="float32"),
                  np.ones((2, 3, 5), dtype="float64"), replace(spec, layout="ragged", ragged_rank=1)):
        with pytest.raises(ImplementationSelectionError):
            method(value)
    assert method.calls == 0


def test_first_call_learning_stays_exact_without_an_explicit_dynamic_contract():
    method = DenseIdentity()
    method.default_batched = True
    method.learn()
    method(np.ones((2, 3, 5), dtype="float32"))
    for shape in ((1, 3, 5), (2, 7, 5)):
        with pytest.raises(PreparedCallMismatchError):
            method(np.ones(shape, dtype="float32"))
    assert method.calls == 1


def test_dynamic_output_constraints_still_reject_fixed_axis_drift():
    class InvalidOutput(DenseIdentity):
        @traits(backend="numpy")
        def numpy(self, value):
            self.calls += 1
            return value[..., :4]

    method = InvalidOutput()
    spec = TensorSpec("float32", shape=(Dynamic, 5), batch=Dynamic, backend="numpy")
    method.learn(spec, output_spec=spec)
    with pytest.raises(ImplementationSelectionError):
        method(np.ones((2, 3, 5), dtype="float32"))
    assert method.calls == 1


def test_equal_dynamic_declarations_do_not_prove_runtime_lengths_equal():
    spec = TensorSpec("float32", shape=(Dynamic, 5), batch=Dynamic, backend="numpy")
    method = Diff()
    method.learn({"x": spec, "y": spec}, output_spec=spec.with_dtype("float64"))
    with pytest.raises(ValueError, match="equal backend and shape"):
        method({"x": np.ones((2, 3, 5), dtype="float32"),
                "y": np.ones((2, 4, 5), dtype="float32")})


def test_batched_sample_inference_generalizes_only_observed_varying_dimensions():
    hint = SpecHint(batch_mode="batched", samples=3)
    spec = detect_spec_tree(variable_batches(), hint)
    assert spec.equal_exact(TensorSpec("float32", shape=(Dynamic, 5), batch=Dynamic, backend="numpy"))
    source = GeneratorDataset(variable_batches, cardinality=3, example_count=6, spec=hint)
    assert source.spec.equal_exact(spec)


def test_merging_variable_batches_does_not_turn_unbatched_values_into_batches():
    element = TensorSpec("float32", shape=(Dynamic, 5), batch_axis_name=None)
    with pytest.raises(ValueError, match="batch"):
        merge_tensor_specs((element, element.with_batch(2, axis_name=None)))


def test_dynamic_composition_retains_graph_specs_and_executes_multiple_lengths():
    spec = TensorSpec("float32", shape=(Dynamic, 5), batch=Dynamic, backend="numpy")
    method = Pipe(Cast("float64"), Scale(dtype="float32"),
                  Project({"flat": Flatten(), "indices": ArgMax()}), Select("indices"))
    expected = TensorSpec("int64", shape=(Dynamic,), batch=Dynamic, backend="numpy")
    assert method.infer_output_spec(spec).equal_exact(expected)
    method.learn(spec, output_spec=expected)
    for value in variable_batches():
        np.testing.assert_array_equal(method(value), value.argmax(axis=-1))
    ports = [spec_from_node(port.spec) for node in method.method_graph().method_nodes for port in node.outputs]
    assert any(isinstance(port, dict) and port["flat"].shape == (Dynamic,) for port in ports)

    source = GeneratorDataset(variable_batches, cardinality=3, example_count=6, spec=spec)
    mapped = Map(source, method, preserves_examples=True)
    graph = mapped.method_graph()
    graph.learn()
    for opener in (mapped.iterator, graph.iterator):
        with opener() as cursor:
            for result, value in zip(cursor, variable_batches(), strict=True):
                np.testing.assert_array_equal(result, value.argmax(axis=-1))
    assert mapped.spec.equal_exact(expected)


def test_dynamic_dataset_model_and_batch_unbatch_paths_keep_declared_contracts():
    spec = TensorSpec("float32", shape=(Dynamic, 5), batch=Dynamic, backend="numpy")
    model = SumFeatures()
    source = GeneratorDataset(variable_batches, cardinality=3, example_count=6, spec=spec)
    mapped = Map(source, model, preserves_examples=True)
    assert mapped.spec.equal_exact(TensorSpec("float32", shape=(Dynamic,), batch=Dynamic, backend="numpy"))
    assert mapped.example_cardinality() == Cardinality.finite(6)
    graph = mapped.method_graph()
    graph.learn()
    for opener in (mapped.iterator, graph.iterator):
        with opener() as cursor:
            results = list(cursor)
        for result, value in zip(results, variable_batches(), strict=True):
            np.testing.assert_array_equal(result, value.sum(axis=-1))
    assert model.call_mode == "eager"

    elements = GeneratorDataset(variable_examples, cardinality=5, spec=spec.without_batch())
    batched = Batch(elements, 2)
    assert batched.spec.equal_exact(spec)
    roundtrip = Unbatch(batched)
    prepared = roundtrip.method_graph()
    prepared.learn()
    for opener in (roundtrip.iterator, prepared.iterator):
        with opener() as cursor:
            assert [value.shape for value in cursor] == [(3, 5), (3, 5), (7, 5), (7, 5), (2, 5)]


def test_dynamic_dimensions_do_not_make_unequal_samples_stackable():
    source = GeneratorDataset(variable_examples, cardinality=5,
                              spec=TensorSpec("float32", shape=(Dynamic, 5), backend="numpy"))
    batches = Batch(source, 3)
    graph = batches.method_graph()
    graph.learn()
    for opener in (batches.iterator, graph.iterator):
        with opener() as cursor, pytest.raises(ValueError, match="shape"):
            next(cursor)


@pytest.mark.parametrize("shape,expected", (
    ((10, Dynamic), (Dynamic,)), ((Dynamic, 0), (0,)), ((0, Dynamic), (0,)),
))
def test_flatten_propagates_dynamic_size_but_zero_dominates(shape, expected):
    method = Flatten()
    spec = TensorSpec("float32", shape=shape, batch=Dynamic, backend="numpy")
    assert method.infer_output_spec(spec).shape == expected


def test_shape_changing_transforms_update_named_dynamic_axes():
    spec = TensorSpec("float32", shape=(Dynamic, 5), batch=Dynamic, backend="numpy",
                      axis_names=("steps", "features"))
    flat = Flatten().infer_output_spec(spec)
    reduced = ArgMax(axis=-1).infer_output_spec(spec)
    assert flat.shape == (Dynamic,)
    assert flat.axis_names == (None,)
    assert reduced.shape == (Dynamic,)
    assert reduced.axis_names == ("steps",)


def test_named_dynamic_contracts_interpret_native_axes_without_weakening_spec_checks():
    spec = TensorSpec("float32", shape=(Dynamic, 5), batch=Dynamic, backend="numpy",
                      axis_names=("steps", "features"), batch_axis_name="examples")
    method = DenseIdentity()
    method.learn(spec, output_spec=spec)
    for value in variable_batches():
        assert method(value) is value
    with pytest.raises(ImplementationSelectionError):
        method(replace(spec, axis_names=("other", "features")))

    flatten = Flatten()
    flatten.learn(spec, output_spec=flatten.infer_output_spec(spec))
    argmax = ArgMax()
    argmax.learn(spec, output_spec=argmax.infer_output_spec(spec))
    for value in variable_batches():
        np.testing.assert_array_equal(flatten(value), value.reshape(len(value), -1))
        np.testing.assert_array_equal(argmax(value), value.argmax(axis=-1))


def test_flatten_accepts_zero_sample_widths_and_empty_batches_without_reshape_ambiguity():
    method = Flatten()
    spec = TensorSpec("float32", shape=(Dynamic, 5), batch=Dynamic, backend="numpy")
    method.learn(spec, output_spec=method.infer_output_spec(spec))
    for shape in ((2, 0, 5), (0, 3, 5), (0, 0, 5)):
        assert method(np.zeros(shape, dtype="float32")).shape == (shape[0], shape[1] * shape[2])


@pytest.mark.parametrize("backend,module", (("torch", "torch"), ("tf", "tensorflow"), ("jax", "jax")))
def test_native_dynamic_dense_contracts_in_fresh_process(backend, module):
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"optional {module} backend is not installed")
    program = textwrap.dedent("""\
        import sys
        import numpy as np
        from dryml.core.tensor_spec import Dynamic, TensorSpec
        from dryml.data import ArgMax, Cast, Flatten, Pipe, Select
        from dryml.methods import ImplementationSelectionError
        from tests.methods.test_dynamic_dense import NativeSumFeatures, variable_batches

        backend = sys.argv[1]
        if backend == "torch":
            import torch
            make = torch.from_numpy
            host = lambda value: value.numpy()
        elif backend == "tf":
            import tensorflow as tf
            make = tf.convert_to_tensor
            host = lambda value: value.numpy()
        else:
            import jax.numpy as jnp
            make = jnp.asarray
            host = np.asarray

        spec = TensorSpec("float32", shape=(Dynamic, 5), batch=Dynamic, backend=backend,
                          axis_names=("steps", "features"), batch_axis_name="examples")
        identity = Select()
        identity.learn(spec, output_spec=spec)
        model = NativeSumFeatures(output_spec=TensorSpec("float32", shape=(Dynamic,), backend=backend))
        model.learn(spec, output_spec=model.infer_output_spec(spec))
        for array in variable_batches():
            value = make(array)
            assert identity(value) is value
            np.testing.assert_allclose(host(model(value)), array.sum(axis=-1))
        try:
            identity(make(np.ones((2, 3, 4), dtype="float32")))
        except ImplementationSelectionError:
            pass
        else:
            raise AssertionError("a fixed feature dimension was not validated")

        if backend != "jax":
            method = Pipe(Cast("float32"), Flatten(), ArgMax())
            method.learn(spec, output_spec=method.infer_output_spec(spec))
            for array in variable_batches():
                np.testing.assert_array_equal(host(method(make(array))), array.reshape(len(array), -1).argmax(axis=-1))
            flatten = Flatten()
            flatten.learn(spec, output_spec=flatten.infer_output_spec(spec))
            for shape in ((2, 0, 5), (0, 3, 5), (0, 0, 5)):
                assert tuple(flatten(make(np.zeros(shape, dtype="float32"))).shape) == (shape[0], shape[1] * shape[2])
            if backend == "tf":
                compiled = tf.function(flatten, input_signature=[tf.TensorSpec((None, None, 5), tf.float32)])
                for array in variable_batches():
                    np.testing.assert_array_equal(host(compiled(make(array))), array.reshape(len(array), -1))
                assert compiled.experimental_get_tracing_count() == 1
        """)
    result = subprocess.run([sys.executable, "-c", program, backend], capture_output=True,
                            text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
