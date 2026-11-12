"""Opaque object declarations remain content-agnostic through Dataset and Method boundaries."""

import gc
import subprocess
import sys
import weakref

import numpy as np
import pytest

from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import Dynamic, TensorSpec, fake_from_spec_tree
from dryml.data import Batch, GeneratorDataset, Map, Unbatch
from dryml.methods import ImplementationSelectionError, Method, traits


OPAQUE = TensorSpec("object", layout="python")


class OpaqueValue:
    """Expose only the field deliberately consumed by the user conversion Method."""

    def __init__(self, number=1):
        self.number = number

    @property
    def shape(self):
        raise AssertionError("framework inspected object shape")

    @property
    def dtype(self):
        raise AssertionError("framework inspected object dtype")

    def __array__(self, *args, **kwargs):
        raise AssertionError("framework coerced opaque content to an array")

    def __iter__(self):
        raise AssertionError("framework iterated opaque content")


class Identity(Method):
    """Return the same opaque reference, recording only admitted target calls."""

    calls = 0

    def __call__(self, value):
        self.calls += 1
        return value

    def infer_output_spec(self, input_spec):
        return input_spec


class ToVector(Method):
    """User-owned conversion from arbitrary object content to a fixed NumPy tensor."""

    calls = 0

    def __call__(self, value):
        self.calls += 1
        return np.asarray([value.number, value.number + 1], dtype="float32")

    def infer_output_spec(self, input_spec):
        return TensorSpec("float32", shape=(2,), backend="numpy")


class BatchToVector(Method):
    """Consume an opaque list/tuple batch while retaining its batch declaration."""

    def __call__(self, batch):
        return np.asarray([[value.number, value.number + 1] for value in batch], dtype="float32")

    def infer_output_spec(self, input_spec):
        return TensorSpec("float32", shape=(2,), batch=input_spec.batch, backend="numpy")


class NumpyTreeIdentity(Method):
    """Select a tensor backend using only the nonopaque fields of a mixed tree."""

    @traits(backend="numpy")
    def numpy(self, value):
        return value

    def infer_output_spec(self, input_spec):
        return input_spec


def values():
    """Create values inside iteration so source metadata captures no mesh objects."""
    for number in range(3):
        yield OpaqueValue(number)


def batches():
    """Yield variable-sized batches with heterogeneous opaque contents."""
    yield [OpaqueValue(), {"nested": [1, 2]}]
    yield (None,)


def mixed_values():
    """Keep opaque and tensor leaves in one declared mapping structure."""
    for item in values():
        yield {"object": item, "tensor": np.ones((2,), dtype="float32")}


def object_containers():
    """Yield containers that must remain opaque instead of becoming tree nodes."""
    yield {"opaque": [object()]}
    yield [object(), object()]


def mixed_backend_values():
    """Require one discovery call across multiple numerical leaves and one opaque leaf."""
    for item in values():
        yield {"object": item, "tensor": (np.ones((2,), dtype="float32"), np.zeros((2,), dtype="float32"))}


def test_opaque_spec_and_numpy_object_dtype_are_distinct():
    assert OPAQUE.is_python_object
    assert OPAQUE.dtype.np() is np.object_
    assert not TensorSpec("object", shape=(2,), backend="numpy").is_python_object
    assert OPAQUE.with_batch(Dynamic).without_batch() == OPAQUE
    with pytest.raises(ValueError, match="synthesize"):
        fake_from_spec_tree(OPAQUE)
    for kwargs in ({"shape": (3,)}, {"backend": "numpy"}, {"ragged_rank": 1}):
        with pytest.raises(ValueError, match="Opaque"):
            TensorSpec("object", layout="python", **kwargs)


def test_dense_numpy_object_arrays_keep_their_tensor_backend_and_axes():
    value = OpaqueValue()
    array = np.empty(2, dtype=object)
    array[0], array[1] = value, None
    spec = TensorSpec("object", shape=(), batch=Dynamic, backend="numpy")
    assert Identity().find_implementation(spec, output_spec=spec)(array) is array
    assert array[0] is value and array[1] is None


@pytest.mark.parametrize("prepared", (False, True))
def test_declared_object_method_accepts_any_class_and_container_without_inspection(prepared):
    method = Identity()
    if prepared:
        method.learn(OPAQUE, output_spec=OPAQUE)
        invoke = method
    else:
        invoke = method.find_implementation(OPAQUE, output_spec=OPAQUE)
    for value in (OpaqueValue(), None, "text", 7, {"values": [OpaqueValue()]},
                  [OpaqueValue(), None], np.ones((4, 2))):
        assert invoke(value) is value
    value = OpaqueValue()
    reference = weakref.ref(value)
    assert invoke(value) is value
    del value
    gc.collect()
    assert reference() is None


def test_object_batch_validation_uses_declared_outer_boundary_and_preserves_cache():
    method = Identity()
    method.learn(OPAQUE.with_batch(Dynamic), output_spec=OPAQUE.with_batch(Dynamic))
    for value in ([OpaqueValue(), None], ("different class",)):
        assert method(value) is value
    assert method.call_mode == "cached"
    before = method.calls
    with pytest.raises(ImplementationSelectionError):
        method(OpaqueValue())
    assert method.calls == before
    fixed = method.find_implementation(OPAQUE.with_batch(2))
    with pytest.raises(ImplementationSelectionError):
        fixed([OpaqueValue()])
    assert method.calls == before
    assert fixed([None, OpaqueValue()])[0] is None


def test_object_spec_does_not_select_or_complete_a_tensor_backend():
    class Alternatives(Method):
        @traits()
        def generic(self, value):
            return value

        @traits(backend="numpy")
        def numeric(self, value):
            raise AssertionError("opaque input selected a tensor implementation")

    method = Alternatives()
    assert [candidate.name for candidate in method.compatible_implementations(OPAQUE)] == ["generic"]
    selected = method.find_implementation(OPAQUE)
    assert selected.consumer_input_specs[0].value.backend is None
    value = OpaqueValue()
    assert selected(value) is value
    with pytest.raises(ImplementationSelectionError, match="conflict"):
        method.find_implementation(OPAQUE, backend="numpy")
    class OnlyNumpy(Method):
        @traits(backend="numpy")
        def numpy(self, value):
            raise AssertionError("opaque input reached a tensor-only implementation")
    with pytest.raises(ImplementationSelectionError) as failure:
        OnlyNumpy().find_implementation(OPAQUE)
    assert failure.value.reason == "no_candidate"


def test_object_batch_result_constraint_rejects_wrong_length_before_raw_callback():
    class DropsOne(Method):
        def __call__(self, batch):
            return batch[:-1]

    spec = OPAQUE.with_batch(2)
    selected = DropsOne().find_implementation(spec, output_spec=spec)
    observed = []
    with pytest.raises(ImplementationSelectionError):
        selected.invoke_with_raw_result(([None, OpaqueValue()],), {}, on_raw_result=observed.append)
    assert observed == []


@pytest.mark.parametrize("graph", (False, True))
def test_generator_objects_map_to_fixed_tensors_without_consumption_during_planning(graph):
    source = GeneratorDataset(values, cardinality=3, spec=OPAQUE)
    method = ToVector()
    result = Map(source, method, preserves_examples=True)
    assert method.calls == 0
    assert source.example_cardinality() == Cardinality.finite(3)
    assert result.spec.equal_exact(TensorSpec("float32", shape=(2,), backend="numpy"))
    if graph:
        plan = result.method_graph()
        plan.learn()
        cursor = plan.iterator()
    else:
        cursor = result.iterator()
    assert method.calls == 0
    with cursor:
        output = list(cursor)
    np.testing.assert_array_equal(np.stack(output), [[0, 1], [1, 2], [2, 3]])
    assert method.calls == 3


@pytest.mark.parametrize("graph", (False, True))
def test_batch_unbatch_preserves_opaque_references_and_short_batches(graph):
    source = GeneratorDataset(batches, cardinality=2, spec=OPAQUE.with_batch(Dynamic))
    assert source.example_cardinality().is_unknown
    assert source.examples_in([OpaqueValue(), None]) == 2
    with pytest.raises(ValueError, match="zero"):
        source.examples_in([])
    with pytest.raises(TypeError, match="lists or tuples"):
        source.examples_in(OpaqueValue())
    result = Unbatch(Batch(Unbatch(source), 2))
    if graph:
        plan = result.method_graph()
        plan.learn()
        cursor = plan.iterator()
    else:
        cursor = result.iterator()
    with cursor:
        output = list(cursor)
    assert len(output) == 3
    assert isinstance(output[0], OpaqueValue)
    assert output[1] == {"nested": [1, 2]}
    assert output[2] is None


def test_fixed_object_batch_shortfall_fails_before_unbatch_yields():
    from dryml.data.split import split_for_spec

    split = split_for_spec(OPAQUE.with_batch(2))
    with pytest.raises(ValueError, match="length"):
        split([OpaqueValue()])


def test_prepared_object_batches_convert_to_fixed_sample_tensors_with_short_final_batch():
    source = Batch(GeneratorDataset(values, cardinality=3, spec=OPAQUE), 2)
    result = Map(source, BatchToVector(), preserves_examples=True)
    assert result.spec.equal_exact(TensorSpec("float32", shape=(2,), batch=Dynamic, backend="numpy"))
    plan = result.method_graph()
    plan.learn()
    with plan.iterator() as cursor:
        output = list(cursor)
    assert [item.shape for item in output] == [(2, 2), (1, 2)]
    np.testing.assert_array_equal(np.concatenate(output), [[0, 1], [1, 2], [2, 3]])


@pytest.mark.parametrize("graph", (False, True))
def test_mixed_backend_discovery_ignores_declared_opaque_values(monkeypatch, graph):
    from dryml.core import backend

    original, inspected = backend.discover_backends, []
    def discover(*values, **kwargs):
        assert all(not isinstance(value, OpaqueValue) for value in values)
        inspected.append(values)
        return original(*values, **kwargs)
    source = GeneratorDataset(mixed_backend_values, cardinality=3, spec={
        "object": OPAQUE, "tensor": (TensorSpec("float32", shape=(2,)), TensorSpec("float32", shape=(2,)))})
    result = Map(source, NumpyTreeIdentity())
    monkeypatch.setattr(backend, "discover_backends", discover)
    if graph:
        plan = result.method_graph()
        plan.learn()
        cursor = plan.iterator()
    else:
        cursor = result.iterator()
    with cursor:
        output = list(cursor)
    assert len(output) == 3
    assert len(inspected) == 1 and len(inspected[0]) == 2
    assert all(isinstance(value, np.ndarray) for value in inspected[0])


def test_mixed_tensor_object_structure_batches_and_validates_by_spec():
    source = GeneratorDataset(mixed_values, cardinality=3, spec={
        "object": OPAQUE, "tensor": TensorSpec("float32", shape=(2,), backend="numpy")})
    batch = next(iter(Batch(source, 2)))
    assert isinstance(batch["object"], list)
    assert batch["tensor"].shape == (2, 2)
    selected = Identity().find_implementation(Batch(source, 2).spec)
    assert selected(batch) is batch
    assert dict(selected.consumer_input_specs[0].value)["object"].value.backend is None
    output = list(Unbatch(Batch(source, 2)))
    assert len(output) == 3
    assert all(isinstance(item["object"], OpaqueValue) for item in output)


def test_object_batching_does_not_import_native_tensor_frameworks():
    code = """
import sys
from dryml.core.tensor_spec import TensorSpec
from dryml.data import Batch, GeneratorDataset, Unbatch
from tests.data.test_object_specs import object_containers
dataset = GeneratorDataset(object_containers, cardinality=2, spec=TensorSpec('object', layout='python'))
items = list(Unbatch(Batch(dataset, 2)))
assert len(items) == 2 and isinstance(items[0], dict) and isinstance(items[1], list)
assert not {'tensorflow', 'torch', 'jax', 'ray'}.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=15)
