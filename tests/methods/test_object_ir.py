"""Opaque values have the same logical input, output and intermediate IR support."""

import gc
import weakref

import numpy as np
import pytest

from dryml.core.tensor_spec import Dynamic, SpecHint, TensorSpec, as_tensor_spec, detect_spec_tree
from dryml.data import Batch, GeneratorDataset, Map, Pipe, Project, Select
from dryml.methods import Accumulator, AccumulatorGroup, Method, PreparedCallMismatchError, traits
from dryml.methods.errors import MethodError
from dryml.methods.signature import runtime_node
from tests.data.test_object_specs import Identity, OpaqueValue, ToVector, values


OBJECT = TensorSpec("object", layout="python")
VECTOR = TensorSpec("float32", shape=(2,), backend="numpy")


class ShiftMesh(Method):
    """Object-to-object work deliberately inspects content only in the user body."""

    calls = 0

    def __call__(self, mesh):
        self.calls += 1
        return OpaqueValue(mesh.number + 10)

    def infer_output_spec(self, input_spec):
        return input_spec


class MeshFromVector(Method):
    """A native tensor input may produce a backend-free opaque object."""

    @traits(backend="numpy")
    def numpy(self, value):
        return OpaqueValue(int(value[0]))

    def infer_output_spec(self, input_spec):
        return OBJECT


class KeepMesh(Accumulator):
    """Carry an opaque result without imposing a tensor backend on either port."""

    def __call__(self, observation, state):
        return observation

    def infer_output_spec(self, observation_spec, state_spec):
        return observation_spec


def tensor_values():
    for number in range(3):
        yield np.asarray([number, number + 1], dtype="float32")


def mesh_batches():
    yield [OpaqueValue(0), OpaqueValue(1)]
    yield [OpaqueValue(2)]


def test_shared_spec_inference_reports_opaque_inputs_and_outputs_without_class_facts():
    value = OpaqueValue()
    assert as_tensor_spec(value).equal_exact(OBJECT)
    result = MeshFromVector()(np.ones((2,), dtype="float32"))
    assert as_tensor_spec(result).equal_exact(OBJECT)
    inferred = as_tensor_spec({"mesh": value, "coordinates": np.ones((2,), dtype="float32")})
    assert inferred["mesh"].equal_exact(OBJECT)
    assert inferred["coordinates"].equal_exact(VECTOR)


def test_native_spec_failures_are_not_reclassified_as_opaque(monkeypatch):
    from dryml.core.backend import Backend, backend_existence_testers, backend_testers

    native = type("Native", (), {"__module__": "torch"})()
    monkeypatch.delitem(backend_testers, Backend.torch, raising=False)
    with pytest.raises(TypeError, match="registered backend adapter"):
        as_tensor_spec(native)
    with pytest.raises(TypeError, match="registered native tensor spec adapter"):
        runtime_node(native)

    def broken_detector(value):
        raise ValueError("detector failed")

    monkeypatch.setitem(backend_testers, Backend.torch, broken_detector)
    monkeypatch.setitem(backend_existence_testers, Backend.torch, lambda: True)
    with pytest.raises(ValueError, match="detector failed"):
        as_tensor_spec(OpaqueValue())
    with pytest.raises(MethodError, match="backend detector failed"):
        runtime_node(OpaqueValue())


def test_native_unsupported_dtype_is_not_reclassified_as_opaque():
    value = np.zeros(2, dtype=[("x", "float32")])
    with pytest.raises((TypeError, ValueError)):
        as_tensor_spec(value)
    with pytest.raises((TypeError, ValueError)):
        runtime_node(value)


def test_spec_hint_infers_variable_object_batches():
    inferred = detect_spec_tree(mesh_batches(), SpecHint(batch_mode="batched", samples=2))
    assert inferred.equal_exact(OBJECT.with_batch(Dynamic))


def test_first_call_learning_and_cached_calls_do_not_retain_objects_or_classes():
    method = Identity()
    method.learn()
    first = OpaqueValue()
    pointer = weakref.ref(first)
    assert method(first) is first
    assert method.call_mode == "cached"
    assert method.cached_signature.args[0].value.equal_exact(OBJECT)
    assert method(object()) is not None
    del first
    gc.collect()
    assert pointer() is None
    for value in (np.ones((2,), dtype="float32"), {"nested": [object()]}, None):
        assert method(value) is value


def test_learning_object_batch_uses_retained_domain_not_element_classes():
    method = Identity()
    method.default_batched = True
    method.learn()
    first = [OpaqueValue(), OpaqueValue(2)]
    assert method(first) is first
    assert method.cached_signature.args[0].value.equal_exact(OBJECT.with_batch(2))
    changed = (None, np.ones((4,), dtype="float32"))
    assert method(changed) is changed
    with pytest.raises(PreparedCallMismatchError):
        method([OpaqueValue()])
    assert method(first) is first


def test_learning_mixed_tree_preserves_object_domain_and_exact_tensor_shape():
    method = Identity()
    method.learn()
    first = {"mesh": OpaqueValue(), "numbers": np.ones((2,), dtype="float32")}
    assert method(first) is first
    changed = {"mesh": {"opaque": [object()]}, "numbers": np.ones((2,), dtype="float32")}
    assert method(changed) is changed
    with pytest.raises(PreparedCallMismatchError):
        method({"mesh": OpaqueValue(), "numbers": np.ones((3,), dtype="float32")})


def test_learning_nested_object_batch_keeps_opaque_items_and_exact_numerical_axes():
    method = Identity()
    method.default_batched = True
    method.learn()
    first = {"meshes": [OpaqueValue(), OpaqueValue(2)],
             "numbers": np.ones((2, 3), dtype="float32")}
    assert method(first) is first
    changed = {"meshes": (None, {"content": [object()]}),
               "numbers": np.zeros((2, 3), dtype="float32")}
    assert method(changed) is changed
    with pytest.raises(PreparedCallMismatchError):
        method({"meshes": [OpaqueValue()], "numbers": first["numbers"]})
    with pytest.raises(PreparedCallMismatchError):
        method({"meshes": first["meshes"], "numbers": np.ones((2, 4), dtype="float32")})


def test_object_observation_learning_does_not_select_from_numeric_carry():
    method = KeepMesh()
    method.learn()
    mesh = OpaqueValue()
    assert method(mesh, np.ones((2,), dtype="float32")) is mesh
    assert method(mesh, np.zeros((2,), dtype="float32")) is mesh


@pytest.mark.parametrize("graph", (False, True))
def test_map_inferred_object_source_transforms_objects_then_returns_tensors(graph):
    source = GeneratorDataset(values, cardinality=3, spec=SpecHint(samples=2))
    shift = ShiftMesh()
    dataset = Map(source, shift, ToVector())
    assert source.spec.equal_exact(OBJECT)
    assert shift.calls == 0
    if graph:
        plan = dataset.method_graph()
        plan.learn()
        assert any(port.spec.value.is_python_object
                   for node in plan.nodes for port in node.outputs
                   if port.spec is not None and port.spec.kind == "tensor")
        cursor = plan.iterator()
    else:
        cursor = dataset.iterator()
    with cursor:
        result = list(cursor)
    np.testing.assert_array_equal(np.stack(result), [[10, 11], [11, 12], [12, 13]])


@pytest.mark.parametrize("prepared", (False, True))
def test_pipe_and_project_preserve_tensor_object_tensor_edges(prepared):
    method = Pipe(MeshFromVector(), Project(mesh=Identity(), vector=ToVector()))
    if prepared:
        graph = method.method_graph()
        graph.learn(VECTOR)
        assert any(node.outputs and node.outputs[0].spec.kind == "tensor"
                   and node.outputs[0].spec.value.is_python_object for node in graph.method_nodes)
    result = method(np.asarray([3, 4], dtype="float32"))
    assert isinstance(result["mesh"], OpaqueValue)
    np.testing.assert_array_equal(result["vector"], [3, 4])


def test_first_call_prepared_pipe_accepts_mesh_and_keeps_opaque_intermediate():
    method = Pipe(ShiftMesh(), ToVector())
    method.learn()
    np.testing.assert_array_equal(method(OpaqueValue(2)), [12, 13])
    np.testing.assert_array_equal(method(OpaqueValue(4)), [14, 15])


def test_selected_object_outputs_and_accumulator_carry_share_the_same_spec_contract():
    method = AccumulatorGroup({"mesh": KeepMesh()})
    carry = {"mesh": OBJECT}
    method.learn(OBJECT, carry, output_spec=carry)
    mesh = OpaqueValue(4)
    assert method(mesh, {"mesh": None})["mesh"] is mesh
    selected = MeshFromVector().find_implementation(VECTOR, output_spec=OBJECT)
    assert isinstance(selected(np.ones((2,), dtype="float32")), OpaqueValue)


@pytest.mark.parametrize("graph", (False, True))
def test_map_tensor_source_can_return_mesh_then_select_it_from_a_mixed_output(graph):
    source = GeneratorDataset(tensor_values, cardinality=3, spec=VECTOR)
    dataset = Map(source, MeshFromVector(), Project(mesh=Identity(), vector=ToVector()),
                  Select("mesh"), ShiftMesh())
    if graph:
        plan = dataset.method_graph()
        plan.learn()
        cursor = plan.iterator()
    else:
        cursor = dataset.iterator()
    with cursor:
        result = list(cursor)
    assert [mesh.number for mesh in result] == [10, 11, 12]


def test_batched_mesh_transform_composes_with_tensor_batching():
    source = GeneratorDataset(values, cardinality=3, spec=OBJECT)
    dataset = Batch(Map(source, ShiftMesh(), ToVector()), 2)
    plan = dataset.method_graph()
    plan.learn()
    with plan.iterator() as cursor:
        batches = list(cursor)
    assert [value.shape for value in batches] == [(2, 2), (1, 2)]
