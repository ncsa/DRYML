"""Focused contracts for the inspectable local Method preparation IR."""

import numpy as np
import pytest

from dryml.core.tensor_spec import TensorSpec
from dryml.data import Cast, Pipe, Project
from dryml.methods import Method, MethodGraph, traits
from dryml.models import AutoEncoder, Model


class Identity(Method):
    """A NumPy identity with pure output facts for graph tests."""

    @traits(backend="numpy")
    def numpy(self, value):
        """Return the supplied value after its retained contract is checked."""

        return value

    def infer_output_spec(self, input_spec):
        """Preserve the input specification without executing the target."""

        return input_spec


class IdentityModel(Model):
    """A direct model fixture with pure identity output inference."""

    def __call__(self, value):
        """Return the supplied model value."""

        return value

    def infer_output_spec(self, input_spec):
        """Preserve the input specification."""

        return input_spec


def test_graph_preparation_retains_occurrence_local_invokers_without_body_calls():
    """One shared Method receives independent retained specs at two graph occurrences."""

    shared = Identity()
    pipe = Pipe(shared, Cast("float64"), shared)
    graph = MethodGraph(pipe)
    source = TensorSpec("float32", shape=(2,), backend="numpy")

    graph.learn(source)

    shared_nodes = [node for node in graph.method_nodes if node.method_type is Identity]
    assert len(shared_nodes) == 2
    assert all(node.selected is not None for node in shared_nodes)
    assert shared_nodes[0].inputs[0].spec.value.dtype.name == "float32"
    assert shared_nodes[1].inputs[0].spec.value.dtype.name == "float64"
    assert shared_nodes[0].outputs[0].spec.value.dtype.name == "float32"
    assert shared_nodes[1].outputs[0].spec.value.dtype.name == "float64"
    assert all(
        node.selected.adapter.receiver_type is node.method_type
        for node in graph.method_nodes
    )
    assert graph.input_specs == (source,)
    assert graph.method_nodes[0].method_type is Pipe
    assert graph.method_nodes[0].outputs[0].spec.value.dtype.name == "float64"
    assert graph.conversion_edges == ()
    with pytest.raises(NotImplementedError):
        graph.iterator()

    graph.eager()
    assert all(node.selected is None for node in graph.method_nodes)


def test_composite_known_spec_output_facts_remain_truthful_without_invocation():
    """Pipe, Project, and AutoEncoder expose their pure output specs during preparation."""

    source = TensorSpec("int32", shape=(2,), backend="numpy")
    expected = TensorSpec("float32", shape=(2,), backend="numpy")
    composites = (
        Pipe(Cast("float32")),
        Project(value=Cast("float32")),
        AutoEncoder(IdentityModel(), IdentityModel()),
    )

    for composite in composites:
        graph = composite.method_graph()
        graph.learn(source)
        output = graph.method_nodes[0].outputs[0].spec
        assert graph.method_nodes[0].method_type is type(composite)
        if isinstance(composite, Project):
            assert output.value[0][1].value.dtype.name == "float32"
        else:
            assert output.value.dtype.name == ("int32" if isinstance(composite, AutoEncoder) else "float32")
    assert expected.dtype.name == "float32"


def test_nested_selection_is_metadata_only_and_graph_shares_final_invokers(monkeypatch):
    """Specialization is passive; executable occurrences compile exactly once."""
    from dryml.core import signatures
    compiled = []
    original = signatures.compile_signature
    monkeypatch.setattr(signatures, "compile_signature",
                        lambda target, **kw: (compiled.append(target), original(target, **kw))[1])
    shared = Identity()
    composite = Project(left=Pipe(shared, Cast("float64"), shared), right=Pipe(shared))
    source = TensorSpec("float32", shape=(2,), backend="numpy")
    compiled.clear()
    composite.find_implementation(source)
    composite.infer_output_spec(source)
    assert compiled == []
    graph = composite.method_graph()
    graph.learn(source)
    assert len(compiled) == len(graph.method_nodes) == 7
    retained = graph.method_nodes[0].selected
    assert retained is not None
    assert retained.children[0] is graph.method_nodes[1].selected
    assert retained.children[0].children[0] is graph.method_nodes[2].selected
    assert retained.children[1] is graph.method_nodes[5].selected
    with monkeypatch.context() as patch:
        patch.setattr(Identity, "implementations", lambda *_: pytest.fail("reselection"))
        result = composite(np.ones(2, dtype=np.float32))
    assert result["left"].dtype == np.float64
    assert result["right"].dtype == np.float32
    assert len(compiled) == 7


def test_converted_and_autoencoder_selection_do_not_compile_or_import(monkeypatch):
    """Converted/nested metadata stays passive; final target compiles once."""
    import sys
    from dryml.core import signatures
    source = TensorSpec("float32", shape=(2,), backend="torch")
    identity = Identity()
    autoencoder = AutoEncoder(IdentityModel(), IdentityModel())
    modules = {name for name in ("torch", "tensorflow", "jax") if name in sys.modules}
    compiled = []
    original = signatures.compile_signature
    monkeypatch.setattr(signatures, "compile_signature",
                        lambda target, **kw: (compiled.append(target), original(target, **kw))[1])
    converted = identity.find_implementation(source)
    assert converted.conversion_edge is not None
    selected = autoencoder.find_implementation(source)
    autoencoder.infer_output_spec(source)
    assert compiled == []
    converted_invoker = converted.prepared_invoker()
    assert len(compiled) == 1
    prepared = selected.prepared_invoker()
    assert len(compiled) == 4
    assert len(prepared.children) == 2
    assert {name for name in ("torch", "tensorflow", "jax") if name in sys.modules} == modules

    from contextvars import copy_context
    from dryml.core import Repo
    from dryml.core.signatures import SignatureError, signature_context
    import dryml.methods.conversion as conversion
    handoffs = []

    def handoff(edge, value):
        handoffs.append(edge)
        return np.ones(2, dtype=np.float32)

    monkeypatch.setattr(conversion, "convert", handoff)
    with signature_context(repo=Repo()):
        np.testing.assert_equal(converted_invoker(source), np.ones(2, dtype=np.float32))
        expired = copy_context()
    assert handoffs == [converted.conversion_edge]
    with pytest.raises(SignatureError):
        expired.run(converted_invoker, source)
    assert handoffs == [converted.conversion_edge]
    assert len(compiled) == 4
