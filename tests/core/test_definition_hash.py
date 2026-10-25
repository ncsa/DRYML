from copy import copy, deepcopy

import numpy as np
import pytest

from tests.core import core_objects as objects
from dryml.core.cdef_graph import ConcreteDefinitionGraph
from dryml.core.cdef_identity import V2_IDENTITY_VERSION
from dryml.core.bound_args import BoundArguments
from dryml.core.definition import ConcreteDefinition, Definition, stable_hash_function
from dryml.core.freeze import FrozenDict, FrozenList, FrozenTuple


def test_definition_hash_1():
    definition1 = Definition(
        objects.TestClass1,
        10, test='a')

    def_hash1 = stable_hash_function(definition1)

    definition2 = Definition(
        objects.TestClass1,
        10, test='a')

    def_hash_2 = stable_hash_function(definition2)

    assert def_hash1 == def_hash_2


def test_definition_hash_2():
    definition1 = Definition(
        objects.TestClass2,
        var1='a',
        var2='b',
        var3='c',
        var4='d')

    def_hash1 = stable_hash_function(definition1)

    definition2 = Definition(
        objects.TestClass2,
        var4='d',
        var3='c',
        var2='b',
        var1='a')

    def_hash2 = stable_hash_function(definition2)

    assert def_hash1 == def_hash2


def test_definition_hash_4():
    definition1 = Definition(
        objects.TestClass2,
        Definition(
            objects.TestClass1,
            10,
            test='A'),
        var1='a',
        var2='b',
        var3='c',
        var4='d')

    def_hash1 = stable_hash_function(definition1)

    definition2 = Definition(
        objects.TestClass2,
        Definition(
            objects.TestClass1,
            10,
            test='A'),
        var4='d',
        var3='c',
        var2='b',
        var1='a')

    def_hash2 = stable_hash_function(definition2)

    assert def_hash1 == def_hash2


def test_definition_hash_5():
    definition1 = Definition(
        objects.TestClass2,
        Definition(
            objects.TestClass1,
            10,
            test='A'),
        var1='a',
        var2='b',
        var3='c',
        var4='d')

    def_hash1 = stable_hash_function(definition1)

    definition2 = Definition(
        objects.TestClass2,
        Definition(
            objects.TestClass1,
            10,
            test='B'),
        var4='d',
        var3='c',
        var2='b',
        var1='a')

    def_hash2 = stable_hash_function(definition2)

    assert def_hash1 != def_hash2


def test_definition_hash_6():
    arr = np.random.random((10,10)).astype(np.float32)
    arr2 = np.copy(arr)

    definition1 = Definition(
        objects.TestClass2,
        Definition(
            objects.TestClass1,
            arr,
            test='A'),
        var1='a',
        var2='b',
        var3='c',
        var4='d')

    def_hash1 = stable_hash_function(definition1)

    definition2 = Definition(
        objects.TestClass2,
        Definition(
            objects.TestClass1,
            arr2,
            test='A'),
        var4='d',
        var3='c',
        var2='b',
        var1='a')

    def_hash2 = stable_hash_function(definition2)

    assert def_hash1 == def_hash2


def test_definition_hash_7():
    arr = np.random.random((10,10)).astype(np.float32)
    arr2 = np.copy(arr)
    arr2[0,0] = 5.

    definition1 = Definition(
        objects.TestClass2,
        Definition(
            objects.TestClass1,
            arr,
            test='A'),
        var1='a',
        var2='b',
        var3='c',
        var4='d')

    def_hash1 = stable_hash_function(definition1)

    definition2 = Definition(
        objects.TestClass2,
        Definition(
            objects.TestClass1,
            arr2,
            test='A'),
        var4='d',
        var3='c',
        var2='b',
        var1='a')

    def_hash2 = stable_hash_function(definition2)

    assert def_hash1 != def_hash2


def test_mapping_insertion_order_is_not_part_of_definition_identity():
    first = Definition(
        objects.TestClass1,
        10,
        test={"outer": {"batch_size": 32, "seed": 7, "dtype": "float32"}},
    ).concretize()
    second = Definition(
        objects.TestClass1,
        10,
        test={"outer": {"dtype": "float32", "seed": 7, "batch_size": 32}},
    ).concretize()

    assert first.graph_equal(second)
    assert first.stable_hash() == second.stable_hash()
    assert len({first, second}) == 2


def test_frozen_dict_hash_matches_order_independent_mapping_equality():
    first = FrozenDict((("items", FrozenList((1, 2))), ("label", "x")))
    second = FrozenDict((("label", "x"), ("items", FrozenList((1, 2)))))

    assert first == second
    assert hash(first) == hash(second)
    assert len({first, second}) == 1


def test_definition_sequence_identity_remains_order_sensitive():
    listed = Definition(objects.TestClass1, 10, test={"items": [1, 2]}).concretize()
    reversed_list = Definition(objects.TestClass1, 10, test={"items": [2, 1]}).concretize()
    tupled = Definition(objects.TestClass1, 10, test={"items": (1, 2)}).concretize()
    reversed_tuple = Definition(objects.TestClass1, 10, test={"items": (2, 1)}).concretize()

    assert listed != reversed_list
    assert tupled != reversed_tuple


def test_private_v2_records_preserve_independent_graph_nodes():
    child_v2 = ConcreteDefinition._from_persisted_record(
        objects.TestClass1,
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((("value", 10), ("test", "child"))),
    )
    root_v2 = ConcreteDefinition._from_persisted_record(
        objects.TestClass1,
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((("values", FrozenTuple((child_v2, child_v2))), ("test", "root"))),
    )

    assert copy(root_v2) is root_v2
    assert deepcopy(root_v2) is root_v2

    graph = ConcreteDefinitionGraph.from_roots((root_v2, root_v2))
    assert graph.roots == (root_v2,)
    assert {node.definition for node in graph.nodes()} == {root_v2, child_v2}


def test_v2_persisted_hash_cache_must_match_identity_record():
    cdef = Definition(objects.TestClass1, 10, test="cached").concretize()

    with pytest.raises(ValueError, match="hash cache"):
        ConcreteDefinition._from_persisted_record(
            cdef.cls,
            identity_version=V2_IDENTITY_VERSION,
            parameters=cdef._bound_args,
            stable_hash_cache="0" * 64,
        )

    restored = ConcreteDefinition._from_persisted_record(
        cdef.cls,
        identity_version=V2_IDENTITY_VERSION,
        parameters=cdef._bound_args,
        stable_hash_cache=cdef.stable_hash(),
    )
    assert restored.graph_equal(cdef)
    assert restored.stable_hash() == cdef.stable_hash()


@pytest.mark.parametrize("depth", (10, 20, 40))
def test_v2_persisted_hash_cache_validation_reuses_validated_nested_hashes(monkeypatch, depth):
    """Current portable Node hydration retains nested topology and warm key reuse."""

    from dryml.core.node_codec import decode_node_graph, encode_node_graph
    from dryml.core.node_prepare import NodePreparation

    cdef = Definition(objects.TestNest2, None).concretize()
    cdef.stable_hash()
    persisted = [cdef]
    for _ in range(depth - 1):
        cdef = Definition(objects.TestNest2, cdef).concretize()
        cdef.stable_hash()
        persisted.append(cdef)

    data = encode_node_graph(NodePreparation((cdef,)))
    restored = decode_node_graph(data)
    assert restored is not cdef and restored.graph_equal(cdef)
    prepared = NodePreparation((restored,))
    assert prepared.exact_key(restored) == cdef.exact_key()
    before = dict(prepared.counts)
    assert prepared.exact_key(restored) == cdef.exact_key()
    assert prepared.counts == before


def test_v2_parameter_order_is_not_part_of_identity():
    first = ConcreteDefinition._from_persisted_record(
        objects.TestClass1,
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((("value", 10), ("test", "ordered"))),
    )
    second = ConcreteDefinition._from_persisted_record(
        objects.TestClass1,
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((("test", "ordered"), ("value", 10))),
    )

    assert first == second
    assert first.stable_hash() == second.stable_hash()
    assert len({first, second}) == 1
