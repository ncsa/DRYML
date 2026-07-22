from __future__ import annotations

from enum import Enum

import numpy as np
import pytest

from dryml.core import ConcreteDefinition, Definition
from dryml.core.cardinality import Cardinality
from dryml.core.config import ConfigRef
from dryml.core.dtype import DType
from dryml.core.factory import FactorySpec
from dryml.core.freeze import FrozenDict, FrozenList, FrozenNDArray, FrozenSet, FrozenTuple
from dryml.core.object import Object
from dryml.core.params import Par, PresentMatcher
from dryml.core.quoted import SelectorSpec
from dryml.core.symbol import ImportRef, SourceSpec
from dryml.core.tensor_spec import Layout, TensorSpec
from dryml.core.utils.stable_hash import (
    StableHashLimitError,
    StableHashLimits,
    bounded_stable_hash_function,
    stable_hash_function,
)


class Choice(Enum):
    A = "a"


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        7,
        1.5,
        "value",
        b"bytes",
        dict,
        np.asarray([1, 2], dtype=np.int32),
        FrozenNDArray.from_array(np.asarray([1, 2], dtype=np.int32)),
        [1, (2, 3)],
        {"a": 1, "b": [2]},
        {1, 2},
        FrozenList([1, 2]),
        FrozenTuple((1, 2)),
        FrozenSet({1, 2}),
        FrozenDict({"a": 1}),
        Definition(dict, value=1),
        ConcreteDefinition(dict, (), {"value": 1}),
        DType("float", 32),
        TensorSpec("float32", (2, 3)),
        Cardinality.finite(3),
        ConfigRef("key"),
        ConfigRef("key", [1, 2]),
        FactorySpec(dict, value=1),
        ImportRef("builtins", "dict"),
        SourceSpec.from_source("class Local: pass", kind="class", name="Local"),
        Layout.DENSE,
        Choice.A,
    ],
)
def test_bounded_hash_preserves_existing_digest(value):
    assert bounded_stable_hash_function(value) == stable_hash_function(value)


@pytest.mark.parametrize(
    "value",
    [
        FrozenList([1]),
        FrozenTuple((1,)),
        FrozenSet({1}),
        FrozenDict({"key": 1}),
    ],
)
def test_frozen_containers_obey_exact_traversal_budgets(value):
    import dryml.core.utils.stable_hash as stable_hash

    expected = stable_hash_function(value)
    probe = stable_hash._BoundedStableHasher(StableHashLimits())
    assert probe.hash(value) == expected

    assert bounded_stable_hash_function(
        value,
        limits=StableHashLimits(max_depth=1),
    ) == expected
    with pytest.raises(StableHashLimitError) as depth:
        bounded_stable_hash_function(value, limits=StableHashLimits(max_depth=0))
    assert (depth.value.limit_name, depth.value.limit, depth.value.observed_lower_bound) == ("depth", 0, 1)

    counts = {
        "max_occurrences": ("occurrences", probe.budget.occurrences),
        "max_edges": ("edges", probe.budget.edges),
        "max_encoded_bytes": ("encoded_bytes", probe.budget.encoded_bytes),
    }
    for field, (limit_name, exact) in counts.items():
        assert bounded_stable_hash_function(
            value,
            limits=StableHashLimits(**{field: exact}),
        ) == expected
        with pytest.raises(StableHashLimitError) as exceeded:
            bounded_stable_hash_function(
                value,
                limits=StableHashLimits(**{field: exact - 1}),
            )
        assert exceeded.value.limit_name == limit_name
        assert exceeded.value.limit == exact - 1
        assert exceeded.value.observed_lower_bound > exceeded.value.limit


def test_frozen_ndarray_obeys_exact_atomic_budgets():
    import dryml.core.utils.stable_hash as stable_hash

    value = FrozenNDArray.from_array(np.asarray([1, 2], dtype=np.int32))
    expected = stable_hash_function(value)
    probe = stable_hash._BoundedStableHasher(StableHashLimits())
    assert probe.hash(value) == expected

    assert bounded_stable_hash_function(
        value,
        limits=StableHashLimits(max_depth=0, max_occurrences=1, max_edges=0),
    ) == expected
    with pytest.raises(StableHashLimitError) as occurrences:
        bounded_stable_hash_function(value, limits=StableHashLimits(max_occurrences=0))
    assert (
        occurrences.value.limit_name,
        occurrences.value.limit,
        occurrences.value.observed_lower_bound,
    ) == ("occurrences", 0, 1)

    exact_bytes = probe.budget.encoded_bytes
    assert bounded_stable_hash_function(
        value,
        limits=StableHashLimits(max_encoded_bytes=exact_bytes),
    ) == expected
    with pytest.raises(StableHashLimitError) as encoded:
        bounded_stable_hash_function(
            value,
            limits=StableHashLimits(max_encoded_bytes=exact_bytes - 1),
        )
    assert encoded.value.limit_name == "encoded_bytes"
    assert encoded.value.limit == exact_bytes - 1
    assert encoded.value.observed_lower_bound > encoded.value.limit


def test_depth_occurrence_edge_and_byte_boundaries():
    value = ["x"]
    assert bounded_stable_hash_function(value, limits=StableHashLimits(max_depth=1))
    with pytest.raises(StableHashLimitError) as depth:
        bounded_stable_hash_function(value, limits=StableHashLimits(max_depth=0))
    assert (depth.value.limit_name, depth.value.limit, depth.value.observed_lower_bound) == ("depth", 0, 1)

    assert bounded_stable_hash_function(value, limits=StableHashLimits(max_occurrences=2))
    with pytest.raises(StableHashLimitError) as occurrences:
        bounded_stable_hash_function(value, limits=StableHashLimits(max_occurrences=1))
    assert occurrences.value.limit_name == "occurrences"
    assert occurrences.value.observed_lower_bound == 2

    assert bounded_stable_hash_function(value, limits=StableHashLimits(max_edges=1))
    with pytest.raises(StableHashLimitError) as edges:
        bounded_stable_hash_function(value, limits=StableHashLimits(max_edges=0))
    assert edges.value.limit_name == "edges"
    assert edges.value.observed_lower_bound == 1

    assert bounded_stable_hash_function("x", limits=StableHashLimits(max_encoded_bytes=2))
    with pytest.raises(StableHashLimitError) as encoded:
        bounded_stable_hash_function("x", limits=StableHashLimits(max_encoded_bytes=1))
    assert encoded.value.limit_name == "encoded_bytes"
    assert encoded.value.observed_lower_bound == 2


def test_integer_and_string_limits_fail_before_encoding():
    assert bounded_stable_hash_function(1 << 7, limits=StableHashLimits(max_integer_bits=8))
    with pytest.raises(StableHashLimitError) as integer:
        bounded_stable_hash_function(1 << 8, limits=StableHashLimits(max_integer_bits=8))
    assert integer.value.limit_name == "integer_bits"

    assert bounded_stable_hash_function("abcd", limits=StableHashLimits(max_string_chars=4))
    with pytest.raises(StableHashLimitError) as string:
        bounded_stable_hash_function("abcde", limits=StableHashLimits(max_string_chars=4))
    assert string.value.limit_name == "string_chars"


def test_byte_payload_limit_is_checked_at_the_exact_leaf_boundary():
    value = b"payload"
    # The existing leaf encoding is one ``Y`` marker byte plus the payload.
    assert bounded_stable_hash_function(
        value,
        limits=StableHashLimits(max_encoded_bytes=1 + len(value)),
    ) == stable_hash_function(value)
    with pytest.raises(StableHashLimitError) as encoded:
        bounded_stable_hash_function(
            value,
            limits=StableHashLimits(max_encoded_bytes=len(value)),
        )
    assert (
        encoded.value.limit_name,
        encoded.value.limit,
        encoded.value.observed_lower_bound,
    ) == ("encoded_bytes", len(value), len(value) + 1)


def test_memo_hit_charges_occurrence_and_edge_without_descendants():
    shared = [1]
    value = [shared, shared]
    assert bounded_stable_hash_function(value, limits=StableHashLimits(max_occurrences=4, max_edges=3))
    with pytest.raises(StableHashLimitError) as occurrences:
        bounded_stable_hash_function(value, limits=StableHashLimits(max_occurrences=3))
    assert occurrences.value.observed_lower_bound == 4


def test_nested_factory_and_config_defaults_share_budgets_and_digest():
    value = FactorySpec(dict, ConfigRef("nested", FactorySpec(list, 1)), answer=ConfigRef("answer", 42))
    assert bounded_stable_hash_function(value) == stable_hash_function(value)
    with pytest.raises(StableHashLimitError):
        bounded_stable_hash_function(value, limits=StableHashLimits(max_occurrences=3))


@pytest.mark.parametrize(
    "value",
    [
        FactorySpec(dict, ConfigRef("nested", FactorySpec(list, 1))),
        ConfigRef("missing"),
        ConfigRef("default", FactorySpec(dict, answer=1)),
    ],
)
def test_factory_and_config_branches_obey_exact_shared_budgets(value):
    import dryml.core.utils.stable_hash as stable_hash

    probe = stable_hash._BoundedStableHasher(StableHashLimits())
    assert probe.hash(value) == stable_hash_function(value)
    counts = {
        "max_occurrences": probe.budget.occurrences,
        "max_edges": probe.budget.edges,
        "max_encoded_bytes": probe.budget.encoded_bytes,
    }
    for field, exact in counts.items():
        assert bounded_stable_hash_function(
            value,
            limits=StableHashLimits(**{field: exact}),
        ) == stable_hash_function(value)
        with pytest.raises(StableHashLimitError):
            bounded_stable_hash_function(
                value,
                limits=StableHashLimits(**{field: exact - 1}),
            )


def test_graph_wrapper_node_kinds_preserve_existing_digests():
    definition = Definition(dict, value=1)
    selector = definition.as_selector()

    class Model(Object):
        pass

    runtime_object = object.__new__(Model)
    runtime_object.__cdef__ = ConcreteDefinition(Model)
    runtime_object.__ws__ = None
    values = (
        definition.ref(),
        definition.mat(),
        definition.quote(),
        selector,
        SelectorSpec(selector),
        Par("value", PresentMatcher()),
        runtime_object,
    )
    for value in values:
        assert bounded_stable_hash_function(value) == stable_hash_function(value)


def test_complex_remains_unsupported():
    with pytest.raises(TypeError):
        stable_hash_function(1 + 2j)
    with pytest.raises(TypeError):
        bounded_stable_hash_function(1 + 2j)


def test_custom_stable_leaf_hook_is_not_invoked():
    class Custom:
        called = False

        def __stable_leaf_bytes__(self):
            type(self).called = True
            return b"custom"

    with pytest.raises(TypeError):
        bounded_stable_hash_function(Custom())
    assert Custom.called is False


def test_atomic_subclass_metaclass_repr_is_not_invoked():
    class HookMeta(type):
        called = False

        def __repr__(cls):
            HookMeta.called = True
            return "HookMetaDType"

    class HookDType(DType, metaclass=HookMeta):
        pass

    class HookImportRef(ImportRef, metaclass=HookMeta):
        pass

    class HookSourceSpec(SourceSpec, metaclass=HookMeta):
        pass

    for value, message in (
        (HookDType("float", 32), "Unsupported identity-value subclass"),
        (HookImportRef("builtins", "dict"), "Unsupported symbol-reference subclass"),
        (HookSourceSpec.from_source("class Local: pass", kind="class", name="Local"), "Unsupported symbol-reference subclass"),
    ):
        HookMeta.called = False
        with pytest.raises(TypeError, match=message):
            bounded_stable_hash_function(value)
        assert HookMeta.called is False


def test_python_pod_subclass_hooks_are_not_invoked():
    class HookInt(int):
        called = False

        def __str__(self):
            type(self).called = True
            return super().__str__()

    class HookString(str):
        called = False

        def encode(self, *args, **kwargs):
            type(self).called = True
            return super().encode(*args, **kwargs)

    class HookMeta(type):
        called = False

        def __repr__(cls):
            HookMeta.called = True
            return "HookMetaInt"

    class HookMetaInt(int, metaclass=HookMeta):
        pass

    for value, value_type in ((HookInt(1), HookInt), (HookString("value"), HookString)):
        with pytest.raises(TypeError, match="Unsupported Python POD subclass"):
            bounded_stable_hash_function(value)
        assert value_type.called is False

    with pytest.raises(TypeError, match="Unsupported Python POD subclass"):
        bounded_stable_hash_function(HookMetaInt(1))
    assert HookMeta.called is False


def test_limit_configuration_is_strict():
    with pytest.raises(TypeError):
        bounded_stable_hash_function(1, limits=StableHashLimits(max_depth=True))
    with pytest.raises(ValueError):
        bounded_stable_hash_function(1, limits=StableHashLimits(max_edges=-1))
