import numpy as np

from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import Dynamic, TensorSpec
from dryml.data.source import GeneratorDataset


def generator_factory(offset=0):
    def data_gen():
        yield offset
        yield offset + 1

    return data_gen


def direct_iterable_factory(offset=0):
    return iter([offset, offset + 1])


def variable_shape_array_factory():
    return iter([
        np.zeros((2,), dtype=np.float32),
        np.zeros((5,), dtype=np.float32),
    ])


def test_generator_dataset_accepts_factory_returning_generator_function():
    ds = GeneratorDataset(
        generator_factory,
        offset=10,
        cardinality=Cardinality.finite(2),
        spec=TensorSpec("int32", shape=()),
    )

    assert list(ds) == [10, 11]
    assert list(ds) == [10, 11]


def test_generator_dataset_accepts_factory_returning_iterable():
    ds = GeneratorDataset(
        direct_iterable_factory,
        offset=20,
        cardinality=Cardinality.finite(2),
        spec=TensorSpec("int32", shape=()),
    )

    assert list(ds) == [20, 21]


def test_generator_dataset_infers_spec_from_hint():
    ds = GeneratorDataset(
        variable_shape_array_factory,
        cardinality=Cardinality.finite(2),
        spec=2,
    )

    assert ds.spec == TensorSpec("float32", shape=(Dynamic,), backend="numpy")
