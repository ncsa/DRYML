from dryml.core2.cardinality import Cardinality
from dryml.core2.tensor_spec import TensorSpec
from dryml.data.source import GeneratorDataset


def generator_factory(offset=0):
    def data_gen():
        yield offset
        yield offset + 1

    return data_gen


def direct_iterable_factory(offset=0):
    return iter([offset, offset + 1])


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
