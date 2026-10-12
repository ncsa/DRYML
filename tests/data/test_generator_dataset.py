import numpy as np
import inspect
import itertools
import pytest

from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import Dynamic, TensorSpec
from dryml.data.dataset import Dataset
from dryml.data.source import GeneratorDataset
from dryml.methods import Method


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


_resources = []


class CloseableResource:
    def __init__(self):
        self.values = iter(np.arange(3, dtype=np.int64))
        self.closes = 0

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.values)

    def close(self):
        self.closes += 1


def closeable_factory():
    resource = CloseableResource()
    _resources.append(resource)
    return resource


def unbounded_factory():
    resource = closeable_factory()
    resource.values = map(np.int64, itertools.count())
    return resource


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


def test_dataset_requires_iteration_without_affecting_supported_generators():
    """Dataset subclasses must provide iteration while current sources stay concrete."""

    class MissingIteration(Dataset):
        pass

    assert inspect.isabstract(Dataset)
    assert inspect.isabstract(MissingIteration)
    with pytest.raises(TypeError, match="__iter__"):
        MissingIteration()
    assert not inspect.isabstract(GeneratorDataset)


@pytest.mark.parametrize("bound, consume", ((None, 0), (None, 1), (2, 1), (2, 3), (None, 4)))
def test_generator_cursor_owns_actual_factory_iterator_lazily(bound, consume):
    """Early close, finite cutoff, and real EOF close the actual factory resource."""
    _resources.clear()

    dataset = GeneratorDataset(
        closeable_factory if consume == 4 else unbounded_factory,
        spec=TensorSpec("int64", shape=(), backend="numpy"),
        cardinality=Cardinality.UNKNOWN if bound is None else Cardinality.finite(bound),
    )
    cursor = dataset.iterator()
    assert _resources == []
    for _ in range(consume):
        try:
            next(cursor)
        except StopIteration:
            break
    cursor.close()
    cursor.close()
    assert [resource.closes for resource in _resources] == ([] if consume == 0 else [1])
    assert cursor.observed_eof is (consume >= 3)


def test_prepared_map_closes_actual_unbounded_factory_iterator_on_early_exit():
    """Prepared Map retains the real factory resource, not only its finite wrapper."""
    from dryml.data import Map

    class Identity(Method):
        def __call__(self, value):
            return value

        def infer_output_spec(self, input_spec):
            return input_spec

    _resources.clear()
    source = GeneratorDataset(
        unbounded_factory, cardinality=Cardinality.finite(2),
        spec=TensorSpec("int64", shape=(), backend="numpy"),
    )
    graph = Map(source, Identity()).method_graph()
    graph.learn()
    cursor = graph.iterator()
    assert _resources == []
    assert next(cursor) == 0
    cursor.close()
    assert [resource.closes for resource in _resources] == [1]
    assert cursor.observed_eof is False
