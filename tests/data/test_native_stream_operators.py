"""Selected native Batch/Unbatch operators agree with the stream fallback."""

import numpy as np
import pytest

from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import TensorSpec
from dryml.data import Batch, Dataset, Unbatch
from dryml.data.collate import default_collate
from dryml.data.split import default_split


class Values(Dataset):
    """Small re-iterable dense source used for operator equivalence checks."""

    def __init__(self, spec):
        super().__init__(spec=spec)
        self.values = ()

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return Cardinality.finite(len(self.values))


class ClosableValues(Values):
    """Track source-generator cleanup under a native Batch graph traversal."""

    def __init__(self, spec):
        super().__init__(spec)
        self.closed = 0

    def __iter__(self):
        try:
            yield from self.values
        finally:
            self.closed += 1


@pytest.mark.parametrize("backend", ("numpy", "torch", "tf", "jax"))
def test_batch_and_unbatch_selected_paths_match_existing_fallback(backend):
    """Native collate/split retain short-batch order and round-trip cardinality."""

    if backend == "torch":
        torch = pytest.importorskip("torch")
        values = [torch.tensor([1, 2]), torch.tensor([3, 4]), torch.tensor([5, 6])]
    elif backend == "tf":
        tf = pytest.importorskip("tensorflow")
        values = [tf.constant([1, 2]), tf.constant([3, 4]), tf.constant([5, 6])]
    elif backend == "jax":
        jnp = pytest.importorskip("jax.numpy")
        values = [jnp.array([1, 2]), jnp.array([3, 4]), jnp.array([5, 6])]
    else:
        values = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
    source = Values(TensorSpec("int64", shape=(2,), backend=backend))
    source.values = tuple(values)
    batches = list(Batch(source, 2))

    assert len(batches) == 2
    assert [len(default_split(batch)) for batch in batches] == [2, 1]
    assert len(list(Unbatch(Batch(source, 2)))) == 3
    # The generic implementation remains a semantics-equivalent fallback.
    assert len(default_split(default_collate([values[0], values[1]]))) == 2


def test_jax_batch_and_unbatch_prepared_streams_match_eager_short_batches():
    """JAX stream plans select stack/split once and retain eager short-batch results."""

    jnp = pytest.importorskip("jax.numpy")
    source = Values(TensorSpec("int32", shape=(2,), backend="jax"))
    source.values = (jnp.array([1, 2]), jnp.array([3, 4]), jnp.array([5, 6]))

    eager_batches = list(Batch(source, 2))
    graph = Batch(source, 2).method_graph()
    graph.learn()
    prepared_batches = list(graph.iterator())
    eager_unbatched = list(Unbatch(Batch(source, 2)))
    unbatch_graph = Unbatch(Batch(source, 2)).method_graph()
    unbatch_graph.learn()
    prepared_unbatched = list(unbatch_graph.iterator())

    assert [np.asarray(value).tolist() for value in prepared_batches] == [np.asarray(value).tolist() for value in eager_batches]
    assert [np.asarray(value).tolist() for value in prepared_unbatched] == [np.asarray(value).tolist() for value in eager_unbatched]


def test_native_batch_graph_retains_fallback_cursor_cleanup_on_partial_consumption():
    """Selecting NumPy stack acceleration does not transfer cursor ownership."""

    source = ClosableValues(TensorSpec("int64", shape=(1,), backend="numpy"))
    source.values = (np.array([1]), np.array([2]), np.array([3]))
    graph = Batch(source, 2).method_graph()
    graph.learn()
    cursor = graph.iterator()

    np.testing.assert_array_equal(next(cursor), np.array([[1], [2]]))
    cursor.close()
    assert source.closed == 1
