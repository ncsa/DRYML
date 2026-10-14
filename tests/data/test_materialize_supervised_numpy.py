"""Strict one-shot NumPy collection respects declarations and cursor ownership."""

import subprocess
import sys

import numpy as np
import pytest

from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import Dynamic, TensorSpec
from dryml.data import ArrayDataset, Batch, Dataset, GeneratorDataset, Take, as_supervised
from dryml.data import DatasetExhaustedError, materialize_supervised


def pair_spec(*, batch=None, width=2):
    return (
        TensorSpec("float32", shape=(width,), batch=batch, backend="numpy"),
        TensorSpec("int64", shape=(), batch=batch, backend="numpy"),
    )


def pair(index=0):
    return np.asarray([index, index + 1], dtype=np.float32), np.int64(index)


class TrackingDataset(Dataset):
    """Expose descriptive counts independently of real, closeable traversal."""

    def __init__(self, values, *, spec=None, cardinality=None):
        self.values = tuple(values)
        self.cardinality = len(self.values) if cardinality is None else cardinality
        self.opens = 0
        self.reads = 0
        self.closes = 0
        self.close_error = None
        super().__init__(spec=pair_spec() if spec is None else spec)

    def __iter__(self):
        self.opens += 1
        return TrackingIterator(self)

    def __len__(self):
        return self.cardinality


class TrackingIterator:
    """Keep read EOF distinct from close failure on a real owned source resource."""

    def __init__(self, source):
        self.source = source
        self.values = iter(source.values)

    def __iter__(self):
        return self

    def __next__(self):
        value = next(self.values)
        self.source.reads += 1
        return value

    def close(self):
        self.source.closes += 1
        if self.source.close_error is not None:
            raise self.source.close_error


@pytest.mark.parametrize("cardinality", (Cardinality.UNKNOWN, Cardinality.INFINITE))
def test_unbounded_logical_count_rejected_without_source_acquisition(cardinality):
    source = TrackingDataset((pair(),), cardinality=cardinality)
    with pytest.raises(ValueError, match="finite"):
        materialize_supervised(source, strict_numpy=True)
    assert (source.opens, source.reads, source.closes) == (0, 0, 0)


@pytest.mark.parametrize("branch", (
    TensorSpec("float32", shape=(2,), backend="torch"),
    TensorSpec("float32", shape=(2,), backend=None),
    TensorSpec("float32", shape=(2,), backend="numpy", layout="sparse"),
    TensorSpec("float32", shape=(2,), backend="numpy", layout="ragged"),
    TensorSpec("bool", shape=(2,), backend="numpy"),
    TensorSpec("float16", shape=(2,), backend="numpy"),
    TensorSpec("complex64", shape=(2,), backend="numpy"),
    TensorSpec("object", shape=(2,), backend="numpy"),
    TensorSpec("string", shape=(2,), backend="numpy"),
    {"nested": TensorSpec("float32", shape=(2,), backend="numpy")},
))
def test_unsupported_declarations_rejected_before_open(branch):
    source = TrackingDataset((pair(),), spec=(branch, pair_spec()[1]))
    with pytest.raises(TypeError):
        materialize_supervised(source, strict_numpy=True)
    assert source.opens == source.reads == 0


@pytest.mark.parametrize("spec", (
    pair_spec(width=0),
    (TensorSpec("float32", shape=None, backend="numpy"), pair_spec()[1]),
    (pair_spec()[0], TensorSpec("int64", shape=(1,), backend="numpy")),
    (pair_spec(batch=2)[0], pair_spec(batch=3)[1]),
    pair_spec(batch=0),
))
def test_invalid_shapes_and_batches_rejected_before_open(spec):
    source = TrackingDataset((pair(),), spec=spec)
    with pytest.raises(ValueError):
        materialize_supervised(source, strict_numpy=True)
    assert source.opens == source.reads == 0


def test_opaque_bounded_dynamic_batches_count_actual_rows_in_order():
    source = TrackingDataset(tuple(pair(i) for i in range(5)))
    authored = Batch(source, 3)
    opaque = GeneratorDataset(source_iterator, authored, spec=authored.spec,
                              cardinality=Cardinality.finite(2))
    assert opaque.example_cardinality() == Cardinality.UNKNOWN
    x, y, examples = materialize_supervised(opaque, strict_numpy=True)
    np.testing.assert_array_equal(x, np.asarray([[i, i + 1] for i in range(5)]))
    np.testing.assert_array_equal(y, np.arange(5))
    assert (x.dtype, y.dtype, examples) == (np.dtype("float32"), np.dtype("int64"), 5)
    assert (source.opens, source.reads, source.closes) == (1, 5, 1)


def source_iterator(source):
    return source.iterator()


def reused_buffer_pairs():
    x = np.zeros(2, dtype=np.float32)
    y = np.zeros((), dtype=np.int64)
    for index in range(3):
        x[:] = [index, index + 1]
        y[()] = index
        yield x, y


def test_collection_retains_each_yield_when_source_reuses_native_buffers():
    source = GeneratorDataset(reused_buffer_pairs, spec=pair_spec(),
                              cardinality=Cardinality.finite(3))
    x, y, examples = materialize_supervised(source, strict_numpy=True)
    np.testing.assert_array_equal(x, [[0, 1], [1, 2], [2, 3]])
    np.testing.assert_array_equal(y, [0, 1, 2])
    assert examples == 3


def test_real_supervised_plain_and_authored_short_batches_are_identical():
    x = np.arange(10, dtype=np.float32).reshape(5, 2)
    y = np.arange(5, dtype=np.int64)
    data = as_supervised(ArrayDataset((x, y)), 0, 1)
    for dataset in (data, Batch(data, 3)):
        actual_x, actual_y, examples = materialize_supervised(dataset, strict_numpy=True)
        np.testing.assert_array_equal(actual_x, x)
        np.testing.assert_array_equal(actual_y, y)
        assert examples == 5


def test_validated_memmap_inputs_preserve_dtype_and_values(tmp_path):
    x = np.lib.format.open_memmap(tmp_path / "features.npy", mode="w+",
                                 dtype="float64", shape=(3, 2))
    x[:] = [[0, 1], [2, 3], [4, 5]]
    y = np.asarray([0, 1, 2], dtype=np.uint8)
    spec = (TensorSpec("float64", shape=(2,), backend="numpy"),
            TensorSpec("uint8", shape=(), backend="numpy"))
    source = ArrayDataset((x, y), spec=spec)
    actual_x, actual_y, count = materialize_supervised(Batch(source, 2), strict_numpy=True)
    assert type(actual_x) is type(actual_y) is np.ndarray
    assert (actual_x.dtype, actual_y.dtype, count) == (x.dtype, y.dtype, 3)
    np.testing.assert_array_equal(actual_x, x)
    np.testing.assert_array_equal(actual_y, y)


def test_nonfinite_values_remain_the_consumers_policy():
    x = np.asarray([[np.nan, np.inf], [-np.inf, 1]], dtype=np.float32)
    y = np.asarray([np.nan, np.inf], dtype=np.float64)
    source = ArrayDataset((x, y), spec=(pair_spec()[0],
                                      TensorSpec("float64", shape=(), backend="numpy")))
    actual_x, actual_y, count = materialize_supervised(source, strict_numpy=True)
    np.testing.assert_array_equal(actual_x, x)
    np.testing.assert_array_equal(actual_y, y)
    assert count == 2


def test_short_logical_traversal_keeps_count_error_over_cleanup_failure():
    source = TrackingDataset((pair(),), cardinality=2)
    source.close_error = RuntimeError("cleanup failed")
    with pytest.raises(DatasetExhaustedError) as error:
        materialize_supervised(source, strict_numpy=True)
    assert (error.value.requested, error.value.yielded) == (2, 1)
    assert source.reads == source.closes == 1
    assert "cleanup failed" in " ".join(error.value.__notes__)


def test_first_excess_logical_yield_stops_before_validation_or_collation(monkeypatch):
    import dryml.data.util as util
    source = TrackingDataset((pair(), None, pair(2)), cardinality=1)
    monkeypatch.setattr(util, "default_collate", lambda _: pytest.fail("generic collation"))
    monkeypatch.setattr(np, "stack", lambda *a, **k: pytest.fail("collation before validation"))
    with pytest.raises(ValueError, match="exceeds.*yield"):
        materialize_supervised(source, strict_numpy=True)
    assert (source.opens, source.reads, source.closes) == (1, 2, 1)


def test_authored_take_never_reads_underlying_excess():
    source = TrackingDataset((pair(), pair(1), None), cardinality=Cardinality.INFINITE)
    x, y, examples = materialize_supervised(Take(source, 2), strict_numpy=True)
    assert x.shape == (2, 2)
    np.testing.assert_array_equal(y, [0, 1])
    assert (examples, source.reads, source.closes) == (2, 2, 1)


@pytest.mark.parametrize("value", (
    [np.ones(2, dtype=np.float32), np.int64(1)],
    (np.ones(2, dtype=np.float32),),
    ([1.0, 2.0], np.int64(1)),
    (np.ones(2, dtype=np.float64), np.int64(1)),
    (np.ones((1, 2), dtype=np.float32), np.int64(1)),
    (np.ones(3, dtype=np.float32), np.int64(1)),
    (np.ones(2, dtype=np.float32), 1),
    (np.ones(2, dtype=np.float32), np.int32(1)),
    (np.asarray([[], []], dtype=object), np.int64(1)),
))
def test_runtime_drift_rejected_before_collation_and_without_later_reads(value, monkeypatch):
    source = TrackingDataset((pair(), value, pair(2)))
    monkeypatch.setattr(np, "stack", lambda *a, **k: pytest.fail("collation before validation"))
    with pytest.raises((TypeError, ValueError)):
        materialize_supervised(source, strict_numpy=True)
    assert (source.reads, source.closes) == (2, 1)


@pytest.mark.parametrize("spec, value", (
    (pair_spec(batch=2), (np.ones((1, 2), dtype=np.float32), np.ones(1, dtype=np.int64))),
    (pair_spec(batch=Dynamic), (np.ones((2, 2), dtype=np.float32), np.ones(3, dtype=np.int64))),
    (pair_spec(batch=Dynamic), (np.ones((0, 2), dtype=np.float32), np.ones(0, dtype=np.int64))),
    (pair_spec(batch=Dynamic), (np.ones((2, 2), dtype=np.float32), np.ones((2, 1), dtype=np.int64))),
))
def test_authored_batch_facts_rejected_before_collation(spec, value, monkeypatch):
    source = TrackingDataset((value, pair()), spec=spec)
    monkeypatch.setattr(np, "stack", lambda *a, **k: pytest.fail("collation before validation"))
    with pytest.raises(ValueError):
        materialize_supervised(source, strict_numpy=True)
    assert (source.reads, source.closes) == (1, 1)


def test_dynamic_feature_width_must_be_positive_and_consistent_before_collation(monkeypatch):
    monkeypatch.setattr(np, "stack", lambda *a, **k: pytest.fail("collation before validation"))
    for values in ((pair(), (np.ones(3, dtype=np.float32), np.int64(1))),
                   ((np.ones(0, dtype=np.float32), np.int64(1)),)):
        source = TrackingDataset(values, spec=pair_spec(width=Dynamic))
        with pytest.raises(ValueError, match="feature width"):
            materialize_supervised(source, strict_numpy=True)
        assert source.closes == 1


def test_convertible_foreign_and_masked_values_are_never_coerced():
    class Convertible:
        shape = (2,)
        dtype = np.dtype("float32")

        def __array__(self, *args, **kwargs):
            pytest.fail("foreign conversion attempted")

    for value in (Convertible(), np.ma.array([1, 2], dtype=np.float32, mask=[0, 1])):
        source = TrackingDataset(())
        source.values = ((value, np.int64(1)), pair())
        source.cardinality = 2
        with pytest.raises(TypeError, match="dense NumPy"):
            materialize_supervised(source, strict_numpy=True)
        assert source.reads == source.closes == 1


def test_strict_control_requires_an_exact_bool_before_open():
    source = TrackingDataset((pair(),))
    with pytest.raises(TypeError, match="exact bool"):
        materialize_supervised(source, strict_numpy=1)
    assert source.opens == 0


def test_source_failure_remains_primary_when_cleanup_also_fails(monkeypatch):
    source = TrackingDataset((pair(),))
    source.close_error = RuntimeError("cleanup failed")
    failure = OSError("source failed")

    def failed_read(iterator):
        raise failure

    monkeypatch.setattr(TrackingIterator, "__next__", failed_read)
    with pytest.raises(OSError) as error:
        materialize_supervised(source, strict_numpy=True)
    assert error.value is failure
    assert source.opens == source.closes == 1
    assert "cleanup failed" in " ".join(error.value.__notes__)


@pytest.mark.parametrize("strict", (False, True))
def test_consumer_error_remains_primary_when_source_close_fails(strict):
    source = TrackingDataset((None,))
    source.close_error = RuntimeError("cleanup failed")
    kwargs = {"strict_numpy": True} if strict else {}
    with pytest.raises(TypeError) as error:
        materialize_supervised(source, **kwargs)
    assert source.closes == 1
    assert "cleanup failed" in " ".join(error.value.__notes__)


def test_interruption_remains_primary_when_cleanup_also_fails(monkeypatch):
    source = TrackingDataset((pair(),))
    source.close_error = RuntimeError("cleanup failed")
    interruption = KeyboardInterrupt("consumer interrupted")

    def interrupted(value):
        raise interruption

    monkeypatch.setattr(source, "examples_in", interrupted)
    with pytest.raises(KeyboardInterrupt) as error:
        materialize_supervised(source, strict_numpy=True)
    assert error.value is interruption
    assert source.reads == source.closes == 1


def test_empty_collection_and_cleanup_only_failure_do_not_return_success():
    with pytest.raises(ValueError, match="empty"):
        materialize_supervised(TrackingDataset(()), strict_numpy=True)
    source = TrackingDataset((pair(),))
    source.close_error = RuntimeError("cleanup failed")
    with pytest.raises(RuntimeError, match="cleanup failed"):
        materialize_supervised(source, strict_numpy=True)
    assert source.closes == 1


def test_strict_numpy_success_and_admission_never_import_optional_frameworks():
    script = """
import sys
attempts = []
class RejectNativeImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'tensorflow', 'torch', 'jax', 'xgboost', 'statsmodels'}:
            attempts.append(fullname)
            raise AssertionError('unexpected native import: ' + fullname)
sys.meta_path.insert(0, RejectNativeImports())
import numpy as np
from dryml.core import TensorSpec
from dryml.core.cardinality import Cardinality
from dryml.data import ArrayDataset, Batch, materialize_supervised
from tests.data.test_materialize_supervised_numpy import TrackingDataset, pair, pair_spec
for cardinality in (Cardinality.UNKNOWN, Cardinality.INFINITE):
    source = TrackingDataset((pair(),), cardinality=cardinality)
    try:
        materialize_supervised(source, strict_numpy=True)
    except ValueError:
        pass
    else:
        raise AssertionError('unbounded source admitted')
    assert source.opens == 0
for backend in ('torch', 'tf', 'jax', None):
    spec = (TensorSpec('float32', shape=(2,), backend=backend), pair_spec()[1])
    source = TrackingDataset((pair(),), spec=spec)
    try:
        materialize_supervised(source, strict_numpy=True)
    except TypeError:
        pass
    else:
        raise AssertionError('foreign source admitted')
    assert source.opens == 0
source = ArrayDataset((np.ones((3, 2), dtype=np.float32), np.arange(3, dtype=np.int64)),
                      spec=pair_spec())
x, y, n = materialize_supervised(Batch(source, 2), strict_numpy=True)
assert (x.shape, y.shape, n) == ((3, 2), (3,), 3)
assert attempts == [], attempts
assert not {'tensorflow', 'torch', 'jax', 'xgboost', 'statsmodels'} & sys.modules.keys()
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
