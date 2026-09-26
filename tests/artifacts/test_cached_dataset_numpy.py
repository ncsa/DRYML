"""Focused NumPy completed-cache behavior."""

from __future__ import annotations

import numpy as np
import pytest

from dryml.artifacts import ArtifactNotReadyError, CachedDataset
from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import Dynamic, TensorSpec
from dryml.data import Dataset
from dryml.managed import ManagedConfig


class ArrayDataset(Dataset):
    """Small re-iterable NumPy source used by cache tests."""

    def __init__(self, values, spec, *, integer_length=False):
        self.values = list(values)
        self.integer_length = integer_length
        super().__init__(spec)

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        if self.integer_length:
            return len(self.values)
        return Cardinality.finite(len(self.values))


class ZeroLeafDataset(Dataset):
    """Finite empty-tree source that does not retain yielded container objects."""

    def __init__(self, count):
        self.count = count
        super().__init__(())

    def __iter__(self):
        for _ in range(self.count):
            yield ()

    def __len__(self):
        return self.count


def test_cache_is_not_ready_before_compute_and_validates_codec_keyword(tmp_path):
    """Construction is inert and exposes no completed Dataset contract."""

    cache = CachedDataset(ArrayDataset([], TensorSpec("int32", shape=(1,), backend="numpy")))

    assert not cache.ready
    with pytest.raises(ArtifactNotReadyError):
        _ = cache.spec
    with pytest.raises(ArtifactNotReadyError):
        len(cache)
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    with pytest.raises(ValueError):
        cache.compute(codec="unknown", managed=ManagedConfig(state_repo=Repo(DirStore(tmp_path / "store"))))


def test_numpy_cache_round_trips_nested_values_and_empty_source(tmp_path):
    """NumPy completed state preserves nested containers and typed emptiness."""

    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    spec = {
        "name": TensorSpec("string", shape=(), backend="numpy"),
        1: (TensorSpec("float32", shape=(Dynamic,), backend="numpy"),),
    }
    source = ArrayDataset([
        {"name": np.asarray("a\x00b"), 1: (np.asarray([1.0, np.nan], dtype=np.float32),)},
        {"name": np.asarray("é"), 1: (np.asarray([], dtype=np.float32),)},
    ], spec)
    cache = CachedDataset(source)
    repo = Repo(DirStore(tmp_path / "store"))

    state = cache.compute(codec="numpy", managed=ManagedConfig(state_repo=repo))

    assert cache.ready
    assert state == cache.last_state_ref
    assert len(cache) == 2
    restored = repo.load_state_ref(state, reuse_live="never")
    values = list(restored)
    assert values[0]["name"].item() == "a\x00b"
    assert values[1]["name"].item() == "é"
    np.testing.assert_equal(values[0][1][0], source.values[0][1][0])
    assert values[1][1][0].shape == (0,)

    empty = CachedDataset(ArrayDataset([], TensorSpec("int32", shape=(1,), backend="numpy")))
    empty.compute(codec="numpy", managed=ManagedConfig(state_repo=repo))
    assert empty.ready and len(empty) == 0
    with pytest.raises(ValueError, match="empty"):
        empty.peek()


def test_numpy_cache_preserves_compact_logical_fidelity_surface(tmp_path):
    """One nested yield covers keys, shapes, special values, and optional bfloat."""

    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    class FidelityDataset(Dataset):
        """Keep bool-key cache evidence out of the general CDef argument codec."""

        def __init__(self):
            from dryml.core.tensor_spec import Dynamic as dynamic

            spec = {
                "float": TensorSpec("float64", shape=(dynamic,), backend="numpy"),
                2: [TensorSpec("uint64", shape=(1,), backend="numpy"), TensorSpec("complex64", shape=(), backend="numpy")],
                False: (TensorSpec("string", shape=(dynamic,), backend="numpy"), TensorSpec("float32", shape=(0,), backend="numpy")),
            }
            value = {
                "float": np.asarray([np.nan, -0.0], dtype=np.float64),
                2: [np.asarray([np.iinfo(np.uint64).max], dtype=np.uint64), np.asarray(1 + 2j, dtype=np.complex64)],
                False: (np.asarray(["", "e\u0301", "a\x00b"]), np.asarray([], dtype=np.float32)),
            }
            try:
                import ml_dtypes
            except ImportError:
                pass
            else:
                spec["bfloat"] = TensorSpec("bfloat16", shape=(1,), backend="numpy")
                value["bfloat"] = np.asarray([1.5], dtype=ml_dtypes.bfloat16)
            self.values = [value]
            super().__init__(spec)

        def __iter__(self):
            return iter(self.values)

        def __len__(self):
            return Cardinality.finite(1)

    cache = CachedDataset(FidelityDataset())
    cache.compute(codec="numpy", managed=ManagedConfig(state_repo=Repo(DirStore(tmp_path / "store"))))
    restored = next(iter(cache))

    assert list(restored.keys())[:3] == ["float", 2, False]
    assert [type(key) for key in list(restored)[:3]] == [str, int, bool]
    assert np.isnan(restored["float"][0])
    assert np.signbit(restored["float"][1])
    np.testing.assert_equal(restored[2][0], np.asarray([np.iinfo(np.uint64).max], dtype=np.uint64))
    np.testing.assert_equal(restored[2][1], np.asarray(1 + 2j, dtype=np.complex64))
    np.testing.assert_equal(restored[False][0], np.asarray(["", "e\u0301", "a\x00b"]))
    assert restored[False][1].shape == (0,)
    if "bfloat" in restored:
        import ml_dtypes

        np.testing.assert_equal(restored["bfloat"], np.asarray([1.5], dtype=ml_dtypes.bfloat16))


@pytest.mark.parametrize(
    ("spec", "value"),
    [
        ({"leaf": TensorSpec("int32", shape=(1,), backend="numpy")}, np.asarray([1], dtype=np.int32)),
        (TensorSpec("object", shape=(), backend="numpy"), np.asarray(object(), dtype=object)),
        (TensorSpec("string", shape=(), backend="numpy"), np.asarray(b"bytes")),
        (TensorSpec("string", shape=(), backend="numpy"), None),
        (TensorSpec("int32", shape=(2,), backend="numpy"), np.asarray([1], dtype=np.int32)),
        (TensorSpec("int32", shape=(2,), backend="numpy"), np.asarray(1, dtype=np.int32)),
        (TensorSpec("int32", shape=None, batch=Dynamic, backend="numpy"), np.asarray(1, dtype=np.int32)),
    ],
    ids=("structure", "object", "bytes", "null", "fixed-shape", "rank", "batched-scalar"),
)
def test_numpy_cache_rejects_invalid_logical_yields(tmp_path, spec, value):
    """Malformed structure, scalar contracts, and unsupported leaves never become ready."""

    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    cache = CachedDataset(ArrayDataset([value], spec))
    with pytest.raises((TypeError, ValueError)):
        cache.compute(codec="numpy", managed=ManagedConfig(state_repo=Repo(DirStore(tmp_path / "store"))))
    assert not cache.ready


def test_numpy_cache_accepts_array_dataset_integer_length(tmp_path):
    """Exact nonnegative integers remain supported Dataset cardinality adapters."""

    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    source = ArrayDataset(
        [np.asarray([1], dtype=np.int32)],
        TensorSpec("int32", shape=(1,), backend="numpy"),
        integer_length=True,
    )
    cache = CachedDataset(source)
    cache.compute(codec="numpy", managed=ManagedConfig(state_repo=Repo(DirStore(tmp_path / "store"))))

    assert [item.item() for item in cache] == [1]


@pytest.mark.exhaustive_only
def test_zero_leaf_stream_partitions_exactly_at_yield_cap(tmp_path):
    """131,073 empty-tree yields seal the required 65,536/65,536/1 chunks."""

    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    cache = CachedDataset(ZeroLeafDataset(131_073))
    cache.compute(codec="numpy", managed=ManagedConfig(state_repo=Repo(DirStore(tmp_path / "store"))))

    assert len(cache) == 131_073
    assert [chunk["count"] for chunk in cache._cache_payload["chunks"]] == [65_536, 65_536, 1]
    assert list(cache)[:1] == [()]


@pytest.mark.exhaustive_only
@pytest.mark.parametrize("dtype", ("bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64", "bfloat16", "complex64", "complex128", "string"))
@pytest.mark.parametrize("shape", ((), (0,), (2,), (2, 1)))
def test_numpy_cache_exhaustive_dtype_and_rank_matrix(tmp_path, dtype, shape):
    """Every concrete NumPy dtype/rank pair remains available to exhaustive runs."""

    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    if dtype == "bool":
        value = np.zeros(shape, dtype=np.bool_)
    elif dtype == "string":
        value = np.full(shape, "x", dtype=np.str_)
    elif dtype == "bfloat16":
        import ml_dtypes

        value = np.zeros(shape, dtype=ml_dtypes.bfloat16)
    else:
        value = np.zeros(shape, dtype=np.dtype(dtype))
    cache = CachedDataset(ArrayDataset([value], TensorSpec(dtype, shape=shape, backend="numpy")))
    cache.compute(codec="numpy", managed=ManagedConfig(state_repo=Repo(DirStore(tmp_path / "store"))))
    np.testing.assert_equal(next(iter(cache)), value)


@pytest.mark.exhaustive_only
@pytest.mark.parametrize("backend", ("numpy", "torch", "tf", "jax"))
@pytest.mark.parametrize("dtype", ("bool", "int32", "uint64", "float32", "bfloat16", "complex64", "string"))
@pytest.mark.parametrize("shape", ((), (0,), (2,), (2, 1)))
def test_numpy_cache_exhaustive_framework_conversion_matrix(tmp_path, backend, dtype, shape):
    """Installed framework leaves retain every supported dtype/rank conversion case."""

    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    if dtype == "bool":
        value = np.zeros(shape, dtype=np.bool_)
    elif dtype == "string":
        value = np.full(shape, "x", dtype=np.str_)
    elif dtype == "bfloat16":
        import ml_dtypes

        value = np.zeros(shape, dtype=ml_dtypes.bfloat16)
    else:
        value = np.zeros(shape, dtype=np.dtype(dtype))
    if backend == "torch":
        if dtype == "string":
            pytest.skip("Torch has no Unicode tensor dtype.")
        torch = pytest.importorskip("torch")
        value = torch.as_tensor(value)
    elif backend == "tf":
        tensorflow = pytest.importorskip("tensorflow")
        try:
            value = tensorflow.convert_to_tensor(value)
        except (TypeError, ValueError):
            pytest.skip("TensorFlow does not support this dtype conversion.")
    elif backend == "jax":
        if dtype in {"string", "bfloat16"}:
            pytest.skip("JAX does not provide this portable dtype conversion.")
        jax_numpy = pytest.importorskip("jax.numpy")
        try:
            value = jax_numpy.asarray(value)
        except (TypeError, ValueError):
            pytest.skip("JAX does not support this dtype conversion.")
    cache = CachedDataset(ArrayDataset([value], TensorSpec(dtype, shape=shape, backend=backend)))
    cache.compute(codec="numpy", managed=ManagedConfig(state_repo=Repo(DirStore(tmp_path / "store"))))
    np.testing.assert_equal(next(iter(cache)), np.asarray(value))


def test_oversized_logical_yield_uses_bounded_physical_segments(tmp_path):
    """One large yield remains one result while its NPZ components stay bounded."""

    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    value = np.arange(6 * 1024 * 1024, dtype=np.uint8)
    cache = CachedDataset(ArrayDataset(
        [value], TensorSpec("uint8", shape=(6 * 1024 * 1024,), backend="numpy"),
    ))
    cache.compute(
        codec="numpy", target_chunk_bytes=1024 * 1024,
        managed=ManagedConfig(state_repo=Repo(DirStore(tmp_path / "store"))),
    )

    descriptor = cache._cache_payload["chunks"][0]
    assert descriptor["count"] == 1
    assert descriptor["segment_bytes"] == 1024 * 1024
    assert len(descriptor["entries"][0][0]["names"]) == 6
    np.testing.assert_equal(next(iter(cache)), value)
