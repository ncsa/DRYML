"""Focused typed Parquet behavior for :class:`dryml.artifacts.CachedDataset`."""

from __future__ import annotations

import os
import hashlib
import subprocess
import sys

import numpy as np
import pytest

from dryml.artifacts import CacheIntegrityError, CachedDataset
from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import Dynamic, TensorSpec
from dryml.data import Dataset
from dryml.managed import ManagedConfig


class ArrayDataset(Dataset):
    """Small re-iterable NumPy source for physical-cache tests."""

    def __init__(self, values, spec):
        self.values = list(values)
        self.opens = 0
        super().__init__(spec)

    def __iter__(self):
        self.opens += 1
        return iter(self.values)

    def __len__(self):
        return Cardinality.finite(len(self.values))


def _require_supported_pyarrow():
    """Return PyArrow only when this environment qualifies the Parquet contract."""

    pyarrow = pytest.importorskip("pyarrow")
    from packaging.version import Version

    if Version(pyarrow.__version__) < Version("25.0.1"):
        pytest.skip("PyArrow 25.0.1 or newer is required for Parquet qualification.")
    return pyarrow


def _repo(tmp_path):
    """Create the local Store authority used by one cache test."""

    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    return Repo(DirStore(tmp_path / "store"))


def _chunk_path(cache):
    """Return the one completed Parquet chunk path from a test cache."""

    return os.path.join(cache._cache_payload["source_dir"], "chunks", "chunk-00000000.parquet")


def test_parquet_missing_dependency_fails_before_source_traversal(tmp_path, monkeypatch):
    """Selecting Parquet checks its optional dependency before opening the source."""

    from dryml.artifacts import _cache_parquet

    source = ArrayDataset([np.asarray([1], dtype=np.int32)], TensorSpec("int32", shape=(1,), backend="numpy"))
    monkeypatch.setattr(
        _cache_parquet,
        "load_pyarrow",
        lambda: (_ for _ in ()).throw(ImportError("CachedDataset Parquet codec requires pyarrow>=25.0.1")),
    )

    with pytest.raises(ImportError, match="pyarrow>=25.0.1"):
        CachedDataset(source).compute(codec="parquet", managed=ManagedConfig(state_repo=_repo(tmp_path)))
    assert source.opens == 0


def test_parquet_cache_round_trips_nested_dynamic_typed_values(tmp_path):
    """Typed Parquet components preserve nested dynamic values and special dtypes."""

    _require_supported_pyarrow()
    spec = {
        "unsigned": TensorSpec("uint64", shape=(Dynamic,), backend="numpy"),
        "nested": [
            TensorSpec("complex64", shape=(), backend="numpy"),
            TensorSpec("string", shape=(Dynamic,), backend="numpy"),
        ],
    }
    value = {
        "unsigned": np.asarray([np.iinfo(np.uint64).max], dtype=np.uint64),
        "nested": [np.asarray(1 + 2j, dtype=np.complex64), np.asarray(["", "e\u0301", "a\x00b"])],
    }
    try:
        import ml_dtypes
    except ImportError:
        pass
    else:
        spec["bfloat"] = TensorSpec("bfloat16", shape=(1,), backend="numpy")
        value["bfloat"] = np.asarray([1.5], dtype=ml_dtypes.bfloat16)

    repo = _repo(tmp_path)
    source = ArrayDataset([value], spec)
    cache = CachedDataset(source)
    state = cache.compute(
        codec="parquet", target_chunk_bytes=1,
        managed=ManagedConfig(state_repo=repo),
    )
    restored = next(iter(repo.load_state_ref(state, reuse_live="never")))

    np.testing.assert_equal(restored["unsigned"], value["unsigned"])
    np.testing.assert_equal(restored["nested"][0], value["nested"][0])
    np.testing.assert_equal(restored["nested"][1], value["nested"][1])
    if "bfloat" in value:
        np.testing.assert_equal(restored["bfloat"], value["bfloat"])
    source.opens = 0
    cache.compute(codec="parquet", managed=ManagedConfig(state_repo=repo, rerun=True))
    assert source.opens == 0


def test_parquet_schema_contains_typed_value_shape_and_complex_columns(tmp_path):
    """The physical schema exposes typed rows, shapes, and paired complex values."""

    pyarrow = _require_supported_pyarrow()
    import pyarrow.parquet as pq

    spec = (TensorSpec("complex128", shape=(2,), backend="numpy"), TensorSpec("uint64", shape=(1,), backend="numpy"))
    cache = CachedDataset(ArrayDataset([
        (np.asarray([1 + 2j, 3 + 4j], dtype=np.complex128), np.asarray([4], dtype=np.uint64)),
    ], spec))
    cache.compute(codec="parquet", managed=ManagedConfig(state_repo=_repo(tmp_path)))

    schema = pq.ParquetFile(_chunk_path(cache)).schema_arrow
    assert schema.get_field_index("shape") >= 0
    assert schema.get_field_index("values_float64") >= 0
    assert schema.get_field_index("values_uint64") >= 0
    assert pyarrow.types.is_list(schema.field("values_float64").type)
    assert schema.field("values_float64").type.value_type == pyarrow.float64()
    assert all(field.type != pyarrow.binary() for field in schema)


def test_parquet_reads_multiple_bounded_batches_and_chunks(tmp_path, monkeypatch):
    """Reading uses real ``iter_batches`` calls across bounded row groups and chunks."""

    _require_supported_pyarrow()
    from dryml.artifacts import _cache_parquet

    original = _cache_parquet.pq.ParquetFile.iter_batches
    calls = []

    def observed_iter_batches(self, *args, **kwargs):
        size = kwargs.get("batch_size", args[0] if args else None)
        for batch in original(self, *args, **kwargs):
            calls.append(size)
            yield batch

    monkeypatch.setattr(_cache_parquet.pq.ParquetFile, "iter_batches", observed_iter_batches)
    values = [np.arange(3 * 1024 * 1024, dtype=np.uint8) + index for index in range(2)]
    cache = CachedDataset(ArrayDataset(values, TensorSpec("uint8", shape=(3 * 1024 * 1024,), backend="numpy")))
    cache.compute(codec="parquet", target_chunk_bytes=1024 * 1024, managed=ManagedConfig(state_repo=_repo(tmp_path)))

    assert len(cache._cache_payload["chunks"]) == 2
    assert len(list(cache)) == 2
    assert len(calls) == 6 and all(type(size) is int and size > 0 for size in calls)


def test_parquet_corruption_fails_before_chunk_yield(tmp_path):
    """Deferred chunk integrity rejects altered bytes before a logical value escapes."""

    _require_supported_pyarrow()
    cache = CachedDataset(ArrayDataset(
        [np.asarray([1], dtype=np.int32)], TensorSpec("int32", shape=(1,), backend="numpy"),
    ))
    cache.compute(codec="parquet", managed=ManagedConfig(state_repo=_repo(tmp_path)))
    with open(_chunk_path(cache), "r+b") as output:
        output.seek(0)
        output.write(b"corrupt")

    with pytest.raises(CacheIntegrityError):
        next(iter(cache))


def test_parquet_distinguishes_empty_cache_container_and_tensor(tmp_path):
    """Empty source, yielded empty tree, and zero-sized tensor remain distinct."""

    _require_supported_pyarrow()
    repo = _repo(tmp_path)
    empty = CachedDataset(ArrayDataset([], TensorSpec("int32", shape=(1,), backend="numpy")))
    empty.compute(codec="parquet", managed=ManagedConfig(state_repo=repo))
    assert len(empty) == 0

    containers = CachedDataset(ArrayDataset([()], ()))
    containers.compute(codec="parquet", managed=ManagedConfig(state_repo=repo))
    assert list(containers) == [()]

    zeros = CachedDataset(ArrayDataset(
        [np.asarray([], dtype=np.float32)], TensorSpec("float32", shape=(0,), backend="numpy"),
    ))
    zeros.compute(codec="parquet", managed=ManagedConfig(state_repo=repo))
    assert next(iter(zeros)).shape == (0,)


def test_parquet_oversized_yield_segments_and_reconstructs_once(tmp_path):
    """A large logical tensor spans physical rows but produces one exact yield."""

    _require_supported_pyarrow()
    import pyarrow.parquet as pq

    value = np.arange(3 * 1024 * 1024, dtype=np.uint8)
    cache = CachedDataset(ArrayDataset(
        [value], TensorSpec("uint8", shape=value.shape, backend="numpy"),
    ))
    cache.compute(codec="parquet", target_chunk_bytes=1024 * 1024, managed=ManagedConfig(state_repo=_repo(tmp_path)))

    assert pq.ParquetFile(_chunk_path(cache)).metadata.num_rows == 3
    np.testing.assert_equal(next(iter(cache)), value)


def test_parquet_import_isolated_from_public_import_and_numpy_cache(tmp_path):
    """Public imports and NumPy caching leave the Parquet dependency unloaded."""

    result = subprocess.run(
        [sys.executable, "-c", "import sys; import dryml.artifacts; assert 'pyarrow' not in sys.modules"],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    probe = tmp_path / "numpy_cache_probe.py"
    probe.write_text("""import sys
import numpy as np
from dryml.artifacts import CachedDataset
from dryml.core import Repo
from dryml.core.cardinality import Cardinality
from dryml.core.store.dir import DirStore
from dryml.core.tensor_spec import TensorSpec
from dryml.data import Dataset
from dryml.managed import ManagedConfig

class Source(Dataset):
    def __init__(self):
        super().__init__(TensorSpec('int32', shape=(1,), backend='numpy'))
    def __iter__(self):
        return iter((np.asarray([1], dtype=np.int32),))
    def __len__(self):
        return Cardinality.finite(1)

CachedDataset(Source()).compute(codec='numpy', managed=ManagedConfig(state_repo=Repo(DirStore(sys.argv[1]))))
assert 'pyarrow' not in sys.modules
""", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(probe), str(tmp_path / "numpy-store")],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.exhaustive_only
@pytest.mark.parametrize("dtype", ("bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64", "bfloat16", "complex64", "complex128", "string"))
@pytest.mark.parametrize("shape", ((), (0,), (2,), (2, 1)))
@pytest.mark.parametrize("tree", ("leaf", "dict-list", "tuple-dict"))
def test_parquet_exhaustive_dtype_rank_and_tree_matrix(tmp_path, dtype, shape, tree):
    """Every supported Parquet dtype/rank/tree product remains exhaustive-only."""

    _require_supported_pyarrow()
    if dtype == "string":
        value = np.full(shape, "x", dtype=np.str_)
    elif dtype == "bfloat16":
        import ml_dtypes

        value = np.zeros(shape, dtype=ml_dtypes.bfloat16)
    else:
        value = np.zeros(shape, dtype=np.dtype(dtype))
    leaf = TensorSpec(dtype, shape=shape, backend="numpy")
    if tree == "dict-list":
        spec, value = {"leaf": [leaf]}, {"leaf": [value]}
    elif tree == "tuple-dict":
        spec, value = ({"leaf": leaf},), ({"leaf": value},)
    else:
        spec = leaf
    cache = CachedDataset(ArrayDataset([value], spec))
    cache.compute(codec="parquet", managed=ManagedConfig(state_repo=_repo(tmp_path)))
    restored = next(iter(cache))
    while isinstance(restored, (dict, tuple, list)):
        restored = restored["leaf"] if isinstance(restored, dict) else restored[0]
    np.testing.assert_equal(restored, value if tree == "leaf" else value["leaf"][0] if tree == "dict-list" else value[0]["leaf"])


@pytest.mark.exhaustive_only
@pytest.mark.parametrize("mutation", ("null", "wrong-type", "missing-column", "extra-column", "bad-offset", "row-count", "metadata", "missing-segment", "duplicate-segment", "reordered-segment"))
def test_parquet_exhaustive_malformed_physical_layouts_fail_before_yield(tmp_path, mutation):
    """Malformed Parquet schema and segment products remain exhaustive-only tests."""

    _require_supported_pyarrow()
    import pyarrow as pa
    import pyarrow.parquet as pq

    value = np.arange(512 * 1024, dtype=np.int32)
    cache = CachedDataset(ArrayDataset(
        [value], TensorSpec("int32", shape=value.shape, backend="numpy"),
    ))
    cache.compute(codec="parquet", target_chunk_bytes=1024 * 1024, managed=ManagedConfig(state_repo=_repo(tmp_path)))
    path = _chunk_path(cache)
    table = pq.read_table(path)
    if mutation == "null":
        field = table.schema.field("values_int32")
        table = table.set_column(table.schema.get_field_index(field.name), field, pa.array([None] * len(table), type=field.type))
    elif mutation == "wrong-type":
        field = pa.field("values_int32", pa.list_(pa.int64()))
        table = table.set_column(table.schema.get_field_index(field.name), field, pa.array([[1]] * len(table), type=field.type))
    elif mutation == "missing-column":
        table = table.remove_column(table.schema.get_field_index("shape"))
    elif mutation == "extra-column":
        table = table.append_column("unexpected", pa.array([0] * len(table), type=pa.int8()))
    elif mutation == "bad-offset":
        field = table.schema.field("start")
        table = table.set_column(table.schema.get_field_index("start"), field, pa.array([1] * len(table), type=field.type))
    elif mutation in {"row-count", "missing-segment"}:
        table = table.slice(0, 1)
    elif mutation == "duplicate-segment":
        table = pa.concat_tables((table, table.slice(0, 1)))
    elif mutation == "reordered-segment":
        table = table.take(pa.array(list(reversed(range(len(table)))), type=pa.int64()))
    elif mutation == "metadata":
        table = table.replace_schema_metadata({b"dryml.cached-dataset.parquet": b"corrupt"})
    else:  # pragma: no cover - the parameter list is closed above.
        raise AssertionError(mutation)
    pq.write_table(table, path, compression="NONE")
    descriptor = cache._cache_payload["chunks"][0]
    descriptor["size"] = os.path.getsize(path)
    descriptor["sha256"] = _sha256(path)

    with pytest.raises(CacheIntegrityError):
        next(iter(cache))


def _sha256(path):
    """Return one test chunk digest after a controlled physical rewrite."""

    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
