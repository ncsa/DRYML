"""Focused NETCDF4 behavior for :class:`dryml.artifacts.CachedDataset`."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

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


def _require_supported_netcdf4():
    """Return netCDF4 only when this environment qualifies the codec contract."""

    netcdf4 = pytest.importorskip("netCDF4")
    from packaging.version import Version

    if Version(netcdf4.__version__) < Version("1.7.4"):
        pytest.skip("netCDF4 1.7.4 or newer is required for NetCDF qualification.")
    return netcdf4


def _repo(tmp_path, name="store"):
    """Create one local Store authority for a cache test."""

    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    return Repo(DirStore(tmp_path / name))


def _chunk_path(cache):
    """Return the first completed NetCDF chunk path from a test cache."""

    return os.path.join(
        cache._cache_payload["source_dir"], "chunks", "chunk-00000000.nc"
    )


def test_netcdf_missing_dependency_fails_before_source_traversal(tmp_path, monkeypatch):
    """Selecting NetCDF admits its optional runtime before opening the source."""

    from dryml.artifacts import _cache_netcdf

    source = ArrayDataset(
        [np.asarray([1], dtype=np.int32)],
        TensorSpec("int32", shape=(1,), backend="numpy"),
    )
    monkeypatch.setattr(
        _cache_netcdf,
        "load_netcdf4",
        lambda: (_ for _ in ()).throw(
            ImportError("CachedDataset NetCDF codec requires netCDF4>=1.7.4")
        ),
    )

    with pytest.raises(ImportError, match="netCDF4>=1.7.4"):
        CachedDataset(source).compute(
            codec="netcdf", managed=ManagedConfig(state_repo=_repo(tmp_path))
        )
    assert source.opens == 0


def test_netcdf_cache_round_trips_nested_dynamic_typed_values(tmp_path):
    """Native and explicit-bit component streams preserve representative values."""

    _require_supported_netcdf4()
    spec = {
        "unsigned": TensorSpec("uint64", shape=(Dynamic,), backend="numpy"),
        "nested": [
            TensorSpec("float16", shape=(2,), backend="numpy"),
            TensorSpec("complex128", shape=(), backend="numpy"),
            TensorSpec("string", shape=(Dynamic,), backend="numpy"),
        ],
    }
    value = {
        "unsigned": np.asarray([np.iinfo(np.uint64).max], dtype=np.uint64),
        "nested": [
            np.asarray([np.inf, -0.0], dtype=np.float16),
            np.asarray(complex(np.nan, -np.inf), dtype=np.complex128),
            np.asarray(["", "e\u0301", "a\x00b"]),
        ],
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
        codec="netcdf",
        target_chunk_bytes=1,
        managed=ManagedConfig(state_repo=repo),
    )
    restored = next(iter(repo.load_state_ref(state, reuse_live="never")))

    np.testing.assert_equal(restored["unsigned"], value["unsigned"])
    np.testing.assert_equal(restored["nested"][0], value["nested"][0])
    np.testing.assert_equal(restored["nested"][1], value["nested"][1])
    np.testing.assert_equal(restored["nested"][2], value["nested"][2])
    if "bfloat" in value:
        np.testing.assert_equal(restored["bfloat"], value["bfloat"])
    source.opens = 0
    cache.compute(codec="netcdf", managed=ManagedConfig(state_repo=repo, rerun=True))
    assert source.opens == 0


def test_netcdf_file_uses_flat_native_nonlossy_component_streams(tmp_path):
    """The physical file is NETCDF4 with fixed typed streams and no lossy controls."""

    netcdf4 = _require_supported_netcdf4()
    spec = (
        TensorSpec("uint64", shape=(1,), backend="numpy"),
        TensorSpec("complex64", shape=(1,), backend="numpy"),
        TensorSpec("string", shape=(1,), backend="numpy"),
    )
    cache = CachedDataset(
        ArrayDataset(
            [
                (
                    np.asarray([2**63 + 1], dtype=np.uint64),
                    np.asarray([1 + 2j], dtype=np.complex64),
                    np.asarray(["text"]),
                )
            ],
            spec,
        )
    )
    cache.compute(codec="netcdf", managed=ManagedConfig(state_repo=_repo(tmp_path)))

    with netcdf4.Dataset(_chunk_path(cache)) as dataset:
        assert dataset.data_model == dataset.file_format == "NETCDF4"
        assert all(
            not dimension.isunlimited() for dimension in dataset.dimensions.values()
        )
        names = set(dataset.variables)
        assert any(
            "_cvalue_" in name and dataset.variables[name].dtype == np.dtype("uint64")
            for name in names
        )
        assert any(
            "_creal_" in name and dataset.variables[name].dtype == np.dtype("float32")
            for name in names
        )
        assert any(
            "_cimaginary_" in name
            and dataset.variables[name].dtype == np.dtype("float32")
            for name in names
        )
        assert any(
            "_cutf8_" in name and dataset.variables[name].dtype == np.dtype("uint8")
            for name in names
        )
        assert any(
            "_cstring_offsets_" in name
            and dataset.variables[name].dtype == np.dtype("uint64")
            for name in names
        )
        for variable in dataset.variables.values():
            assert variable.ncattrs() == []
            assert variable.quantization() is None
            assert not any(
                bool(value)
                for key, value in variable.filters().items()
                if key != "complevel"
            )


def test_netcdf_zero_sized_tensor_dimensions_are_metadata(tmp_path):
    """Zero tensor axes survive without creating zero/unlimited NetCDF dimensions."""

    netcdf4 = _require_supported_netcdf4()
    value = np.empty((0, 3), dtype=np.float32)
    cache = CachedDataset(
        ArrayDataset(
            [value],
            TensorSpec("float32", shape=(0, 3), backend="numpy"),
        )
    )
    cache.compute(codec="netcdf", managed=ManagedConfig(state_repo=_repo(tmp_path)))

    with netcdf4.Dataset(_chunk_path(cache)) as dataset:
        assert all(
            len(dimension) > 0 and not dimension.isunlimited()
            for dimension in dataset.dimensions.values()
        )
        shape_variables = [
            variable
            for name, variable in dataset.variables.items()
            if "shape_values" in name
        ]
        assert any(0 in variable[:] for variable in shape_variables)
    assert next(iter(cache)).shape == (0, 3)


def test_netcdf_schema_corruption_fails_before_first_chunk_yield(tmp_path):
    """An authenticated impossible stream length exposes no chunk values."""

    netcdf4 = _require_supported_netcdf4()
    cache = CachedDataset(
        ArrayDataset(
            [np.asarray([1], dtype=np.int32), np.asarray([2], dtype=np.int32)],
            TensorSpec("int32", shape=(1,), backend="numpy"),
        )
    )
    cache.compute(codec="netcdf", managed=ManagedConfig(state_repo=_repo(tmp_path)))
    path = _chunk_path(cache)
    with netcdf4.Dataset(path, "r+") as dataset:
        name = next(name for name in dataset.variables if "shape_offsets" in name)
        dataset.variables[name][-1] = np.iinfo(np.uint64).max
    _refresh_descriptor(cache, path)

    with pytest.raises(CacheIntegrityError, match="length|offset"):
        next(iter(cache))


def test_netcdf_resume_tail_cleanup_uses_nc_chunk_sequence(tmp_path):
    """Resume cleanup removes only contiguous unassociated ``.nc`` chunks."""

    from dryml.artifacts.dataset import _discard_unassociated_tail

    workdir = tmp_path / "work"
    chunks = workdir / "data" / "chunks"
    chunks.mkdir(parents=True)
    (chunks / "chunk-00000000.nc").write_bytes(b"associated")
    (chunks / "chunk-00000001.nc").write_bytes(b"tail")
    (chunks / "chunk-00000001.parquet").write_bytes(b"preserve")

    _discard_unassociated_tail(
        os.fspath(workdir),
        [{"file": "chunk-00000000.nc"}],
        "netcdf",
    )

    assert (chunks / "chunk-00000000.nc").exists()
    assert not (chunks / "chunk-00000001.nc").exists()
    assert (chunks / "chunk-00000001.parquet").exists()


def test_netcdf_native_access_uses_one_reentrant_process_lock(tmp_path, monkeypatch):
    """Concurrent codec actions open native files only while owning the shared RLock."""

    _require_supported_netcdf4()
    from dryml.artifacts import _cache_netcdf

    module = _cache_netcdf.load_netcdf4()
    original = module.Dataset
    observed = []

    def locked_dataset(*args, **kwargs):
        observed.append(_cache_netcdf._NETCDF_LOCK._is_owned())
        with _cache_netcdf._NETCDF_LOCK:
            return original(*args, **kwargs)

    monkeypatch.setattr(module, "Dataset", locked_dataset)

    def build(index):
        cache = CachedDataset(
            ArrayDataset(
                [np.asarray([index], dtype=np.int64)],
                TensorSpec("int64", shape=(1,), backend="numpy"),
            )
        )
        cache.compute(
            codec="netcdf",
            managed=ManagedConfig(state_repo=_repo(tmp_path, f"store-{index}")),
        )
        return next(iter(cache)).item()

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(build, (1, 2))) == [1, 2]
    assert observed and all(observed)


def test_netcdf_import_isolated_from_public_numpy_and_parquet_paths(tmp_path):
    """Public, NumPy, and Parquet paths leave the optional native module unloaded."""

    probe = tmp_path / "netcdf_import_probe.py"
    probe.write_text(
        """import sys
import numpy as np
from dryml.artifacts import CachedDataset, _cache_parquet
from dryml.core import Repo
from dryml.core.cardinality import Cardinality
from dryml.core.store.dir import DirStore
from dryml.core.tensor_spec import TensorSpec
from dryml.data import Dataset
from dryml.managed import ManagedConfig

assert 'netCDF4' not in sys.modules

class Source(Dataset):
    def __init__(self):
        super().__init__(TensorSpec('int32', shape=(1,), backend='numpy'))
    def __iter__(self):
        return iter((np.asarray([1], dtype=np.int32),))
    def __len__(self):
        return Cardinality.finite(1)

CachedDataset(Source()).compute(codec='numpy', managed=ManagedConfig(state_repo=Repo(DirStore(sys.argv[1]))))
assert 'netCDF4' not in sys.modules
try:
    import pyarrow
except ImportError:
    pass
else:
    from packaging.version import Version
    if Version(pyarrow.__version__) >= Version('25.0.1'):
        CachedDataset(Source()).compute(codec='parquet', managed=ManagedConfig(state_repo=Repo(DirStore(sys.argv[2]))))
assert _cache_parquet is not None
assert 'netCDF4' not in sys.modules
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(probe),
            str(tmp_path / "numpy-store"),
            str(tmp_path / "parquet-store"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.exhaustive_only
@pytest.mark.parametrize(
    "dtype",
    (
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "complex64",
        "complex128",
        "string",
    ),
)
@pytest.mark.parametrize("shape", ((), (0,), (2,), (2, 1)))
@pytest.mark.parametrize("tree", ("leaf", "dict-list", "tuple-dict"))
def test_netcdf_exhaustive_dtype_rank_and_tree_matrix(tmp_path, dtype, shape, tree):
    """Every required NetCDF dtype/rank/tree product remains exhaustive-only."""

    _require_supported_netcdf4()
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
    cache.compute(codec="netcdf", managed=ManagedConfig(state_repo=_repo(tmp_path)))
    restored = next(iter(cache))
    while isinstance(restored, (dict, tuple, list)):
        restored = restored["leaf"] if isinstance(restored, dict) else restored[0]
    expected = (
        value
        if tree == "leaf"
        else value["leaf"][0] if tree == "dict-list" else value[0]["leaf"]
    )
    np.testing.assert_equal(restored, expected)


@pytest.mark.exhaustive_only
@pytest.mark.parametrize(
    "mutation",
    (
        "missing-variable",
        "extra-variable",
        "wrong-dtype",
        "wrong-dimension",
        "bad-offset",
        "masked",
        "bad-utf8",
        "yield-count",
    ),
)
def test_netcdf_exhaustive_malformed_physical_layouts_fail_before_yield(
    tmp_path, mutation
):
    """Malformed NetCDF variable, metadata, and content products stay exhaustive-only."""

    netcdf4 = _require_supported_netcdf4()
    string_case = mutation == "bad-utf8"
    value = np.asarray(["text"]) if string_case else np.asarray([1], dtype=np.int32)
    dtype = "string" if string_case else "int32"
    cache = CachedDataset(
        ArrayDataset([value], TensorSpec(dtype, shape=(1,), backend="numpy"))
    )
    cache.compute(codec="netcdf", managed=ManagedConfig(state_repo=_repo(tmp_path)))
    path = _chunk_path(cache)
    with netcdf4.Dataset(path, "r+") as dataset:
        component = next(
            name
            for name in dataset.variables
            if name.startswith("component_") and "string_offsets" not in name
        )
        if mutation == "missing-variable":
            dataset.renameVariable(component, f"{component}_missing")
        elif mutation == "extra-variable":
            dataset.createDimension("extra_dim", 1)
            dataset.createVariable("extra", "u1", ("extra_dim",), fill_value=False)[
                :
            ] = [0]
        elif mutation == "wrong-dtype":
            original = dataset.variables[component]
            dimension = original.dimensions
            dataset.renameVariable(component, f"{component}_wrong")
            dataset.createVariable(component, "i8", dimension, fill_value=False)[:] = [
                0
            ]
        elif mutation == "wrong-dimension":
            original = dataset.variables[component]
            dataset.renameVariable(component, f"{component}_wrong")
            dataset.createDimension("wrong_dim", 1)
            dataset.createVariable(
                component, original.dtype, ("wrong_dim",), fill_value=False
            )[:] = [0]
        elif mutation == "bad-offset":
            name = next(name for name in dataset.variables if "value_offsets" in name)
            dataset.variables[name][0] = 1
        elif mutation == "masked":
            dataset.variables[component].setncattr("missing_value", np.int32(1))
        elif mutation == "bad-utf8":
            dataset.variables[component][0] = 255
        elif mutation == "yield-count":
            dataset.setncattr("yield_count", np.int64(2))
        else:  # pragma: no cover - the parameter set above is closed.
            raise AssertionError(mutation)
    _refresh_descriptor(cache, path)

    with pytest.raises(CacheIntegrityError):
        next(iter(cache))


def _refresh_descriptor(cache, path):
    """Update test manifest integrity after a controlled physical rewrite."""

    descriptor = cache._cache_payload["chunks"][0]
    descriptor["size"] = os.path.getsize(path)
    descriptor["sha256"] = _sha256(path)


def _sha256(path):
    """Return one test chunk digest after a controlled physical rewrite."""

    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
