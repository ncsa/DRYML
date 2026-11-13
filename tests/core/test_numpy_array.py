"""Array payload ownership and exact SavedNode recovery stay outside definitions."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from dryml.core import AnyValue, Definition, Par, Repo
from dryml.core.dtype import DType
from dryml.core.repo import RepoLoadError, RepoSaveError
from dryml.core.store.dir import DirStore
from dryml.numpy import NumpyArray


pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")


@pytest.fixture
def repo(tmp_path, request):
    result = Repo(DirStore(tmp_path / "store", query_index=getattr(request, "param", "none")))
    yield result
    result.close(flush=False)


def test_assignment_owns_readonly_array_content_without_changing_definition(repo):
    points = NumpyArray(shape=(1024, 3), dtype="float32", repo=repo)
    definition = points.definition
    assert points.has_content
    assert points.shape == (1024, 3) and points.dtype == np.dtype("float32")
    np.testing.assert_array_equal(points.array, np.zeros((1024, 3), dtype="float32"))
    assert not points.array.flags.writeable
    source = np.arange(1024 * 3, dtype="float32").reshape(1024, 3)
    expected = source.copy()
    points.array = source
    source.fill(-1)
    assert points.has_content
    assert points.definition.recipe_equal(definition)
    assert dict(points.definition.parameters) == {"shape": (1024, 3), "dtype": "float32"}
    np.testing.assert_array_equal(points.array, expected)
    with pytest.raises(ValueError):
        points.array[0, 0] = 1
    with pytest.raises(ValueError):
        points.array.setflags(write=True)


def test_construction_and_save_use_existing_active_repo_without_constructor_repo_argument(repo):
    from dryml.core.session import config

    with config(repo=repo):
        points = NumpyArray(shape=(1, 3), dtype="float32")
        points.array = np.asarray([[1, 2, 3]], dtype="float32")
        saved = points.save()
    assert dict(saved.parameters) == {"shape": (1, 3), "dtype": "float32"}
    assert Path(repo.default_store.state_data_path(saved.state.key), "array.npy").is_file()
    np.testing.assert_array_equal(repo.load_state_ref(saved, reuse_live="never", cache="none").array, [[1, 2, 3]])


def test_shape_and_dtype_aliases_normalize_without_allocating_during_authoring(repo, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("metadata authoring allocated a zero array")
    monkeypatch.setattr(np, "zeros", forbidden)
    expected = Definition(NumpyArray, shape=(1024, 3), dtype="float32").concretize()
    for shape, dtype in (([1024, 3], np.float32), ((np.int64(1024), 3), np.dtype("f4")),
                         ((1024, 3), DType("float", 32))):
        actual = NumpyArray.defn(shape=shape, dtype=dtype).concretize()
        assert actual.recipe_equal(expected)
    template = NumpyArray.defn(shape=(Par("count"), 3), dtype=Par("kind"))
    assert template.sub(count=1024, kind="float32").concretize().recipe_equal(expected)
    vector = NumpyArray.defn(1024, "float32").concretize()
    assert vector.parameters["shape"] == (1024,)


@pytest.mark.parametrize("shape,dtype", (
    ((-1, 3), "float32"), ((True, 3), "float32"), ((2.5, 3), "float32"),
    ((1,), "object"), ((1,), np.dtype([("x", "float32")])),
    ((1,), np.dtype("float32", metadata={"units": "meters"})), ((1,), "U"),
))
def test_invalid_constructor_metadata_is_rejected_before_allocation(repo, monkeypatch, shape, dtype):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid metadata reached array allocation")
    monkeypatch.setattr(np, "zeros", forbidden)
    with pytest.raises((TypeError, ValueError)):
        NumpyArray(shape, dtype, repo=repo)


@pytest.mark.parametrize("repo", ("none", "sqlite"), indirect=True)
def test_shape_dtype_metadata_is_searchable_without_loading_array_content(repo, monkeypatch):
    for shape, dtype in (((1024, 3), "float32"), ((2048, 3), "float32"), ((1024, 3), "float64")):
        NumpyArray(shape, dtype, repo=repo).save(repo=repo)
    def forbidden(*args, **kwargs):
        pytest.fail("metadata search allocated or decoded array content")
    monkeypatch.setattr(np, "zeros", forbidden)
    monkeypatch.setattr(np.lib.format, "read_array", forbidden)
    query = repo.query().saved()
    assert query.sel(Definition(NumpyArray, shape=(1024, 3), dtype="float32")).count() == 1
    assert query.sel(Definition(NumpyArray, shape=(1024, 3))).collect().count() == 2
    assert query.sel(Definition(NumpyArray, dtype="float32")).count() == 2
    assert query.sel(NumpyArray.defn(shape=(AnyValue(), 3), dtype="float32")).count() == 2


@pytest.mark.parametrize("value", (
    np.arange(1024 * 3, dtype="float32").reshape(1024, 3),
    np.asfortranarray(np.arange(18, dtype="float64").reshape(6, 3)),
    np.array(-0.0, dtype=">f8"),
    np.empty((0, 3), dtype="int16"),
))
def test_saved_array_roundtrip_preserves_dtype_shape_bytes_and_receipt(repo, value):
    points = NumpyArray(shape=value.shape, dtype=value.dtype, repo=repo)
    points.array = value
    saved = points.save(repo=repo)
    assert saved.parameters["shape"] == value.shape
    assert set(saved.parameters) == {"shape", "dtype"}
    assert saved.state.key.codec_hash.startswith("npy1-")
    payload = Path(repo.default_store.state_data_path(saved.state.key))
    assert {path.name for path in payload.iterdir()} == {"array.npy"}
    restored = repo.load_state_ref(saved, reuse_live="never", cache="none")
    assert restored is not points and restored.has_content
    assert restored.object_id == points.object_id
    assert restored.last_state_ref.graph_equal(saved)
    assert restored.array.dtype == value.dtype
    assert restored.array.shape == value.shape
    assert restored.array.tobytes() == value.tobytes()
    assert restored.array.flags.f_contiguous == points.array.flags.f_contiguous
    assert not restored.array.flags.writeable


def test_replacement_clears_live_receipt_and_saved_revisions_remain_independent(repo):
    points = NumpyArray((1024, 3), "float32", repo=repo)
    points.array = np.ones((1024, 3), dtype="float32")
    first = points.save(repo=repo)
    view = points.array
    points.array = np.full((1024, 3), 2, dtype="float32")
    assert points.last_state_ref is None and points._last_state_hash is None
    original = repo.load_state_ref(first, cache="none")
    assert original is not points
    np.testing.assert_array_equal(original.array, np.ones((1024, 3), dtype="float32"))
    np.testing.assert_array_equal(view, original.array)
    assert points.array.shape == (1024, 3)
    second = points.save(repo=repo)
    assert first.creation_digest() == second.creation_digest()
    assert first.object_id == second.object_id
    assert first.state.key != second.state.key
    current = repo.load_state_ref(second, reuse_live="never", cache="none")
    assert current.array.dtype == np.dtype("float32")
    np.testing.assert_array_equal(current.array, np.full((1024, 3), 2))


@pytest.mark.parametrize("invalid", (
    [1, 2, 3],
    np.asarray([None], dtype=object),
    np.zeros(1, dtype=[("x", "float32")]),
    np.zeros(1, dtype=np.dtype("float32", metadata={"units": "meters"})),
))
def test_invalid_assignment_preserves_prior_content_and_saved_receipt(repo, invalid):
    points = NumpyArray((2,), "int32", repo=repo)
    points.array = np.asarray([3, 4], dtype="int32")
    previous = points.save(repo=repo)
    with pytest.raises(TypeError):
        points.array = invalid
    np.testing.assert_array_equal(points.array, [3, 4])
    assert points.last_state_ref.graph_equal(previous)


@pytest.mark.parametrize("replacement,error", (
    (np.zeros((3,), dtype="float32"), ValueError),
    (np.zeros((2,), dtype="float64"), TypeError),
))
def test_declared_shape_and_dtype_are_enforced_on_assignment(repo, replacement, error):
    points = NumpyArray((2,), "float32", repo=repo)
    previous = points.save(repo=repo)
    with pytest.raises(error):
        points.array = replacement
    np.testing.assert_array_equal(points.array, [0, 0])
    assert points.last_state_ref.graph_equal(previous)
    with pytest.raises(AttributeError):
        points.shape = (3,)
    with pytest.raises(AttributeError):
        points.dtype = np.dtype("float64")


def test_assignment_rejects_custom_dtype_that_npy_would_silently_reduce(repo):
    ml_dtypes = pytest.importorskip("ml_dtypes")
    points = NumpyArray((1,), "float32", repo=repo)
    with pytest.raises(TypeError, match="preserved"):
        points.array = np.asarray([1], dtype=ml_dtypes.bfloat16)
    assert points.has_content
    np.testing.assert_array_equal(points.array, [0])
    with pytest.raises(TypeError, match="preserved"):
        NumpyArray((1,), ml_dtypes.bfloat16, repo=repo)


def test_zero_initialization_can_be_saved_and_alternative_codec_fails_before_io(repo, tmp_path):
    points = NumpyArray((2, 3), "float32", repo=repo)
    saved = points.save(repo=repo)
    restored = repo.load_state_ref(saved, reuse_live="never", cache="none")
    np.testing.assert_array_equal(restored.array, np.zeros((2, 3), dtype="float32"))
    assert not repo.default_store.authority_cut().pending
    destination = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="npy1"):
        points.save_state_to_dir_imp(str(destination), codec="pkl")
    with pytest.raises(ValueError, match="npy1"):
        points.restore_state_from_dir_imp(str(destination), codec="pkl")
    assert not destination.exists()


@pytest.mark.parametrize("damage", ("truncated", "trailing", "object", "structured"))
def test_restore_hook_failure_preserves_existing_array(repo, tmp_path, damage):
    points = NumpyArray((3,), "int64", repo=repo)
    points.array = np.asarray([1, 2, 3], dtype="int64")
    previous = points.save(repo=repo)
    payload = tmp_path / "payload"
    payload.mkdir()
    path = payload / "array.npy"
    if damage == "object":
        np.save(path, np.asarray([None], dtype=object), allow_pickle=True)
    elif damage == "structured":
        np.save(path, np.zeros(1, dtype=[("x", "float32")]), allow_pickle=False)
    else:
        np.save(path, np.arange(4), allow_pickle=False)
        if damage == "truncated":
            path.write_bytes(path.read_bytes()[:12])
        else:
            with path.open("ab") as stream:
                stream.write(b"extra")
    with pytest.raises((ValueError, TypeError, EOFError)):
        points.restore_state_from_dir_imp(str(payload), codec="npy1")
    np.testing.assert_array_equal(points.array, [1, 2, 3])
    assert points.last_state_ref.graph_equal(previous)


@pytest.mark.parametrize("payload,error", (
    (np.zeros((4,), dtype="int64"), ValueError),
    (np.zeros((3,), dtype="float32"), TypeError),
))
def test_restore_rejects_payload_that_disagrees_with_constructor_metadata(repo, tmp_path, payload, error):
    points = NumpyArray((3,), "int64", repo=repo)
    points.array = np.asarray([1, 2, 3], dtype="int64")
    previous = points.save(repo=repo)
    folder = tmp_path / "mismatched"
    folder.mkdir()
    np.save(folder / "array.npy", payload, allow_pickle=False)
    with pytest.raises(error):
        points.restore_state_from_dir_imp(str(folder), codec="npy1")
    np.testing.assert_array_equal(points.array, [1, 2, 3])
    assert points.last_state_ref.graph_equal(previous)


def test_partial_write_aborts_without_changing_earlier_saved_content(repo, monkeypatch):
    points = NumpyArray((2,), "float32", repo=repo)
    points.array = np.asarray([1, 2], dtype="float32")
    first = points.save(repo=repo)
    points.array = np.asarray([3, 4], dtype="float32")
    def fail(stream, *args, **kwargs):
        stream.write(b"partial")
        raise OSError("controlled array write failure")
    with monkeypatch.context() as patch:
        patch.setattr(np, "save", fail)
        with pytest.raises(RepoSaveError) as failure:
            points.save(repo=repo)
        cause = failure.value
        while cause.__cause__ is not None:
            cause = cause.__cause__
        assert isinstance(cause, OSError) and str(cause) == "controlled array write failure"
    assert not repo.default_store.authority_cut().pending
    assert points.last_state_ref is None
    np.testing.assert_array_equal(repo.load_state_ref(first, cache="none").array, [1, 2])
    second = points.save(repo=repo)
    np.testing.assert_array_equal(repo.load_state_ref(second, reuse_live="never", cache="none").array, [3, 4])


def test_input_memmap_is_copied_without_retaining_its_file(repo, tmp_path):
    path = tmp_path / "sampled.npy"
    source = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(1024, 3))
    source.fill(2)
    points = NumpyArray((1024, 3), "float32", repo=repo)
    points.array = source
    del source
    path.unlink()
    saved = points.save(repo=repo)
    restored = repo.load_state_ref(saved, reuse_live="never", cache="none")
    assert type(restored.array) is np.ndarray
    np.testing.assert_array_equal(restored.array, np.full((1024, 3), 2))


def test_store_payload_corruption_is_detected_before_live_reuse(repo):
    points = NumpyArray((2,), "int32", repo=repo)
    points.array = np.asarray([1, 2], dtype="int32")
    saved = points.save(repo=repo)
    path = Path(repo.default_store.state_data_path(saved.state.key)) / "array.npy"
    path.write_bytes(b"truncated")
    with pytest.raises(RepoLoadError):
        repo.load_state_ref(saved)


def test_core_failed_restore_invalidates_live_access_and_fresh_load_recovers(repo, monkeypatch):
    points = NumpyArray((2,), "int32", repo=repo)
    points.array = np.asarray([1, 2], dtype="int32")
    saved = points.save(repo=repo)
    points.array = np.asarray([3, 4], dtype="int32")
    def fail(*args, **kwargs):
        raise ValueError("controlled decode failure")
    with monkeypatch.context() as patch:
        patch.setattr(np.lib.format, "read_array", fail)
        with pytest.raises(RepoLoadError):
            repo.restore_state_ref_into(points, saved)
    assert not points.has_content
    with pytest.raises(RuntimeError, match="invalidated"):
        _ = points.array
    with pytest.raises(RuntimeError, match="invalidated"):
        points.array = np.asarray([5, 6], dtype="int32")
    recovered = repo.load_state_ref(saved, reuse_live="never", cache="none")
    np.testing.assert_array_equal(recovered.array, [1, 2])


def test_numpy_adapters_keep_container_export_lazy_without_optional_frameworks():
    code = """
import sys
import dryml.numpy as backend
assert 'NumpyArray' in dir(backend)
assert 'dryml.numpy.array' not in sys.modules
from dryml.numpy import NumpyArray
from dryml.core import Serializable
assert issubclass(NumpyArray, Serializable)
assert not {'torch', 'tensorflow', 'jax', 'ray', 'sqlite3'}.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=15)
