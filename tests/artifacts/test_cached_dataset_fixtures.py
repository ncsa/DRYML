"""Direct-reader compatibility tests for committed CachedDataset v1 fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from dryml.core import Repo
from dryml.core.store.dir import DirStore
from dryml.data import Batch, Map, Select, Unbatch
from tests import cached_dataset_fixture_types as fixture_types


FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "cached_dataset" / "v1"


def _manifest() -> dict:
    """Read the committed fixture manifest without invoking its generator."""

    return json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="ascii"))


def _fresh_repo(tmp_path) -> tuple[Repo, dict]:
    """Copy committed authority bytes and open them through one fresh Repo."""

    manifest = _manifest()
    authority_root = FIXTURE_ROOT / "dir-store"
    actual = {
        path.relative_to(authority_root).as_posix()
        for path in authority_root.rglob("*")
        if path.is_file()
    }
    assert actual == set(manifest["authority_sha256"])
    for relative, expected in manifest["authority_sha256"].items():
        assert hashlib.sha256((authority_root / relative).read_bytes()).hexdigest() == expected
    destination = tmp_path / "store"
    shutil.copytree(FIXTURE_ROOT / "dir-store", destination)
    return Repo(DirStore.open_existing(destination, query_index="none")), manifest


def _load(repo: Repo, digest: str):
    """Restore one exact fixture state while forbidding source construction."""

    record = repo.default_store.read_state_ref_record(digest)
    assert record is not None
    before = fixture_types.CONSTRUCTION_COUNT
    fixture_types.REJECT_CONSTRUCTION = True
    try:
        cache = repo.load_state_ref(record.state_ref, reuse_live="never")
    finally:
        fixture_types.REJECT_CONSTRUCTION = False
    assert fixture_types.CONSTRUCTION_COUNT == before
    return cache


def _require_codec(codec: str) -> None:
    """Skip fixture reads when their qualified optional codec is unavailable."""

    if codec == "parquet":
        pytest.importorskip("pyarrow", minversion="25.0.1")
    elif codec == "netcdf":
        pytest.importorskip("netCDF4", minversion="1.7.4")


@pytest.mark.parametrize("codec", ("numpy", "parquet", "netcdf"))
def test_v1_fixtures_restore_in_a_fresh_repo_without_source_or_codec_input(
        tmp_path, codec):
    """Read every committed codec, nested value, and empty cache directly."""

    _require_codec(codec)
    repo, manifest = _fresh_repo(tmp_path)
    fidelity = _load(repo, manifest["states"][codec]["fidelity"])
    empty = _load(repo, manifest["states"][codec]["empty"])

    assert fidelity.ready and len(fidelity) == 2
    assert empty.ready and len(empty) == 0 and list(empty) == []
    first, second = list(fidelity)
    assert list(first) == ["bool", 2, False]
    np.testing.assert_equal(first["bool"], np.asarray([True, False]))
    assert first[2][0].item() == -7
    assert first[2][1][-1] == np.iinfo(np.uint64).max
    assert np.isnan(first[2][2][0]) and np.isposinf(first[2][2][1])
    assert np.signbit(first[2][2][2])
    assert first[False][1][0] == np.asarray(1 + 2j, dtype=np.complex128)
    assert first[False][1][1] == np.asarray(-3 + 0.5j, dtype=np.complex128)
    np.testing.assert_equal(first[False][2], np.asarray(["", "e\u0301", "a\x00b"]))
    assert second["bool"].shape == (0,)
    assert second[2][2].shape == (0,)
    assert second[False][0].shape == (0,)
    assert second[False][1].shape == (0,)
    assert second[False][2].item() == "caf\u00e9"
    assert len(list(fidelity)) == 2


def test_polynomial_numpy_and_parquet_fixtures_are_equivalent_and_reusable(
        tmp_path):
    """Reuse one exact saved NumPy cache across two ordinary workflow runs."""

    _require_codec("parquet")
    repo, manifest = _fresh_repo(tmp_path)
    numpy_digest = manifest["states"]["numpy"]["polynomial"]
    parquet_cache = _load(repo, manifest["states"]["parquet"]["polynomial"])
    numpy_cache = _load(repo, numpy_digest)
    numpy_values = list(numpy_cache)
    parquet_values = list(parquet_cache)

    assert len(numpy_values) == len(parquet_values) == 5
    for numpy_value, parquet_value in zip(numpy_values, parquet_values):
        np.testing.assert_equal(numpy_value["x"], parquet_value["x"])
        np.testing.assert_equal(numpy_value["y"], parquet_value["y"])
        x = numpy_value["x"][0]
        assert numpy_value["y"].item() == 1.25 - 0.5 * x + 2.0 * x * x

    def workflow_run():
        workflow_repo = Repo(DirStore.open_existing(
            tmp_path / "store", query_index="none",
        ))
        try:
            cache = _load(workflow_repo, numpy_digest)
            ordinary = Unbatch(Batch(Map(cache, Select("x")), 2))
            return np.stack(tuple(ordinary))
        finally:
            workflow_repo.close(flush=False)

    first = workflow_run()
    second = workflow_run()
    np.testing.assert_equal(first, second)
    np.testing.assert_equal(first, np.stack([value["x"] for value in numpy_values]))
