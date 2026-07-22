from __future__ import annotations

import json

import numpy as np
import pytest

from dryml.artifacts.representations.numpy_sequence import (
    NumpySequenceCorruptError,
    iter_numpy_sequence,
    write_numpy_sequence,
)


def _canonical(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_canonical(item) for item in value)
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def test_nested_tensor_tree_round_trips_in_order(tmp_path):
    rows = [
        {
            "features": np.array([index, index + 1], dtype=np.float32),
            "target": (np.int64(index % 2), np.array(index, dtype=np.int32)),
        }
        for index in range(7)
    ]

    index = write_numpy_sequence(rows, tmp_path, shard_rows=3, shard_bytes=4096)

    assert index.count == 7
    assert [shard.stop - shard.start for shard in index.shards] == [3, 3, 1]
    assert [_canonical(item) for item in iter_numpy_sequence(tmp_path)] == [
        _canonical(item) for item in rows
    ]


def test_shard_rows_and_bytes_bound_files_not_elements(tmp_path):
    rows = [np.full((64,), index, dtype=np.int32) for index in range(25)]

    index = write_numpy_sequence(rows, tmp_path, shard_rows=6, shard_bytes=1800)

    payload_files = sorted(path for path in tmp_path.rglob("*") if path.is_file())
    assert len(index.shards) < len(rows)
    assert len(payload_files) == len(index.shards) + 1
    assert all(shard.stop - shard.start <= 6 for shard in index.shards)
    assert all(shard.size <= 1800 for shard in index.shards)
    assert [_canonical(item) for item in iter_numpy_sequence(tmp_path)] == [
        _canonical(item) for item in rows
    ]


@pytest.mark.parametrize("damage", ["missing", "corrupt", "index"])
def test_missing_or_corrupt_shard_and_manifest_are_rejected(tmp_path, damage):
    write_numpy_sequence(
        [np.array([index], dtype=np.int32) for index in range(5)],
        tmp_path,
        shard_rows=2,
    )
    index_path = tmp_path / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_path = tmp_path / index["shards"][0]["path"]
    if damage == "missing":
        shard_path.unlink()
    elif damage == "corrupt":
        shard_path.write_bytes(b"not-an-npz")
    else:
        index["count"] += 1
        index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(NumpySequenceCorruptError):
        list(iter_numpy_sequence(tmp_path))


def test_empty_sequence_has_compact_index_and_no_shards(tmp_path):
    index = write_numpy_sequence([], tmp_path)

    assert index.count == 0
    assert index.shards == ()
    assert [path.name for path in tmp_path.iterdir()] == ["index.json"]
    assert list(iter_numpy_sequence(tmp_path)) == []
