from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from dryml.artifacts.representations.parquet import (
    ParquetCompatibilityError,
    iter_parquet_sequence,
    read_parquet_index,
    write_parquet_sequence,
)


pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def test_partitioned_parquet_preserves_schema_and_row_order(tmp_path):
    rows = [np.array([index, index + 10], dtype=np.int32) for index in range(7)]

    index = write_parquet_sequence(rows, tmp_path, partition_rows=3)

    assert index.count == 7
    assert [partition.stop - partition.start for partition in index.partitions] == [3, 3, 1]
    assert [partition.path for partition in index.partitions] == [
        "partitions/00000000.parquet",
        "partitions/00000001.parquet",
        "partitions/00000002.parquet",
    ]
    first = pq.read_table(tmp_path / index.partitions[0].path)
    assert first.column_names == ["value_000000", "value_000001"]
    assert first.schema.field("value_000000").type == pa.int32()
    assert [value.tolist() for value in iter_parquet_sequence(tmp_path)] == [
        value.tolist() for value in rows
    ]
    assert read_parquet_index(tmp_path) == index


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([np.zeros((2, 2), dtype=np.float32)], "scalar or one-dimensional"),
        ([np.zeros((2,), dtype=np.float32), np.zeros((3,), dtype=np.float32)], "stable"),
        ([{"x": np.array(1)}], "nested"),
        ([np.empty((0,), dtype=np.float32)], "at least one column"),
    ],
)
def test_parquet_rejects_non_tabular_shapes(tmp_path, rows, message):
    with pytest.raises(ParquetCompatibilityError, match=message):
        write_parquet_sequence(rows, tmp_path, partition_rows=2)


def test_representation_import_does_not_load_optional_frameworks():
    source = """
import sys
import dryml.artifacts
import dryml.artifacts.representations.parquet
heavy = {'pyarrow', 'tensorflow', 'torch'}
assert not ({name.split('.', 1)[0] for name in sys.modules} & heavy)
"""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
