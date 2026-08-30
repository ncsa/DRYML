import os
from pathlib import Path
import subprocess
import sys

import pytest

from dryml.core.utils.graph import (
    Arg,
    GraphPath,
    Index,
    Key,
    Kwarg,
    Parameter,
    SetMember,
)
from dryml.core.utils.graph.path import (
    QueryPathError,
    canonical_key_bytes,
    graph_path_sort_key,
)


def test_tagged_path_bytes_distinguish_every_segment_kind():
    """The authority order must not collapse lookalike segment payloads."""

    paths = [
        GraphPath((Parameter("x"),)),
        GraphPath((Kwarg("x"),)),
        GraphPath((Arg(0),)),
        GraphPath((Index(0),)),
        GraphPath((Key(0),)),
        GraphPath((Key("0"),)),
        GraphPath((SetMember("x", 0),)),
    ]

    assert len({path.to_bytes() for path in paths}) == len(paths)
    assert sorted(paths, key=graph_path_sort_key) == sorted(
        paths, key=lambda path: path.to_bytes()
    )


def test_mapping_key_order_is_typed_and_process_deterministic():
    """Heterogeneous canonical keys use bytes rather than Python comparison."""

    local = [
        canonical_key_bytes(key).hex()
        for key in sorted(("2", 2, -1, "a"), key=canonical_key_bytes)
    ]
    code = """
from dryml.core.utils.graph.path import canonical_key_bytes
print(','.join(canonical_key_bytes(key).hex() for key in sorted(('2', 2, -1, 'a'), key=canonical_key_bytes)))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[2],
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).parents[2] / "src"),
        },
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.strip().split(",") == local


def test_path_bytes_reject_noncanonical_mapping_key_types():
    """Boolean keys cannot silently share the integer-key authority domain."""

    with pytest.raises(QueryPathError):
        GraphPath((Key(True),)).to_bytes()
