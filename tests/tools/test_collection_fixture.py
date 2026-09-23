"""Meta-tests for maintained-node collection administration."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

from tests.tools import test_buckets


ROOT = Path(__file__).resolve().parents[2]


def test_maintained_collection_fixture_is_shared_by_two_tests(tmp_path):
    count_path = tmp_path / "collector-calls.txt"
    (tmp_path / "conftest.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "from tests.tools import test_buckets\n"
        "pytest_plugins = ('tests.tools.conftest',)\n"
        "def fake_collection():\n"
        "    path = Path(os.environ['DRYML_COLLECTION_COUNT'])\n"
        "    with path.open('a', encoding='utf-8') as stream:\n"
        "        stream.write('called\\n')\n"
        "    return {'tests/example.py::test_example'}\n"
        "test_buckets.collected_test_nodeids = fake_collection\n"
    )
    (tmp_path / "test_fixture.py").write_text(
        "def test_first(maintained_test_nodeids):\n"
        "    assert maintained_test_nodeids == "
        "{'tests/example.py::test_example'}\n"
        "def test_second(maintained_test_nodeids):\n"
        "    assert maintained_test_nodeids == "
        "{'tests/example.py::test_example'}\n"
    )
    env = os.environ.copy()
    env["DRYML_COLLECTION_COUNT"] = str(count_path)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("PYTEST_ADDOPTS", None)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "test_fixture.py"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "2 passed" in result.stdout
    assert count_path.read_text().splitlines() == ["called"]


def test_collection_failure_is_not_converted_to_an_empty_snapshot(monkeypatch):
    result = SimpleNamespace(
        returncode=1,
        stdout="",
        stderr="synthetic collection failure",
    )
    monkeypatch.setattr(test_buckets.subprocess, "run", lambda *a, **k: result)

    with pytest.raises(AssertionError, match="synthetic collection failure"):
        test_buckets.collected_test_nodeids()
