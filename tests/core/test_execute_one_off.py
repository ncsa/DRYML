"""Explicit-backend core one-off execution coverage."""

from __future__ import annotations

from dryml.core import Repo
from dryml.core.execute import CoreOptions, run, submit
from dryml.core.store.dir import DirStore
from dryml.execute.subprocess import SubProcessConfig


def _multiply(value):
    """Return a simple deterministic one-off value."""
    return value * 2


def test_core_one_off_submit_and_run_require_explicit_backend(tmp_path):
    """One-off calls expose adapted results while retaining generic hidden ownership."""
    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    (tmp_path / "spool").mkdir()
    config = SubProcessConfig(spool_directory=tmp_path / "spool")
    options = CoreOptions(repo=repo, return_objects=False)

    future = submit(_multiply, 4, backend=config, core=options)
    assert future.result(timeout=10) == 8
    future.cleanup(timeout=5)
    assert run(_multiply, 5, backend=config, core=options) == 10
