"""Bound Object execution through backend-hosted Dispatch."""

from __future__ import annotations

import pytest

import dryml.dispatch as dispatch
from dryml.core import Repo, Serializable
from dryml.core.execute import CoreOptions
from dryml.core.store.dir import DirStore
from dryml.execute.subprocess import SubProcessConfig
from dryml.managed import managed_operation


class _BoundValue(Serializable):
    """
    Small Object fixture whose bound method is core-transported exactly once.
    """

    @managed_operation()
    def add(self, value: int, *, managed) -> int:
        """Return an ordinary scalar through the selected bound receiver."""

        return value + 3

    def save_state_to_dir_imp(self, dest_dir, *, codec) -> None:
        """
        Keep this stateless fixture compatible with persisted Object transport.
        """

    def restore_state_from_dir_imp(self, src_dir, *, codec) -> None:
        """Restore no local state for this stateless fixture."""


@pytest.fixture(autouse=True)
def _clear_dispatch_state() -> None:
    """Keep the process-wide Dispatch registry isolated for this module."""

    dispatch._state._reset_for_testing()
    yield
    dispatch._state._reset_for_testing()


def test_dispatch_runs_a_bound_object_method_through_core_transport(
        tmp_path) -> None:
    """
    Preserve the selected receiver and return the recovered ordinary result.
    """

    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    spool = tmp_path / "spool"
    spool.mkdir()
    dispatch.set_execute_backend_default(
        SubProcessConfig(spool_directory=spool),
        core=CoreOptions(repo=repo, return_objects=False),
    )
    value = _BoundValue(repo=repo)
    repo.save(value, deep_capture=True)

    assert dispatch.run(value.add, 4) == 7
