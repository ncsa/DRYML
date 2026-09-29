"""Managed completion contracts for v3 snapshot publication reports."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.usefixtures("fixed_managed_snapshot_environment")

from dryml.core import Repo, SaveRouting, Selector
from dryml.core.object import Pickleable
from dryml.core.repo import RepoSaveError
from dryml.core.store.dir import DirStore
from dryml.managed import ManagedConfig, managed_operation


class ManagedReportValue(Pickleable):
    """Managed receiver whose final save exercises replica report completion."""

    def __init__(self, value=0):
        self.value = value

    @managed_operation()
    def increment(self, *, managed):
        """Mutate the receiver before the managed final-publication boundary."""

        self.value += 1
        return self.value


def test_managed_completion_rejects_a_partial_snapshot_report(tmp_path, monkeypatch):
    """A completed replica cannot make managed final association succeed alone."""

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    repo = Repo(
        (first, second),
        save_routing=SaveRouting(
            ((Selector(ManagedReportValue), first), (Selector(ManagedReportValue), second)),
            match_mode="all",
        ),
    )
    value = ManagedReportValue(repo=repo)
    monkeypatch.setattr(
        second,
        "publish_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("second replica failed")),
    )

    with pytest.raises(RepoSaveError) as raised:
        value.increment(managed=ManagedConfig(state_repo=repo, control_store=first))

    report = raised.value.report
    assert report is not None
    assert [(item.store, item.status) for item in report.publications if item.phase == "snapshot"] == [
        (first, "completed"), (second, "failed"),
    ]
    status = value.increment.status(state_repo=repo, control_store=first)
    assert (status.state, status.final_state_ref) == ("failed", None)
