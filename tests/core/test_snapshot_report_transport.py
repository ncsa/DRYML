"""V3 snapshot publication report transport contracts."""

from __future__ import annotations

import pytest

from dryml.core import Repo, SavePublication, Serializable, StoreReport
from dryml.core.execute import decode_core_outcome
from dryml.core.execute_codec import _outcome, _report_data
from dryml.core.repo import RepoSaveError
from dryml.core.repo_plan import PUBLICATION_PHASES
from dryml.core.store.dir import DirStore


class ReportTransportValue(Serializable):
    """Minimal stateful payload for v3 report transport tests."""

    def __init__(self, value=0):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        """Write the value as one codec-owned snapshot payload."""

        from pathlib import Path

        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_report_transport_accepts_every_closed_v3_phase(tmp_path):
    """Detached worker evidence preserves every StoreReport phase without handles."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    state = repo.save_object(ReportTransportValue(1, repo=repo))
    phases = tuple(sorted(PUBLICATION_PHASES))
    report = StoreReport(
        (store,), {}, (store,), publications=tuple(
            SavePublication(store, None, None, state, phase, "unattempted")
            for phase in phases
        ),
    )

    outcome = decode_core_outcome(_outcome(
        False,
        publications=_report_data(report, repo, state),
        reason="result outcome exceeds configured bound after publication",
        limit_bytes=1_000_000,
    ), repo=repo)

    assert tuple(item.phase for item in outcome.evidence.publications) == phases
    assert all(item.status == "unattempted" for item in outcome.evidence.publications)
    assert {item.store_index for item in outcome.evidence.publications} == {0}


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_snapshot_failure_reports_unattempted_late_work_before_hooks(tmp_path, monkeypatch):
    """An early snapshot error retains planned derived work as unattempted evidence."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    monkeypatch.setattr(
        store,
        "publish_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("snapshot failure")),
    )

    with pytest.raises(RepoSaveError) as raised:
        repo.save_object(ReportTransportValue(1, repo=repo))

    report = raised.value.report
    assert report is not None
    assert next(item.status for item in report.publications if item.phase == "snapshot") == "failed"
    assert all(
        item.status == "unattempted"
        for item in report.publications
        if item.phase in {"membership", "index"}
    )
