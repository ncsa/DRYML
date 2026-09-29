import pytest
from pathlib import Path

from dryml.core import (
    Repo, SaveAnnotations, SavePublication, SaveRouting, SavedSnapshot, Selector, Serializable,
    StoreReport,
)
from dryml.core.repo import RepoSaveError, save_object
from dryml.core.store.dir import DirStore
from dryml.core.store.zip import ZipStore

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")


class ReportState(Serializable):
    def __init__(self, value=0):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        from pathlib import Path

        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        from pathlib import Path

        self.value = int(Path(src_dir, "value").read_text(encoding="ascii"))


def test_store_report_is_ephemeral_complete_and_not_state_identity(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store, save_routing=SaveRouting())
    state, report = ReportState(repo=repo).save(repo=repo, report_stores=True)

    assert isinstance(report, StoreReport)
    assert report.target_stores == (store,)
    assert tuple(report.state_stores) == tuple(state.states)
    assert all(stores == (store,) for stores in report.state_stores.values())
    assert report.required_stores == (store,)
    assert SavedSnapshot(state, (store,), (store,)) in report.snapshots
    assert all(isinstance(publication, SavePublication) for publication in report.publications)
    assert all(publication.status == "completed" for publication in report.publications)
    assert {publication.phase for publication in report.publications} >= {
        "definition", "state", "snapshot", "membership", "index", "main",
    }
    assert "StoreReport" not in repr(state)


def test_unconfigured_repo_save_commits_each_store_once_with_completed_report(tmp_path):
    """The unified save ledger owns one completed flush entry per connected Store."""

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    repo = Repo([first, second])

    _, report = repo.save(ReportState(repo=repo), report_stores=True)

    commits = [item for item in report.publications if item.phase == "commit"]
    assert [(item.store, item.status) for item in commits] == [
        (first, "completed"),
        (second, "completed"),
    ]


@pytest.mark.parametrize("temporary", [False, True], ids=["repo", "convenience"])
def test_explicit_new_zip_target_is_committed_and_reported(tmp_path, temporary):
    """A flush-owning save includes a newly selected archive in its commit ledger."""

    target = tmp_path / "target.zip"
    obj = ReportState()
    if temporary:
        state, report = save_object(
            obj, repo=DirStore(tmp_path / "unrelated"), store=target, report_stores=True,
        )
    else:
        repo = Repo()
        state, report = repo.save(obj, store=target, report_stores=True)

    commits = [item for item in report.publications if item.phase == "commit"]
    assert [(item.store.archive_path, item.status) for item in commits if isinstance(item.store, ZipStore)] == [
        (str(target), "completed"),
    ]
    reopened = ZipStore.open_existing(target)
    try:
        assert Repo(reopened).load_state_ref(state, reuse_live="never").value == 0
    finally:
        reopened.close()


def test_bounded_dirty_commits_preplan_later_required_work(tmp_path, monkeypatch):
    """A first bounded archive commit failure retains later required work as unattempted."""

    first = ZipStore(tmp_path / "first.zip")
    second = ZipStore(tmp_path / "second.zip")
    repo = Repo([first, second])
    obj = ReportState(repo=repo)
    _, report = repo.save_object(obj, deep_capture=True, report_stores=True)
    first._archive_dirty = second._archive_dirty = True
    repo._aliases_dirty = True
    monkeypatch.setattr(
        first, "commit", lambda: (_ for _ in ()).throw(OSError("first commit failed")),
    )

    from dryml.core.repo import _commit_save_report

    with pytest.raises(RepoSaveError) as caught:
        _commit_save_report(repo, obj.last_state_ref, report, stores=(first, second), dirty_only=True)

    commits = [item for item in caught.value.report.publications if item.phase == "commit"]
    assert [(item.store, item.status) for item in commits] == [
        (first, "failed"), (second, "unattempted"),
    ]
    assert repo._aliases_dirty is True


@pytest.mark.parametrize("operation_fails", [False, True], ids=["after_success", "after_failure"])
def test_late_publication_checker_error_is_uncertain(tmp_path, operation_fails):
    """A failed read-back never claims a late boundary either completed or failed."""

    from dryml.core.repo import _record_late_publication

    store = DirStore(tmp_path / "store")
    report = StoreReport((), {}, ())

    def operation():
        if operation_fails:
            raise OSError("publication operation failed")

    def checker():
        raise OSError("publication checker failed")

    with pytest.raises(RepoSaveError) as caught:
        _record_late_publication(
            report, store=store, phase="commit", state_ref=None,
            operation=operation, checker=checker,
        )

    assert caught.value.report.publications[-1].status == "uncertain"


def test_replica_failure_keeps_the_old_root_receipt_and_exposes_route_order(tmp_path, monkeypatch):
    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    initial_repo = Repo(first)
    obj = ReportState(1, repo=initial_repo)
    old = obj.save(repo=initial_repo)
    repo = Repo(
        [first, second],
        save_routing=SaveRouting(((Selector(ReportState), first), (Selector(ReportState), second)), "all"),
    )
    obj.value = 2

    monkeypatch.setattr(
        second, "publish_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("second snapshot failed")),
    )
    with pytest.raises(RepoSaveError) as raised:
        obj.save(repo=repo, deep_capture=True, alias="latest")

    report = raised.value.report
    snapshots = [item for item in report.publications if item.phase == "snapshot" and not item.path]
    assert [(item.store, item.status) for item in snapshots] == [
        (first, "completed"), (second, "failed"),
    ]
    assert report.snapshots == (SavedSnapshot(
        report.snapshots[0].state_ref, (first,), (first,),
    ),)
    assert all(
        item.status == "unattempted"
        for item in report.publications
        if item.phase in {"index", "alias"}
    )
    assert obj.last_state_ref == old
    assert first.read_state_ref_record(old.digest()).state_ref == old
    assert Repo(first).load_state_ref(report.snapshots[0].state_ref, reuse_live="never").value == 2


def test_later_alias_failure_keeps_earlier_name_and_completed_authority(tmp_path, monkeypatch):
    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    repo = Repo(
        [first, second],
        save_routing=SaveRouting(
            ((Selector(ReportState), first), (Selector(ReportState), second)), "all",
        ),
    )
    obj = ReportState(repo=repo)
    monkeypatch.setattr(
        second,
        "write_object_alias",
        lambda record: (_ for _ in ()).throw(OSError("second alias failed")),
    )

    with pytest.raises(RepoSaveError) as raised:
        repo.save_object(obj, alias="latest")

    report = raised.value.report
    assert [(item.store, item.status) for item in report.publications if item.phase == "alias"] == [
        (first, "completed"), (second, "failed"),
    ]
    assert first.read_object_alias("latest").object_ref == obj.object_ref
    assert second.read_object_alias("latest") is None
    assert report.snapshots


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("after", [False, True], ids=["before", "after"])
def test_snapshot_control_flow_preserves_identity_and_readback_status(
        tmp_path, monkeypatch, error_type, after):
    """Snapshot interruption retains its control-flow identity and exact ledger evidence."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store, save_routing=SaveRouting())
    obj = ReportState(1, repo=repo)
    original = store.publish_snapshot

    def interrupt(*args, **kwargs):
        if not after:
            raise error_type("snapshot interruption")
        original(*args, **kwargs)
        raise error_type("snapshot interruption")

    monkeypatch.setattr(store, "publish_snapshot", interrupt)
    with pytest.raises(error_type, match="snapshot interruption") as raised:
        repo.save_object(obj, deep_capture=True)

    report = raised.value.report
    snapshot = next(item for item in report.publications if item.phase == "snapshot")
    membership = next(item for item in report.publications if item.phase == "membership")
    assert snapshot.status == ("completed" if after else "failed")
    assert membership.status == "unattempted"
    assert (store.read_state_ref_record(snapshot.state_ref.digest()) is not None) is after


def test_state_install_error_after_authority_reports_the_confirmed_local_store(tmp_path, monkeypatch):
    """A post-install failure cannot hide independently readable local-state authority."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store, save_routing=SaveRouting())
    obj = ReportState(1, repo=repo)
    original = store.publish_snapshot

    def publish_then_fail(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("post-snapshot dirty-marker failure")

    monkeypatch.setattr(store, "publish_snapshot", publish_then_fail)
    with pytest.raises(RepoSaveError, match="publication") as raised:
        repo.save_object(obj, deep_capture=True)

    report = raised.value.report
    state = next(item for item in report.publications if item.phase == "state")
    assert state.status == "completed"
    assert store.validate_local_state(state.state_ref, state.path)


def test_post_membership_error_retains_completed_snapshot_and_new_receipt(tmp_path, monkeypatch):
    """A membership error after readback leaves complete immutable authority usable."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store, save_routing=SaveRouting())
    obj = ReportState(1, repo=repo)
    old = repo.save_object(obj, deep_capture=True)
    obj.value = 2
    original = store.write_definition_record

    def write_then_fail(record, *, stored_root=True):
        result = original(record, stored_root=stored_root)
        if stored_root:
            raise OSError("post-membership failure")
        return result

    monkeypatch.setattr(store, "write_definition_record", write_then_fail)
    with pytest.raises(RepoSaveError, match="publication") as raised:
        repo.save_object(obj, deep_capture=True)

    report = raised.value.report
    membership = next(item for item in report.publications if item.phase == "membership")
    assert membership.status == "completed"
    assert store.read_state_ref_record(membership.state_ref.digest()).state_ref == membership.state_ref
    assert obj.last_state_ref == old
    assert Repo(store).load_state_ref(membership.state_ref, reuse_live="never").value == 2


@pytest.mark.parametrize(
    ("damage", "expected_status"),
    (("missing", "failed"), ("corrupt", "uncertain")),
)
def test_membership_readback_failure_retains_dirty_authority_and_late_ledger(
        tmp_path, monkeypatch, damage, expected_status):
    """Missing or corrupt membership authority blocks late work without clearing its dirty marker."""
    store = DirStore(tmp_path / "store")
    repo = Repo(store, save_routing=SaveRouting())
    obj = ReportState(1, repo=repo)
    original = store.write_definition_record

    def write_then_damage(record, *, stored_root=True):
        result = original(record, stored_root=stored_root)
        if stored_root:
            root_path = Path(store._stored_root_path(record.digest))
            if damage == "missing":
                root_path.unlink()
            else:
                root_path.write_bytes(b"malformed stored-root authority")
        return result

    monkeypatch.setattr(store, "write_definition_record", write_then_damage)
    with pytest.raises(RepoSaveError) as raised:
        repo.save_object(obj, deep_capture=True, main=True)

    report = raised.value.report
    membership = next(item for item in report.publications if item.phase == "membership")
    assert membership.status == expected_status
    assert store.query_index_is_dirty()
    assert all(
        item.status == "unattempted"
        for item in report.publications
        if item.phase in {"index", "main"}
    )


def test_index_failure_delays_all_requested_names(tmp_path, monkeypatch):
    """Names remain unattempted when derived registration fails after authority."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store, save_routing=SaveRouting())
    obj = ReportState(1, repo=repo)
    monkeypatch.setattr(
        repo._query_index,
        "register_saved_graph",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("index failure")),
    )

    with pytest.raises(RepoSaveError, match="publication") as raised:
        repo.save_object(
            obj, main=True, alias="latest", deep_capture=True,
            annotations=SaveAnnotations(object={"project": "retained"}, state={"score": 1}),
        )

    report = raised.value.report
    assert all(
        item.status == "unattempted"
        for item in report.publications
        if item.phase in {"main", "alias"}
    )
    assert store.read_main_ref() is None
    assert store.read_object_alias("latest") is None
    state = next(item.state_ref for item in report.publications if item.phase == "snapshot")
    assert store.read_metadata(state.object) == {"project": "retained"}
    assert store.read_metadata(state) == {"score": 1}
    assert store.query_index_is_dirty()


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("after", [False, True], ids=["before", "after"])
@pytest.mark.parametrize(
    ("boundary", "save_options", "expected_after"),
    [
        ("definition", {}, "completed"),
        ("state", {}, "completed"),
        ("membership", {}, "completed"),
        ("index", {}, "completed"),
        ("main", {"main": True}, "completed"),
        ("alias", {"alias": "latest"}, "completed"),
    ],
)
def test_routed_control_flow_boundaries_preserve_identity_and_ledger(
        tmp_path, monkeypatch, error_type, after, boundary, save_options, expected_after):
    """Every routed authority boundary reports an interruption without conversion."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store, save_routing=SaveRouting())
    obj = ReportState(1, repo=repo)
    if boundary == "definition":
        original = store.write_definition_record

        def operation(record, *, stored_root=True):
            if stored_root:
                return original(record, stored_root=stored_root)
            if not after:
                raise error_type("definition interruption")
            original(record, stored_root=stored_root)
            raise error_type("definition interruption")

        monkeypatch.setattr(store, "write_definition_record", operation)
    elif boundary == "state":
        original = store.publish_snapshot

        def operation(*args, **kwargs):
            if not after:
                raise error_type("state interruption")
            original(*args, **kwargs)
            raise error_type("state interruption")

        monkeypatch.setattr(store, "publish_snapshot", operation)
    elif boundary == "membership":
        original = store.write_definition_record

        def operation(record, *, stored_root=True):
            if not stored_root:
                return original(record, stored_root=stored_root)
            if not after:
                raise error_type("membership interruption")
            original(record, stored_root=stored_root)
            raise error_type("membership interruption")

        monkeypatch.setattr(store, "write_definition_record", operation)
    elif boundary == "index":
        original = repo._query_index.register_saved_graph

        def operation(*args, **kwargs):
            if not after:
                raise error_type("index interruption")
            original(*args, **kwargs)
            raise error_type("index interruption")

        monkeypatch.setattr(repo._query_index, "register_saved_graph", operation)
    elif boundary == "main":
        original = store.write_main_ref

        def operation(record):
            if not after:
                raise error_type("main interruption")
            original(record)
            raise error_type("main interruption")

        monkeypatch.setattr(store, "write_main_ref", operation)
    else:
        original = store.write_object_alias

        def operation(record):
            if not after:
                raise error_type("alias interruption")
            original(record)
            raise error_type("alias interruption")

        monkeypatch.setattr(store, "write_object_alias", operation)

    with pytest.raises(error_type, match=f"{boundary} interruption") as raised:
        repo.save_object(obj, deep_capture=True, **save_options)

    report = raised.value.report
    publication = next(item for item in report.publications if item.phase == boundary)
    # Every entered boundary is classified by direct read-back, including
    # composite v3 authority boundaries interrupted around their operation.
    expected = expected_after if after else "failed"
    assert publication.status == expected
