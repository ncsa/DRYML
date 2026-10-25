import pytest
from pathlib import Path

from dryml.core import (
    Repo, SaveAnnotations, SavePublication, SaveRouting, SavedSnapshot, Selector, Serializable,
    StoreReport,
)
from dryml.core.repo import RepoSaveError, save_object
from dryml.core.store.dir import DirStore
from dryml.core.store.zip import ZipStore
from dryml.core.store.store import StoreCapabilityError
from tests.core.test_repo_aliases import _saved_digest

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
    assert tuple(report.state_stores) == tuple(state.state_spec())
    assert all(stores == (store,) for stores in report.state_stores.values())
    assert report.required_stores == (store,)
    assert SavedSnapshot(state, (store,), (store,)) in report.snapshots
    assert all(isinstance(publication, SavePublication) for publication in report.publications)
    assert all(publication.status == "completed" for publication in report.publications)
    assert {publication.phase for publication in report.publications} == {"snapshot"}
    publication = next(p for p in report.publications if p.phase == "snapshot")
    assert publication.state_keys == publication.protected_state_keys == (state.state.key,)
    assert publication.mutation_id and publication.retention_token
    assert "StoreReport" not in repr(state)


@pytest.mark.parametrize("boundary", ("keys", "snapshot"))
@pytest.mark.parametrize("changes", (1, 8), ids=("settled", "exhausted"))
def test_save_readback_retries_cut_drift_without_recapture_or_republication(
        tmp_path, monkeypatch, boundary, changes):
    """Unrelated writers invalidate read evidence, not the completed save itself."""
    from dryml.core import Definition

    store = DirStore(tmp_path / "store", query_index="none")
    peer = DirStore.open_existing(store.base_dir, query_index="none")
    repo = Repo(store)
    captures, publications, moves = [], [], []
    capture = ReportState.save_state_to_dir_imp
    publish = store.publish_save_batch
    reader_name = "_read_variant" if boundary == "keys" else "_read_membership"
    reader = getattr(store, reader_name)

    def counted_capture(self, *args, **kwargs):
        captures.append(self)
        return capture(self, *args, **kwargs)

    def counted_publish(batch):
        result = publish(batch)
        publications.append(result)
        return result

    def changed_read(*args, **kwargs):
        result = reader(*args, **kwargs)
        if publications and len(moves) < changes:
            moves.append(len(moves))
            peer.publish_node(Definition(ReportState, value=100 + len(moves)).concretize())
        return result

    monkeypatch.setattr(ReportState, "save_state_to_dir_imp", counted_capture)
    monkeypatch.setattr(store, "publish_save_batch", counted_publish)
    monkeypatch.setattr(store, reader_name, changed_read)
    obj = ReportState(repo=repo)
    if changes == 8:
        with pytest.raises(RepoSaveError, match="read-back"):
            repo.save_object(obj)
        assert obj.last_state_ref is None
    else:
        saved, report = repo.save_object(obj, report_stores=True)
        assert saved is obj.last_state_ref
        assert report.required_stores == (store,)
        assert report.publications[0].protected_state_keys == (saved.state.key,)
        assert peer.read_node(_saved_digest(saved), kind="saved").graph_equal(saved)
    assert captures == [obj]
    assert len(publications) == 1 and len(moves) == changes


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
    """Retained saves refuse buffered authority before capture or archive publication."""

    target = tmp_path / "target.zip"
    obj = ReportState()
    with pytest.raises(StoreCapabilityError, match="durable"):
        if temporary:
            save_object(obj, repo=DirStore(tmp_path / "unrelated"), store=target, report_stores=True)
        else:
            Repo().save(obj, store=target, report_stores=True)
    assert obj.last_state_ref is None
    assert not target.exists()


def test_bounded_dirty_commits_preplan_later_required_work(tmp_path, monkeypatch):
    """A first bounded archive commit failure retains later required work as unattempted."""

    first = ZipStore(tmp_path / "first.zip")
    second = ZipStore(tmp_path / "second.zip")
    repo = Repo([first, second])
    obj = ReportState(repo=repo)
    first.publish_node(obj.definition)
    second.publish_node(obj.definition)
    report = StoreReport((first, second), {}, ())
    first._archive_dirty = second._archive_dirty = True
    repo._aliases_dirty = True
    monkeypatch.setattr(
        first, "commit", lambda: (_ for _ in ()).throw(OSError("first commit failed")),
    )

    from dryml.core.repo import _commit_save_report

    with pytest.raises(RepoSaveError) as caught:
        _commit_save_report(repo, None, report, stores=(first, second), dirty_only=True)

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
        second, "publish_save_batch",
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
    assert obj.last_state_ref.graph_equal(report.snapshots[0].state_ref)
    assert first.read_node(_saved_digest(old), kind="saved").graph_equal(old)
    assert first.read_node_name("latest").root_digest == _saved_digest(obj.last_state_ref)
    assert second.read_node_name("latest") is None
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
    original = second._atomic_write
    def fail_name(path, data):
        if str(path).replace("\\", "/").endswith("/names/latest.record"):
            raise OSError("second alias failed")
        return original(path, data)
    monkeypatch.setattr(second, "_atomic_write", fail_name)

    with pytest.raises(RepoSaveError) as raised:
        repo.save_object(obj, alias="latest")

    report = raised.value.report
    assert [(item.store, item.status) for item in report.publications if item.phase == "snapshot"] == [
        (first, "completed"), (second, "uncertain"),
    ]
    assert first.read_node_name("latest").root_digest == _saved_digest(obj.last_state_ref)
    assert second.read_node_name("latest") is None
    assert report.snapshots


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("after", [False, True], ids=["before", "after"])
def test_snapshot_control_flow_preserves_identity_and_readback_status(
        tmp_path, monkeypatch, error_type, after):
    """Snapshot interruption retains its control-flow identity and exact ledger evidence."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store, save_routing=SaveRouting())
    obj = ReportState(1, repo=repo)
    original = store.publish_save_batch

    def interrupt(*args, **kwargs):
        if not after:
            raise error_type("snapshot interruption")
        original(*args, **kwargs)
        raise error_type("snapshot interruption")

    monkeypatch.setattr(store, "publish_save_batch", interrupt)
    with pytest.raises(error_type, match="snapshot interruption") as raised:
        repo.save_object(obj, deep_capture=True)

    report = raised.value.report
    snapshot = next(item for item in report.publications if item.phase == "snapshot")
    assert snapshot.status == "failed"
    assert (store.read_node(_saved_digest(snapshot.state_ref), kind="saved") is not None) is after
    assert bool(report.snapshots) is after
    assert bool(report.required_stores) is after


def test_state_install_error_after_authority_reports_the_confirmed_local_store(tmp_path, monkeypatch):
    """A post-install failure cannot hide independently readable local-state authority."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store, save_routing=SaveRouting())
    obj = ReportState(1, repo=repo)
    original = store.publish_save_batch

    def publish_then_fail(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("post-snapshot dirty-marker failure")

    monkeypatch.setattr(store, "publish_save_batch", publish_then_fail)
    with pytest.raises(RepoSaveError, match="publication") as raised:
        repo.save_object(obj, deep_capture=True)

    report = raised.value.report
    publication = next(item for item in report.publications if item.phase == "snapshot")
    assert publication.status == "failed"
    assert publication.state_keys == publication.protected_state_keys == (publication.state_ref.state.key,)
    with store.read_state(publication.state_ref.state.key) as source:
        assert Path(source.data_directory, "value").read_text() == "1"


def test_post_membership_error_retains_completed_snapshot_and_new_receipt(tmp_path, monkeypatch):
    """A membership error after readback leaves complete immutable authority usable."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store, save_routing=SaveRouting())
    obj = ReportState(1, repo=repo)
    old = repo.save_object(obj, deep_capture=True)
    obj.value = 2
    original = store.publish_save_batch
    def write_then_fail(batch):
        original(batch)
        raise OSError("post-membership failure")
    monkeypatch.setattr(store, "publish_save_batch", write_then_fail)
    with pytest.raises(RepoSaveError, match="publication") as raised:
        repo.save_object(obj, deep_capture=True)

    report = raised.value.report
    membership = next(item for item in report.publications if item.phase == "snapshot")
    assert membership.status == "failed"
    assert store.read_node(_saved_digest(membership.state_ref), kind="saved").graph_equal(membership.state_ref)
    assert obj.last_state_ref.graph_equal(membership.state_ref)
    assert not obj.last_state_ref.graph_equal(old)
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
    original = store._atomic_write

    def write_then_damage(path, payload):
        result = original(path, payload)
        if "/membership/saved/" in str(path).replace("\\", "/"):
            root_path = Path(path)
            if damage == "missing":
                root_path.unlink()
            else:
                root_path.write_bytes(b"malformed stored-root authority")
            raise OSError("membership readback damaged")
        return result

    monkeypatch.setattr(store, "_atomic_write", write_then_damage)
    with pytest.raises(RepoSaveError) as raised:
        repo.save_object(obj, deep_capture=True, main=True)

    report = raised.value.report
    membership = next(item for item in report.publications if item.phase == "snapshot")
    assert membership.status == "uncertain"
    assert store.authority_cut().pending
    assert store.read_node_name("main") is None
    assert not report.snapshots and not report.required_stores


def test_index_failure_delays_all_requested_names(tmp_path, monkeypatch):
    """Names remain unattempted when derived registration fails after authority."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store, save_routing=SaveRouting())
    obj = ReportState(1, repo=repo)
    index = store.open_query_index()
    index.maintain_store()
    original = index.maintain_store
    def fail_after_authority(*args, **kwargs):
        if store.authority_cut().pending:
            raise RuntimeError("index failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(index, "maintain_store", fail_after_authority)

    with pytest.raises(RepoSaveError, match="publication") as raised:
        repo.save_object(
            obj, main=True, alias="latest", deep_capture=True,
            annotations=SaveAnnotations(object={"project": "retained"}, state={"score": 1}),
        )

    report = raised.value.report
    assert report.publications[0].status == "uncertain"
    state = next(item.state_ref for item in report.publications if item.phase == "snapshot")
    assert store.read_node_name("main").root_kind == "recipe"
    assert store.read_node_name("latest").root_digest == _saved_digest(state)
    assert store.read_metadata(state.object_view()) == {"project": "retained"}
    assert store.read_metadata(state) == {"score": 1}
    assert store.authority_cut().pending
    monkeypatch.setattr(index, "maintain_store", original)
    original()
    assert not store.authority_cut().pending


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
    observed = []
    if boundary == "index":
        index = store.open_query_index()
        index.maintain_store()
        original = index.maintain_store

        def operation(*args, **kwargs):
            if not store.authority_cut().pending:
                return original(*args, **kwargs)
            observed.append(boundary)
            if after:
                original(*args, **kwargs)
            raise error_type("index interruption")

        monkeypatch.setattr(index, "maintain_store", operation)
    else:
        fragment = {"definition": "/definitions/blocks/", "state": "/grants/",
                    "membership": "/membership/saved/", "main": "/names/main.record",
                    "alias": "/names/latest.record"}[boundary]
        original = store._atomic_write

        def operation(path, payload):
            if fragment not in str(path):
                return original(path, payload)
            observed.append(boundary)
            if after:
                original(path, payload)
            raise error_type(f"{boundary} interruption")

        monkeypatch.setattr(store, "_atomic_write", operation)

    with pytest.raises(error_type, match=f"{boundary} interruption") as raised:
        repo.save_object(obj, deep_capture=True, **save_options)

    report = raised.value.report
    publication = next(item for item in report.publications if item.phase == "snapshot")
    assert observed
    assert publication.status == "uncertain"
    assert publication.mutation_id and publication.retention_token
    if boundary in {"main", "alias"}:
        assert (store.read_node_name("main" if boundary == "main" else "latest") is not None) is after
    if boundary != "index" or not after:
        assert store.authority_cut().pending
