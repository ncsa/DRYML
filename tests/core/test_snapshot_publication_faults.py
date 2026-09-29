"""U7 deterministic fault coverage for snapshot publication boundaries."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

import dryml.filesystem as filesystem
import dryml.environments as envs
from dryml.core import Object, Repo, SaveAnnotations, SaveRouting, Selector, Serializable
from dryml.core.repo import RepoLoadError, RepoSaveError
from dryml.core.store.dir import DirStore
from dryml.core.store.store import StoreAuthorityError, StoreCapabilityError
from dryml.core.store.zip import ZipStore


class FaultPayload(Serializable):
    """Small stateful value with a deterministic snapshot payload."""

    def __init__(self, value="value"):
        self.value = value

    def save_state_to_dir_imp(self, directory, *, codec):
        Path(directory, "value").write_text(self.value, encoding="ascii")


class FaultContainer(Object):
    """Stateless root with multiple lineage-bearing children."""

    def __init__(self, left, right):
        self.left = left
        self.right = right


class LegacySignatureStore(DirStore):
    """Backend override using the pre-U7 snapshot publication signature."""

    def publish_snapshot(
            self, reference, *, evidence, annotations=None, local_states,
            children=None):
        return super().publish_snapshot(
            reference, evidence=evidence, annotations=annotations,
            local_states=local_states, children=children,
        )


def _phases(report):
    """Return publication statuses keyed by their unique phase names."""

    return {item.phase: item.status for item in report.publications}


def _observe_environment(observed, value):
    """Record one probe and return deterministic valid environment evidence."""

    observed.append(value)
    return envs.EnvironmentRecord(
        python=envs.PythonRecord("3.12.0", "CPython"),
        platform=envs.PlatformRecord("Linux", "1", "v", "x86_64", "Linux-x86_64"),
        distributions={},
        dryml=envs.DrymlRuntimeRecord(),
    )


@pytest.mark.parametrize("after", [False, True], ids=["before", "after"])
def test_lineage_interruption_reports_only_entered_authority(tmp_path, monkeypatch, after):
    """Lineage read-back classifies its boundary without attempting snapshots."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    original = store.write_lineage_metadata

    def interrupt(lineage):
        if not after:
            raise KeyboardInterrupt("lineage interruption")
        original(lineage)
        raise KeyboardInterrupt("lineage interruption")

    monkeypatch.setattr(store, "write_lineage_metadata", interrupt)
    with pytest.raises(KeyboardInterrupt, match="lineage interruption") as raised:
        repo.save_object(FaultPayload(repo=repo))

    status = _phases(raised.value.report)
    assert status["definition"] == "completed"
    assert status["lineage"] == ("completed" if after else "failed")
    assert status["state"] == status["snapshot"] == status["membership"] == "unattempted"


def test_post_snapshot_current_object_interruption_reports_completed_snapshot_and_mapping(tmp_path, monkeypatch):
    """A post-write interruption retains exact authority without false success."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    original = store.write_metadata

    def interrupt_after_object_write(target, values):
        original(target, values)
        if target.__class__.__name__ == "ObjectRef":
            raise KeyboardInterrupt("object mapping interruption")

    monkeypatch.setattr(store, "write_metadata", interrupt_after_object_write)
    with pytest.raises(KeyboardInterrupt, match="object mapping") as raised:
        repo.save_object(
            FaultPayload(repo=repo),
            annotations=SaveAnnotations(object={"object": "installed"}, state={"state": "later"}),
        )

    report = raised.value.report
    status = _phases(report)
    state = next(item.state_ref for item in report.publications if item.phase == "snapshot")
    assert status["definition"] == status["lineage"] == status["state"] == "completed"
    assert status["snapshot"] == status["object_metadata"] == "completed"
    assert status["state_metadata"] == "unattempted"
    assert store.read_state_ref_record(state.digest()).state_ref == state
    assert store.read_metadata(state.object) == {"object": "installed"}
    assert store.read_metadata(state) is None


def test_descendant_lineage_failure_preserves_exact_completed_path_statuses(tmp_path, monkeypatch):
    """Each primary lineage path retains its own publication outcome."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    original = store.write_lineage_metadata
    calls = 0

    def fail_second(lineage):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("descendant lineage failed")
        return original(lineage)

    monkeypatch.setattr(store, "write_lineage_metadata", fail_second)
    with pytest.raises(RepoSaveError) as raised:
        repo.save_object(FaultContainer(
            FaultPayload("left", repo=repo), FaultPayload("right", repo=repo), repo=repo,
        ))

    lineages = [item for item in raised.value.report.publications if item.phase == "lineage"]
    assert [item.status for item in lineages] == ["completed", "failed", "unattempted"]
    assert len({item.path for item in lineages}) == 3


def test_legacy_store_snapshot_override_requires_callback_only_for_annotations(tmp_path):
    """Legacy overrides save normally but reject unreportable annotation phases."""

    store = LegacySignatureStore(tmp_path / "store")
    repo = Repo(store)
    value = FaultPayload(repo=repo)
    state = repo.save_object(value)

    with pytest.raises(StoreCapabilityError, match="phase callbacks"):
        repo.save_object(
            value,
            annotations=SaveAnnotations(object={"object": True}, state={"state": True}),
        )

    assert store.read_state_ref_record(state.digest()).state_ref == state
    assert store.read_metadata(state.object) is None
    assert store.read_metadata(state) is None


def test_annotation_callback_gate_applies_only_to_destinations_with_actual_writes(tmp_path):
    """Legacy child targets and absent fork copies need no annotation callback."""

    root_store = DirStore(tmp_path / "root")
    child_store = LegacySignatureStore(tmp_path / "child")
    routed = Repo(
        [root_store, child_store],
        save_routing=SaveRouting(
            ((Selector(FaultContainer), root_store), (Selector(FaultPayload), child_store)),
            graph_mode="per-object",
        ),
    )
    state = routed.save_object(
        FaultContainer(FaultPayload(repo=routed), FaultPayload(repo=routed), repo=routed),
        annotations=SaveAnnotations(object={"root": True}),
    )
    assert root_store.read_metadata(state.object) == {"root": True}

    source = DirStore(tmp_path / "source")
    target = LegacySignatureStore(tmp_path / "target")
    source_repo = Repo(source)
    original = source_repo.save_object(FaultPayload(repo=source_repo))
    fork = Repo([source, target]).fork_state_ref(
        original, store=target, source_store=source,
        copy_annotations=("object", "state"),
    )
    assert target.read_state_ref_record(fork.digest()).state_ref == fork


def test_post_snapshot_current_state_interruption_reports_both_completed_mappings(tmp_path, monkeypatch):
    """Read-back recognizes an installed mapping even when its call interrupts."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    original = store.write_metadata

    def interrupt_after_state_write(target, values):
        original(target, values)
        if target.__class__.__name__ == "StateRef":
            raise KeyboardInterrupt("state mapping interruption")

    monkeypatch.setattr(store, "write_metadata", interrupt_after_state_write)
    with pytest.raises(KeyboardInterrupt, match="state mapping") as raised:
        repo.save_object(
            FaultPayload(repo=repo),
            annotations=SaveAnnotations(object={"object": "installed"}, state={"state": "installed"}),
        )

    report = raised.value.report
    status = _phases(report)
    state = next(item.state_ref for item in report.publications if item.phase == "snapshot")
    assert status["snapshot"] == status["object_metadata"] == status["state_metadata"] == "completed"
    assert store.read_metadata(state.object) == {"object": "installed"}
    assert store.read_metadata(state) == {"state": "installed"}


def test_snapshot_post_publish_interruption_retains_completed_authority(tmp_path, monkeypatch):
    """A post-publication failure is reported from snapshot read-back."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    original_publish = filesystem.publish_directory

    def interrupt_after_snapshot_publish(source, destination):
        installs_snapshot = (
            Path(source).is_dir()
            and Path(destination).parent.name == Path(destination).name[:2]
        )
        result = original_publish(source, destination)
        if installs_snapshot:
            raise OSError("snapshot post-publication interruption")
        return result

    monkeypatch.setattr(
        filesystem, "publish_directory", interrupt_after_snapshot_publish,
    )
    with pytest.raises(RepoSaveError, match="publication") as raised:
        repo.save_object(FaultPayload(repo=repo))

    report = raised.value.report
    state = next(item.state_ref for item in report.publications if item.phase == "snapshot")
    assert _phases(report)["snapshot"] == "completed"
    assert store.read_state_ref_record(state.digest()).state_ref == state
    assert store.query_index_is_dirty()


def test_retry_before_snapshot_install_captures_new_evidence(tmp_path, monkeypatch):
    """A failed staged candidate creates no association for a retry to preserve."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    observed = []
    original = filesystem.publish_directory

    def fail_snapshot_replace(source, destination):
        if Path(source).name.startswith("snapshot-"):
            raise OSError("staged snapshot replacement failed")
        return original(source, destination)

    monkeypatch.setattr(filesystem, "publish_directory", fail_snapshot_replace)
    value = FaultPayload(repo=repo)
    with pytest.raises(RepoSaveError) as raised:
        repo.save_object(value, _snapshot_observer=lambda: _observe_environment(observed, "first"))
    assert "staged snapshot replacement" in str(raised.value.__cause__)
    state = next(item.state_ref for item in raised.value.report.publications if item.phase == "snapshot")
    assert store.read_state_ref_record(state.digest()) is None
    assert _phases(raised.value.report)["snapshot"] == "failed"

    monkeypatch.setattr(filesystem, "publish_directory", original)
    assert repo.save_object(
        value, _snapshot_observer=lambda: _observe_environment(observed, "retry"),
    ) == state
    assert observed == ["first", "retry"]


def test_retry_after_post_snapshot_failure_reuses_evidence_without_replaying_annotations(tmp_path, monkeypatch):
    """A retry uses installed evidence and treats new annotations as a new LWW write."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    value = FaultPayload("same", repo=repo)
    original = store.write_metadata
    calls = 0

    def fail_once(target, values):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("first current mapping fails")
        return original(target, values)

    monkeypatch.setattr(store, "write_metadata", fail_once)
    with pytest.raises(RepoSaveError) as raised:
        repo.save_object(value, annotations=SaveAnnotations(object={"first": 1}))
    state = next(item.state_ref for item in raised.value.report.publications if item.phase == "snapshot")
    captured = store.read_snapshot_metadata(state.digest())
    reopened = Repo(DirStore(store.base_dir))
    reopened.set_metadata(state.object, {"intervening": 2})
    assert reopened.save_object(value) == state
    assert reopened.get_metadata(state.object) == {"intervening": 2}
    assert reopened.get_snapshot_metadata(state) == captured
    assert reopened.save_object(value, annotations=SaveAnnotations(object={"retry": 3})) == state
    assert reopened.get_metadata(state.object) == {"retry": 3}
    assert reopened.get_snapshot_metadata(state) == captured


def test_malformed_final_snapshot_is_preserved_and_not_recaptured(tmp_path, monkeypatch):
    """Malformed final-path authority fails closed rather than being overwritten."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    value = FaultPayload("same", repo=repo)
    state = repo.save_object(value)
    metadata_path = store.get_snapshot_directory(state) / "metadata.json"
    original = metadata_path.read_bytes()
    metadata_path.write_bytes(b"not metadata")
    observed = []

    with pytest.raises(RepoSaveError) as raised:
        repo.save_object(value, _snapshot_observer=lambda: observed.append(True))

    assert isinstance(raised.value.__cause__, StoreAuthorityError)
    assert observed == []
    assert metadata_path.read_bytes() == b"not metadata"
    assert original != metadata_path.read_bytes()


def test_zip_commit_interruption_after_replace_reports_completed_and_reopens(tmp_path, monkeypatch):
    """Archive commit read-back retains completed buffered authority after interruption."""

    archive = tmp_path / "store.zip"
    store = ZipStore(archive)
    repo = Repo(store)
    original = store.commit

    def commit_then_interrupt():
        original()
        raise KeyboardInterrupt("archive bookkeeping interruption")

    monkeypatch.setattr(store, "commit", commit_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match="archive bookkeeping") as raised:
        repo.save(FaultPayload(repo=repo), report_stores=True)

    report = raised.value.report
    state = next(item.state_ref for item in report.publications if item.phase == "snapshot")
    assert _phases(report)["commit"] == "completed"
    reopened = ZipStore.open_existing(archive)
    try:
        assert reopened.read_state_ref_record(state.digest()).state_ref == state
    finally:
        reopened.close()


def test_current_metadata_post_publish_failure_retains_mapping_and_dirty_fallback(
        tmp_path, monkeypatch):
    """Post-publication failure exposes complete authority as dirty state."""

    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    state = repo.save_object(FaultPayload(repo=repo))
    target = Path(store._metadata_path(state.object))
    original = filesystem.publish_file

    def fail_after_metadata_publish(source, destination, *, replace=False):
        result = original(source, destination, replace=replace)
        if Path(destination) == target:
            raise OSError("metadata post-publication failure")
        return result

    monkeypatch.setattr(
        filesystem, "publish_file", fail_after_metadata_publish,
    )
    with pytest.raises(OSError, match="metadata post-publication"):
        repo.set_metadata(state.object, {"complete": True})

    assert store.read_metadata(state.object) == {"complete": True}
    assert store.query_index_is_dirty()
    assert DirStore(store.base_dir, query_index="none").read_metadata(state.object) == {"complete": True}


def test_conflicting_copy_and_corrupt_or_expired_sources_do_not_claim_destination(
        tmp_path, monkeypatch):
    """Copy preflight preserves source evidence and withholds target snapshots."""

    first = DirStore(tmp_path / "first")
    conflicting = DirStore(tmp_path / "conflicting")
    repo = Repo([first, conflicting])
    value = FaultPayload(repo=repo)
    state = Repo(first).save_object(value)
    Repo(conflicting).save_object(value)
    first_evidence = first.read_snapshot_metadata(state.digest())
    conflicting_evidence = conflicting.read_snapshot_metadata(state.digest())
    assert first_evidence != conflicting_evidence

    with pytest.raises(RepoSaveError) as raised:
        repo.save_object(value, store=conflicting, source_store=first)
    assert isinstance(raised.value.__cause__, Exception)
    assert first.read_snapshot_metadata(state.digest()) == first_evidence
    assert conflicting.read_snapshot_metadata(state.digest()) == conflicting_evidence

    corrupt_target = DirStore(tmp_path / "corrupt-target")
    manifest = next(first.get_snapshot_directory(state).glob("local-state/*/*/manifest.record"))
    manifest.write_bytes(b"corrupt source manifest")
    with pytest.raises(RepoLoadError, match="complete snapshot-local payload closure"):
        Repo([first, corrupt_target]).fork_state_ref(state, store=corrupt_target, source_store=first)
    assert tuple(corrupt_target.iter_state_ref_records()) == ()

    archive = tmp_path / "source.zip"
    zip_source = ZipStore(archive)
    zip_repo = Repo(zip_source)
    zip_state = zip_repo.save_object(FaultPayload(repo=zip_repo))
    zip_source.commit()
    expired_target = DirStore(tmp_path / "expired-target")
    original_open = zip_source.open_local_state

    def close_after_open(*args, **kwargs):
        borrowed = original_open(*args, **kwargs)
        zip_source.close()
        return borrowed

    monkeypatch.setattr(zip_source, "open_local_state", close_after_open)
    with pytest.raises(RepoLoadError, match="complete snapshot-local payload closure"):
        Repo([zip_source, expired_target]).fork_state_ref(
            zip_state, store=expired_target, source_store=zip_source,
        )
    assert tuple(expired_target.iter_state_ref_records()) == ()
