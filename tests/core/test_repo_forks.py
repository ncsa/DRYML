import pytest

import dryml.environments as envs
from dryml.core import Object, Repo, Serializable, object_namespace
from dryml.core.repo import RepoSaveError
from dryml.core.store.dir import DirStore
from dryml.core.store.records import ClaimRecord


class ForkValue(Serializable):
    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        pass


class SeedWrapper(Object):
    def __init__(self, child):
        self.child = child


def _environment(version):
    """Return deterministic fork provenance for one synthetic process."""

    return envs.EnvironmentRecord(
        python=envs.PythonRecord(version, "CPython"),
        platform=envs.PlatformRecord("Linux", "1", "v", "x86_64", "Linux-x86_64"),
        distributions={},
        dryml=envs.DrymlRuntimeRecord(),
    )


def test_object_and_state_forks_rekey_ids_and_preserve_source_namespace(tmp_path):
    source = DirStore(tmp_path / "source")
    target = DirStore(tmp_path / "target")
    repo = Repo([source, target])
    with object_namespace("source", "run"):
        state = repo.save_object(ForkValue(1, repo=repo), store=source)

    with object_namespace("ignored"):
        object_fork = repo.fork_object_ref(state.object, store=target)
        state_fork = repo.fork_state_ref(state, store=target)

    assert object_fork.object_id != state.object_id
    assert object_fork.object_id.namespace == ("source", "run")
    assert target.read_claim_record(object_fork.digest()) == ClaimRecord(object_fork.digest(), 0, "available")
    assert state_fork.object_id != state.object_id
    assert state_fork.object_id.namespace == ("source", "run")
    assert target.read_state_ref_record(state_fork.digest()).state_ref == state_fork
    assert target.validate_local_state(state_fork, ())


def test_state_fork_rekeys_materializing_seed_references_and_copies_their_records(tmp_path):
    source = DirStore(tmp_path / "source")
    target = DirStore(tmp_path / "target")
    repo = Repo([source, target])
    child = repo.save_object(ForkValue(1, repo=repo), store=source)
    parent = repo.save_object(SeedWrapper(child, repo=repo), store=source)

    fork = repo.fork_state_ref(parent, store=target)

    child_path = next(iter(parent.object.objects))
    assert fork.object.objects[child_path] != parent.object.objects[child_path]
    assert target.read_state_ref_record(fork.digest()).state_ref == fork
    seeds = [record.state_ref for record in target.iter_state_ref_records() if record.state_ref != fork]
    assert len(seeds) == 1
    assert seeds[0].object_id != child.object_id
    assert seeds[0].states == child.states

    # U7 owns actual restoration. U6 proves the target carries the entire
    # authoritative closure that an exact load will require after disconnect.
    target_only = Repo(target)
    target_records = [record.state_ref for record in target.iter_state_ref_records()]
    assert fork in target_records
    assert seeds[0] in target_records
    for reference in target_records:
        for path in reference.states:
            assert target_only.default_store.validate_local_state(reference, path)
    loaded = target_only.load_state_ref(fork, reuse_live="never")
    assert loaded.child.object_id == fork.object.objects[child_path]


def test_fork_copies_verified_snapshot_local_payloads(tmp_path):
    source = DirStore(tmp_path / "source")
    target = DirStore(tmp_path / "target")
    repo = Repo([source, target])
    state = repo.save_object(ForkValue(1, repo=repo), store=source)

    fork = repo.fork_state_ref(state, store=target)

    assert target.read_state_ref_record(fork.digest()).state_ref == fork
    assert target.validate_local_state(fork, ())
    assert source.validate_local_state(state, ())


def test_state_fork_captures_its_own_environment_instead_of_relabeling_source(
        tmp_path, monkeypatch):
    source = DirStore(tmp_path / "source")
    target = DirStore(tmp_path / "target")
    repo = Repo([source, target])
    observed = [_environment("3.12.1")]
    monkeypatch.setattr(
        "dryml.environments.introspection.inspect_current", lambda: observed[0],
    )
    state = repo.save_object(ForkValue(1, repo=repo), store=source)
    source_evidence = source.read_snapshot_metadata(state.digest())
    observed[0] = _environment("3.12.2")

    fork = repo.fork_state_ref(state, store=target, source_store=source)
    fork_evidence = target.read_snapshot_metadata(fork.digest())

    assert source_evidence.environment == _environment("3.12.1")
    assert fork_evidence.environment == _environment("3.12.2")
    assert fork_evidence.saved_at > source_evidence.saved_at
    assert fork_evidence.requirements_coverage == "incomplete"


def test_fork_failure_before_final_boundaries_leaves_no_new_authority(tmp_path, monkeypatch):
    source = DirStore(tmp_path / "source")
    target = DirStore(tmp_path / "target")
    repo = Repo([source, target])
    state = repo.save_object(ForkValue(1, repo=repo), store=source)

    monkeypatch.setattr(
        target,
        "publish_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("final boundary failed")),
    )
    with pytest.raises(RepoSaveError, match="publication") as raised:
        repo.fork_state_ref(state, store=target)

    phases = {item.phase: item.status for item in raised.value.report.publications}
    assert phases == {
        "definition": "completed", "lineage": "completed", "state": "failed",
        "snapshot": "failed", "membership": "unattempted",
    }
    assert tuple(target.iter_state_ref_records()) == ()
    assert tuple(target.iter_declaration_records()) == ()


def test_interruption_after_state_fork_boundary_leaves_complete_discoverable_authority(tmp_path, monkeypatch):
    source = DirStore(tmp_path / "source")
    target = DirStore(tmp_path / "target")
    repo = Repo([source, target])
    state = repo.save_object(ForkValue(1, repo=repo), store=source)
    original = target.publish_snapshot

    def install_then_interrupt(*args, **kwargs):
        original(*args, **kwargs)
        raise KeyboardInterrupt("interrupted after final boundary")

    monkeypatch.setattr(target, "publish_snapshot", install_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match="interrupted after final boundary") as raised:
        repo.fork_state_ref(state, store=target)

    phases = {item.phase: item.status for item in raised.value.report.publications}
    assert phases == {
        "definition": "completed", "lineage": "completed", "state": "completed",
        "snapshot": "completed", "membership": "unattempted",
    }
    records = tuple(target.iter_state_ref_records())
    assert len(records) == 1
    fork = records[0].state_ref
    for path in fork.states:
        assert target.validate_local_state(fork, path)


def test_object_fork_failure_before_declaration_leaves_no_declaration_authority(tmp_path, monkeypatch):
    source = DirStore(tmp_path / "source")
    target = DirStore(tmp_path / "target")
    repo = Repo([source, target])
    state = repo.save_object(ForkValue(1, repo=repo), store=source)

    monkeypatch.setattr(
        target,
        "write_declaration_record",
        lambda record: (_ for _ in ()).throw(OSError("declaration boundary failed")),
    )
    with pytest.raises(RepoSaveError, match="publication") as raised:
        repo.fork_object_ref(state.object, store=target)

    phases = {item.phase: item.status for item in raised.value.report.publications}
    assert phases == {
        "definition": "completed", "lineage": "completed", "membership": "unattempted",
    }
    assert tuple(target.iter_declaration_records()) == ()
