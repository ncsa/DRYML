import pytest

import dryml.environments as envs
from dryml.core import Object, ObjectId, Repo, Serializable, object_namespace
from dryml.core.repo import RepoSaveError
from dryml.core.store.dir import DirStore
from tests.core.test_repo_aliases import _saved_digest


def _rekey(saved):
    """Explicitly rebind owned identities while preserving namespace, aliases and keys."""
    return saved.replace_bindings(identities={path: ObjectId(identity.namespace)
                                             for path, identity in saved.identity_spec().items()})


def _save_rekeyed(repo, saved, target, **kwargs):
    """Build selected state under fresh identities and publish a verified target closure."""
    live = _rekey(saved).build(repo=repo, cache="none")
    return repo.save_object(live, states=saved.state_spec(), store=target, **kwargs)


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


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_object_and_state_forks_rekey_ids_and_preserve_source_namespace(tmp_path):
    source = DirStore(tmp_path / "source")
    target = DirStore(tmp_path / "target")
    repo = Repo([source, target])
    with object_namespace("source", "run"):
        state = repo.save_object(ForkValue(1, repo=repo), store=source)

    with object_namespace("ignored"):
        object_fork = _rekey(state.object_view())
        record = target.publish_node(object_fork)
        state_fork = _save_rekeyed(repo, state, target)

    assert object_fork.object_id != state.object_id
    assert object_fork.object_id.namespace == ("source", "run")
    assert target.read_node(record.digest).graph_equal(object_fork)
    assert state_fork.object_id != state.object_id
    assert state_fork.object_id.namespace == ("source", "run")
    assert target.read_node(_saved_digest(state_fork), kind="saved").graph_equal(state_fork)
    assert Repo(target).load(state_fork).object_id == state_fork.object_id


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_state_fork_rekeys_materializing_seed_references_and_copies_their_records(tmp_path):
    source = DirStore(tmp_path / "source")
    target = DirStore(tmp_path / "target")
    repo = Repo([source, target])
    child = repo.save_object(ForkValue(1, repo=repo), store=source)
    parent = repo.save_object(SeedWrapper(child, repo=repo), store=source)

    fork = _save_rekeyed(repo, parent, target)

    child_path = '$[@param("child")]'
    assert fork.at(child_path).object_id != child.object_id
    assert fork.at(child_path).state == child.state
    assert target.read_node(_saved_digest(fork), kind="saved").graph_equal(fork)
    target_only = Repo(target)
    assert target_only.query().saved().one().graph_equal(fork)
    loaded = target_only.load_state_ref(fork, reuse_live="never")
    assert loaded.child.object_id == fork.at(child_path).object_id


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_fork_copies_verified_snapshot_local_payloads(tmp_path):
    source = DirStore(tmp_path / "source")
    target = DirStore(tmp_path / "target")
    repo = Repo([source, target])
    state = repo.save_object(ForkValue(1, repo=repo), store=source)

    fork = _save_rekeyed(repo, state, target)

    assert target.read_node(_saved_digest(fork), kind="saved").graph_equal(fork)
    assert fork.state.key == state.state.key
    with target.read_state(fork.state.key):
        pass
    with source.read_state(state.state.key):
        pass


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
    source_evidence = repo.get_snapshot_metadata(state, store=source)
    observed[0] = _environment("3.12.2")

    fork = _save_rekeyed(repo, state, target, source_store=source)
    fork_evidence = repo.get_snapshot_metadata(fork, store=target)

    assert source_evidence.environment == _environment("3.12.1")
    assert fork_evidence.environment == _environment("3.12.2")
    assert fork_evidence.saved_at > source_evidence.saved_at
    assert fork_evidence.requirements_coverage == "complete"


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_fork_failure_before_final_boundaries_leaves_no_new_authority(tmp_path, monkeypatch):
    source = DirStore(tmp_path / "source")
    target = DirStore(tmp_path / "target")
    repo = Repo([source, target])
    state = repo.save_object(ForkValue(1, repo=repo), store=source)

    monkeypatch.setattr(
        target,
        "publish_save_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("final boundary failed")),
    )
    with pytest.raises(RepoSaveError, match="publication") as raised:
        _save_rekeyed(repo, state, target)

    phases = {item.phase: item.status for item in raised.value.report.publications}
    assert phases == {"snapshot": "failed"}
    assert not Repo(target).query().saved().exists()
    assert not Repo(target).query().object_refs().exists()


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_interruption_after_state_fork_boundary_leaves_complete_discoverable_authority(tmp_path, monkeypatch):
    source = DirStore(tmp_path / "source")
    target = DirStore(tmp_path / "target")
    repo = Repo([source, target])
    state = repo.save_object(ForkValue(1, repo=repo), store=source)
    original = target.publish_save_batch

    def install_then_interrupt(*args, **kwargs):
        original(*args, **kwargs)
        raise KeyboardInterrupt("interrupted after final boundary")

    monkeypatch.setattr(target, "publish_save_batch", install_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match="interrupted after final boundary") as raised:
        _save_rekeyed(repo, state, target)

    phases = {item.phase: item.status for item in raised.value.report.publications}
    assert phases == {"snapshot": "failed"}
    records = tuple(Repo(target).query().saved().collect())
    assert len(records) == 1
    fork = records[0]
    for _, local in fork.state_spec().items():
        with target.read_state(local.key):
            pass


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_object_fork_failure_before_declaration_leaves_no_declaration_authority(tmp_path, monkeypatch):
    source = DirStore(tmp_path / "source")
    target = DirStore(tmp_path / "target")
    repo = Repo([source, target])
    state = repo.save_object(ForkValue(1, repo=repo), store=source)

    monkeypatch.setattr(
        target,
        "_prepare_and_apply",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("declaration boundary failed")),
    )
    with pytest.raises(OSError, match="declaration boundary failed"):
        target.publish_node(_rekey(state.object_view()))
    assert not Repo(target).query().object_refs().exists()
