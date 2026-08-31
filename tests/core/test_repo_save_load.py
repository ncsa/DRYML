"""U5 direct-record save, structural-load, and future exact-restore contracts."""

import inspect
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from dryml import session
from dryml.core import Object, ObjectRef, Repo, Serializable, StateRef
from dryml.core.repo import make_store
from dryml.core.repo import RepoSaveError
from dryml.core.store.dir import DirStore
from dryml.core.store.records import DefinitionRecord, MainRefRecord
from dryml.core.store.zip import ZipStore
from dryml.runtime.errors import RuntimeTransitionError


class SaveLoadValue(Serializable):
    saves = 0

    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        type(self).saves += 1
        Path(dest_dir, "value.txt").write_text(str(self.value), encoding="ascii")


class SaveLoadNode(Object):
    def __init__(self, children):
        self.children = children


class RestoreValue(Serializable):
    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "value.txt").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        self.value = int(Path(src_dir, "value.txt").read_text(encoding="ascii"))


class FailingSave(Serializable):
    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "partial.txt").write_text("partial", encoding="ascii")
        raise RuntimeError("serializer failed")


def _run_fresh_process(code: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` with this checkout's source tree first on ``PYTHONPATH``."""

    src = Path(__file__).resolve().parents[2] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(src), env.get("PYTHONPATH", "")))
    return subprocess.run(
        (sys.executable, "-c", code),
        text=True,
        capture_output=True,
        env=env,
    )


def test_public_save_returns_state_ref_and_publishes_local_state(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    obj = SaveLoadValue(10, repo=repo)

    state = obj.save(repo=repo, main=False)

    assert isinstance(state, StateRef)
    assert state.object == obj.object_ref
    assert store.read_state_ref_record(state.digest()).state_ref == state
    path, state_hash = next(iter(state.states.items()))
    assert path in state.object.objects
    assert store.validate_local_state(obj.definition, state_hash)


def test_save_publishes_state_ref_before_main_and_object_alias(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    obj = SaveLoadValue(10, repo=repo)
    calls = []
    write_state_ref = store.write_state_ref_record
    write_main = store.write_main_ref
    write_alias = store.write_object_alias

    monkeypatch.setattr(store, "write_state_ref_record", lambda record: (calls.append("state"), write_state_ref(record))[1])
    monkeypatch.setattr(store, "write_main_ref", lambda record: (calls.append("main"), write_main(record))[1])
    monkeypatch.setattr(store, "write_object_alias", lambda record: (calls.append("alias"), write_alias(record))[1])

    state = repo.save_object(obj, main=True, alias="latest")

    assert calls == ["state", "main", "alias"]
    assert repo.main_def.graph_equal(obj.definition)
    assert store.read_main_ref() == MainRefRecord(DefinitionRecord(obj.definition).digest)
    alias = store.read_object_alias("latest")
    assert isinstance(alias.object_ref, ObjectRef)
    assert alias.object_ref == state.object


def test_shared_nodes_save_once_and_equal_independent_nodes_stay_distinct(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    SaveLoadValue.saves = 0
    shared = SaveLoadValue("same", repo=repo)

    shared_state = SaveLoadNode([shared, shared], repo=repo).save(repo=repo, main=False)

    assert SaveLoadValue.saves == 1
    assert len(shared_state.states) == 1

    SaveLoadValue.saves = 0
    first = SaveLoadValue("same", repo=repo)
    second = SaveLoadValue("same", repo=repo)
    independent_state = SaveLoadNode([first, second], repo=repo).save(repo=repo, main=False)

    assert SaveLoadValue.saves == 2
    assert len(independent_state.states) == 2


def test_save_copies_reusable_state_by_default_and_can_federate_it(tmp_path):
    source = DirStore(tmp_path / "source")
    copied = DirStore(tmp_path / "copied")
    federated = DirStore(tmp_path / "federated")
    repo = Repo([source, copied, federated])
    child = SaveLoadValue("child", repo=repo)
    child_state = repo.save_object(child, store=source)
    root = SaveLoadNode(child, repo=repo)

    copied_state, copied_report = repo.save_object(root, store=copied, report_stores=True)
    copied_path = next(path for path, object_id in copied_state.object.objects.items() if object_id == child.object_id)
    assert copied_report.required_stores == (copied,)
    assert copied.validate_local_state(child.definition, copied_state.states[copied_path])

    federated_state, federated_report = repo.save_object(
        root, store=federated, federated=True, report_stores=True
    )
    federated_path = next(path for path, object_id in federated_state.object.objects.items() if object_id == child.object_id)
    assert federated_state.states[federated_path] == child_state.states[next(iter(child_state.states))]
    assert federated_report.state_stores[federated_path] is source
    assert federated_report.required_stores == (federated, source)


def test_failed_local_state_save_never_publishes_a_state_ref(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    obj = FailingSave(repo=repo)

    with pytest.raises(RepoSaveError, match="local state publication"):
        repo.save_object(obj)

    assert obj._last_state_hash is None
    assert not (tmp_path / "store" / "state-refs").exists()
    staging = tmp_path / "store" / ".staging"
    assert not staging.exists() or not any(staging.iterdir())


def test_orchestrator_guard_rejects_live_save_before_publication(tmp_path):
    store_path = tmp_path / "store"
    completed = _run_fresh_process(
        f"""
from pathlib import Path

from dryml import session
from dryml.core import Repo
from dryml.core.store.dir import DirStore
from dryml.runtime.errors import RuntimeTransitionError
from tests.core.test_repo_save_load import SaveLoadValue


store_path = Path({str(store_path)!r})
repo = Repo(DirStore(store_path))
obj = SaveLoadValue(10, repo=repo)
session.set_mode("orchestrator")
try:
    try:
        repo.save_object(obj)
    except RuntimeTransitionError as error:
        assert "prohibits Object materialization" in str(error)
    else:
        raise AssertionError("orchestrator save unexpectedly materialized an Object")
finally:
    session.reset()

assert not (store_path / "state-refs").exists()
"""
    )

    assert completed.returncode == 0, completed.stderr


def test_changed_save_surface_rejects_retired_revision_options_and_generation_keywords():
    for callable_ in (Repo.save_object, Repo.save, SaveLoadValue.save):
        names = set(inspect.signature(callable_).parameters)
        assert not {"revision", "options", "generation", "ephemeral_depth"} & names

    repo = Repo()
    for keyword, value in (("revision", "retired"), ("options", {}), ("generation", 1)):
        with pytest.raises(TypeError):
            repo.save_object(SaveLoadValue(1), **{keyword: value})


def test_make_store_accepts_a_delegating_binary_file_wrapper():
    with tempfile.NamedTemporaryFile() as wrapped:
        store = make_store(wrapped)
        try:
            assert isinstance(store, ZipStore)
        finally:
            store.close()


def test_direct_layout_has_no_retired_object_or_generation_paths(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    state = repo.save_object(SaveLoadValue(10, repo=repo))

    paths = {path.name for path in Path(store.base_dir).rglob("*")}

    assert store.read_state_ref_record(state.digest()).state_ref == state
    assert "objects" not in paths
    assert ".state-generations" not in paths
    assert ".state-current.pkl" not in paths
    assert "generation" not in " ".join(str(path) for path in Path(store.base_dir).rglob("*"))


def test_structural_load_without_state_restore_builds_from_definition_authority(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    saved = RestoreValue(10, repo=repo)
    saved.value = 20
    repo.save_object(saved)

    loaded = Repo(DirStore(store.base_dir)).load_object(saved.definition)

    assert isinstance(loaded, RestoreValue)
    assert loaded.value == 10


def test_reopen_hydrates_definition_authority_and_discovers_queries_without_materializing(tmp_path):
    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(store)
    first = SaveLoadValue("first", repo=repo)
    second = SaveLoadValue("second", repo=repo)
    repo.save_object(first)
    repo.save_object(second)

    reopened = Repo(DirStore(store.base_dir, query_index="memory"))

    assert reopened.find_defs(None, refresh=False).count() == 0
    assert reopened._num_constructions == 0
    assert set(reopened.find_defs(None, refresh=True)) == {first.definition, second.definition}
    assert reopened._num_constructions == 0
    assert set(reopened.default_store.hydrate_index()) == {first.definition, second.definition}


def test_reopening_main_definition_is_read_only_hydration(tmp_path):
    source = DirStore(tmp_path / "source")
    target = DirStore(tmp_path / "target")
    definition = DefinitionRecord(SaveLoadValue(3).definition)
    source.write_definition_record(definition)
    source.write_main_ref(MainRefRecord(definition.digest))
    before = tuple(Path(target.base_dir).rglob("*"))

    repo = Repo([target, source])

    assert repo.main_def == definition.definition
    assert target.read_main_ref() is None
    assert target.read_definition_record(definition.digest) is None
    assert tuple(Path(target.base_dir).rglob("*")) == before


def test_u7_exact_state_ref_restores_published_local_state(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    saved = RestoreValue(10, repo=repo)
    saved.value = 20
    state = repo.save_object(saved)

    loaded = Repo(DirStore(store.base_dir)).load_state_ref(state, reuse_live="never")

    assert loaded.value == 20


def test_u7_exact_state_ref_restores_nested_independent_local_states(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    first = RestoreValue(1, repo=repo)
    second = RestoreValue(2, repo=repo)
    first.value = 10
    second.value = 20
    state = repo.save_object(SaveLoadNode([SaveLoadNode(first, repo=repo), second], repo=repo))

    loaded = Repo(DirStore(store.base_dir)).load_state_ref(state, reuse_live="never")

    assert loaded.children[0].children.value == 10
    assert loaded.children[1].value == 20


def test_u7_exact_state_ref_restores_shared_nodes_and_reuses_the_exact_graph(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    child = RestoreValue(10, repo=repo)
    child.value = 20
    state = repo.save_object(SaveLoadNode([child, child], repo=repo))
    reopened = Repo(DirStore(store.base_dir))

    first = reopened.load_state_ref(state)
    second = reopened.load_state_ref(state)

    assert first.children[0] is first.children[1]
    assert first.children[0].value == 20
    assert second is not first
    assert second.children[0] is first.children[0]
