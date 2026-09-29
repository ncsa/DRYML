"""Focused U4 contracts for current and captured Repo metadata authority."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from dryml.core import (
    MetadataConflictError, Object, Repo, SaveAnnotations, Serializable,
    read_snapshot_metadata, save_object,
)
from dryml.core.store.dir import DirStore
from dryml.core.store.zip import ZipStore
from dryml.core.cdef_graph import EdgeKind
from dryml.core.links import DefLink
from dryml.core.utils.graph.path import GraphPath, Index, Parameter


class MetadataValue(Serializable):
    """Small stateful fixture whose payload access is observable in tests."""

    saves = 0

    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, directory, *, codec):
        type(self).saves += 1
        Path(directory, "value").write_text(str(self.value), encoding="ascii")


class MetadataContainer(Object):
    """Stateless fixture with shared materializing and Ref-only children."""

    def __init__(self, children, reference):
        self.children = children
        self.reference = reference


def test_current_metadata_crud_is_scope_local_and_nonmaterializing(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    state = repo.save_object(MetadataValue(1, repo=repo))

    monkeypatch.setattr(repo, "_materialize_cdef", lambda *_args, **_kwargs: pytest.fail("metadata materialized an Object"))
    repo.set_metadata(state.object, {"project": "one", "null": None})
    repo.set_metadata(state, {"score": 1.0})

    assert repo.get_metadata(state.object) == {"project": "one", "null": None}
    assert repo.get_metadata(state) == {"score": 1.0}
    assert repo.delete_metadata(state) is True
    assert repo.delete_metadata(state) is False
    assert repo.get_metadata(state) is None
    assert repo.get_snapshot_metadata(state).captured_object_annotations is None
    assert repo.get_snapshot_metadata(state).captured_state_annotations is None


def test_absent_empty_and_null_round_trip_and_capture_once(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    value = MetadataValue(1, repo=repo)
    state = repo.save_object(
        value,
        annotations=SaveAnnotations(object={}, state={"value": None}),
    )

    assert repo.get_metadata(state.object) == {}
    assert repo.get_metadata(state) == {"value": None}
    captured = repo.get_snapshot_metadata(state)
    assert captured.captured_object_annotations == {}
    assert captured.captured_state_annotations == {"value": None}

    repo.save_object(value, annotations=SaveAnnotations(object={"later": True}))
    assert repo.get_metadata(state.object) == {"later": True}
    assert repo.get_snapshot_metadata(state) == captured

    reopened = Repo(DirStore.open_existing(tmp_path / "store"))
    assert reopened.get_metadata(state.object) == {"later": True}
    assert reopened.get_metadata(state) == {"value": None}


def test_repeated_explicit_annotations_update_current_lww_not_capture(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    value = MetadataValue(1, repo=repo)
    state = repo.save_object(value, annotations=SaveAnnotations(object={"version": 1}))

    repo.save_object(value, annotations=SaveAnnotations(object={"version": 2}))
    repo.save_object(value, annotations=SaveAnnotations(object={"version": 3}))

    assert repo.get_metadata(state.object) == {"version": 3}
    assert repo.get_snapshot_metadata(state).captured_object_annotations == {"version": 1}


def test_metadata_multi_store_conflicts_are_order_independent(tmp_path):
    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    writer = Repo([first, second])
    root = MetadataValue(1, repo=writer)
    state = writer.save_object(root, store=first)
    writer.save_object(root, store=second, source_store=first)

    first_repo = Repo([first, second])
    first_repo.set_metadata(state.object, {"same": []}, store=first)
    first_repo.set_metadata(state.object, {"same": []}, store=second)
    assert first_repo.get_metadata(state.object) == {"same": []}

    first_repo.set_metadata(state.object, {"different": True}, store=second)
    with pytest.raises(MetadataConflictError):
        first_repo.get_metadata(state.object)
    with pytest.raises(MetadataConflictError):
        Repo([second, first]).get_metadata(state.object)
    assert first_repo.get_metadata(state.object, store=first) == {"same": []}
    with pytest.raises(Exception, match="explicit Store"):
        first_repo.set_metadata(state.object, {"ambiguous": True})


def test_save_rejects_conflicting_completed_destination_evidence(tmp_path):
    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    first_repo = Repo(first)
    value = MetadataValue(1, repo=first_repo)
    state = first_repo.save_object(
        value, annotations=SaveAnnotations(object={"captured": "first"}),
    )
    second_repo = Repo(second)
    assert second_repo.save_object(
        value, annotations=SaveAnnotations(object={"captured": "second"}),
    ) == state

    joined = Repo([first, second])
    with pytest.raises(MetadataConflictError):
        joined.save_object(value, store=second)


def test_save_rejects_source_store_entries_outside_its_closure(tmp_path):
    source = DirStore(tmp_path / "source")
    target = DirStore(tmp_path / "target")
    repo = Repo([source, target])
    value = MetadataValue(1, repo=repo)
    unrelated = repo.save_object(MetadataValue(2, repo=repo), store=source)
    repo.save_object(value, store=source)

    with pytest.raises(ValueError, match="outside the saved snapshot closure"):
        repo.save_object(value, store=target, source_stores={unrelated: source})


def test_metadata_rejects_unknown_and_disconnected_targets_before_mutation(tmp_path):
    store = DirStore(tmp_path / "store")
    other = DirStore(tmp_path / "other")
    repo = Repo(store)
    other_repo = Repo(other)
    state = other_repo.save_object(MetadataValue(1, repo=other_repo))

    with pytest.raises(KeyError):
        repo.get_metadata(state)
    with pytest.raises(KeyError):
        repo.set_metadata(state, {})
    with pytest.raises(ValueError, match="connected"):
        repo.get_snapshot_directory(state, store=other)


def test_nested_materializing_object_ref_is_a_metadata_holder_in_closure(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    child = MetadataValue(1, repo=repo)
    reference_only = MetadataValue(2, repo=repo)
    root = MetadataContainer(
        [child, child],
        DefLink.finalized(EdgeKind.REF, reference_only.object_ref),
        repo=repo,
    )
    state = repo.save_object(root, store=store)
    child_path = GraphPath((Parameter("children"), Index(0)))
    alias_path = GraphPath((Parameter("children"), Index(1)))
    target = state.object.at(child_path)

    assert state.object.at(alias_path) == target
    assert store.read_declaration_record(target.digest()) is None
    assert all(record.state_ref.object != target for record in store.iter_state_ref_records())
    repo.set_metadata(target, {"nested": ["shared"]})
    assert repo.get_metadata(target) == {"nested": ["shared"]}
    assert repo.get_lineage_metadata(target).object_ref == target
    with pytest.raises(KeyError):
        repo.get_metadata(reference_only.object_ref)


def test_directory_lookup_and_all_save_wrappers_preserve_return_shapes(tmp_path):
    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    repo = Repo([first, second])

    root = MetadataValue(1, repo=repo)
    state = repo.save_object(root, store=first, annotations=SaveAnnotations(object={"repo": 1}))
    assert repo.get_snapshot_directory(state, store=first) == first.get_snapshot_directory(state)
    assert read_snapshot_metadata(repo.get_snapshot_directory(state, store=first)).captured_object_annotations == {"repo": 1}

    state_from_repo_save = repo.save(root, store=first, annotations=SaveAnnotations(state={}))
    assert state_from_repo_save == state
    assert repo.get_metadata(state, store=first) == {}
    state_from_object = root.save(repo=repo, store=first, annotations=SaveAnnotations(object={"object": 2}))
    assert state_from_object == state
    state_from_global = save_object(root, repo=repo, store=first, annotations=SaveAnnotations(state={"global": 3}))
    assert state_from_global == state
    assert repo.get_metadata(state, store=first) == {"global": 3}

    replica = repo.save_object(root, store=second, source_store=first)
    assert replica == state
    with pytest.raises(Exception, match="ambiguous"):
        repo.get_snapshot_directory(state)


def test_forks_copy_selected_current_annotations_by_value(tmp_path):
    source = DirStore(tmp_path / "source")
    target = DirStore(tmp_path / "target")
    repo = Repo([source, target])
    state = repo.save_object(MetadataValue(1, repo=repo), store=source)
    repo.set_metadata(state.object, {"object": ["source"]}, store=source)
    repo.set_metadata(state, {"state": {"score": 1}}, store=source)

    object_fork = repo.fork_object_ref(
        state.object, store=target, copy_annotations=True, source_store=source,
    )
    state_fork = repo.fork_state_ref(
        state, store=target, copy_annotations=("object", "state"), source_store=source,
    )
    default_object_fork = repo.fork_object_ref(state.object, store=target)
    default_state_fork = repo.fork_state_ref(state, store=target)

    assert repo.get_metadata(object_fork, store=target) == {"object": ["source"]}
    assert repo.get_metadata(state_fork.object, store=target) == {"object": ["source"]}
    assert repo.get_metadata(state_fork, store=target) == {"state": {"score": 1}}
    assert repo.get_snapshot_metadata(state_fork, store=target).captured_state_annotations == {
        "state": {"score": 1}
    }
    copied = repo.get_metadata(state_fork.object, store=target)
    copied["object"].append("changed")
    assert repo.get_metadata(state.object, store=source) == {"object": ["source"]}
    assert repo.get_metadata(default_object_fork, store=target) is None
    assert repo.get_metadata(default_state_fork.object, store=target) is None
    assert repo.get_metadata(default_state_fork, store=target) is None
    assert repo.get_snapshot_metadata(
        default_state_fork, store=target,
    ).captured_object_annotations is None


def test_zip_metadata_mutations_remain_buffered_until_normal_flush(tmp_path):
    archive = tmp_path / "metadata.zip"
    store = ZipStore(archive)
    repo = Repo(store)
    state = repo.save(
        MetadataValue(1, repo=repo), annotations=SaveAnnotations(object={"first": 1}),
    )
    repo.set_metadata(state.object, {"after": 2}, store=store)
    assert repo.get_metadata(state.object, store=store) == {"after": 2}
    repo.close()

    reopened = Repo(ZipStore.open_existing(archive))
    assert reopened.get_metadata(state.object) == {"after": 2}
