"""Derived definition-index integration tests for current Store authority."""

from pathlib import Path

import pytest

from dryml.core import Definition, Object, Repo, Serializable
from dryml.core.query import QueryIndexError
from dryml.core.store.dir import DirStore
from dryml.core.store.records import DefinitionRecord


class IndexLeaf(Serializable):
    """Minimal stateful value used to publish current definition records."""

    def __init__(self, name):
        self.name = name

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        """Leave this value without local payload files."""


class IndexParent(Object):
    """Structural wrapper used to distinguish definition and reference queries."""

    def __init__(self, child):
        self.child = child


def test_definition_query_refreshes_from_definition_records(tmp_path):
    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(store)
    first = IndexLeaf("first", repo=repo)
    second = IndexLeaf("second", repo=repo)
    repo.save_object(first)
    repo.save_object(second)

    reopened = Repo(DirStore(store.base_dir, query_index="memory"))
    results = reopened.query(Definition(IndexLeaf, "first")).stored(refresh=True).defs()

    assert list(results) == [first.definition]
    assert tuple(reopened.default_store.iter_definition_records()) == tuple(
        sorted(
            (DefinitionRecord(first.definition), DefinitionRecord(second.definition)),
            key=lambda record: record.digest,
        )
    )


def test_exact_definition_query_confirms_authoritative_record_after_hash_hit(tmp_path, monkeypatch):
    from dryml.core.definition import ConcreteDefinition

    monkeypatch.setattr(ConcreteDefinition, "stable_hash", lambda self: "0" * 64)
    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(store)
    stored = IndexLeaf("stored", repo=repo)
    queried = IndexLeaf("queried", repo=repo)
    repo.save_object(stored)

    reopened = Repo(DirStore(store.base_dir, query_index="memory"))

    assert reopened.query(queried.definition).stored(refresh=True).count() == 0
    assert reopened.query(stored.definition).stored(refresh=True).count() == 1


def test_deleting_derived_index_rebuilds_without_changing_authority(tmp_path):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    state = Repo(store).save_object(IndexLeaf("indexed"))
    before = tuple(store.iter_definition_records())
    index = store.open_query_index()

    # The sidecar is derived state; removing it must trigger an authority scan.
    assert index is not None
    index.path.unlink()
    reopened = Repo(DirStore(store.base_dir, query_index="sqlite"))

    assert reopened.references().object_id(state.object_id).state_refs().one() == state
    assert tuple(store.iter_definition_records()) == before
    assert reopened.default_store.validate_query_index(thorough=True).ok


def test_reference_results_preserve_exact_occurrences_while_definitions_dedupe(tmp_path):
    repo = Repo(DirStore(tmp_path / "store", query_index="memory"))
    child = repo.save_object(IndexLeaf("shared", repo=repo))
    first = repo.save_object(IndexParent(child, repo=repo))
    second = repo.save_object(IndexParent(child, repo=repo))

    structural = repo.query(Definition(IndexParent, child)).stored(refresh=True).defs()
    occurrences = repo.references().object_id(child.object_id).object_refs().occurrences()
    owners = {
        occurrence.owner
        for occurrence in occurrences
        if occurrence.owner in {first.object, second.object}
    }

    assert list(structural) == [first.definition]
    assert owners == {first.object, second.object}


@pytest.mark.parametrize("query_index", ["memory", "sqlite"])
def test_closure_definition_does_not_become_stored_root_after_reopen(
    tmp_path, query_index
):
    store = DirStore(tmp_path / "store", query_index=query_index)
    repo = Repo(store)
    child = IndexLeaf("closure", repo=repo)
    parent = IndexParent(child, repo=repo)
    repo.save_object(parent, deep_capture=True)

    reopened = Repo(DirStore(store.base_dir, query_index=query_index))
    selector = Definition(IndexLeaf, "closure")

    assert reopened.query(selector).stored(refresh=True).count() == 0
    assert list(reopened.query(selector).nested().definitions().defs()) == [
        child.definition
    ]


def test_reference_query_never_treats_retired_object_roots_as_authority(tmp_path):
    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(store)
    state = repo.save_object(IndexLeaf("current", repo=repo))
    retired = Path(store.base_dir, "objects", "retired")
    retired.mkdir(parents=True)
    (retired / "definition.pkl").write_bytes(b"retired authority")

    reopened = Repo(DirStore(store.base_dir, query_index="memory"))

    assert reopened.references().exact(state.object).object_refs().one() == state.object
    assert list(reopened.query(Definition(IndexLeaf, "retired")).stored(refresh=True).defs()) == []


def test_strict_index_guard_rejects_unindexed_broad_query(tmp_path):
    repo = Repo(DirStore(tmp_path / "store", query_index="sqlite"))
    repo.save_object(IndexLeaf("indexed", repo=repo))

    with pytest.raises(QueryIndexError):
        repo.query(None).stored(refresh=True).require_indexed().count()
