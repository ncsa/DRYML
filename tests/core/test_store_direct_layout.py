"""Direct format-4 roots and explicit membership, independent of state buckets."""

from pathlib import Path

import pytest

from dryml.core import Definition, Object
from dryml.core.store.dir import DirStore
from dryml.core.store.records import NodeMembershipRecord, NodeRootRecord
from dryml.core.store.store import StoreAuthorityError


class DirectLayoutObject(Object):
    pass


def test_node_roots_use_direct_digest_paths_without_object_generations(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    definition = Definition(DirectLayoutObject).concretize()
    record = store.publish_node(definition)
    path = Path(store._node_path("definitions/roots", record.digest))
    assert path.is_file()
    assert NodeRootRecord.from_bytes(path.read_bytes()) == record
    assert store.read_node(record.digest).graph_equal(definition)
    assert not list(Path(store.base_dir).rglob(".state-generations"))
    assert not list(Path(store.base_dir).rglob(".state-current.pkl"))


@pytest.mark.parametrize("consumer", ("query", "rebuild"))
def test_old_query_consumers_refuse_new_authority_without_inventory(tmp_path, monkeypatch, consumer):
    writer = DirStore(tmp_path / "store", query_index="none")
    root = Definition(DirectLayoutObject).concretize()
    writer.publish_node(root)
    store = DirStore(writer.base_dir, query_index="sqlite")
    monkeypatch.setattr(store, "iter_definition_records", lambda: pytest.fail("legacy inventory"))
    if consumer == "rebuild":
        store.rebuild_query_index()
    assert store.query().cdefs().stored().one().graph_equal(root)
    assert Path(store.query_index_path).exists()


@pytest.mark.parametrize("damage", ("definition", "marker", "marker-path", "target", "kind", "legacy"))
def test_direct_reader_preserves_malformed_authority_diagnostics(tmp_path, monkeypatch, damage):
    from dryml.core.store.records import DefinitionRecord
    store = DirStore(tmp_path / "store", query_index="none")
    definition = Definition(DirectLayoutObject).concretize()
    record = store.publish_node(definition)
    definition_path = Path(store._node_path("definitions/roots", record.digest))
    marker_path = Path(store._node_path("membership/recipe", record.digest))
    if damage == "definition":
        definition_path.write_bytes(b"incomplete")
    elif damage == "marker":
        marker_path.write_bytes(b"incomplete")
    elif damage == "marker-path":
        marker_path.write_bytes(NodeMembershipRecord("recipe", "0" * 64).to_bytes())
    elif damage == "target":
        definition_path.unlink()
    elif damage == "kind":
        marker_path.write_bytes(NodeMembershipRecord("saved", record.digest).to_bytes())
    else:
        definition_path.write_bytes(DefinitionRecord(definition).to_bytes())
    files = {path: path.read_bytes() for path in Path(store.base_dir).rglob("*.record")}
    monkeypatch.setattr(store, "read_state", lambda *_a, **_kw: pytest.fail("reader opened payloads"))
    with pytest.raises(StoreAuthorityError):
        store.read_node(record.digest)
    assert files == {path: path.read_bytes() for path in Path(store.base_dir).rglob("*.record")}


def test_bucket_header_does_not_promote_recipe_or_saved_membership(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    root = Definition(DirectLayoutObject).concretize()
    digest = store.install_creation(root)
    assert store.read_creation(digest).graph_equal(root)
    assert store.read_node(digest) is None
    assert store.read_node(digest, kind="saved") is None
    assert not Path(store.base_dir, "membership").exists()
