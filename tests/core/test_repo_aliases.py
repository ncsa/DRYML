import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from tests.core import core_objects as objects
from dryml.core import Serializable
from dryml.core.repo import Repo, make_store
from dryml.core.store.dir import DirStore
from dryml.core.store.records import NodeNameRecord
from dryml.core.store.store import StoreAliasConflictError
from dryml.core.store.store import StoreAuthorityError
from dryml.core.node_prepare import NodePreparation
from dryml.core.query.model import QueryError


def _saved_digest(node):
    """Return the saved-root address, distinct from the graph-exact digest domain."""
    return NodePreparation((node,)).digest(node, domain="dryml-saved-root/v1")


def _publish_named(repo, obj, name):
    """Publish stateless saved metadata and its name through the buffered-capable seam."""
    if isinstance(repo.default_store, DirStore) and repo.default_store.publication_capabilities.durability == "durable":
        return repo.save(obj, alias=name)
    node = repo.describe(obj)
    store = repo.default_store
    root = store.publish_node(node, kind="saved")
    store.compare_and_set_node_name(NodeNameRecord(name, "saved", root.digest), expected=None)
    return node


def _resolve_named(store, name):
    """Read exact named metadata from one explicit Store, without materialization."""
    record = store.read_node_name(name)
    if record is None:
        raise KeyError(name)
    return store.read_node(record.root_digest, kind=record.root_kind)


def _require_writable_reference_store(primary_store_set):
    """Skip file-like Zip fixtures that cannot atomically publish aliases."""
    if not primary_store_set.stores[0].publication_capabilities.writable:
        pytest.skip("file-like ZipStore deliberately rejects mutable reference authority")


def _require_writable(repo):
    if not repo.default_store.publication_capabilities.writable:
        pytest.skip("reference publication requires a writable Store")


class CurrentAliasValue(Serializable):
    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        from pathlib import Path

        Path(dest_dir, "value.txt").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        from pathlib import Path

        self.value = int(Path(src_dir, "value.txt").read_text(encoding="ascii"))


def test_repo_object_alias_resolves_typed_reference_for_structural_load(primary_store_set):
    _require_writable_reference_store(primary_store_set)
    repo = Repo(stores=primary_store_set.stores)
    _require_writable(repo)
    obj = objects.TestClass1(10, test="train")

    saved = _publish_named(repo, obj, "train_data_1")

    reference = _resolve_named(repo.default_store, "train_data_1")

    assert reference.graph_equal(saved)
    assert reference.build(repo=Repo()).definition.graph_equal(obj.definition)


def test_repo_object_alias_persists_across_reopen(primary_store_set):
    _require_writable_reference_store(primary_store_set)
    repo = Repo(stores=primary_store_set.stores)
    _require_writable(repo)
    obj = objects.TestClass1(10, test="train")
    saved = _publish_named(repo, obj, "train_data_1")
    repo.close(flush=True)

    repo2 = Repo(stores=primary_store_set.fresh_stores())
    reference = _resolve_named(repo2.default_store, "train_data_1")
    loaded = reference.build(repo=Repo())

    assert reference.graph_equal(saved)
    assert loaded.definition.graph_equal(obj.definition)
    assert loaded.x == 10
    assert loaded.test == "train"


def test_object_alias_resolution_requires_an_explicit_repo(primary_store_set, tmp_path):
    _require_writable_reference_store(primary_store_set)
    obj = objects.TestClass1(20, test="eval")
    repo = Repo(stores=primary_store_set.stores)
    _publish_named(repo, obj, "eval_data")
    repo.close(flush=True)

    reopened = Repo(stores=primary_store_set.fresh_stores())
    reference = _resolve_named(reopened.default_store, "eval_data")
    loaded = reference.build(repo=Repo())

    assert loaded.definition.graph_equal(obj.definition)
    with pytest.raises(KeyError):
        _resolve_named(DirStore(tmp_path / "unrelated"), "eval_data")

    assert loaded.x == 20
    assert loaded.test == "eval"


def test_repo_state_alias_resolves_an_exact_state_ref(primary_store_set):
    _require_writable_reference_store(primary_store_set)
    repo = Repo(stores=primary_store_set.stores)
    obj = objects.TestClass1(25, test="checkpoint")
    state = _publish_named(repo, obj, "checkpoint")
    repo.close(flush=True)

    reopened = Repo(stores=primary_store_set.fresh_stores())
    resolved = _resolve_named(reopened.default_store, "checkpoint")

    assert resolved.graph_equal(state)
    if reopened.default_store.publication_capabilities.durability == "durable":
        assert reopened.load_state_ref(resolved, reuse_live="never").definition.graph_equal(obj.definition)
    else:
        with pytest.raises(StoreAuthorityError, match="captured metadata"):
            reopened.load(resolved)
        assert resolved.build(repo=Repo()).definition.graph_equal(obj.definition)


def test_repo_alias_deletion_is_not_a_legacy_cdef_mutation(primary_store_set):
    _require_writable_reference_store(primary_store_set)
    repo = Repo(stores=primary_store_set.stores)
    _require_writable(repo)
    obj = objects.TestClass1(30, test="delete")
    saved = _publish_named(repo, obj, "old_alias")
    store = repo.default_store
    old = store.read_node_name("old_alias")
    store.delete_node_name("old_alias", expected=old)
    assert store.read_node_name("old_alias") is None
    assert store.read_node(_saved_digest(saved), kind="saved").graph_equal(saved)


def test_repo_unknown_alias_raises(primary_store_set):
    repo = Repo(stores=primary_store_set.stores)

    assert not repo.query().alias("missing").exists()


def test_repo_rejects_bad_alias_name(primary_store_set):
    repo = Repo(stores=primary_store_set.stores)
    _require_writable(repo)
    obj = objects.TestClass1(40, test="bad")

    from dryml.core.store.records import StoreRecordError
    for alias in (123, ""):
        with pytest.raises((TypeError, ValueError, StoreRecordError)):
            repo.save(obj, alias=alias)


def test_repo_alias_conflict_across_stores_raises(store_resource_factory):
    res1 = store_resource_factory("directory", prefix="alias_conflict_a")
    res2 = store_resource_factory("directory", prefix="alias_conflict_b")

    store1 = make_store(res1.resource)
    store2 = make_store(res2.resource)

    repo1 = Repo(stores=store1)
    repo1.save_object(objects.TestClass1(1, test="a"), alias="shared")
    repo1.close(flush=True)

    repo2 = Repo(stores=store2)
    repo2.save_object(objects.TestClass1(2, test="b"), alias="shared")
    repo2.close(flush=True)

    repo = Repo(stores=[make_store(res1.resource), make_store(res2.resource)])
    with pytest.raises(QueryError, match="conflicts"):
        repo.query().saved().alias("shared").collect()


def test_dirstore_object_alias_replaces_only_its_current_record(tmp_path):
    store_path = tmp_path / "shared-store"
    repo = Repo(stores=DirStore(store_path, query_index="memory"))
    first = repo.save_object(objects.TestClass1(1, test="first"))
    replacement = repo.save_object(objects.TestClass1(3, test="replacement"))

    store = repo.default_store
    old = NodeNameRecord("current", "saved", _saved_digest(first))
    store.compare_and_set_node_name(old, expected=None)
    current = NodeNameRecord("current", "saved", _saved_digest(replacement))
    store.compare_and_set_node_name(current, expected=old)
    with pytest.raises(StoreAliasConflictError):
        store.compare_and_set_node_name(old, expected=old)

    record = DirStore(store_path, query_index="memory").read_node_name("current")
    assert record == current
    assert store.read_node(_saved_digest(first), kind="saved").graph_equal(first)


def test_selected_state_alias_ignores_conflicting_unselected_replica(tmp_path):
    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    first_repo = Repo(first)
    first_state = first_repo.save_object(
        CurrentAliasValue(1, repo=first_repo), deep_capture=True,
    )
    first_repo.add_store(second)
    first_repo.save_object(first_repo.load(first_state), store=second, states=first_state.state_spec(), source_store=first)
    second_repo = Repo(second)
    second_value = second_repo.load_state_ref(first_state, reuse_live="never", cache="none")
    second_value.value = 2
    second_state = second_repo.save_object(second_value, store=second, deep_capture=True)
    first.compare_and_set_node_name(NodeNameRecord("current", "saved", _saved_digest(first_state)), expected=None)
    second.compare_and_set_node_name(NodeNameRecord("current", "saved", _saved_digest(second_state)), expected=None)
    repo = Repo([first, second])

    with pytest.raises(QueryError, match="conflicts"):
        repo.query().saved().alias("current").count()
    assert _resolve_named(first, "current").graph_equal(first_state)
