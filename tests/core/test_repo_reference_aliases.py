import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from dryml.core import Definition, Object, Repo, Serializable, require_saved
from dryml.core.store.dir import DirStore
from dryml.core.store.records import NodeNameRecord
from dryml.core.store.store import StoreAuthorityError
from tests.core.test_repo_aliases import _resolve_named, _saved_digest
from dryml.core.query.model import QueryError


class AliasValue(Serializable):
    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        pass


class SelectorHolder(Object):
    def __init__(self, selected):
        self.selected = selected


def test_object_and_state_aliases_are_scoped_by_exact_object_ref(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    state = repo.save_object(AliasValue(1, repo=repo))

    store = repo.default_store
    recipe = store.publish_node(state.object_view())
    store.compare_and_set_node_name(NodeNameRecord("model", "recipe", recipe.digest), expected=None)
    store.compare_and_set_node_name(NodeNameRecord("best", "saved", _saved_digest(state)), expected=None)

    assert _resolve_named(store, "model").graph_equal(state.object_view())
    assert _resolve_named(store, "best").graph_equal(state)


def test_state_selector_is_resolved_before_definition_finalization(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    state = repo.save_object(AliasValue(1, repo=repo))
    store = repo.default_store
    store.compare_and_set_node_name(NodeNameRecord("best", "saved", _saved_digest(state)), expected=None)

    definition = Definition(SelectorHolder, _resolve_named(store, "best")).concretize(repo=repo)

    assert require_saved(definition.at('$[@param("selected")]')).graph_equal(state)


def test_state_alias_scope_is_the_complete_ephemeral_aggregate_object_ref(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    child = repo.save_object(AliasValue(1, repo=repo))
    aggregate = repo.save_object(SelectorHolder(child, repo=repo))

    assert aggregate.object_id is None
    store = repo.default_store
    store.compare_and_set_node_name(NodeNameRecord("best", "saved", _saved_digest(aggregate)), expected=None)

    assert _resolve_named(store, "best").graph_equal(aggregate)
    assert _resolve_named(store, "best").at('$[@param("selected")]').graph_equal(child)


def test_alias_mutation_requires_same_store_authority_and_unambiguous_target(tmp_path):
    source = DirStore(tmp_path / "source")
    target = DirStore(tmp_path / "target")
    repo = Repo([source, target])
    state = repo.save_object(AliasValue(1, repo=repo), store=source)

    with pytest.raises(StoreAuthorityError):
        target.compare_and_set_node_name(NodeNameRecord("best", "saved", _saved_digest(state)), expected=None)
    assert target.read_node_name("best") is None
    assert source.read_node(_saved_digest(state), kind="saved").graph_equal(state)


def test_identical_alias_replicas_dedupe_and_conflicts_report_all_targets(tmp_path):
    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    first_repo = Repo(first)
    second_repo = Repo(second)
    first_state = first_repo.save_object(AliasValue(1, repo=first_repo))
    second.publish_node(first_state, kind="saved")
    record = NodeNameRecord("shared", "saved", _saved_digest(first_state))
    second.compare_and_set_node_name(record, expected=None)
    first.compare_and_set_node_name(record, expected=None)
    repo = Repo([first, second])
    assert repo.query().saved().alias("shared").one().graph_equal(first_state)

    conflicting = second_repo.save_object(AliasValue(2, repo=second_repo))
    second.compare_and_set_node_name(NodeNameRecord("shared", "saved", _saved_digest(conflicting)), expected=record)
    with pytest.raises(QueryError, match="conflicts"):
        repo.query().saved().alias("shared").collect()
