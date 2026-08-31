import pytest

from dryml.core import Definition, Object, Repo, Serializable, StateRef
from dryml.core.repo import RepoLoadError, RepoSaveError
from dryml.core.store.dir import DirStore


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

    repo.set_alias("model", state.object)
    repo.set_state_alias("best", state)

    assert repo.get_alias("model") == state.object
    assert repo.resolve_state_selector(state.object.state("best")) == state


def test_state_selector_is_resolved_before_definition_finalization(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    state = repo.save_object(AliasValue(1, repo=repo))
    repo.set_state_alias("best", state)

    definition = Definition(SelectorHolder, state.object.state("best")).concretize(repo=repo)

    assert isinstance(definition.parameters["selected"], StateRef)
    assert definition.parameters["selected"] == state


def test_state_alias_scope_is_the_complete_ephemeral_aggregate_object_ref(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    child = repo.save_object(AliasValue(1, repo=repo))
    aggregate = repo.save_object(SelectorHolder(child, repo=repo))

    assert aggregate.object.object_id is None
    repo.set_state_alias("best", aggregate)

    assert repo.resolve_state_selector(aggregate.object.state("best")) == aggregate


def test_alias_mutation_requires_same_store_authority_and_unambiguous_target(tmp_path):
    source = DirStore(tmp_path / "source")
    target = DirStore(tmp_path / "target")
    repo = Repo([source, target])
    state = repo.save_object(AliasValue(1, repo=repo), store=source)

    with pytest.raises(RepoLoadError, match="same-Store"):
        repo.set_alias("model", state.object, store=target)
    with pytest.raises(RepoLoadError, match="same-Store"):
        repo.set_state_alias("best", state, store=target)
    with pytest.raises(RepoSaveError, match="exactly one writable"):
        repo.set_alias("model", state.object)


def test_identical_alias_replicas_dedupe_and_conflicts_report_all_targets(tmp_path):
    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    first_repo = Repo(first)
    second_repo = Repo(second)
    first_state = first_repo.save_object(AliasValue(1, repo=first_repo))
    second.write_state_ref_record(type(first).read_state_ref_record(first, first_state.digest()))
    # Definition authority is not needed for alias resolution; replicate the
    # exact StateRef and prove equivalent aliases collapse across Stores.
    second_repo.set_alias("shared", first_state.object)
    first_repo.set_alias("shared", first_state.object)
    repo = Repo([first, second])
    assert repo.get_alias("shared") == first_state.object

    conflicting = second_repo.save_object(AliasValue(2, repo=second_repo))
    second_repo.set_alias("shared", conflicting.object)
    with pytest.raises(RepoLoadError, match="conflicts across connected Stores"):
        repo.get_alias("shared")
