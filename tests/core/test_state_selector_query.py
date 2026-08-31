import pytest

from dryml.core import Definition, Repo, Serializable
from dryml.core.store.dir import DirStore


class SelectorQueryValue(Serializable):
    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        pass


class SelectorQueryConsumer(Serializable):
    def __init__(self, selected):
        self.selected = selected

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        pass


def test_query_resolves_soft_state_selector_before_index_access(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    state = repo.save_object(SelectorQueryValue(1, repo=repo))
    repo.set_state_alias("best", state)

    query = repo.query(Definition(SelectorQueryConsumer, state.object.state("best")))

    assert query.selector.parameters["selected"] == state


def test_query_soft_selector_failure_happens_before_domain_execution(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    state = repo.save_object(SelectorQueryValue(1, repo=repo))

    with pytest.raises(KeyError, match="missing"):
        repo.query(Definition(SelectorQueryConsumer, state.object.state("missing")))


def test_query_resolves_one_reused_soft_selector_once_before_index_access(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    state = repo.save_object(SelectorQueryValue(1, repo=repo))
    repo.set_state_alias("best", state)
    selector = state.object.state("best")
    calls = 0
    original = repo.resolve_state_selector

    def resolve(value):
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(repo, "resolve_state_selector", resolve)
    query = repo.query(Definition(SelectorQueryConsumer, [selector, selector]))

    assert calls == 1
    assert tuple(query.selector.parameters["selected"]) == (state, state)
