import pytest

from dryml.core2 import Definition, Object, Repo, SKIP_ARGS
from dryml.core2.query import QueryDomainError
from dryml.core2.store.store import Store
from dryml.core2.store.dir import DirStore


class QueryRoot(Object):
    def __init__(self, name):
        super().__init__()
        self.name = name


class CountingStore(Store):
    def __init__(self, store):
        self.store = store
        self.has_count = 0
        self.hydrate_count = 0
        self.restore_count = 0

    @property
    def base_dir(self):
        return self.store.base_dir

    @property
    def object_root_dir(self):
        return self.store.object_root_dir

    def has(self, cdef):
        self.has_count += 1
        return self.store.has(cdef)

    def hydrate_index(self):
        self.hydrate_count += 1
        return self.store.hydrate_index()

    def _object_dir(self, cdef):
        return self.store.object_dir(cdef)

    def save_object(self, obj, *, revision=None):
        return self.store.save_object(obj, revision=revision)

    def restore_object(self, obj, *, revision=None):
        self.restore_count += 1
        return self.store.restore_object(obj, revision=revision)

    def read_definition(self, cdef):
        self.has_count += 1
        return self.store.read_definition(cdef)

    def read_main_def(self):
        return self.store.read_main_def()

    def set_main_def(self, main_def):
        return self.store.set_main_def(main_def)

    def write_main_def(self, main_def):
        return self.store.write_main_def(main_def)

    def read_aliases(self):
        return self.store.read_aliases()

    def write_aliases(self, aliases):
        return self.store.write_aliases(aliases)

    def commit(self):
        return self.store.commit()


def test_definition_query_does_not_materialize_until_object_terminal(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = QueryRoot("saved", repo=repo)
    repo.save_object(obj)

    counting_store = CountingStore(DirStore(store.base_dir))
    repo2 = Repo(stores=counting_store)

    defs = repo2.query(Definition(QueryRoot, SKIP_ARGS)).stored().defs()
    assert defs.count() == 1
    assert repo2._num_constructions == 0
    assert counting_store.restore_count == 0

    objs = defs.objects(restore_state=False)
    assert objs.one().name == "saved"
    assert repo2._num_constructions == 1


def test_exact_root_query_uses_store_has_without_full_hydration(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = QueryRoot("saved", repo=repo)
    repo.save_object(obj)

    counting_store = CountingStore(DirStore(store.base_dir))
    repo2 = Repo(stores=counting_store)

    defs = repo2.query(obj.definition).stored().defs()

    assert list(defs) == [obj.definition]
    assert counting_store.has_count >= 1
    assert counting_store.hydrate_count == 0


def test_query_requires_domain_before_execution():
    repo = Repo()
    with pytest.raises(QueryDomainError):
        repo.query(None).defs()


def test_repo_get_rejects_build_missing_and_returns_mapping(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = QueryRoot("saved", repo=repo)
    repo.save_object(obj)

    repo2 = Repo(stores=DirStore(store.base_dir))
    result = repo2.get(Definition(QueryRoot, SKIP_ARGS), restore_state=False)

    assert list(result.keys()) == [obj.definition]
    assert result[obj.definition].name == "saved"
    with pytest.raises(ValueError, match="load_or_build"):
        repo2.get(build_missing=True)
