import pytest

from dryml.core import Definition, Object, Repo, SKIP_ARGS
from dryml.core.query import QueryDomainError
from dryml.core.store.dir import DirStore


class QueryRoot(Object):
    def __init__(self, name):
        super().__init__()
        self.name = name


class CountingStore(DirStore):
    def __init__(self, store):
        super().__init__(store.base_dir, query_index=store.query_index)
        self.definition_read_count = 0
        self.hydrate_count = 0
        self.restore_count = 0

    def read_definition_record(self, digest):
        self.definition_read_count += 1
        return super().read_definition_record(digest)

    def catalog_key(self):
        return f"{DirStore.__module__}.{DirStore.__qualname__}:{self.base_dir}"

    def iter_definition_records(self):
        self.hydrate_count += 1
        return super().iter_definition_records()

    def open_local_state(self, graph_hash, state_hash):
        self.restore_count += 1
        return super().open_local_state(graph_hash, state_hash)


def test_definition_query_does_not_materialize_until_object_terminal(tmp_path):
    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(stores=store)
    obj = QueryRoot("saved", repo=repo)
    repo.save_object(obj)

    counting_store = CountingStore(DirStore(store.base_dir, query_index="memory"))
    repo2 = Repo(stores=counting_store)

    defs = repo2.query(Definition(QueryRoot, SKIP_ARGS)).stored().defs()
    assert defs.count() == 1
    assert repo2._num_constructions == 0
    assert counting_store.restore_count == 0

    objs = defs.objects()
    assert objs.one().name == "saved"
    assert repo2._num_constructions == 1


def test_exact_root_query_reads_definition_record_without_full_hydration(tmp_path):
    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(stores=store)
    obj = QueryRoot("saved", repo=repo)
    repo.save_object(obj)

    counting_store = CountingStore(DirStore(store.base_dir, query_index="memory"))
    repo2 = Repo(stores=counting_store)

    defs = repo2.query(obj.definition).stored().defs()

    assert list(defs) == [obj.definition]
    assert counting_store.definition_read_count >= 1
    assert counting_store.hydrate_count == 0


def test_query_requires_domain_before_execution():
    repo = Repo()
    with pytest.raises(QueryDomainError):
        repo.query(None).defs()


def test_query_object_terminal_structurally_loads_stored_definitions(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = QueryRoot("saved", repo=repo)
    repo.save_object(obj)

    repo2 = Repo(stores=DirStore(store.base_dir))
    result = repo2.query(Definition(QueryRoot, SKIP_ARGS)).stored().objects()

    assert result.one().definition == obj.definition
    assert result.one().name == "saved"
