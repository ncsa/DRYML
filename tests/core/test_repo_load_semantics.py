import pytest

from dryml.core2 import Definition, Object, Repo, Serializable
from dryml.core2.policies import RepoLoadOptions
from dryml.core2.repo import RepoLoadError
from dryml.core2.store.dir import DirStore


class LoadLeaf(Object):
    prepare_count = 0

    @classmethod
    def __prepare_args__(cls, *args, **kwargs):
        cls.prepare_count += 1
        return args, kwargs

    def __init__(self, name):
        super().__init__()
        self.name = name


class PersistentLoadLeaf(Serializable):
    def __init__(self, name):
        super().__init__()
        self.name = name


def test_repo_load_requires_exact_concrete_definition(tmp_path):
    repo = Repo(stores=DirStore(tmp_path / "store"))
    with pytest.raises(TypeError):
        repo.load(Definition(LoadLeaf, "x"))


def test_load_or_build_concretizes_definition_once(tmp_path):
    repo = Repo(stores=DirStore(tmp_path / "store"))
    LoadLeaf.prepare_count = 0
    definition = Definition(LoadLeaf, "x")

    obj = repo.load_or_build(definition)

    assert obj.name == "x"
    assert LoadLeaf.prepare_count == 1
    assert obj.definition.identity_version == 1


def test_load_or_build_does_not_choose_compatible_sibling(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    existing = LoadLeaf("existing", repo=repo)
    repo.save_object(existing)

    new_def = Definition(LoadLeaf, "new")
    loaded = repo.load_or_build(new_def)

    assert loaded.definition != existing.definition
    assert loaded.name == "new"


def test_repo_load_overrides_build_missing_options(tmp_path):
    repo = Repo(stores=DirStore(tmp_path / "store"))
    cdef = Definition(PersistentLoadLeaf, "missing").concretize(repo=repo)

    with pytest.raises(RepoLoadError, match="Missing stored state"):
        repo.load(cdef, options=RepoLoadOptions(build_missing=True))
