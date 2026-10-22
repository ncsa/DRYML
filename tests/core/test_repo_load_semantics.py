import pytest

from dryml.core import Definition, Object, Repo, Serializable
from dryml.core.cdef_identity import V2_IDENTITY_VERSION
from dryml.core.repo import RepoLoadError
from dryml.core.store.dir import DirStore
from dryml.core.utils.graph.path import GraphPath, Parameter

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")


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


class HookProbeLoadLeaf(Serializable):
    prepare_count = 0

    @classmethod
    def __prepare_args__(cls, *args, **kwargs):
        cls.prepare_count += 1
        return args, kwargs

    def __init__(self, name):
        self.name = name


def test_repo_load_accepts_stateless_saved_metadata_without_capture(tmp_path):
    repo = Repo(stores=DirStore(tmp_path / "store"))
    saved = LoadLeaf("x", repo=repo)
    metadata = repo.save_object(saved)
    loaded = Repo(stores=DirStore(tmp_path / "store")).load(metadata)

    assert loaded.definition.graph_equal(saved.definition)
    assert loaded.name == "x"


def test_load_or_build_concretizes_definition_once(tmp_path):
    repo = Repo(stores=DirStore(tmp_path / "store"))
    LoadLeaf.prepare_count = 0
    definition = Definition(LoadLeaf, "x")

    obj = repo.load_or_build(definition)

    assert obj.name == "x"
    assert LoadLeaf.prepare_count == 0
    assert obj.definition.identity_version == V2_IDENTITY_VERSION


def test_load_or_build_does_not_choose_compatible_sibling(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    existing = LoadLeaf("existing", repo=repo)
    repo.save_object(existing)

    new_def = Definition(LoadLeaf, "new")
    loaded = repo.load_or_build(new_def)

    assert loaded.definition != existing.definition
    assert loaded.name == "new"


def test_repo_load_requires_complete_saved_state_not_structural_fallback(tmp_path):
    repo = Repo(stores=DirStore(tmp_path / "store"))
    cdef = Definition(PersistentLoadLeaf, "missing").concretize(repo=repo)

    with pytest.raises(ValueError, match="Saved completeness"):
        repo.load(cdef)

    loaded = repo.load_or_build(cdef)

    assert loaded.name == "missing"


def test_exact_saved_restore_never_invokes_retired_constructor_hook(tmp_path):
    """Exact StateRef restoration uses stored bound arguments without hooks."""

    repo = Repo(stores=DirStore(tmp_path / "store"))
    HookProbeLoadLeaf.prepare_count = 0
    saved = repo.save_object(HookProbeLoadLeaf("saved", repo=repo))

    restored = Repo(stores=DirStore(tmp_path / "store")).load_state_ref(saved, reuse_live="never")

    assert restored.name == "saved"
    assert HookProbeLoadLeaf.prepare_count == 0


def test_saved_load_missing_selected_data_fails_before_any_constructor_or_hook(tmp_path):
    from pathlib import Path
    from tests.core.test_exact_load_metadata_evidence import EvidenceRoot, EvidenceValue
    store = DirStore(tmp_path / "store")
    writer = Repo(store)
    saved = writer.save(EvidenceRoot(EvidenceValue(1, repo=writer), repo=writer))
    child = saved.at(GraphPath((Parameter("child"),)))
    Path(store.state_data_path(child.state.key), "value").unlink()
    EvidenceValue.constructions = EvidenceValue.restores = 0
    with pytest.raises(RepoLoadError, match="preflight is incomplete"):
        Repo(store).load(saved)
    assert EvidenceValue.constructions == EvidenceValue.restores == 0


def test_repeated_load_revalidates_data_and_never_uses_historical_receipt(tmp_path):
    from pathlib import Path
    from tests.core.test_exact_load_metadata_evidence import EvidenceValue
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    original = EvidenceValue(1, repo=repo)
    saved = repo.save(original)
    repo._evict_live(original)
    EvidenceValue.restores = 0
    first = repo.load_state_ref(saved, reuse_live="never")
    first.value = 99
    second = repo.load_state_ref(saved, reuse_live="matching")
    assert second is first and second.value == 1
    assert EvidenceValue.restores == 2
    Path(store.state_data_path(saved.state.key), "value").unlink()
    with pytest.raises(RepoLoadError, match="preflight is incomplete"):
        repo.load(saved)
    assert not first._restore_failed


def test_independent_equal_node_handles_never_collapse_on_build(tmp_path):
    from tests.core.test_state_ref_restore_into import RestoreIntoValue, RestoreIntoIndependentPair
    repo = Repo(DirStore(tmp_path / "store"))
    first, second = Definition(RestoreIntoValue, 1), Definition(RestoreIntoValue, 1)
    graph = Definition(RestoreIntoIndependentPair, first, second).concretize()
    built = repo.build(graph)
    assert built.first is not built.second
    assert built.first.object_id != built.second.object_id
    loaded = repo.load_state_ref(repo.save(built), reuse_live="never")
    assert loaded.first is not loaded.second
    assert loaded.first.object_id == built.first.object_id
    assert loaded.second.object_id == built.second.object_id
