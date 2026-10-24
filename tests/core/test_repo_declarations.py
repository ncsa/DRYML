import pytest

from dryml.core import Definition, Object, ObjectId, Repo, Serializable
from dryml.core.store.dir import DirStore
from dryml.core.store.store import StoreAuthorityError

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")


class DeclaredValue(Serializable):
    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        pass


class EphemeralValue(Object):
    pass


class EphemeralWrapper(Object):
    def __init__(self, child):
        self.child = child


def test_declaration_publishes_definition_claim_then_object_ref(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)

    reference = Definition(DeclaredValue, 1).with_identity(ObjectId(("run",))).concretize()
    record = store.publish_node(reference)
    assert store.read_node(record.digest).graph_equal(reference)
    assert repo.query().object_refs().object_id(reference.object_id).one().graph_equal(reference)
    assert not repo.query().saved().exists()
    assert reference.object_id.namespace == ("run",)


def test_declaration_rejects_all_ephemeral_graphs(tmp_path):
    """Stateless Node membership needs no invented durable identity or claim."""
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    node = Definition(EphemeralValue).concretize()
    record = store.publish_node(node)
    assert store.read_node(record.digest).object_id is None
    assert repo.query().cdefs().stored().one().graph_equal(node)


def test_definition_failure_creates_no_claim_or_declaration(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    reference = Definition(DeclaredValue, 1).with_identity(ObjectId()).concretize()

    monkeypatch.setattr(
        store,
        "_prepare_and_apply",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("definition failure")
        ),
    )

    with pytest.raises(OSError, match="definition failure"):
        store.publish_node(reference)
    assert not repo.query().object_refs().exists()
    assert not repo.query().cdefs().stored().exists()


def test_claim_without_declaration_has_no_authority_and_missing_claim_rejects(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    reference = Definition(DeclaredValue, 1).with_identity(ObjectId()).concretize()
    digest = store.install_creation(reference.creation_view())
    assert store.read_creation(digest).graph_equal(reference.creation_view())
    assert not repo.query().object_refs().exists()
    record = store.publish_node(reference)
    marker = store._node_path("membership/recipe", record.digest)
    from pathlib import Path
    Path(marker).write_bytes(b"incomplete membership")
    with pytest.raises(StoreAuthorityError):
        store.read_node(record.digest)


def test_declaration_rejects_graphs_with_only_imported_lineage(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    child = repo.save_object(DeclaredValue(1, repo=repo))
    cdef = Definition(EphemeralWrapper, child).concretize(repo=repo)
    record = repo.default_store.publish_node(cdef)
    assert repo.default_store.read_node(record.digest).graph_equal(cdef)
    assert cdef.object_id is None
    assert cdef.at('$[@param("child")]').object_id == child.object_id


def test_get_or_declare_reuses_the_selected_store_claimed_declaration(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    cdef = DeclaredValue(1).definition

    reference = cdef.with_identity(ObjectId())
    record = store.publish_node(reference)
    before = store.authority_cut()
    assert store.publish_node(reference) == record
    assert store.authority_cut() == before
    assert repo.query().object_refs().one().graph_equal(reference)
