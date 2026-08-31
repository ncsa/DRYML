import pytest

from dryml.core import Definition, Object, Repo, Serializable
from dryml.core.repo import RepoLoadError
from dryml.core.store.dir import DirStore
from dryml.core.store.records import ClaimRecord, DeclarationRecord, DefinitionRecord


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

    reference = repo.declare_object(DeclaredValue(1).definition, namespace=("run",))

    assert store.read_definition_record(DefinitionRecord(reference.definition).digest)
    assert store.read_claim_record(reference.digest()) == ClaimRecord(reference.digest(), 0, "available")
    assert store.read_declaration_record(reference.digest()) == DeclarationRecord(reference)
    assert reference.object_id.namespace == ("run",)


def test_declaration_rejects_all_ephemeral_graphs(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))

    try:
        repo.declare_object(EphemeralValue().definition)
    except ValueError as error:
        assert "all-ephemeral" in str(error)
    else:
        raise AssertionError("all-ephemeral declarations must be rejected")


def test_definition_failure_creates_no_claim_or_declaration(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    reference = repo._declaration_reference(DeclaredValue(1).definition, None)

    monkeypatch.setattr(
        store,
        "write_definition_record",
        lambda record, **kwargs: (_ for _ in ()).throw(
            OSError("definition failure")
        ),
    )

    with pytest.raises(OSError, match="definition failure"):
        repo._register_declaration(reference, store)

    assert store.read_claim_record(reference.digest()) is None
    assert store.read_declaration_record(reference.digest()) is None


def test_claim_without_declaration_has_no_authority_and_missing_claim_rejects(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    reference = repo._declaration_reference(DeclaredValue(1).definition, None)
    store.write_claim_record(ClaimRecord(reference.digest(), 0, "available"))

    assert repo.find_object_refs(object_id=reference.object_id) == ()
    assert store.read_declaration_record(reference.digest()) is None

    store.write_declaration_record(DeclarationRecord(reference))
    # Simulate a declaration boundary whose preceding claim was lost.
    store._claim_path(reference.digest())
    claim_path = store._claim_path(reference.digest())
    import os
    os.unlink(claim_path)

    with pytest.raises(RepoLoadError, match="ClaimRecord"):
        repo.build_object_ref(reference)


def test_declaration_rejects_graphs_with_only_imported_lineage(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    child = repo.save_object(DeclaredValue(1, repo=repo))
    cdef = Definition(EphemeralWrapper, child.object).concretize(repo=repo)

    with pytest.raises(ValueError, match="no new durable lineage"):
        repo.declare_object(cdef)
