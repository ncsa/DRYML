from pathlib import Path
from abc import abstractmethod, update_abstractmethods

import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from dryml.core import ConcreteDefinition, Definition, Object, Ref, Repo, Serializable
from dryml.core.repo import RepoLoadError
from dryml.core.store.dir import DirStore
from dryml.core.store.records import DefinitionRecord


class PreflightValue(Serializable):
    constructions = 0

    def __init__(self, value):
        type(self).constructions += 1
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")


class PreflightPair(Object):
    def __init__(self, first, second):
        self.first = first
        self.second = second


class ReferenceBarrierRoot(Serializable):
    """Stateful root retaining one CDef as non-materializing constructor data."""

    def __init__(self, target: Ref[ConcreteDefinition]):
        self.target = target

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "value").write_text("root", encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        Path(src_dir, "value").read_text(encoding="ascii")


class MaterializingTarget(Object):
    """Structural wrapper whose child remains in the materializing closure."""

    def __init__(self, child):
        self.child = child


class AbstractablePreflightValue(PreflightValue):
    restores = 0

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        type(self).restores += 1


def test_exact_preflight_rejects_currently_abstract_class_before_greedy_restore(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    saved = AbstractablePreflightValue(1, repo=repo)
    state = repo.save_object(saved)
    repo.pin(saved)

    def required(self):
        """Return the newly required implementation result."""

    AbstractablePreflightValue.required = abstractmethod(required)
    update_abstractmethods(AbstractablePreflightValue)
    try:
        with pytest.raises(TypeError, match="required"):
            repo.load_state_ref(state, reuse_live="greedy")
        with pytest.raises(TypeError, match="required"):
            repo.restore_state_ref_into(saved, state)
    finally:
        del AbstractablePreflightValue.required
        update_abstractmethods(AbstractablePreflightValue)

    assert AbstractablePreflightValue.restores == 0
    assert saved in repo._all_live_candidates()


def test_exact_preflight_reports_missing_authority_before_construction(tmp_path):
    source = DirStore(tmp_path / "source")
    repo = Repo(source)
    saved = PreflightValue(1, repo=repo)
    state = repo.save_object(saved)
    PreflightValue.constructions = 0

    with pytest.raises(RepoLoadError, match="StateRefRecord"):
        Repo(DirStore(tmp_path / "empty")).load_state_ref(state)

    assert PreflightValue.constructions == 0


def test_exact_preflight_reports_every_missing_local_state_before_construction(tmp_path):
    source = DirStore(tmp_path / "source")
    repo = Repo(source)
    first = PreflightValue(1, repo=repo)
    second = PreflightValue(2, repo=repo)
    state = repo.save_object(PreflightPair(first, second, repo=repo))
    payloads = [source.open_local_state(state, path).handle for path in state.states]
    for payload in payloads:
        Path(payload, "data", "value").unlink()
    PreflightValue.constructions = 0

    with pytest.raises(RepoLoadError) as error:
        Repo(DirStore(tmp_path / "source")).load_state_ref(state)

    message = str(error.value)
    assert "Exact StateRef preflight is incomplete" in message
    for state_hash in state.states.values():
        assert state_hash in message
    assert PreflightValue.constructions == 0


def test_exact_preflight_fails_before_live_candidate_reservation(tmp_path):
    source = DirStore(tmp_path / "source")
    repo = Repo(source)
    saved = PreflightValue(1, repo=repo)
    state = repo.save_object(saved)
    path, state_hash = next(iter(state.states.items()))
    Path(source.open_local_state(state, path).handle, "data", "value").unlink()

    class ReservationProbe:
        attempts = 0

        def acquire(self, *, blocking):
            type(self).attempts += 1
            return False

        def release(self):
            raise AssertionError("preflight did not acquire this reservation")

    saved._save_load_reservation = ReservationProbe()

    with pytest.raises(RepoLoadError, match="preflight is incomplete"):
        repo.load_state_ref(state)

    assert ReservationProbe.attempts == 0


def test_exact_preflight_keeps_ref_held_nested_state_refs_inert(tmp_path):
    """Ref CDef data does not require its own declaration or nested snapshot authority."""

    source_repo = Repo(DirStore(tmp_path / "source"))
    missing_state = source_repo.save_object(PreflightValue(1, repo=source_repo))
    reference_target = Definition(MaterializingTarget, missing_state).concretize()
    store = DirStore(tmp_path / "root")
    writer = Repo(store)
    state = writer.save_object(ReferenceBarrierRoot(reference_target, repo=writer))

    loaded = Repo(store).load_state_ref(state, reuse_live="never")

    assert loaded.target.graph_equal(reference_target)


def test_exact_preflight_still_requires_materializing_child_declaration(tmp_path, monkeypatch):
    """A missing declaration on a materializing occurrence fails before construction."""

    store = DirStore(tmp_path / "store")
    writer = Repo(store)
    state = writer.save_object(
        MaterializingTarget(PreflightValue(1, repo=writer), repo=writer),
        deep_capture=True,
    )
    child_definition = next(
        definition
        for path, definition in (
            (path, state.object.at(path).definition) for path in state.object.objects
        )
        if path
    )
    hidden_digest = DefinitionRecord(child_definition).digest
    original_read = store.read_definition_record

    monkeypatch.setattr(
        store,
        "read_definition_record",
        lambda digest: None if digest == hidden_digest else original_read(digest),
    )
    PreflightValue.constructions = 0

    with pytest.raises(RepoLoadError, match="DefinitionRecord"):
        Repo(store).load_state_ref(state, reuse_live="never")

    assert PreflightValue.constructions == 0
