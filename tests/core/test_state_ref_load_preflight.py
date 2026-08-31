from pathlib import Path

import pytest

from dryml.core import Object, Repo, Serializable
from dryml.core.repo import RepoLoadError
from dryml.core.store.dir import DirStore


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
    for path, state_hash in state.states.items():
        definition = state.object.at(path).definition
        local_state = source._local_state_path(definition.graph_hash(), state_hash)
        Path(local_state, "data", "value").unlink()
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
    definition = state.object.at(path).definition
    Path(source._local_state_path(definition.graph_hash(), state_hash), "data", "value").unlink()

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
