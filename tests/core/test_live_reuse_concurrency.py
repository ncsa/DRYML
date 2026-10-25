from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from dryml.core import Repo, Serializable
from dryml.core.repo import RepoLoadError
from dryml.core.store.dir import DirStore


class ReservedValue(Serializable):
    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")


@pytest.mark.parametrize("reuse_live", ["matching", "greedy"])
def test_contended_exact_candidate_builds_fresh_without_waiting(tmp_path, reuse_live):
    """Active identity ownership refuses overlap; release permits exact reuse."""
    repo = Repo(DirStore(tmp_path / "store"))
    saved = ReservedValue(1, repo=repo)
    state = repo.save_object(saved)
    saved._save_load_reservation.acquire()
    try:
        with pytest.raises(RepoLoadError, match="reserved"):
            repo.load_state_ref(state, reuse_live=reuse_live)
    finally:
        saved._save_load_reservation.release()

    loaded = repo.load_state_ref(state, reuse_live=reuse_live)
    assert loaded is saved
    assert loaded.object_id == saved.object_id
