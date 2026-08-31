from pathlib import Path

import pytest

from dryml.core import Repo, Serializable
from dryml.core.store.dir import DirStore


class ReservedValue(Serializable):
    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")


@pytest.mark.parametrize("reuse_live", ["matching", "greedy"])
def test_contended_exact_candidate_builds_fresh_without_waiting(tmp_path, reuse_live):
    repo = Repo(DirStore(tmp_path / "store"))
    saved = ReservedValue(1, repo=repo)
    state = repo.save_object(saved)
    saved._save_load_reservation.acquire()
    try:
        loaded = repo.load_state_ref(state, reuse_live=reuse_live)
    finally:
        saved._save_load_reservation.release()

    assert loaded is not saved
    assert loaded.object_id == saved.object_id
