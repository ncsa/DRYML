from pathlib import Path

from dryml.core import Repo, Serializable
from dryml.core.store.dir import DirStore


class SnapshotDirectoryValue(Serializable):
    def __init__(self, value="payload"):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "value.txt").write_text(self.value, encoding="utf-8")


def test_saved_state_is_complete_snapshot_local_authority(tmp_path):
    store = DirStore(tmp_path / "store")
    state_ref = Repo(store).save_object(SnapshotDirectoryValue())

    snapshot = Path(store.base_dir, "snapshots", state_ref.digest()[:2], state_ref.digest())
    assert snapshot.is_dir()
    assert {path.name for path in snapshot.iterdir()} == {
        "state-ref.record", "placement.json", "metadata.json", "snapshot.json", "local-state",
    }
    assert not Path(store.base_dir, "state-refs").exists()
    assert not Path(store.base_dir, "local-state").exists()
    assert store.get_snapshot_directory(state_ref) == snapshot
