from pathlib import Path

import pytest

from dryml.core import Repo, Serializable
from dryml.core.store.dir import DirStore


class SnapshotDirectoryValue(Serializable):
    def __init__(self, value="payload"):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "value.txt").write_text(self.value, encoding="utf-8")


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_saved_state_is_complete_snapshot_local_authority(tmp_path):
    store = DirStore(tmp_path / "store")
    state_ref = Repo(store).save_object(SnapshotDirectoryValue(), alias="snapshot")

    assert store.read_node(store.read_node_name("snapshot").root_digest, kind="saved").graph_equal(state_ref)
    key = state_ref.state.key
    bucket = Path(store.base_dir, "states", key.cdef_digest[:2], key.cdef_digest)
    assert (bucket / "def.pkl").is_file()
    assert (bucket / key.codec_hash / "manifest.record").is_file()
    with store.read_state(key) as read:
        assert Path(read.data_directory, "value.txt").read_text() == "payload"
    assert not Path(store.base_dir, "snapshots").exists()
    assert not Path(store.base_dir, "state-refs").exists()
    assert not Path(store.base_dir, "local-state").exists()
    with pytest.raises(TypeError, match="StateRef"):
        store.get_snapshot_directory(state_ref)
