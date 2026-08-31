from pathlib import Path

from dryml.core import Repo, Serializable
from dryml.core.store.dir import DirStore


class StructuralValue(Serializable):
    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        self.value = int(Path(src_dir, "value").read_text(encoding="ascii"))


def test_repo_load_remains_structural_while_exact_load_restores_state(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    saved = StructuralValue(1, repo=repo)
    saved.value = 2
    state = repo.save_object(saved)
    reopened = Repo(DirStore(tmp_path / "store"))

    structural = reopened.load_or_build(saved.definition, instance="new", cache="none")
    exact = reopened.load_state_ref(state, reuse_live="never")

    assert structural.value == 1
    assert structural.object_id != saved.object_id
    assert exact.value == 2
    assert exact.object_id == saved.object_id
